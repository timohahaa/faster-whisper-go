package silerovad

import (
	"fmt"
	"math"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// streamSampleRate is the audio sample rate the Silero VAD model runs at.
// Callers must feed 16kHz mono float32 audio to a Stream (resample upstream).
const streamSampleRate = 16000

// windowDurationSec is the wall-clock duration of one 512-sample VAD window.
var windowDurationSec = float64(windowSamples) / float64(streamSampleRate)

// EventType enumerates streaming VAD endpointing events.
type EventType int

const (
	// EventStartOfSpeech is emitted once speech has been sustained for at least
	// MinSpeechDurationMs.
	EventStartOfSpeech EventType = iota + 1
	// EventEndOfSpeech is emitted once silence has been sustained for at least
	// MinSilenceDurationMs after speech (or on Flush while speaking). Its Samples
	// field carries the full speech segment (prefix padding + speech + trailing
	// silence) at 16kHz, ready to hand to a transcriber.
	EventEndOfSpeech
)

// Event is a streaming VAD endpointing event. Times are relative to the start
// of the stream.
type Event struct {
	Type      EventType
	Samples   []float32     // EndOfSpeech only: the speech segment (16kHz mono f32)
	StartTime time.Duration // approximate start of the emitted segment
	EndTime   time.Duration // time at which the event was decided
	Prob      float32       // last (smoothed) speech probability
}

// StreamConfig controls streaming VAD endpointing. Zero-value fields are filled
// with sensible defaults (Silero/LiveKit values).
type StreamConfig struct {
	// Threshold is the activation probability: a frame at or above it counts as
	// speech. Default: 0.5.
	Threshold float32
	// NegThreshold is the deactivation probability while already speaking: a
	// frame at or below it counts as silence. 0 means auto = max(Threshold-0.15, 0.01).
	NegThreshold float32
	// MinSpeechDurationMs is the minimum sustained speech before StartOfSpeech.
	// Default: 50.
	MinSpeechDurationMs int
	// MinSilenceDurationMs is the minimum sustained silence before EndOfSpeech.
	// Default: 550.
	MinSilenceDurationMs int
	// SpeechPadMs is the lead-in (prefix) padding kept before speech and carried
	// into the emitted segment. Default: 500.
	SpeechPadMs int
	// MaxBufferedSpeechS caps the buffered speech segment length. Default: 60.
	MaxBufferedSpeechS float64
	// SmoothingAlpha is the EMA factor applied to raw probabilities
	// (y = a*x + (1-a)*prev). Default: 0.35. A negative value disables smoothing.
	SmoothingAlpha float32
}

func (c *StreamConfig) applyDefaults() {
	if c.Threshold == 0 {
		c.Threshold = 0.5
	}
	if c.NegThreshold == 0 {
		c.NegThreshold = float32(math.Max(float64(c.Threshold)-0.15, 0.01))
	}
	if c.MinSpeechDurationMs == 0 {
		c.MinSpeechDurationMs = 50
	}
	if c.MinSilenceDurationMs == 0 {
		c.MinSilenceDurationMs = 550
	}
	if c.SpeechPadMs == 0 {
		c.SpeechPadMs = 500
	}
	if c.MaxBufferedSpeechS == 0 {
		c.MaxBufferedSpeechS = 60
	}
	if c.SmoothingAlpha == 0 {
		c.SmoothingAlpha = 0.35
	}
}

// Stream is a per-source streaming VAD endpointer. It holds its own LSTM state
// (over the VAD's shared session) plus the endpointing state machine and a
// rolling speech buffer.
//
// A Stream is NOT safe for concurrent use by multiple goroutines: drive one
// Stream from a single goroutine (e.g. one per audio track). Different Streams
// created from the same VAD may run concurrently.
type Stream struct {
	vad *VAD
	cfg StreamConfig

	// per-stream LSTM state tensors (session is shared on vad).
	hIn, cIn, hOut, cOut *ort.Tensor[float32]

	context  [contextSamples]float32 // trailing samples of the previous window
	inputRow []float32               // reused context+window row (len 576)
	pending  []float32               // leftover (<512) between Push calls

	// rolling speech buffer (16kHz); speechIdx is the write cursor.
	speechBuf     []float32
	speechIdx     int
	prefixSamples int
	maxBufSamples int

	speaking     bool
	speechDur    float64 // seconds of sustained speech
	silenceDur   float64 // seconds of sustained silence
	segStartTime time.Duration
	totalSamples int // total samples processed (for timestamps)

	ema     float32
	emaInit bool
}

// NewStream creates a streaming endpointer over this VAD's shared session.
func (v *VAD) NewStream(cfg StreamConfig) (_ *Stream, err error) {
	if v == nil || v.session == nil {
		return nil, fmt.Errorf("silerovad: VAD is closed")
	}
	cfg.applyDefaults()

	s := &Stream{
		vad:      v,
		cfg:      cfg,
		inputRow: make([]float32, contextSamples+windowSamples),
	}
	defer func() {
		if err != nil {
			s.Close()
		}
	}()

	stateShape := ort.NewShape(1, 1, stateLen)
	if s.hIn, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		return nil, fmt.Errorf("alloc h tensor: %w", err)
	}
	if s.cIn, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		return nil, fmt.Errorf("alloc c tensor: %w", err)
	}
	if s.hOut, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		return nil, fmt.Errorf("alloc hn tensor: %w", err)
	}
	if s.cOut, err = ort.NewEmptyTensor[float32](stateShape); err != nil {
		return nil, fmt.Errorf("alloc cn tensor: %w", err)
	}
	// Ensure the initial state is zeroed (start of a fresh sequence).
	clearFloat32(s.hIn.GetData())
	clearFloat32(s.cIn.GetData())

	s.prefixSamples = cfg.SpeechPadMs * streamSampleRate / 1000
	s.maxBufSamples = int(cfg.MaxBufferedSpeechS*float64(streamSampleRate)) + s.prefixSamples
	s.speechBuf = make([]float32, s.maxBufSamples)

	return s, nil
}

// Push feeds an arbitrary-length chunk of 16kHz mono float32 audio. Audio is
// buffered into 512-sample windows internally; any endpointing events produced
// while consuming this chunk are returned in order.
func (s *Stream) Push(samples []float32) ([]Event, error) {
	if s == nil || s.vad == nil {
		return nil, fmt.Errorf("silerovad: stream is closed")
	}

	// Prepend any leftover from the previous Push.
	if len(s.pending) > 0 {
		s.pending = append(s.pending, samples...)
		samples = s.pending
		s.pending = nil
	}

	var events []Event
	i := 0
	for ; i+windowSamples <= len(samples); i += windowSamples {
		ev, err := s.processWindow(samples[i : i+windowSamples])
		if err != nil {
			return events, err
		}
		if ev != nil {
			events = append(events, *ev)
		}
	}

	// Retain the sub-window remainder (copied so we don't alias caller memory).
	if rem := len(samples) - i; rem > 0 {
		s.pending = append(s.pending[:0], samples[i:]...)
	}
	return events, nil
}

// processWindow runs one 512-sample window through the model and advances the
// endpointing state machine, returning an event if one was produced.
func (s *Stream) processWindow(window []float32) (*Event, error) {
	// Build the context+window row and run inference.
	copy(s.inputRow[:contextSamples], s.context[:])
	copy(s.inputRow[contextSamples:], window)

	p, err := s.vad.runStep(s.inputRow, s.hIn, s.cIn, s.hOut, s.cOut)
	if err != nil {
		return nil, err
	}

	// Carry the trailing context for the next window.
	copy(s.context[:], window[windowSamples-contextSamples:])

	// Optional EMA smoothing.
	if s.cfg.SmoothingAlpha > 0 {
		if !s.emaInit {
			s.ema, s.emaInit = p, true
		} else {
			a := s.cfg.SmoothingAlpha
			s.ema = a*p + (1-a)*s.ema
		}
		p = s.ema
	}

	s.writeToBuf(window)
	s.totalSamples += windowSamples

	speechLike := p >= s.cfg.Threshold || (s.speaking && p > s.cfg.NegThreshold)

	if speechLike {
		s.speechDur += windowDurationSec
		s.silenceDur = 0

		if !s.speaking && s.speechDur >= float64(s.cfg.MinSpeechDurationMs)/1000 {
			s.speaking = true
			// The buffer currently holds prefix padding + the confirmed speech.
			segLen := s.speechIdx
			s.segStartTime = s.timestamp() - samplesToDuration(segLen)
			return &Event{
				Type:      EventStartOfSpeech,
				StartTime: s.segStartTime,
				EndTime:   s.timestamp(),
				Prob:      p,
			}, nil
		}
		return nil, nil
	}

	// Silence.
	s.silenceDur += windowDurationSec
	s.speechDur = 0

	if !s.speaking {
		// Not speaking: keep only the rolling prefix padding.
		s.resetWriteCursor()
		return nil, nil
	}

	if s.silenceDur >= float64(s.cfg.MinSilenceDurationMs)/1000 {
		s.speaking = false
		ev := s.emitEnd(p)
		s.resetWriteCursor()
		return ev, nil
	}
	return nil, nil
}

// Flush forces an EndOfSpeech if a speech segment is currently open (e.g. the
// track ended). Returns nil when not speaking.
func (s *Stream) Flush() ([]Event, error) {
	if s == nil || s.vad == nil {
		return nil, fmt.Errorf("silerovad: stream is closed")
	}
	if !s.speaking {
		return nil, nil
	}
	s.speaking = false
	ev := s.emitEnd(0)
	s.resetWriteCursor()
	s.speechDur, s.silenceDur = 0, 0
	return []Event{*ev}, nil
}

// emitEnd builds an EndOfSpeech event from the current speech buffer.
func (s *Stream) emitEnd(prob float32) *Event {
	seg := make([]float32, s.speechIdx)
	copy(seg, s.speechBuf[:s.speechIdx])
	end := s.timestamp()
	return &Event{
		Type:      EventEndOfSpeech,
		Samples:   seg,
		StartTime: end - samplesToDuration(len(seg)),
		EndTime:   end,
		Prob:      prob,
	}
}

// writeToBuf appends a window to the speech buffer, up to the max cap.
func (s *Stream) writeToBuf(window []float32) {
	space := len(s.speechBuf) - s.speechIdx
	n := windowSamples
	if n > space {
		n = space
	}
	if n > 0 {
		copy(s.speechBuf[s.speechIdx:s.speechIdx+n], window[:n])
		s.speechIdx += n
	}
	// n == 0: MaxBufferedSpeech reached; further audio for this segment is dropped.
}

// resetWriteCursor keeps only the trailing prefix padding at the buffer start,
// so the next speech segment includes a lead-in.
func (s *Stream) resetWriteCursor() {
	if s.speechIdx <= s.prefixSamples {
		return
	}
	copy(s.speechBuf[:s.prefixSamples], s.speechBuf[s.speechIdx-s.prefixSamples:s.speechIdx])
	s.speechIdx = s.prefixSamples
}

func (s *Stream) timestamp() time.Duration {
	return samplesToDuration(s.totalSamples)
}

func samplesToDuration(n int) time.Duration {
	return time.Duration(float64(n) / float64(streamSampleRate) * float64(time.Second))
}

// Close releases the per-stream state tensors. The shared VAD session is not
// affected. Safe to call multiple times.
func (s *Stream) Close() error {
	if s == nil {
		return nil
	}
	if s.hIn != nil {
		s.hIn.Destroy()
		s.hIn = nil
	}
	if s.cIn != nil {
		s.cIn.Destroy()
		s.cIn = nil
	}
	if s.hOut != nil {
		s.hOut.Destroy()
		s.hOut = nil
	}
	if s.cOut != nil {
		s.cOut.Destroy()
		s.cOut = nil
	}
	s.vad = nil
	return nil
}
