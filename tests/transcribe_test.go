package tests

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	whisper "github.com/timohahaa/faster-whisper-go"
)

const testdataDir = ".testdata"

func TestTranscribeWAV(t *testing.T) {
	wavPath := filepath.Join(testdataDir, "test.wav")

	samples, err := readWAV(wavPath)
	if err != nil {
		t.Fatalf("readWAV(%q): %v", wavPath, err)
	}
	if len(samples) == 0 {
		t.Fatal("WAV file contains no samples")
	}
	t.Logf("loaded %d samples (%.2fs of audio)", len(samples), float64(len(samples))/16000)

	model, err := whisper.Load("tiny", whisper.DefaultModelConfig())
	if err != nil {
		t.Fatalf("Load model: %v", err)
	}
	defer model.Close()

	cfg := whisper.DefaultTranscribeConfig()
	cfg.Language = "en"

	result, err := model.Transcribe(context.Background(), samples, cfg)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}

	if result.Text == "" {
		t.Error("expected non-empty transcription text")
	}
	if len(result.Segments) == 0 {
		t.Error("expected at least one segment")
	}

	t.Logf("transcription: %s", result.Text)
	t.Logf("segments: %d, language: %s (prob=%.2f)",
		len(result.Segments), result.Info.Language, result.Info.LanguageProbability)
}

// readWAV reads a 16-bit PCM mono 16kHz WAV file and returns float32 samples in [-1, 1].
// Handles WAV files with arbitrary chunks between "fmt " and "data".
func readWAV(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var riffHeader struct {
		ID   [4]byte
		Size uint32
		Type [4]byte
	}
	if err := binary.Read(f, binary.LittleEndian, &riffHeader); err != nil {
		return nil, fmt.Errorf("read RIFF header: %w", err)
	}
	if string(riffHeader.ID[:]) != "RIFF" || string(riffHeader.Type[:]) != "WAVE" {
		return nil, fmt.Errorf("not a RIFF/WAVE file")
	}

	var (
		fmtFound  bool
		audioFmt  uint16
		nChannels uint16
		sampleR   uint32
		bps       uint16
		dataSize  uint32
	)

	for {
		var chunkID [4]byte
		var chunkSize uint32
		if err := binary.Read(f, binary.LittleEndian, &chunkID); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("read chunk ID: %w", err)
		}
		if err := binary.Read(f, binary.LittleEndian, &chunkSize); err != nil {
			return nil, fmt.Errorf("read chunk size: %w", err)
		}

		switch string(chunkID[:]) {
		case "fmt ":
			var fmtChunk struct {
				AudioFormat   uint16
				NumChannels   uint16
				SampleRate    uint32
				ByteRate      uint32
				BlockAlign    uint16
				BitsPerSample uint16
			}
			if err := binary.Read(f, binary.LittleEndian, &fmtChunk); err != nil {
				return nil, fmt.Errorf("read fmt chunk: %w", err)
			}
			audioFmt = fmtChunk.AudioFormat
			nChannels = fmtChunk.NumChannels
			sampleR = fmtChunk.SampleRate
			bps = fmtChunk.BitsPerSample
			fmtFound = true
			if extra := int64(chunkSize) - 16; extra > 0 {
				if _, err := f.Seek(extra, io.SeekCurrent); err != nil {
					return nil, fmt.Errorf("skip fmt extra bytes: %w", err)
				}
			}

		case "data":
			if !fmtFound {
				return nil, fmt.Errorf("data chunk before fmt chunk")
			}
			if audioFmt != 1 {
				return nil, fmt.Errorf("unsupported audio format %d (want PCM=1)", audioFmt)
			}
			if nChannels != 1 {
				return nil, fmt.Errorf("unsupported channels %d (want mono=1)", nChannels)
			}
			if sampleR != 16000 {
				return nil, fmt.Errorf("unsupported sample rate %d (want 16000)", sampleR)
			}
			if bps != 16 {
				return nil, fmt.Errorf("unsupported bits per sample %d (want 16)", bps)
			}
			dataSize = chunkSize
			nSamples := int(dataSize) / 2
			raw := make([]int16, nSamples)
			if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
				return nil, fmt.Errorf("read PCM data: %w", err)
			}
			samples := make([]float32, nSamples)
			for i, s := range raw {
				samples[i] = float32(s) / 32768.0
			}
			return samples, nil

		default:
			if _, err := f.Seek(int64(chunkSize), io.SeekCurrent); err != nil {
				return nil, fmt.Errorf("skip chunk %q: %w", chunkID, err)
			}
		}
	}

	return nil, fmt.Errorf("no data chunk found")
}
