// Command bench runs the Go faster-whisper port over the shared benchmark
// config and writes results in the same JSON schema as the Python baseline
// (see ../python/bench.py), so the two can be diffed with ../compare.py.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	whisper "github.com/timohahaa/faster-whisper-go"
)

const sampleRate = 16000

type fileSpec struct {
	Path     string `json:"path"`
	Language string `json:"language"`
}

type vadSpec struct {
	Threshold            float32 `json:"threshold"`
	MinSpeechDurationMs  int     `json:"min_speech_duration_ms"`
	MaxSpeechDurationS   float64 `json:"max_speech_duration_s"`
	MinSilenceDurationMs int     `json:"min_silence_duration_ms"`
	SpeechPadMs          int     `json:"speech_pad_ms"`
}

type config struct {
	Model       string `json:"model"`
	Device      string `json:"device"`
	DeviceIndex int    `json:"device_index"`
	ComputeType string `json:"compute_type"`
	CPUThreads  int    `json:"cpu_threads"`
	NumWorkers  int    `json:"num_workers"`

	BatchSize      int  `json:"batch_size"`
	WordTimestamps bool `json:"word_timestamps"`
	VadFilter      bool `json:"vad_filter"`
	Warmup         bool `json:"warmup"`

	BeamSize                  int       `json:"beam_size"`
	BestOf                    int       `json:"best_of"`
	Temperature               []float32 `json:"temperature"`
	CompressionRatioThreshold float32   `json:"compression_ratio_threshold"`
	LogProbThreshold          float32   `json:"log_prob_threshold"`
	NoSpeechThreshold         float32   `json:"no_speech_threshold"`
	ConditionOnPreviousText   bool      `json:"condition_on_previous_text"`

	Vad vadSpec `json:"vad"`

	Datadir string     `json:"datadir"`
	Files   []fileSpec `json:"files"`
}

type wordOut struct {
	Start       float64 `json:"start"`
	End         float64 `json:"end"`
	Word        string  `json:"word"`
	Probability float64 `json:"probability"`
}

type segmentOut struct {
	ID    int       `json:"id"`
	Start float64   `json:"start"`
	End   float64   `json:"end"`
	Text  string    `json:"text"`
	Words []wordOut `json:"words"`
}

type fileResult struct {
	File                string       `json:"file"`
	LanguageRequested   string       `json:"language_requested"`
	LanguageDetected    string       `json:"language_detected"`
	LanguageProbability float64      `json:"language_probability"`
	AudioDurationSec    float64      `json:"audio_duration_sec"`
	TranscribeTimeSec   float64      `json:"transcribe_time_sec"`
	RTF                 float64      `json:"rtf"`
	SpeedupVsRealtime   float64      `json:"speedup_vs_realtime"`
	NumSegments         int          `json:"num_segments"`
	NumWords            int          `json:"num_words"`
	Text                string       `json:"text"`
	Segments            []segmentOut `json:"segments"`
}

type payload struct {
	Implementation string       `json:"implementation"`
	Mode           string       `json:"mode"`
	Model          string       `json:"model"`
	Device         string       `json:"device"`
	DeviceIndex    int          `json:"device_index"`
	ComputeType    string       `json:"compute_type"`
	BatchSize      int          `json:"batch_size"`
	WordTimestamps bool         `json:"word_timestamps"`
	VadFilter      bool         `json:"vad_filter"`
	ModelLoadSec   float64      `json:"model_load_sec"`
	Results        []fileResult `json:"results"`
}

func main() {
	defaultConfig := "../config.json"
	if _, err := os.Stat(defaultConfig); err != nil {
		defaultConfig = "config.json"
	}
	configPath := flag.String("config", defaultConfig, "path to shared benchmark config.json")
	mode := flag.String("mode", "both", "batched | sequential | both")
	outDir := flag.String("out", "../results", "output directory for result JSON files")
	filesFilter := flag.String("files", "", "comma-separated list of file names to run (default: all)")
	flag.Parse()

	cfg, cfgDir, err := loadConfig(*configPath)
	if err != nil {
		fail("load config: %v", err)
	}

	datadir := cfg.Datadir
	if datadir == "" {
		datadir = "../.testdata"
	}
	if !filepath.IsAbs(datadir) {
		datadir = filepath.Join(cfgDir, datadir)
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fail("create out dir: %v", err)
	}

	files := cfg.Files
	if *filesFilter != "" {
		wanted := map[string]bool{}
		for _, w := range strings.Split(*filesFilter, ",") {
			wanted[strings.TrimSpace(w)] = true
		}
		var kept []fileSpec
		for _, f := range files {
			if wanted[f.Path] || wanted[filepath.Base(f.Path)] {
				kept = append(kept, f)
			}
		}
		files = kept
	}

	var modes []string
	switch *mode {
	case "both":
		modes = []string{"batched", "sequential"}
	default:
		modes = []string{*mode}
	}

	fmt.Printf("loading model=%s device=%s compute_type=%s ...\n", cfg.Model, cfg.Device, cfg.ComputeType)
	loadStart := time.Now()
	model, err := whisper.Load(cfg.Model, whisper.ModelConfig{
		Device:      cfg.Device,
		ComputeType: cfg.ComputeType,
		DeviceIndex: []int{cfg.DeviceIndex},
		CPUThreads:  cfg.CPUThreads,
		NumWorkers:  cfg.NumWorkers,
	})
	if err != nil {
		fail("load model: %v", err)
	}
	defer model.Close()
	modelLoadSec := time.Since(loadStart).Seconds()
	fmt.Printf("model loaded in %.2fs\n", modelLoadSec)

	// Read every WAV once up front so file IO is excluded from the timed region.
	decoded := map[string][]float32{}
	for _, f := range files {
		p := filepath.Join(datadir, f.Path)
		samples, err := readWAV(p)
		if err != nil {
			fmt.Printf("  WARN: cannot read %s: %v, skipping\n", p, err)
			continue
		}
		decoded[f.Path] = samples
	}

	ctx := context.Background()

	for _, m := range modes {
		if cfg.Warmup && len(decoded) > 0 {
			var warm []float32
			for _, f := range files {
				if s, ok := decoded[f.Path]; ok {
					warm = s
					break
				}
			}
			if n := sampleRate * 30; len(warm) > n {
				warm = warm[:n]
			}
			fmt.Printf("[%s] warmup ...\n", m)
			_, _ = transcribe(ctx, model, m, warm, "en", cfg)
		}

		var results []fileResult
		for _, f := range files {
			samples, ok := decoded[f.Path]
			if !ok {
				continue
			}
			audioDur := float64(len(samples)) / sampleRate
			fmt.Printf("[%s] transcribing %s (%.1fs) ...\n", m, f.Path, audioDur)

			start := time.Now()
			res, err := transcribe(ctx, model, m, samples, f.Language, cfg)
			elapsed := time.Since(start).Seconds()
			if err != nil {
				fail("transcribe %s: %v", f.Path, err)
			}

			fr := buildFileResult(f, res, audioDur, elapsed)
			results = append(results, fr)
			fmt.Printf("    -> %.2fs (%.1fx realtime, lang=%s, segments=%d)\n",
				fr.TranscribeTimeSec, fr.SpeedupVsRealtime, fr.LanguageDetected, fr.NumSegments)
		}

		out := payload{
			Implementation: "go",
			Mode:           m,
			Model:          cfg.Model,
			Device:         cfg.Device,
			DeviceIndex:    cfg.DeviceIndex,
			ComputeType:    cfg.ComputeType,
			BatchSize:      cfg.BatchSize,
			WordTimestamps: cfg.WordTimestamps,
			VadFilter:      cfg.VadFilter,
			ModelLoadSec:   round(modelLoadSec, 3),
			Results:        results,
		}
		outPath := filepath.Join(*outDir, fmt.Sprintf("go_%s.json", m))
		if err := writeJSON(outPath, out); err != nil {
			fail("write %s: %v", outPath, err)
		}
		fmt.Printf("[%s] wrote %s\n", m, outPath)
	}
}

func transcribe(ctx context.Context, model *whisper.Model, mode string, samples []float32, language string, cfg config) (*whisper.Result, error) {
	tc := whisper.TranscribeConfig{
		Language:                  language,
		BeamSize:                  cfg.BeamSize,
		BestOf:                    cfg.BestOf,
		Temperature:               cfg.Temperature,
		CompressionRatioThreshold: f32ptr(cfg.CompressionRatioThreshold),
		LogProbThreshold:          f32ptr(cfg.LogProbThreshold),
		NoSpeechThreshold:         f32ptr(cfg.NoSpeechThreshold),
		ConditionOnPreviousText:   boolptr(cfg.ConditionOnPreviousText),
		WordTimestamps:            cfg.WordTimestamps,
		VadFilter:                 cfg.VadFilter,
		VadConfig: &whisper.VadConfig{
			Threshold:            cfg.Vad.Threshold,
			MinSpeechDurationMs:  cfg.Vad.MinSpeechDurationMs,
			MaxSpeechDurationS:   cfg.Vad.MaxSpeechDurationS,
			MinSilenceDurationMs: cfg.Vad.MinSilenceDurationMs,
			SpeechPadMs:          cfg.Vad.SpeechPadMs,
		},
		BatchSize: cfg.BatchSize,
	}
	if mode == "batched" {
		return model.TranscribeBatched(ctx, samples, tc)
	}
	return model.Transcribe(ctx, samples, tc)
}

func buildFileResult(f fileSpec, res *whisper.Result, audioDur, elapsed float64) fileResult {
	segs := make([]segmentOut, 0, len(res.Segments))
	numWords := 0
	var textParts []string
	for _, s := range res.Segments {
		var words []wordOut
		for _, w := range s.Words {
			words = append(words, wordOut{
				Start:       round(w.Start.Seconds(), 3),
				End:         round(w.End.Seconds(), 3),
				Word:        w.Word,
				Probability: round(float64(w.Probability), 4),
			})
		}
		numWords += len(words)
		segs = append(segs, segmentOut{
			ID:    s.ID,
			Start: round(s.Start.Seconds(), 3),
			End:   round(s.End.Seconds(), 3),
			Text:  s.Text,
			Words: words,
		})
		textParts = append(textParts, strings.TrimSpace(s.Text))
	}

	rtf := 0.0
	speedup := 0.0
	if audioDur > 0 {
		rtf = elapsed / audioDur
	}
	if elapsed > 0 {
		speedup = audioDur / elapsed
	}

	return fileResult{
		File:                f.Path,
		LanguageRequested:   f.Language,
		LanguageDetected:    res.Info.Language,
		LanguageProbability: round(float64(res.Info.LanguageProbability), 4),
		AudioDurationSec:    round(audioDur, 3),
		TranscribeTimeSec:   round(elapsed, 3),
		RTF:                 round(rtf, 4),
		SpeedupVsRealtime:   round(speedup, 2),
		NumSegments:         len(segs),
		NumWords:            numWords,
		Text:                strings.TrimSpace(strings.Join(textParts, " ")),
		Segments:            segs,
	}
}

func loadConfig(path string) (config, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return config{}, "", err
	}
	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return config{}, "", err
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return config{}, "", err
	}
	return cfg, filepath.Dir(abs), nil
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// round rounds half away from zero, matching the Python baseline's rnd().
func round(v float64, places int) float64 {
	p := math.Pow(10, float64(places))
	return math.Round(v*p) / p
}

func f32ptr(v float32) *float32 { return &v }
func boolptr(v bool) *bool      { return &v }

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}

// readWAV reads a 16-bit PCM mono 16kHz WAV file and returns float32 samples in
// [-1, 1). Input must already be in this format.
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
			if sampleR != sampleRate {
				return nil, fmt.Errorf("unsupported sample rate %d (want %d)", sampleR, sampleRate)
			}
			if bps != 16 {
				return nil, fmt.Errorf("unsupported bits per sample %d (want 16)", bps)
			}
			nSamples := int(chunkSize) / 2
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
