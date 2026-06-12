//go:build e2e

package whisper

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

func TestDownloadModelFromHF(t *testing.T) {
	cacheDir := t.TempDir()
	cfg := ModelConfig{CacheDir: &cacheDir}

	repoID, err := resolveModelName("tiny")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Repo: %s", repoID)
	t.Logf("Cache: %s", cacheDir)

	dir, err := downloadModel(repoID, cfg)
	if err != nil {
		t.Fatalf("downloadModel: %v", err)
	}
	t.Logf("Downloaded to: %s", dir)

	if err := validateModelDir(dir); err != nil {
		t.Fatalf("validation failed: %v", err)
	}

	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		info, _ := e.Info()
		t.Logf("  %-30s %s", e.Name(), formatSize(info.Size()))
	}

	t.Log("OK: all required files present")

	// Second call should use cache — no re-download.
	dir2, err := downloadModel(repoID, cfg)
	if err != nil {
		t.Fatalf("second downloadModel: %v", err)
	}
	if dir2 != dir {
		t.Errorf("cache miss: got %q, want %q", dir2, dir)
	}
	if err := validateModelDir(dir2); err != nil {
		t.Fatalf("validation after cache hit: %v", err)
	}
	t.Log("OK: cache hit, no re-download")
}

// TestFullFlow downloads a model, loads it, transcribes a real audio file, and checks the output.
func TestFullFlow(t *testing.T) {
	// JFK "ask not what your country can do for you" — classic Whisper test sample.
	const audioURL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"
	audioPath := downloadTestAudio(t, audioURL)

	samples := decodeWAV(t, audioPath)
	t.Logf("Audio: %d samples (%.1f sec)", len(samples), float64(len(samples))/16000)

	// Load model via our loader (downloads tiny from HF + caches).
	cacheDir := t.TempDir()
	cfg := DefaultModelConfig()
	cfg.CacheDir = &cacheDir

	model, err := Load("tiny", cfg)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer model.Close()
	t.Log("Model loaded")

	// Transcribe.
	tcfg := DefaultTranscribeConfig()
	tcfg.Language = "en"
	result, err := model.Transcribe(context.Background(), samples, tcfg)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}

	t.Logf("Language: %s", result.Language)
	t.Logf("Text:     %s", result.Text)
	for _, seg := range result.Segments {
		t.Logf("  [%s → %s] %s", seg.Start, seg.End, seg.Text)
	}

	if strings.TrimSpace(result.Text) == "" {
		t.Fatal("transcription returned empty text")
	}
	t.Log("OK: full flow works")
}

// downloadTestAudio fetches a WAV file to a temp dir.
func downloadTestAudio(t *testing.T, url string) string {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("download audio: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("download audio: HTTP %s", resp.Status)
	}

	f, err := os.CreateTemp(t.TempDir(), "test-*.wav")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if _, err := io.Copy(f, resp.Body); err != nil {
		t.Fatalf("save audio: %v", err)
	}
	return f.Name()
}

// decodeWAV reads a 16-bit PCM WAV and returns float32 samples normalized to [-1, 1].
// Resamples from source sample rate to 16kHz if needed (simple nearest-neighbor).
func decodeWAV(t *testing.T, path string) []float32 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	var riffID [4]byte
	var fileSize uint32
	var waveID [4]byte
	binary.Read(f, binary.LittleEndian, &riffID)
	binary.Read(f, binary.LittleEndian, &fileSize)
	binary.Read(f, binary.LittleEndian, &waveID)

	if string(riffID[:]) != "RIFF" || string(waveID[:]) != "WAVE" {
		t.Fatal("not a WAV file")
	}

	var sampleRate uint32
	var bitsPerSample uint16
	var numChannels uint16

	var dataSize uint32
	for {
		var chunkID [4]byte
		var chunkSize uint32
		if err := binary.Read(f, binary.LittleEndian, &chunkID); err != nil {
			t.Fatalf("read chunk: %v", err)
		}
		binary.Read(f, binary.LittleEndian, &chunkSize)

		switch string(chunkID[:]) {
		case "fmt ":
			var audioFormat uint16
			binary.Read(f, binary.LittleEndian, &audioFormat)
			binary.Read(f, binary.LittleEndian, &numChannels)
			binary.Read(f, binary.LittleEndian, &sampleRate)
			f.Seek(6, io.SeekCurrent)
			binary.Read(f, binary.LittleEndian, &bitsPerSample)
			if chunkSize > 16 {
				f.Seek(int64(chunkSize-16), io.SeekCurrent)
			}
		case "data":
			dataSize = chunkSize
			goto readData
		default:
			f.Seek(int64(chunkSize), io.SeekCurrent)
		}
	}

readData:
	if bitsPerSample != 16 {
		t.Fatalf("expected 16-bit WAV, got %d-bit", bitsPerSample)
	}
	t.Logf("WAV: %d Hz, %d ch, %d bits, %d bytes data", sampleRate, numChannels, bitsPerSample, dataSize)

	numSamples := int(dataSize) / int(numChannels) / 2
	raw := make([]int16, numSamples*int(numChannels))
	binary.Read(f, binary.LittleEndian, &raw)

	mono := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		var sum float64
		for ch := 0; ch < int(numChannels); ch++ {
			sum += float64(raw[i*int(numChannels)+ch])
		}
		mono[i] = float32(sum / float64(numChannels) / 32768.0)
	}

	if sampleRate == 16000 {
		return mono
	}
	ratio := float64(sampleRate) / 16000.0
	outLen := int(float64(numSamples) / ratio)
	out := make([]float32, outLen)
	for i := range out {
		srcIdx := int(float64(i) * ratio)
		if srcIdx >= len(mono) {
			srcIdx = len(mono) - 1
		}
		out[i] = mono[srcIdx]
	}
	t.Logf("Resampled %d Hz → 16000 Hz (%d → %d samples)", sampleRate, numSamples, outLen)
	return out
}

func formatSize(b int64) string {
	switch {
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(b)/float64(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(b)/float64(1<<10))
	default:
		return fmt.Sprintf("%d B", b)
	}
}
