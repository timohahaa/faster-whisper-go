package whisper

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

var knownModels = map[string]string{
	"tiny.en":          "Systran/faster-whisper-tiny.en",
	"tiny":             "Systran/faster-whisper-tiny",
	"base.en":          "Systran/faster-whisper-base.en",
	"base":             "Systran/faster-whisper-base",
	"small.en":         "Systran/faster-whisper-small.en",
	"small":            "Systran/faster-whisper-small",
	"medium.en":        "Systran/faster-whisper-medium.en",
	"medium":           "Systran/faster-whisper-medium",
	"large-v1":         "Systran/faster-whisper-large-v1",
	"large-v2":         "Systran/faster-whisper-large-v2",
	"large-v3":         "Systran/faster-whisper-large-v3",
	"large":            "Systran/faster-whisper-large-v3",
	"distil-large-v2":  "Systran/faster-distil-whisper-large-v2",
	"distil-medium.en": "Systran/faster-distil-whisper-medium.en",
	"distil-small.en":  "Systran/faster-distil-whisper-small.en",
	"distil-large-v3":  "Systran/faster-distil-whisper-large-v3",
	"large-v3-turbo":   "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
	"turbo":            "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

var modelFiles = []string{
	"model.bin",
	"config.json",
	"tokenizer.json",
	"preprocessor_config.json",
	"vocabulary.json",
}

// AvailableModels returns the list of known model size names.
func AvailableModels() []string {
	names := make([]string, 0, len(knownModels))
	for k := range knownModels {
		names = append(names, k)
	}
	return names
}

func resolveModelPath(sizeOrPath string, cfg ModelConfig) (string, error) {
	if info, err := os.Stat(sizeOrPath); err == nil && info.IsDir() {
		return sizeOrPath, nil
	}

	repoID, ok := knownModels[sizeOrPath]
	if !ok {
		return "", fmt.Errorf(
			"invalid model size %q, expected one of: %s, or a path to a local model directory",
			sizeOrPath, strings.Join(AvailableModels(), ", "),
		)
	}

	cacheDir := cfg.CacheDir
	if cacheDir == "" {
		cacheDir = defaultCacheDir()
	}

	modelDir := filepath.Join(cacheDir, repoDirName(repoID))

	if cfg.LocalFilesOnly {
		if !modelFilesExist(modelDir) {
			return "", fmt.Errorf(
				"model %q not found in cache %q and local_files_only is set",
				sizeOrPath, modelDir,
			)
		}
		return modelDir, nil
	}

	if modelFilesExist(modelDir) {
		return modelDir, nil
	}

	if err := downloadModel(repoID, modelDir); err != nil {
		return "", fmt.Errorf("download model %q: %w", sizeOrPath, err)
	}

	return modelDir, nil
}

func defaultCacheDir() string {
	if dir := os.Getenv("XDG_CACHE_HOME"); dir != "" {
		return filepath.Join(dir, "faster-whisper-go")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "faster-whisper-go")
}

// repoDirName converts "Systran/faster-whisper-large-v3" to "models--Systran--faster-whisper-large-v3"
func repoDirName(repoID string) string {
	return "models--" + strings.ReplaceAll(repoID, "/", "--")
}

func modelFilesExist(dir string) bool {
	for _, f := range modelFiles {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return false
		}
	}
	return true
}

func downloadModel(repoID, destDir string) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}

	for _, filename := range modelFiles {
		dest := filepath.Join(destDir, filename)
		if _, err := os.Stat(dest); err == nil {
			continue
		}

		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, filename)
		if err := downloadFile(url, dest); err != nil {
			if filename == "vocabulary.json" {
				// Some models use vocabulary.txt instead.
				altURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/vocabulary.txt", repoID)
				altDest := filepath.Join(destDir, "vocabulary.txt")
				if err2 := downloadFile(altURL, altDest); err2 == nil {
					continue
				}
			}
			os.Remove(dest)
			return fmt.Errorf("download %s: %w", filename, err)
		}
	}

	return nil
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}

	tmp := dest + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}

	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}

	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}

	return os.Rename(tmp, dest)
}
