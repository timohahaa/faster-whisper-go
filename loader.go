package whisper

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

var (
	// Short model name → full Hugging Face repo ID.
	modelAliases = map[string]string{
		"tiny":      "Systran/faster-whisper-tiny",
		"tiny.en":   "Systran/faster-whisper-tiny.en",
		"base":      "Systran/faster-whisper-base",
		"base.en":   "Systran/faster-whisper-base.en",
		"small":     "Systran/faster-whisper-small",
		"small.en":  "Systran/faster-whisper-small.en",
		"medium":    "Systran/faster-whisper-medium",
		"medium.en": "Systran/faster-whisper-medium.en",
		"large-v1":  "Systran/faster-whisper-large-v1",
		"large-v2":  "Systran/faster-whisper-large-v2",
		"large-v3":  "Systran/faster-whisper-large-v3",
	}

	requiredFiles = []string{"model.bin", "tokenizer.json", "config.json"}
	optionalFiles = []string{"vocabulary.json", "preprocessor_config.json"}

	hfBaseURL = "https://huggingface.co"
)

// resolveModelName converts a short name or full repo ID into a HF repo ID.
// Short names (e.g. "large-v3") are looked up in modelAliases.
// Strings with "/" are treated as full repo IDs. Anything else is an error.
func resolveModelName(name string) (string, error) {
	if repo, ok := modelAliases[name]; ok {
		return repo, nil
	}
	if strings.Contains(name, "/") {
		return name, nil
	}
	return "", fmt.Errorf("unknown model name %q; use a short name (e.g. \"large-v3\") or a full repo ID (e.g. \"Systran/faster-whisper-large-v3\")", name)
}

// resolveToken returns the HF auth token: from config if set, otherwise from HF_TOKEN env.
func resolveToken(cfg ModelConfig) string {
	if cfg.Token != nil {
		return *cfg.Token
	}
	return os.Getenv("HF_TOKEN")
}

// resolveCacheDir builds the local cache path for a given repo.
// Priority: cfg.CacheDir → $XDG_CACHE_HOME/faster-whisper-go → ~/.cache/faster-whisper-go.
// The repoID (e.g. "Systran/faster-whisper-large-v3") becomes a subdirectory.
func resolveCacheDir(cfg ModelConfig, repoID string) (string, error) {
	var base string
	if cfg.CacheDir != nil {
		base = *cfg.CacheDir
	} else if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
		base = filepath.Join(xdg, "faster-whisper-go")
	} else {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve cache dir: %w", err)
		}
		base = filepath.Join(home, ".cache", "faster-whisper-go")
	}
	return filepath.Join(base, filepath.FromSlash(repoID)), nil
}

// validateModelDir checks that the directory contains all required model files.
func validateModelDir(dir string) error {
	var missing []string
	for _, f := range requiredFiles {
		if _, err := os.Stat(filepath.Join(dir, f)); errors.Is(err, os.ErrNotExist) {
			missing = append(missing, f)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required files in %s: %s", dir, strings.Join(missing, ", "))
	}
	return nil
}

// fetchRepoFiles queries the HF API for the file list in a repo,
// then returns only the files we need (required + optional if present).
func fetchRepoFiles(repoID, token string) ([]string, error) {
	url := hfBaseURL + "/api/models/" + repoID
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch model info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HF API returned %s for %s", resp.Status, repoID)
	}

	var info struct {
		Siblings []struct {
			Filename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("parse model info: %w", err)
	}

	// Build a set of filenames available in the remote repo.
	remote := make(map[string]bool, len(info.Siblings))
	for _, s := range info.Siblings {
		remote[s.Filename] = true
	}

	var files []string
	for _, f := range requiredFiles {
		if !remote[f] {
			return nil, fmt.Errorf("required file %q not found in repo %s", f, repoID)
		}
		files = append(files, f)
	}
	for _, f := range optionalFiles {
		if remote[f] {
			files = append(files, f)
		}
	}
	return files, nil
}

// downloadFile downloads a single file from HF to destDir.
// Skips download if the file already exists in cache.
func downloadFile(repoID, filename, destDir, token string) error {
	// Already cached — skip.
	dest := filepath.Join(destDir, filename)
	if _, err := os.Stat(dest); err == nil {
		return nil
	}

	url := hfBaseURL + "/" + repoID + "/resolve/main/" + filename
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", filename, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %s", filename, resp.Status)
	}

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}

	// Write to a temp file first, then rename — avoids partial files on failure.
	tmp, err := os.CreateTemp(destDir, filename+".*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer func() {
		tmp.Close()
		os.Remove(tmpPath)
	}()

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		return fmt.Errorf("download %s: %w", filename, err)
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	return os.Rename(tmpPath, dest)
}

// downloadModel downloads all required (and available optional) model files
// from HF into the local cache directory. Returns the path to that directory.
func downloadModel(repoID string, cfg ModelConfig) (string, error) {
	token := resolveToken(cfg)

	dir, err := resolveCacheDir(cfg, repoID)
	if err != nil {
		return "", err
	}

	files, err := fetchRepoFiles(repoID, token)
	if err != nil {
		return "", err
	}

	for _, f := range files {
		if err := downloadFile(repoID, f, dir, token); err != nil {
			return "", err
		}
	}

	return dir, nil
}
