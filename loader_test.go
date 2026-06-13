package whisper

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveModelName(t *testing.T) {
	tests := []struct {
		input   string
		want    string
		wantErr bool
	}{
		{"tiny", "Systran/faster-whisper-tiny", false},
		{"tiny.en", "Systran/faster-whisper-tiny.en", false},
		{"base", "Systran/faster-whisper-base", false},
		{"base.en", "Systran/faster-whisper-base.en", false},
		{"small", "Systran/faster-whisper-small", false},
		{"small.en", "Systran/faster-whisper-small.en", false},
		{"medium", "Systran/faster-whisper-medium", false},
		{"medium.en", "Systran/faster-whisper-medium.en", false},
		{"large-v1", "Systran/faster-whisper-large-v1", false},
		{"large-v2", "Systran/faster-whisper-large-v2", false},
		{"large-v3", "Systran/faster-whisper-large-v3", false},
		{"Systran/faster-whisper-large-v3", "Systran/faster-whisper-large-v3", false},
		{"someorg/custom-model", "someorg/custom-model", false},
		{"nonexistent", "", true},
		{"", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := resolveModelName(tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("resolveModelName(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("resolveModelName(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestResolveToken(t *testing.T) {
	t.Run("from config", func(t *testing.T) {
		tok := "cfg-token"
		cfg := ModelConfig{Token: &tok}
		if got := resolveToken(cfg); got != "cfg-token" {
			t.Errorf("got %q, want %q", got, "cfg-token")
		}
	})

	t.Run("from env", func(t *testing.T) {
		t.Setenv("HF_TOKEN", "env-token")
		cfg := ModelConfig{}
		if got := resolveToken(cfg); got != "env-token" {
			t.Errorf("got %q, want %q", got, "env-token")
		}
	})

	t.Run("empty", func(t *testing.T) {
		t.Setenv("HF_TOKEN", "")
		cfg := ModelConfig{}
		if got := resolveToken(cfg); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})

	t.Run("config overrides env", func(t *testing.T) {
		t.Setenv("HF_TOKEN", "env-token")
		tok := "cfg-token"
		cfg := ModelConfig{Token: &tok}
		if got := resolveToken(cfg); got != "cfg-token" {
			t.Errorf("got %q, want %q", got, "cfg-token")
		}
	})

	t.Run("explicit empty config ignores env", func(t *testing.T) {
		t.Setenv("HF_TOKEN", "env-token")
		tok := ""
		cfg := ModelConfig{Token: &tok}
		if got := resolveToken(cfg); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
}

func TestResolveCacheDir(t *testing.T) {
	t.Run("from config", func(t *testing.T) {
		dir := "/custom/cache"
		cfg := ModelConfig{CacheDir: &dir}
		got, err := resolveCacheDir(cfg, "Systran/faster-whisper-tiny")
		if err != nil {
			t.Fatal(err)
		}
		want := filepath.Join("/custom/cache", "Systran", "faster-whisper-tiny")
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("from XDG", func(t *testing.T) {
		t.Setenv("XDG_CACHE_HOME", "/xdg/cache")
		cfg := ModelConfig{}
		got, err := resolveCacheDir(cfg, "Systran/faster-whisper-tiny")
		if err != nil {
			t.Fatal(err)
		}
		want := filepath.Join("/xdg/cache", "faster-whisper-go", "Systran", "faster-whisper-tiny")
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("fallback to home", func(t *testing.T) {
		t.Setenv("XDG_CACHE_HOME", "")
		cfg := ModelConfig{}
		got, err := resolveCacheDir(cfg, "org/model")
		if err != nil {
			t.Fatal(err)
		}
		home, _ := os.UserHomeDir()
		want := filepath.Join(home, ".cache", "faster-whisper-go", "org", "model")
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("config overrides XDG", func(t *testing.T) {
		t.Setenv("XDG_CACHE_HOME", "/xdg/cache")
		dir := "/custom"
		cfg := ModelConfig{CacheDir: &dir}
		got, err := resolveCacheDir(cfg, "org/model")
		if err != nil {
			t.Fatal(err)
		}
		want := filepath.Join("/custom", "org", "model")
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

func TestValidateModelDir(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		dir := t.TempDir()
		for _, f := range requiredFiles {
			os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o644)
		}
		if err := validateModelDir(dir); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("missing all", func(t *testing.T) {
		dir := t.TempDir()
		err := validateModelDir(dir)
		if err == nil {
			t.Fatal("expected error")
		}
		for _, f := range requiredFiles {
			if !contains(err.Error(), f) {
				t.Errorf("error %q should mention %q", err, f)
			}
		}
	})

	t.Run("missing one", func(t *testing.T) {
		dir := t.TempDir()
		os.WriteFile(filepath.Join(dir, "model.bin"), []byte("x"), 0o644)
		os.WriteFile(filepath.Join(dir, "config.json"), []byte("x"), 0o644)
		err := validateModelDir(dir)
		if err == nil {
			t.Fatal("expected error")
		}
		if !contains(err.Error(), "tokenizer.json") {
			t.Errorf("error %q should mention tokenizer.json", err)
		}
	})
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func newTestHFServer(t *testing.T, files map[string]string, token string) *httptest.Server {
	t.Helper()

	type sibling struct {
		Filename string `json:"rfilename"`
	}

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if token != "" {
			if got := r.Header.Get("Authorization"); got != "Bearer "+token {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
		}

		// API endpoint: /api/models/{org}/{repo}
		if len(r.URL.Path) > len("/api/models/") && r.URL.Path[:len("/api/models/")] == "/api/models/" {
			var siblings []sibling
			for name := range files {
				siblings = append(siblings, sibling{Filename: name})
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"siblings": siblings})
			return
		}

		// Download endpoint: /{org}/{repo}/resolve/main/{filename}
		parts := splitPath(r.URL.Path)
		if len(parts) >= 5 && parts[len(parts)-3] == "resolve" && parts[len(parts)-2] == "main" {
			filename := parts[len(parts)-1]
			if content, ok := files[filename]; ok {
				w.Write([]byte(content))
				return
			}
			http.NotFound(w, r)
			return
		}

		http.NotFound(w, r)
	}))
}

func splitPath(p string) []string {
	var parts []string
	for _, s := range filepath.SplitList(p) {
		for _, part := range split(s, '/') {
			if part != "" {
				parts = append(parts, part)
			}
		}
	}
	return parts
}

func split(s string, sep byte) []string {
	var parts []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == sep {
			parts = append(parts, s[start:i])
			start = i + 1
		}
	}
	parts = append(parts, s[start:])
	return parts
}

func TestFetchRepoFiles(t *testing.T) {
	t.Run("required and optional", func(t *testing.T) {
		srv := newTestHFServer(t, map[string]string{
			"model.bin":                "weights",
			"tokenizer.json":          "{}",
			"config.json":             "{}",
			"vocabulary.txt":          "vocab",
			"vocabulary.json":         "{}",
			"preprocessor_config.json": "{}",
			"README.md":               "readme",
		}, "")
		defer srv.Close()

		old := hfBaseURL
		hfBaseURL = srv.URL
		defer func() { hfBaseURL = old }()

		files, err := fetchRepoFiles("org/repo", "")
		if err != nil {
			t.Fatal(err)
		}

		want := map[string]bool{
			"model.bin": true, "tokenizer.json": true, "config.json": true,
			"vocabulary.txt": true, "vocabulary.json": true, "preprocessor_config.json": true,
		}
		if len(files) != len(want) {
			t.Fatalf("got %d files, want %d", len(files), len(want))
		}
		for _, f := range files {
			if !want[f] {
				t.Errorf("unexpected file %q", f)
			}
		}
	})

	t.Run("only required", func(t *testing.T) {
		srv := newTestHFServer(t, map[string]string{
			"model.bin":       "weights",
			"tokenizer.json":  "{}",
			"config.json":     "{}",
			"vocabulary.txt":  "vocab",
		}, "")
		defer srv.Close()

		old := hfBaseURL
		hfBaseURL = srv.URL
		defer func() { hfBaseURL = old }()

		files, err := fetchRepoFiles("org/repo", "")
		if err != nil {
			t.Fatal(err)
		}
		if len(files) != 4 {
			t.Fatalf("got %d files, want 4", len(files))
		}
	})

	t.Run("missing required file", func(t *testing.T) {
		srv := newTestHFServer(t, map[string]string{
			"tokenizer.json": "{}",
			"config.json":    "{}",
			"vocabulary.txt": "vocab",
		}, "")
		defer srv.Close()

		old := hfBaseURL
		hfBaseURL = srv.URL
		defer func() { hfBaseURL = old }()

		_, err := fetchRepoFiles("org/repo", "")
		if err == nil {
			t.Fatal("expected error for missing model.bin")
		}
		if !contains(err.Error(), "model.bin") {
			t.Errorf("error %q should mention model.bin", err)
		}
	})

	t.Run("auth token sent", func(t *testing.T) {
		srv := newTestHFServer(t, map[string]string{
			"model.bin":      "w",
			"tokenizer.json": "{}",
			"config.json":    "{}",
			"vocabulary.txt": "vocab",
		}, "secret")
		defer srv.Close()

		old := hfBaseURL
		hfBaseURL = srv.URL
		defer func() { hfBaseURL = old }()

		_, err := fetchRepoFiles("org/repo", "wrong")
		if err == nil {
			t.Fatal("expected auth error")
		}

		files, err := fetchRepoFiles("org/repo", "secret")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(files) != 4 {
			t.Fatalf("got %d files, want 4", len(files))
		}
	})
}

func TestDownloadFile(t *testing.T) {
	t.Run("downloads file", func(t *testing.T) {
		srv := newTestHFServer(t, map[string]string{
			"model.bin": "fake-weights-data",
		}, "")
		defer srv.Close()

		old := hfBaseURL
		hfBaseURL = srv.URL
		defer func() { hfBaseURL = old }()

		dir := t.TempDir()
		err := downloadFile("org/repo", "model.bin", dir, "")
		if err != nil {
			t.Fatal(err)
		}

		data, err := os.ReadFile(filepath.Join(dir, "model.bin"))
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != "fake-weights-data" {
			t.Errorf("got %q, want %q", data, "fake-weights-data")
		}
	})

	t.Run("skips existing", func(t *testing.T) {
		srv := newTestHFServer(t, map[string]string{
			"model.bin": "new-data",
		}, "")
		defer srv.Close()

		old := hfBaseURL
		hfBaseURL = srv.URL
		defer func() { hfBaseURL = old }()

		dir := t.TempDir()
		os.WriteFile(filepath.Join(dir, "model.bin"), []byte("old-data"), 0o644)

		err := downloadFile("org/repo", "model.bin", dir, "")
		if err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(filepath.Join(dir, "model.bin"))
		if string(data) != "old-data" {
			t.Errorf("file was overwritten: got %q, want %q", data, "old-data")
		}
	})

	t.Run("404 error", func(t *testing.T) {
		srv := newTestHFServer(t, map[string]string{}, "")
		defer srv.Close()

		old := hfBaseURL
		hfBaseURL = srv.URL
		defer func() { hfBaseURL = old }()

		err := downloadFile("org/repo", "missing.bin", t.TempDir(), "")
		if err == nil {
			t.Fatal("expected error for missing file")
		}
	})
}

func TestDownloadModel(t *testing.T) {
	srv := newTestHFServer(t, map[string]string{
		"model.bin":       "weights",
		"tokenizer.json":  `{"model":{"vocab":{}}}`,
		"config.json":     `{}`,
		"vocabulary.txt":  "vocab",
		"vocabulary.json": `{}`,
	}, "")
	defer srv.Close()

	old := hfBaseURL
	hfBaseURL = srv.URL
	defer func() { hfBaseURL = old }()

	cacheDir := t.TempDir()
	cfg := ModelConfig{CacheDir: &cacheDir}

	dir, err := downloadModel("org/repo", cfg)
	if err != nil {
		t.Fatal(err)
	}

	for _, f := range []string{"model.bin", "tokenizer.json", "config.json", "vocabulary.txt", "vocabulary.json"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Errorf("expected %s to exist: %v", f, err)
		}
	}

	if err := validateModelDir(dir); err != nil {
		t.Errorf("validation failed after download: %v", err)
	}
}
