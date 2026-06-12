//go:build integration

package whisper

import (
	"fmt"
	"os"
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
