// Command vadbench measures pyannote VAD (onnxruntime, CPU) latency on a 16kHz
// mono WAV across different intra-op thread counts. It isolates the VAD stage so
// its cost can be compared against the rest of the transcription pipeline.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/timohahaa/faster-whisper-go/pyannotevad"
)

func main() {
	wavPath := flag.String("wav", "", "path to 16kHz mono s16le WAV")
	threadsCSV := flag.String("threads", "1,2,4,8", "comma-separated intra-op thread counts to test")
	reps := flag.Int("reps", 3, "timed repetitions per thread count")
	dump := flag.Bool("dump", false, "dump score coverage/regions at onset threshold instead of timing")
	onset := flag.Float64("onset", 0.5, "onset threshold for -dump coverage")
	flag.Parse()

	if *wavPath == "" {
		fmt.Fprintln(os.Stderr, "error: -wav is required")
		os.Exit(2)
	}

	samples, err := readWav(*wavPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read wav:", err)
		os.Exit(1)
	}
	durSec := float64(len(samples)) / 16000.0
	fmt.Printf("wav=%s samples=%d duration=%.1fs NumCPU=%d\n", *wavPath, len(samples), durSec, runtime.NumCPU())

	var counts []int
	for _, s := range strings.Split(*threadsCSV, ",") {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		n, err := strconv.Atoi(s)
		if err != nil {
			fmt.Fprintln(os.Stderr, "bad thread count:", s)
			os.Exit(2)
		}
		counts = append(counts, n)
	}

	if *dump {
		vad, err := pyannotevad.NewWithThreads(4)
		if err != nil {
			fmt.Fprintln(os.Stderr, "new vad:", err)
			os.Exit(1)
		}
		defer vad.Close()
		sc, err := vad.Scores(samples)
		if err != nil {
			fmt.Fprintln(os.Stderr, "scores:", err)
			os.Exit(1)
		}
		fs := pyannotevad.FrameStep
		// Score histogram.
		var hist [10]int
		above := 0
		for _, s := range sc {
			b := int(s * 10)
			if b > 9 {
				b = 9
			}
			if b < 0 {
				b = 0
			}
			hist[b]++
			if float64(s) >= *onset {
				above++
			}
		}
		fmt.Printf("frames=%d resolution=%.5fs onset=%.3f frames>=onset=%d (%.1f%%)\n",
			len(sc), fs, *onset, above, 100*float64(above)/float64(len(sc)))
		fmt.Print("score hist [0..1]:")
		for i, h := range hist {
			fmt.Printf(" %.1f:%d", float64(i)/10, h)
		}
		fmt.Println()
		// Merge frames >= onset into coarse regions (>0.2s gap splits).
		var regs [][2]float64
		inReg := false
		var start float64
		var lastActive float64
		for i, s := range sc {
			t := float64(i) * fs
			if float64(s) >= *onset {
				if !inReg {
					start = t
					inReg = true
				}
				lastActive = t
			} else if inReg && t-lastActive > 0.2 {
				regs = append(regs, [2]float64{start, lastActive})
				inReg = false
			}
		}
		if inReg {
			regs = append(regs, [2]float64{start, lastActive})
		}
		var covered float64
		for _, r := range regs {
			covered += r[1] - r[0]
		}
		fmt.Printf("simple-threshold coverage: %.1fs across %d regions (audio=%.1fs)\n", covered, len(regs), durSec)

		// Replicate binarizeVadScores (hysteresis + min-cut @30s) to isolate
		// whether the chunking stage collapses coverage.
		ch := binarize(sc, fs, float32(*onset), 0.363, 30.0)
		var chCov float64
		for _, c := range ch {
			chCov += c[1] - c[0]
		}
		fmt.Printf("hysteresis binarize (onset=%.3f offset=0.363 max=30s): %d chunks, coverage=%.1fs\n", *onset, len(ch), chCov)
		fmt.Println("binarize regions [start-end]:")
		for _, c := range ch {
			fmt.Printf("  %7.1f - %7.1f  (%.1fs)\n", c[0], c[1], c[1]-c[0])
		}
		return
	}

	fmt.Printf("%-8s %-10s %-12s %-10s\n", "threads", "vad_sec", "xRealtime", "frames")
	for _, n := range counts {
		vad, err := pyannotevad.NewWithThreads(n)
		if err != nil {
			fmt.Fprintln(os.Stderr, "new vad:", err)
			os.Exit(1)
		}
		// Warm-up (session init, first-run allocations) is excluded from timing.
		if _, err := vad.Scores(samples[:min(len(samples), 16000*10)]); err != nil {
			fmt.Fprintln(os.Stderr, "warmup:", err)
			os.Exit(1)
		}
		best := time.Duration(1<<62 - 1)
		var frames int
		for r := 0; r < *reps; r++ {
			t := time.Now()
			sc, err := vad.Scores(samples)
			d := time.Since(t)
			if err != nil {
				fmt.Fprintln(os.Stderr, "scores:", err)
				os.Exit(1)
			}
			frames = len(sc)
			if d < best {
				best = d
			}
		}
		_ = vad.Close()
		secs := best.Seconds()
		fmt.Printf("%-8d %-10.3f %-12.1f %-10d\n", n, secs, durSec/secs, frames)
	}
}

// binarize is a copy of whisper.binarizeVadScores returning [start,end] seconds,
// used to diagnose the chunking stage in isolation.
func binarize(scores []float32, frameStep float64, onset, offset float32, maxDurationS float64) [][2]float64 {
	n := len(scores)
	if n == 0 {
		return nil
	}
	ts := func(i int) float64 { return (float64(i) + 0.5) * frameStep }
	var regions [][2]float64
	emit := func(a, b float64) {
		if b > a {
			regions = append(regions, [2]float64{a, b})
		}
	}
	start := ts(0)
	isActive := scores[0] > onset
	curScores := []float32{scores[0]}
	curTimes := []float64{ts(0)}
	t := ts(0)
	for i := 1; i < n; i++ {
		t = ts(i)
		y := scores[i]
		if isActive {
			curDuration := t - start
			if curDuration > maxDurationS {
				searchAfter := len(curScores) / 2
				minIdx := searchAfter
				minVal := curScores[searchAfter]
				for k := searchAfter + 1; k < len(curScores); k++ {
					if curScores[k] < minVal {
						minVal = curScores[k]
						minIdx = k
					}
				}
				emit(start, curTimes[minIdx])
				start = curTimes[minIdx]
				curScores = append([]float32(nil), curScores[minIdx+1:]...)
				curTimes = append([]float64(nil), curTimes[minIdx+1:]...)
			} else if y < offset {
				emit(start, t)
				start = t
				isActive = false
				curScores = curScores[:0]
				curTimes = curTimes[:0]
			}
			curScores = append(curScores, y)
			curTimes = append(curTimes, t)
		} else {
			if y > onset {
				start = t
				isActive = true
			}
		}
	}
	if isActive {
		emit(start, t)
	}
	return regions
}

func readWav(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 44 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a RIFF/WAVE file")
	}
	// Walk chunks to find "data".
	off := 12
	var data []byte
	for off+8 <= len(b) {
		id := string(b[off : off+4])
		sz := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		body := off + 8
		if body+sz > len(b) {
			sz = len(b) - body
		}
		if id == "data" {
			data = b[body : body+sz]
			break
		}
		off = body + sz
		if sz%2 == 1 {
			off++
		}
	}
	if data == nil {
		return nil, fmt.Errorf("no data chunk")
	}
	n := len(data) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		s := int16(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
		out[i] = float32(s) / 32768.0
	}
	return out, nil
}
