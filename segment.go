package whisper

const timePrecision = 0.02

// segmentSplitResult holds the output of SplitSegmentsByTimestamps.
type segmentSplitResult struct {
	segments              []rawSegment
	seek                  int
	singleTimestampEnding bool
}

// splitSegmentsByTimestamps parses timestamp tokens to determine segment boundaries
// and how far to advance the seek position, matching the Python _split_segments_by_timestamps.
func (t *tokenizer) splitSegmentsByTimestamps(
	tokens []int32,
	timeOffset float64,
	segmentSize int,
	segmentDuration float64,
	seek int,
) segmentSplitResult {
	singleTimestampEnding := len(tokens) >= 2 &&
		tokens[len(tokens)-2] < t.timestampBegin &&
		tokens[len(tokens)-1] >= t.timestampBegin

	var consecutiveTimestamps []int
	for i := 1; i < len(tokens); i++ {
		if tokens[i] >= t.timestampBegin && tokens[i-1] >= t.timestampBegin {
			consecutiveTimestamps = append(consecutiveTimestamps, i)
		}
	}

	var segments []rawSegment

	if len(consecutiveTimestamps) > 0 {
		slices := append([]int(nil), consecutiveTimestamps...)
		if singleTimestampEnding {
			slices = append(slices, len(tokens))
		}

		lastSlice := 0
		for _, currentSlice := range slices {
			slicedTokens := tokens[lastSlice:currentSlice]
			startPos := slicedTokens[0] - t.timestampBegin
			endPos := slicedTokens[len(slicedTokens)-1] - t.timestampBegin
			startTime := timeOffset + float64(startPos)*timePrecision
			endTime := timeOffset + float64(endPos)*timePrecision

			segments = append(segments, rawSegment{
				start:  startTime,
				end:    endTime,
				tokens: copyTokens(slicedTokens),
			})
			lastSlice = currentSlice
		}

		if singleTimestampEnding {
			seek += segmentSize
		} else {
			lastTSPos := tokens[lastSlice-1] - t.timestampBegin
			seek += int(lastTSPos) * inputStride
		}
	} else {
		duration := segmentDuration
		var timestamps []int32
		for _, tok := range tokens {
			if tok >= t.timestampBegin {
				timestamps = append(timestamps, tok)
			}
		}
		if len(timestamps) > 0 && timestamps[len(timestamps)-1] != t.timestampBegin {
			lastTSPos := timestamps[len(timestamps)-1] - t.timestampBegin
			duration = float64(lastTSPos) * timePrecision
		}

		segments = append(segments, rawSegment{
			start:  timeOffset,
			end:    timeOffset + duration,
			tokens: copyTokens(tokens),
		})
		seek += segmentSize
	}

	return segmentSplitResult{
		segments:              segments,
		seek:                  seek,
		singleTimestampEnding: singleTimestampEnding,
	}
}

// rawSegment is an intermediate segment before converting to public Segment type.
type rawSegment struct {
	start  float64
	end    float64
	tokens []int32
}

func copyTokens(src []int32) []int32 {
	out := make([]int32, len(src))
	copy(out, src)
	return out
}
