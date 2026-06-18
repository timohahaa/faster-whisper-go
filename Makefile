# Runtime requirements:
#   - CTranslate2 shared library (linked via cgo for inference).
#   - onnxruntime shared library (loaded at runtime for the Silero VAD).
#     The VAD locates libonnxruntime.so automatically in common system paths
#     (e.g. /usr/lib/x86_64-linux-gnu, /usr/local/lib). Override the location
#     by exporting ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.so
#     before running tests or binaries.

PLATFORM := $(shell uname)

ifeq ($(PLATFORM), Darwin)
	BUILD_ENVPARMS := GOOS=darwin GOARCH=arm64 CGO_ENABLED=1
endif
ifeq ($(PLATFORM), Linux)
	BUILD_ENVPARMS := GOOS=linux GOARCH=amd64 CGO_ENABLED=1
endif

.PHONY: build test test-e2e clean

build:
	$(BUILD_ENVPARMS) go build ./...

test:
	$(BUILD_ENVPARMS) go test -v ./...

clean:
	go clean ./...
