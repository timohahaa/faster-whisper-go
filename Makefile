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
	$(BUILD_ENVPARMS) go test ./...

test-e2e:
	$(BUILD_ENVPARMS) go test -tags e2e -v ./tests/...

clean:
	go clean ./...
