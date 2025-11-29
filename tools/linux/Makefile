# Makefile for building cumm on ARM64 (aarch64)
# Usage: cd tools/linux && make <target>

.PHONY: help build-base dev-shell build-wheel clean

PYTHON_VERSION ?= 3.12
BASE_IMAGE = cumm-arm64-base
TAG = latest

# Root directory of cumm repo (two levels up from tools/linux)
ROOT_DIR = $(realpath ../..)

help:
	@echo "cumm ARM64 Build Targets:"
	@echo ""
	@echo "  make build-base   - Build base Docker image"
	@echo "  make dev-shell    - Interactive dev shell (volume mount)"
	@echo "  make build-wheel  - Build wheel in Docker"
	@echo "  make clean        - Remove Docker images and build artifacts"
	@echo ""
	@echo "Environment variables:"
	@echo "  PYTHON_VERSION=$(PYTHON_VERSION)"
	@echo ""

build-base:
	@echo "Building base image: $(BASE_IMAGE):$(TAG)"
	docker build \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		-f Dockerfile.arm64 \
		-t $(BASE_IMAGE):$(TAG) \
		../..
	@echo "Base image built successfully"

dev-shell: build-base
	@echo "Starting interactive dev shell (volume mounted)..."
	docker run --rm -it \
		--gpus all \
		-v $(ROOT_DIR):/cumm \
		-w /cumm \
		$(BASE_IMAGE):$(TAG) /bin/bash

build-wheel: build-base
	@echo "Building wheel..."
	docker run --rm \
		-v $(ROOT_DIR):/cumm \
		-w /cumm \
		-e MAX_JOBS=4 \
		$(BASE_IMAGE):$(TAG) \
		python setup.py bdist_wheel
	@echo "Wheel built: $(ROOT_DIR)/dist/"
	@ls -lh $(ROOT_DIR)/dist/*.whl 2>/dev/null || echo "No wheel found"

clean:
	@echo "Cleaning up..."
	-docker rmi $(BASE_IMAGE):$(TAG)
	-rm -rf $(ROOT_DIR)/dist $(ROOT_DIR)/build $(ROOT_DIR)/*.egg-info
	@echo "Clean complete"
