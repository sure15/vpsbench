#!/usr/bin/env sh
set -eu

APP_NAME="vpsbench"
SRC="vpsbench.go"
OUT_DIR="dist"

if ! command -v go >/dev/null 2>&1; then
  echo "error: go is not installed or not in PATH" >&2
  exit 1
fi

if [ ! -f "$SRC" ]; then
  echo "error: $SRC not found" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p ".gocache"

if [ -z "${GOCACHE:-}" ]; then
  export GOCACHE="$(pwd)/.gocache"
fi

build() {
  goos="$1"
  goarch="$2"
  ext=""
  if [ "$goos" = "windows" ]; then
    ext=".exe"
  fi

  output="$OUT_DIR/${APP_NAME}-${goos}-${goarch}${ext}"
  echo "building $output"
  GOOS="$goos" GOARCH="$goarch" CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o "$output" "$SRC"
}

build linux amd64
build linux arm64
build darwin amd64
build darwin arm64
build windows amd64
build windows arm64

echo
echo "done. binaries are in $OUT_DIR/"
