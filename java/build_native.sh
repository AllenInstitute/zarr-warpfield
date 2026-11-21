#!/usr/bin/env bash
# Build the native library for this machine and put it where the jar picks it up
# (java/native/org/warpfield/native/<os>-<arch>/), then build and test the jar.
#   java/build_native.sh            # cargo build + mvn package
#   java/build_native.sh --no-mvn   # native library only
# Other platforms come from CI (.github/workflows/java.yml), which assembles one
# jar holding every platform's library.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"

case "$(uname -s)" in
  Darwin) os=macos; lib=libwarpfield_jni.dylib ;;
  Linux) os=linux; lib=libwarpfield_jni.so ;;
  MINGW* | MSYS* | CYGWIN*) os=windows; lib=warpfield_jni.dll ;;
  *) echo "unsupported OS $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64 | amd64) arch=x86_64 ;;
  arm64 | aarch64) arch=aarch64 ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

cargo build --manifest-path "$root/rust/Cargo.toml" --release -p warpfield-jni
dest="$here/native/org/warpfield/native/$os-$arch"
mkdir -p "$dest"
cp "$root/rust/target/release/$lib" "$dest/"
echo "native library: $dest/$lib"

if [[ "${1:-}" != "--no-mvn" ]]; then
  (cd "$here" && mvn -B package)
fi
