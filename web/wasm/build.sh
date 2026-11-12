#!/bin/sh
# Builds kernel.wasm (hand-written SIMD intrinsics, about 1.5 kB) from kernel.c.
#
#   sh web/wasm/build.sh
#
# The compiler is `zig cc`, taken from the `ziglang` PyPI wheel through uv, so nothing is installed
# globally (the wheel is downloaded into uv's cache on first use). Set ZIG to use another zig binary,
# e.g. ZIG=zig sh web/wasm/build.sh. The compiled kernel.wasm is committed, so this is only needed
# after editing kernel.c.
set -e
cd "$(dirname "$0")"
ZIG="${ZIG:-uv run --no-project --with ziglang python -m ziglang}"
$ZIG cc --target=wasm32-freestanding -O3 -msimd128 -nostdlib \
  -Wl,--no-entry -Wl,--import-memory -Wl,--export=__heap_base -Wl,--strip-all \
  -o kernel.wasm kernel.c
chmod 644 kernel.wasm
ls -l kernel.wasm
