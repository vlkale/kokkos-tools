#!/usr/bin/env bash
set -euo pipefail

echo "[configure] Running CMake configure..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tee build/configure_min.log

echo "[build] Building minimal runner..."
cmake --build build --target externalfeedback_min_runner -j4 2>&1 | tee build/build_min.log

echo "[run] Running minimal runner..."
./build/profiling/externalfeedback-profiler/externalfeedback_min_runner
