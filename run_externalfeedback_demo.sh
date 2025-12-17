#!/usr/bin/env bash
set -euo pipefail

echo "[configure] Running CMake configure..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tee build/configure.log

echo "[build] Building POSIX demo runner..."
cmake --build build --target externalfeedback_demo_runner_posix -j4 2>&1 | tee build/build_posix.log

echo "[run] Running POSIX demo runner..."
./build/profiling/externalfeedback-profiler/externalfeedback_demo_runner_posix
