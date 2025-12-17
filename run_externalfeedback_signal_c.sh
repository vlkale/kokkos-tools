#!/usr/bin/env bash
set -euo pipefail

echo "[configure] Running CMake configure..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tee build/configure_signal_c.log

echo "[build] Building pure C signal-driven demo runner..."
cmake --build build --target externalfeedback_demo_runner_signal_c -j4 2>&1 | tee build/build_signal_c.log

echo "[run] Running pure C signal-driven demo runner..."
./build/profiling/externalfeedback-profiler/externalfeedback_demo_runner_signal_c &
PID=$!
# Give it a moment to start and install the handler
sleep 1
# Send SIGUSR1 to trigger handshake
kill -USR1 ${PID}
wait ${PID}
