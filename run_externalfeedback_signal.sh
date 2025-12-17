#!/usr/bin/env bash
set -euo pipefail

echo "[configure] Running CMake configure..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tee build/configure_signal.log

echo "[build] Building signal-driven demo runner..."
cmake --build build --target externalfeedback_demo_runner_signal -j4 2>&1 | tee build/build_signal.log

echo "[run] Running signal-driven demo runner (daemon disabled)..."
# Disable the periodic daemon; use signal-driven triggering only
export KOKKOS_TOOLS_EXTERNALFEEDBACK_DISABLE_DAEMON=1
./build/profiling/externalfeedback-profiler/externalfeedback_demo_runner_signal &
PID=$!
# Give it a moment to start and install the handler
sleep 1
# Send SIGUSR1 to trigger handshake
kill -USR1 $(pgrep -n externalfeedback_demo_runner_signal || echo ${PID})
wait ${PID}
