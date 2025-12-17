#!/usr/bin/env bash
# Pure C build script - no CMake, no C++ required
set -euo pipefail

cd profiling/externalfeedback-profiler

echo "Building pure C external feedback library..."
cc -c -o externalfeedback_c.o externalfeedback_c.c
ar rcs libexternalfeedback_c.a externalfeedback_c.o

echo "Building signal demo..."
cc -o demo_runner_signal_c demo_runner_signal_c.c libexternalfeedback_c.a -lpthread

echo "Running signal demo..."
./demo_runner_signal_c &
PID=$!
sleep 1
echo "Sending SIGUSR1 to PID $PID..."
kill -USR1 $PID
wait $PID

echo "Done!"
