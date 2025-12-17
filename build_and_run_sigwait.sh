#!/usr/bin/env bash
# Build and run sigwait pattern demo (inspired by IceUtil::CtrlCHandler)
set -euo pipefail

cd profiling/externalfeedback-profiler

echo "Building sigwait-based external feedback library..."
cc -c -o externalfeedback_sigwait.o externalfeedback_sigwait.c
ar rcs libexternalfeedback_sigwait.a externalfeedback_sigwait.o

echo "Building sigwait demo..."
cc -o demo_runner_sigwait demo_runner_sigwait.c libexternalfeedback_sigwait.a -lpthread

echo "Running sigwait demo..."
./demo_runner_sigwait &
PID=$!
sleep 2
echo "Sending SIGUSR1 to PID $PID..."
kill -USR1 $PID
sleep 1
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true

echo "Done!"
