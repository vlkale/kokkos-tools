// Demo using sigwait pattern (inspired by IceUtil::CtrlCHandler)
// Advantages over async signal handlers:
// 1. No signal-safety restrictions (can use printf, malloc, etc.)
// 2. Cleaner synchronization with mutexes
// 3. Easier to reason about - signal handling is just another thread

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include "externalfeedback_sigwait.h"

int main() {
    printf("[sigwait-demo] Starting external feedback with sigwait pattern...\n");
    
    // Initialize the feedback system (starts sigwait thread)
    if (externalfeedback_init(1) != 0) {
        fprintf(stderr, "Failed to initialize external feedback\n");
        return 1;
    }
    
    pid_t pid = getpid();
    printf("[sigwait-demo] PID=%d. Send: kill -USR1 %d\n", (int)pid, (int)pid);
    printf("[sigwait-demo] Waiting for SIGUSR1...\n");
    
    // Wait for signal (in real app, this would be doing actual work)
    sleep(30);
    
    // Shutdown
    printf("[sigwait-demo] Shutting down...\n");
    externalfeedback_shutdown();
    printf("[sigwait-demo] Completed.\n");
    
    return 0;
}
