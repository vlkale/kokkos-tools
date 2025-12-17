// External feedback library using sigwait pattern (inspired by IceUtil::CtrlCHandler)
// Uses sigwait to synchronously wait for signals in a dedicated thread
// This is cleaner than async signal handlers and avoids signal-safety issues

#include "externalfeedback_sigwait.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

// Internal state
typedef struct {
    pthread_t signal_thread;
    volatile int running;
    pthread_mutex_t mutex;
    int verbose;
    int handshake_done;
} ExternalFeedbackState;

static ExternalFeedbackState g_state = {
    .signal_thread = 0,
    .running = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .verbose = 0,
    .handshake_done = 0
};

// Simple hash function
static unsigned long make_simple_hash(const char* payload) {
    unsigned long hash = (unsigned long)getpid();
    if (payload) {
        size_t len = strlen(payload);
        hash ^= len;
    }
    static unsigned long counter = 0;
    hash ^= __sync_fetch_and_add(&counter, 1);
    return hash;
}

// Process external message (Hello -> hash -> World)
static void process_message(const char* message) {
    pthread_mutex_lock(&g_state.mutex);
    
    if (g_state.verbose) {
        printf("[ExternalFeedback] (sigwait) Received: '%s'\n", message);
    }
    
    if (strcmp(message, "Hello") == 0 && !g_state.handshake_done) {
        unsigned long hash = make_simple_hash(message);
        if (g_state.verbose) {
            printf("[ExternalFeedback] Hashed to: %lu\n", hash);
        }
        puts("world");
        fflush(stdout);
        g_state.handshake_done = 1;
    }
    
    pthread_mutex_unlock(&g_state.mutex);
}

// Signal waiting thread (inspired by IceUtil::CtrlCHandler's sigwait approach)
static void* signal_wait_thread(void* arg) {
    (void)arg;
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGUSR1);
    
    while (g_state.running) {
        int sig;
        int ret = sigwait(&sigset, &sig);
        
        if (ret == 0 && sig == SIGUSR1 && g_state.running) {
            process_message("Hello");
        }
    }
    
    return NULL;
}

// Initialize the external feedback system
int externalfeedback_init(int verbose) {
    pthread_mutex_lock(&g_state.mutex);
    
    if (g_state.running) {
        pthread_mutex_unlock(&g_state.mutex);
        return 0; // Already initialized
    }
    
    g_state.verbose = verbose;
    g_state.handshake_done = 0;
    g_state.running = 1;
    
    // Block SIGUSR1 in all threads so sigwait can catch it
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &sigset, NULL);
    
    // Start signal waiting thread
    int ret = pthread_create(&g_state.signal_thread, NULL, signal_wait_thread, NULL);
    if (ret != 0) {
        g_state.running = 0;
        pthread_mutex_unlock(&g_state.mutex);
        return -1;
    }
    
    pthread_mutex_unlock(&g_state.mutex);
    return 0;
}

// Shutdown the external feedback system
void externalfeedback_shutdown(void) {
    pthread_mutex_lock(&g_state.mutex);
    
    if (!g_state.running) {
        pthread_mutex_unlock(&g_state.mutex);
        return;
    }
    
    g_state.running = 0;
    pthread_mutex_unlock(&g_state.mutex);
    
    // Wake up the sigwait thread by sending a signal to ourselves
    pthread_kill(g_state.signal_thread, SIGUSR1);
    pthread_join(g_state.signal_thread, NULL);
}

// Trigger handshake programmatically (for testing)
void externalfeedback_trigger_hello(void) {
    process_message("Hello");
}
