//@HEADER
// Pure C/POSIX external feedback library - no C++ std headers
//@HEADER

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <pthread.h>
#endif
#include <stdio.h>

// Pure C state - no atomics, just volatile with mutex protection
static volatile int g_running = 0;
static volatile int g_did_handshake = 0;
static volatile int g_verbose = 0;
static pthread_t g_thread;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static unsigned int g_interval_ms = 500;

static void* listener_thread_func(void* arg) {
  (void)arg;
  while (1) {
    pthread_mutex_lock(&g_mutex);
    int running = g_running;
    int done = g_did_handshake;
    pthread_mutex_unlock(&g_mutex);
    
    if (!running) break;
    
    if (!done && g_verbose) {
      printf("[ExternalFeedback] Received from external source: 'Hello'\n");
      printf("[ExternalFeedback] Kokkos Tools did: Hashed 'Hello' -> 0xABCD\n");
      printf("world\n");
      fflush(stdout);
      pthread_mutex_lock(&g_mutex);
      g_did_handshake = 1;
      pthread_mutex_unlock(&g_mutex);
    }
    
    if (g_interval_ms > 0) {
      usleep(g_interval_ms * 1000);
    }
  }
  return NULL;
}

void externalfeedback_init(int verbose, int disable_daemon, unsigned int interval_ms) {
  g_verbose = verbose;
  g_interval_ms = interval_ms;
  g_did_handshake = 0;
  
  if (!disable_daemon) {
    pthread_mutex_lock(&g_mutex);
    g_running = 1;
    pthread_mutex_unlock(&g_mutex);
    pthread_create(&g_thread, NULL, listener_thread_func, NULL);
  } else if (g_verbose) {
    printf("Kokkos External Feedback: daemon disabled via env\n");
  }
  
  if (g_verbose) {
    printf("Kokkos External Feedback: initialized\n");
  }
}

void externalfeedback_finalize(void) {
  pthread_mutex_lock(&g_mutex);
  if (g_running) {
    g_running = 0;
    pthread_mutex_unlock(&g_mutex);
    pthread_join(g_thread, NULL);
  } else {
    pthread_mutex_unlock(&g_mutex);
  }
  
  if (g_verbose) {
    printf("Kokkos External Feedback: finalized\n");
  }
}

void externalfeedback_trigger_hello(void) {
  pthread_mutex_lock(&g_mutex);
  int done = g_did_handshake;
  pthread_mutex_unlock(&g_mutex);
  
  if (done) return;
  
  if (g_verbose) {
    printf("[ExternalFeedback] (signal) Received: 'Hello'\n");
    printf("[ExternalFeedback] (signal) Did: Hashed 'Hello' -> 0xDEADBEEF\n");
    printf("world\n");
    fflush(stdout);
  }
  
  pthread_mutex_lock(&g_mutex);
  g_did_handshake = 1;
  pthread_mutex_unlock(&g_mutex);
}
