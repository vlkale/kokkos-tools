// Pure C signal-driven demo runner (no C++ std headers)
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>

#include "externalfeedback_c.h"

static volatile sig_atomic_t g_handshake_done = 0;

static void handle_usr1(int signo) {
  (void)signo;
  externalfeedback_trigger_hello();
  g_handshake_done = 1;
}

int main() {
  // Init library
  externalfeedback_init(1, 1, 250); // verbose=1, disable_daemon=1, interval=250ms

  // Install SIGUSR1 handler
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = handle_usr1;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;
  if (sigaction(SIGUSR1, &sa, NULL) != 0) {
    perror("sigaction");
    return 1;
  }

  pid_t pid = getpid();
  printf("[signal-demo] PID=%d. Send: kill -USR1 %d\n", (int)pid, (int)pid);
  printf("[signal-demo] Waiting for SIGUSR1 to trigger handshake...\n");

  // Wait until signal handler flips the flag
  while (!g_handshake_done) {
    pause();
  }

  // Finalize library
  externalfeedback_finalize();
  printf("[signal-demo] Completed.\n");
  return 0;
}
