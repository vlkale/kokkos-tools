// POSIX signal-driven demo runner
// Sets up a SIGUSR1 handler that triggers the Kokkos Tools handshake
// and prints "world" from the library side.

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h> // getpid, pause

#ifdef __cplusplus
extern "C" {
#endif
struct Kokkos_Tools_ToolSettings { int requires_global_fencing; unsigned int padding[8]; };
struct Kokkos_Profiling_KokkosPDeviceInfo { int dummy; };

void kokkosp_request_tool_settings(const unsigned int num_actions, struct Kokkos_Tools_ToolSettings* settings);
void kokkosp_init_library(const int loadSeq, const unsigned long long interfaceVer,
                          const unsigned int devInfoCount,
                          struct Kokkos_Profiling_KokkosPDeviceInfo* deviceInfo);
void kokkosp_finalize_library();
void kokkosp_externalfeedback_trigger_hello();
#ifdef __cplusplus
}
#endif

static volatile sig_atomic_t g_handshake_done = 0;

static void handle_usr1(int signo) {
  (void)signo;
  kokkosp_externalfeedback_trigger_hello();
  g_handshake_done = 1;
}

int main() {
  // Verbose output from the library
  setenv("KOKKOS_TOOLS_EXTERNALFEEDBACK_VERBOSE", "1", 1);

  // Init library (simulate Kokkos startup)
  kokkosp_init_library(1, 40000ULL, 0U, NULL);

  // Install SIGUSR1 handler
  struct sigaction sa;
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

  // Finalize library (simulate Kokkos shutdown)
  kokkosp_finalize_library();
  printf("[signal-demo] Completed.\n");
  return 0;
}
