// Minimal POSIX demo runner for external feedback handshake
// Avoids C++ std headers; uses C + pthreads/usleep

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep

#ifdef __cplusplus
extern "C" {
#endif
// Prototypes provided by the externalfeedback profiler (C++ library)
struct Kokkos_Tools_ToolSettings { int requires_global_fencing; unsigned int padding[8]; };
struct Kokkos_Profiling_KokkosPDeviceInfo { int dummy; };

void kokkosp_request_tool_settings(const unsigned int num_actions, struct Kokkos_Tools_ToolSettings* settings);
void kokkosp_init_library(const int loadSeq, const unsigned long long interfaceVer,
                          const unsigned int devInfoCount,
                          struct Kokkos_Profiling_KokkosPDeviceInfo* deviceInfo);
void kokkosp_finalize_library();
#ifdef __cplusplus
}
#endif

int main() {
  // Verbose output
  setenv("KOKKOS_TOOLS_EXTERNALFEEDBACK_VERBOSE", "1", 1);
  setenv("KOKKOS_TOOLS_FEEDBACK_INTERVAL_MS", "250", 1);

  // Init the library (simulating Kokkos runtime startup)
  kokkosp_init_library(/*loadSeq=*/1, /*interfaceVer=*/40000ULL, /*devInfoCount=*/0U, /*deviceInfo=*/NULL);

  // Allow the background listener to tick and perform the Hello→hash→World handshake
  usleep(750 * 1000);

  // Finalize (simulating Kokkos runtime shutdown)
  kokkosp_finalize_library();

  printf("Demo (POSIX) completed.\n");
  return 0;
}
