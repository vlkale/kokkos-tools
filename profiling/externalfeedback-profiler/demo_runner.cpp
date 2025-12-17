//@HEADER
// ************************************************************************
// Minimal demo runner for external feedback handshake
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//@HEADER

#include <iostream>
#include <thread>
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

extern "C" {
// Provide minimal prototypes for the exposed Kokkos Tools hooks implemented in kp_externalfeedback_profiler.cpp
void kokkosp_request_tool_settings(const uint32_t num_actions, struct Kokkos_Tools_ToolSettings* settings);
void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                          const uint32_t devInfoCount,
                          struct Kokkos_Profiling_KokkosPDeviceInfo* deviceInfo);
void kokkosp_finalize_library();
}

// Provide tiny stand-in structs to satisfy prototypes (not used by our demo)
struct Kokkos_Tools_ToolSettings { bool requires_global_fencing; uint32_t padding[8]; };
struct Kokkos_Profiling_KokkosPDeviceInfo { int dummy; };

int main() {
  // Verbose output and select a mock source for deterministic logs
  setenv("KOKKOS_TOOLS_EXTERNALFEEDBACK_VERBOSE", "1", 1);
  setenv("KOKKOS_TOOLS_FEEDBACK_SOURCE", "MOCK", 1);
  setenv("KOKKOS_TOOLS_FEEDBACK_INTERVAL_MS", "250", 1);

  // Init the library (simulating Kokkos runtime startup)
  kokkosp_init_library(/*loadSeq=*/1, /*interfaceVer=*/40000, /*devInfoCount=*/0, /*deviceInfo=*/nullptr);

  // Allow the background listener to tick and perform the Hello→hash→World handshake
  // Sleep briefly to let the listener tick at least once
#if defined(__unix__) || defined(__APPLE__)
  usleep(750 * 1000);
#else
  // Fallback: minimal spin using threads
  std::this_thread::sleep_for(std::chrono::milliseconds(750));
#endif

  // Finalize (simulating Kokkos runtime shutdown)
  kokkosp_finalize_library();

  std::cout << "Demo completed." << std::endl;
  return 0;
}
