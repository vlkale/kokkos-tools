//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <atomic>
#include <sstream>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include "kp_core.hpp"  // Provides the Kokkos Tools hook prototypes/macros
#include "externalfeedback.hpp"

namespace KokkosTools::ExternalFeedbackOld {

// Global state for the minimal external feedback demo
struct State {
  std::mutex mtx;
  std::atomic<bool> verbose{false};
  std::optional<ExternalFeedback::Daemon> daemon;  // lazy constructed
  std::atomic<bool> did_handshake{false};

  static State& get() {
    static State s;
    return s;
  }
};

static uint64_t make_hash_from_env(const std::string& payload) {
  // Combine timestamp, pid, and payload to produce a stable-ish number
  using clock = std::chrono::steady_clock;
  auto now    = clock::now().time_since_epoch();
  uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
#if defined(__unix__) || defined(__APPLE__)
  uint64_t pid = static_cast<uint64_t>(::getpid());
#else
  uint64_t pid = 1u;
#endif
  // Simple mix
  std::hash<std::string> shash;
  uint64_t ph = static_cast<uint64_t>(shash(payload));
  uint64_t x  = ns ^ (pid + 0x9e3779b97f4a7c15ULL + (ns<<6) + (ns>>2));
  x ^= ph + 0x9e3779b97f4a7c15ULL + (x<<6) + (x>>2);
  return x;
}

// This function represents the "Kokkos Tools environment" processing.
// It receives a message, performs a trivial parameter-related operation (hash),
// and returns a response string describing what it did.
static std::string process_external_message(const std::string& msg_in, uint64_t& out_hash) {
  out_hash = make_hash_from_env(msg_in);
  std::ostringstream oss;
  oss << "Hashed '" << msg_in << "' -> 0x" << std::hex << out_hash;
  return oss.str();
}

// The listener callback executed by the background daemon.
static void listener_tick() {
  auto& S = State::get();
  // For demo: perform a single handshake the first time, then just idle
  if (S.did_handshake.load()) return;
  const std::string hello = "Hello";
  uint64_t hashed_value   = 0;
  std::string what_happened = process_external_message(hello, hashed_value);
  if (S.verbose.load()) {
    std::cout << "[ExternalFeedback-Old] Received from external source: '" << hello << "'\n";
    std::cout << "[ExternalFeedback-Old] Kokkos Tools did: " << what_happened << "\n";
    std::cout << "[ExternalFeedback-Old] Responding back to external source: 'World'\n";
  }
  // Mark done so we don't spam logs every tick
  S.did_handshake.store(true);
}

} // namespace KokkosTools::ExternalFeedbackOld

extern "C" {

// Tool settings — minimal: no global fencing required
void kokkosp_request_tool_settings(const uint32_t, Kokkos_Tools_ToolSettings* settings) {
  settings->requires_global_fencing = false;
  settings->padding[0] = 0;
}

// Library init: construct and start the listener daemon
void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                          const uint32_t devInfoCount,
                          Kokkos_Profiling_KokkosPDeviceInfo* deviceInfo) {
  (void)loadSeq; (void)interfaceVer; (void)devInfoCount; (void)deviceInfo;
  using namespace KokkosTools::ExternalFeedbackOld;

  const char* verbose_env = std::getenv("KOKKOS_TOOLS_EXTERNALFEEDBACK_VERBOSE");
  if (verbose_env && (std::string(verbose_env) == "1" || std::string(verbose_env) == "ON")) {
    State::get().verbose.store(true);
  }

  // Construct the daemon lazily to avoid static init order issues
  {
    auto& S = State::get();
    std::lock_guard<std::mutex> lock(S.mtx);
    if (!S.daemon.has_value()) {
      S.daemon.emplace(listener_tick, std::chrono::milliseconds(500));
    }
    S.daemon->start();
  }

  if (State::get().verbose.load()) {
    std::cout << "Kokkos External Feedback (old): initialized" << std::endl;
  }
}

// Library finalize: stop the listener daemon
void kokkosp_finalize_library() {
  using namespace KokkosTools::ExternalFeedbackOld;
  auto& S = State::get();
  std::lock_guard<std::mutex> lock(S.mtx);
  if (S.daemon.has_value()) {
    S.daemon->stop();
  }
  if (S.verbose.load()) {
    std::cout << "Kokkos External Feedback (old): finalized" << std::endl;
  }
}

// We keep all other hooks unimplemented for this minimal example.

} // extern "C"
