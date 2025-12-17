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

namespace KokkosTools::ExternalFeedbackDemo {

// Global state for the minimal external feedback demo
struct State {
  std::mutex mtx;
  std::atomic<bool> verbose{false};
  std::optional<KokkosTools::ExternalFeedback::Daemon> daemon;  // lazy constructed
  std::atomic<bool> did_handshake{false};
  bool printed_world{false};

  static State& get() {
    static State s;
    return s;
  }
};

static unsigned long long make_simple_hash(const std::string& payload) {
  // Very simple hash: pid + payload length + a static counter
  static unsigned long long counter = 0;
  ++counter;
#if defined(__unix__) || defined(__APPLE__)
  unsigned long long pid = static_cast<unsigned long long>(::getpid());
#else
  unsigned long long pid = 1ULL;
#endif
  return pid ^ (static_cast<unsigned long long>(payload.size()) + counter * 2654435761ULL);
}

// This function represents the "Kokkos Tools environment" processing.
// It receives a message, performs a trivial parameter-related operation (hash),
// and returns a response string describing what it did.
static std::string process_external_message(const std::string& msg_in, unsigned long long& out_hash) {
  out_hash = make_simple_hash(msg_in);
  std::ostringstream oss;
  oss << "Hashed '" << msg_in << "' -> 0x" << std::hex << out_hash;
  return oss.str();
}

// The listener callback executed by the background daemon.
static void listener_tick() {
  auto& S = State::get();
  // Perform the Hello->hash->World handshake only once
  if (S.did_handshake.load()) return;
  const std::string hello = "Hello";
  unsigned long long hashed_value   = 0;
  std::string what_happened = process_external_message(hello, hashed_value);
  if (S.verbose.load()) {
    std::cout << "[ExternalFeedback] Received from external source: '" << hello << "'\n";
    std::cout << "[ExternalFeedback] Kokkos Tools did: " << what_happened << "\n";
    std::cout << "world" << std::endl;
  }
  // Mark done so we don't spam logs every tick
  S.did_handshake.store(true);
}

} // namespace KokkosTools::ExternalFeedbackDemo

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
  using namespace KokkosTools::ExternalFeedbackDemo;

  const char* verbose_env = std::getenv("KOKKOS_TOOLS_EXTERNALFEEDBACK_VERBOSE");
  if (verbose_env && (std::string(verbose_env) == "1" || std::string(verbose_env) == "ON")) {
    State::get().verbose.store(true);
  }
  // Optionally disable the periodic daemon when using signal-driven triggering
  const char* disable_env = std::getenv("KOKKOS_TOOLS_EXTERNALFEEDBACK_DISABLE_DAEMON");
  const bool disable_daemon = (disable_env && (std::string(disable_env) == "1" || std::string(disable_env) == "ON"));

  // No external sources needed for minimal example

  // Construct the daemon lazily to avoid static init order issues (unless disabled)
  if (!disable_daemon) {
    auto& S = State::get();
    std::lock_guard<std::mutex> lock(S.mtx);
    // Allow interval override via env KOKKOS_TOOLS_FEEDBACK_INTERVAL_MS
    unsigned long interval_ms = 500;
    if (const char* int_env = std::getenv("KOKKOS_TOOLS_FEEDBACK_INTERVAL_MS")) {
      interval_ms = static_cast<unsigned long>(std::strtoul(int_env, nullptr, 10));
      if (interval_ms == 0) interval_ms = 500;
    }
    if (!S.daemon.has_value()) {
      S.daemon.emplace(listener_tick, static_cast<unsigned int>(interval_ms));
    }
    S.daemon->start();
  } else if (State::get().verbose.load()) {
    std::cout << "Kokkos External Feedback: daemon disabled via env" << std::endl;
  }

  if (State::get().verbose.load()) {
    std::cout << "Kokkos External Feedback: initialized" << std::endl;
  }
}

// Library finalize: stop the listener daemon
void kokkosp_finalize_library() {
  using namespace KokkosTools::ExternalFeedbackDemo;
  auto& S = State::get();
  std::lock_guard<std::mutex> lock(S.mtx);
  if (S.daemon.has_value()) {
    S.daemon->stop();
  }
  if (S.verbose.load()) {
    std::cout << "Kokkos External Feedback: finalized" << std::endl;
  }
}

// We keep all other hooks unimplemented for this minimal example.

} // extern "C"

// Alternate trigger path: expose a C-callable function to simulate an external
// interrupt-driven event delivering "Hello" to the Kokkos Tools environment.
extern "C" void kokkosp_externalfeedback_trigger_hello() {
  using namespace KokkosTools::ExternalFeedbackDemo;
  auto& S = State::get();
  // Perform the Hello->hash->World handshake only once
  if (S.did_handshake.load()) return;
  const std::string hello = "Hello";
  unsigned long long hashed_value = 0ULL;
  std::string what_happened = process_external_message(hello, hashed_value);
  if (S.verbose.load()) {
    std::cout << "[ExternalFeedback] (signal) Received: '" << hello << "'\n";
    std::cout << "[ExternalFeedback] (signal) Did: " << what_happened << "\n";
    std::cout << "world" << std::endl;
  }
  S.did_handshake.store(true);
}
