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

#include "externalfeedback.hpp"

#include <thread>

namespace KokkosTools::ExternalFeedback {

void Daemon::start() {
  if (!running_.load()) {
    running_.store(true);
    thread_ = std::thread(&Daemon::run, this);
  }
}

void Daemon::stop() {
  if (running_.load()) {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
  }
}

void Daemon::run() {
  while (running_.load()) {
    if (func_) func_();
#if defined(__unix__) || defined(__APPLE__)
    if (interval_ms_ > 0) usleep(interval_ms_ * 1000);
#else
    // Fallback: simple busy wait for a very small interval (not ideal)
    if (interval_ms_ > 0) std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
#endif
  }
}

}  // namespace KokkosTools::ExternalFeedback
