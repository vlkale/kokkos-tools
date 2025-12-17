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

#ifndef KOKKOSP_ExternalFeedback_HPP
#define KOKKOSP_ExternalFeedback_HPP

#include <chrono>
#include <functional>
#include <thread>
#include <atomic>

namespace KokkosTools::ExternalFeedback {
// Simple periodic listener thread that invokes a callback at a given cadence.
class Daemon {
 public:
  using Duration = std::chrono::milliseconds;

  Daemon(std::function<void()> func, Duration interval)
      : interval_(interval), func_(std::move(func)) {}

  void start();
  void stop();
  bool is_running() const { return running_.load(); }
  std::thread& get_thread() { return thread_; }

 private:
  void run();

  Duration interval_;
  std::atomic<bool> running_{false};
  std::function<void()> func_;
  std::thread thread_;
};
}  // namespace KokkosTools::ExternalFeedback
#endif
