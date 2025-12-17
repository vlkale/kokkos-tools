//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include "data_sources.hpp"

#include <chrono>
#include <random>

namespace KokkosTools::Feedback {

static uint64_t now_ns() {
  using clock = std::chrono::steady_clock;
  auto now    = clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

std::optional<DataPoint> LdmsSource::poll() {
  // Stub: pretend LDMS streams returned a bandwidth sample
  static std::mt19937_64 rng{12345};
  static std::uniform_real_distribution<double> dist(100.0, 1000.0);
  DataPoint dp;
  dp.source = "LDMS";
  dp.metric = "bandwidth";
  dp.value  = dist(rng);
  dp.timestamp_ns = now_ns();
  return dp;
}

std::optional<DataPoint> DcgmSource::poll() {
  // Stub: pretend DCGM returned a GPU temperature sample
  static std::mt19937_64 rng{54321};
  static std::uniform_real_distribution<double> dist(40.0, 90.0);
  DataPoint dp;
  dp.source = "DCGM";
  dp.metric = "temp";
  dp.value  = dist(rng);
  dp.timestamp_ns = now_ns();
  return dp;
}

std::optional<DataPoint> MockSource::poll() {
  DataPoint dp;
  dp.source = "Mock";
  dp.metric = "ping";
  dp.value  = 1.0;
  dp.timestamp_ns = now_ns();
  return dp;
}

} // namespace KokkosTools::Feedback
