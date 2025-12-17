//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#ifndef KOKKOSTOOLS_DATA_SOURCES_HPP
#define KOKKOSTOOLS_DATA_SOURCES_HPP

#include <cstdint>
#include <string>
#include <optional>

namespace KokkosTools::Feedback {

struct DataPoint {
  std::string source;   // "LDMS" or "DCGM" or "Mock"
  std::string metric;   // e.g. "bandwidth", "temp"
  double value{0.0};
  uint64_t timestamp_ns{0};
};

struct IDataSource {
  virtual ~IDataSource() = default;
  virtual std::optional<DataPoint> poll() = 0; // return one sample if available
};

struct LdmsSource : public IDataSource {
  std::optional<DataPoint> poll() override;
};

struct DcgmSource : public IDataSource {
  std::optional<DataPoint> poll() override;
};

struct MockSource : public IDataSource {
  std::optional<DataPoint> poll() override;
};

} // namespace KokkosTools::Feedback

#endif // KOKKOSTOOLS_DATA_SOURCES_HPP
