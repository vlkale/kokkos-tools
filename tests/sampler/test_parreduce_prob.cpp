#include <iostream>
#include <sstream>
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

struct Tester {
  template <typename execution_space>
  explicit Tester(const execution_space& space) {
    //! Explicitly launch a kernel with a name, and run it 150 times with kernel
    //! logger. Use a probabilistic sampling with 100% probability. This should print
    //! out all 150 invocations, and there is a single matcher with a regular
    //! expression to check the last two.

    double sum;
    for (int iter = 0; iter < 150; iter++) {
      sum = 0;
      Kokkos::parallel_reduce("named kernel",
                              Kokkos::RangePolicy<execution_space>(space, 0, 1),
                              *this, sum);
    }
  }

  KOKKOS_FUNCTION void operator()(const int, const int) const {}
};

static const std::vector<std::string> matchers{
    "(.*)KokkosP: sample 148 calling child-begin function...(.*)",
    "(.*)KokkosP: sample 148 finished with child-begin function.(.*)",
    "(.*)KokkosP: sample 148 calling child-end function...(.*)",
    "(.*)KokkosP: sample 148 finished with child-end function.(.*)",
    "(.*)KokkosP: sample 149 calling child-begin function...(.*)",
    "(.*)KokkosP: sample 149 finished with child-begin function.(.*)",
    "(.*)KokkosP: sample 149 calling child-end function...(.*)",
    "(.*)KokkosP: sample 149 finished with child-end function.(.*)"};

/**
 * @test This test checks that the tool effectively samples with probability.
 *

 */
TEST(SamplerTest, ktoEnvVarDefault) {
  //! Initialize @c Kokkos.
  Kokkos::initialize();

  //! Redirect output for later analysis.
  std::cout.flush();
  std::ostringstream output;
  std::streambuf* coutbuf = std::cout.rdbuf(output.rdbuf());

  //! Run tests. @todo Replace this with Google Test.
  Tester tester(Kokkos::DefaultExecutionSpace{});

  //! Finalize @c Kokkos.
  Kokkos::finalize();

  //! Restore output buffer.
  // std::cout.flush();
  std::cout.rdbuf(coutbuf);
  std::cout << output.str() << std::endl;

  //! Analyze test output.
  for (const auto& matcher : matchers) {
    EXPECT_THAT(output.str(), ::testing::ContainsRegex(matcher));
  }  // end TEST
}
