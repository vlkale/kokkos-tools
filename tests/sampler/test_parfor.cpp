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
    //! logger. Use a periodic sampling with skip rate 51. This should print
    //! out 2 invocations, and there is a single matcher with a regular
    //! expression to check this.

    for (int iter = 0; iter < 150; iter++) {
      Kokkos::parallel_for("named kernel",
                           Kokkos::RangePolicy<execution_space>(space, 0, 1),
                           *this);
    }
  }

  KOKKOS_FUNCTION void operator()(const int) const {}
};

static const std::vector<std::string> matchers{
    "KokkosP: sample 51 calling child-begin function...",
    "KokkosP: sample 51 finished with child-begin function.",
    "KokkosP: sample 51 calling child-end function...",
    "KokkosP: sample 51 finished with child-end function.",
    "KokkosP: sample 102 calling child-begin function...",
    "KokkosP: sample 102 finished with child-begin function.",
    "KokkosP: sample 102 calling child-end function...",
    "KokkosP: sample 102 finished with child-end function."};

/**
 * @test This test checks that the tool effectively samples.
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
  std::cout.flush();
  std::cout.rdbuf(coutbuf);
  std::cout << output.str() << std::endl;

  //! Analyze test output.
  for (const auto& matcher : matchers) {
    EXPECT_THAT(output.str(), ::testing::HasSubstr(matcher));
  }  // end TEST

  EXPECT_THAT(output.str(),
              ::testing::Contains("calling child-begin function...")
                  .::testing::Times("2"));

  EXPECT_THAT(output.str(),
              ::testing::Contains("finished with child-begin function.")
                  .::testing::Times("2"));
  EXPECT_THAT(
      output.str(),
      ::testing::Contains("calling child-end function...").::testing::Times("2"));

  EXPECT_THAT(output.str(),
              ::testing::Contains("finished with child-end function.")
                  .::testing::Times("2"));

  EXPECT_THAT(output.str(),
              ::testing::Not(::testing::HasSubstr("KokkosP: FATAL: No child library of sampler utility library to call")));

  EXPECT_THAT(output.str(),
              ::testing::Not(::testing::HasSubstr("KokkosP: FATAL: Kokkos Tools Programming "
                             "Interface's tool-invoked Fence is NULL!")));
}
