#include <iostream>
#include <sstream>
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "parscan.hpp"

static const std::vector<std::string> matchers{
    "(.*)KokkosP: sample 6 calling child-begin function...(.*)",
    "(.*)KokkosP: sample 6 finished with child-begin function.(.*)",
    "(.*)KokkosP: sample 6 calling child-end function...(.*)",
    "(.*)KokkosP: sample 6 finished with child-end function.(.*)",
    "(.*)KokkosP: sample 12 calling child-begin function...(.*)",
    "(.*)KokkosP: sample 12 finished with child-begin function.(.*)",
    "(.*)KokkosP: sample 12 calling child-end function...(.*)",
    "(.*)KokkosP: sample 12 finished with child-end function.(.*)"};

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
    EXPECT_THAT(output.str(), ::testing::ContainsRegex(matcher));
  }  // end TEST
}
