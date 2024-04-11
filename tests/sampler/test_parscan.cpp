#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

using ::testing::HasSubstr;
using ::testing::Not;

struct Tester {
  template <typename execution_space>
  explicit Tester(const execution_space& space) {
    //! Explicitly launch a kernel with a name, and run it 15 times with kernel
    //! logger. Use a periodic sampling with skip rate 5. This should print
    //! out 2 invocations, and there is a single matcher with a regular
    //! expression to check this.

    long int N = 1024;
    long int result;

    for (int iter = 0; iter < 15; iter++) {
      result = 0;
      Kokkos::parallel_scan("named kernel scan", N, *this, result);
    }
  }

  KOKKOS_FUNCTION void operator()(const int, long int&, bool) const {}
};

static const std::vector<std::string> matchers{
    "KokkosP: sample 6 calling child-begin function...",
    "KokkosP: sample 6 finished with child-begin function.",
    "KokkosP: sample 6 calling child-end function...",
    "KokkosP: sample 6 finished with child-end function.",
    "KokkosP: sample 11 calling child-begin function...",
    "KokkosP: sample 11 finished with child-begin function.",
    "KokkosP: sample 11 calling child-end function...",
    "KokkosP: sample 11 finished with child-end function."};

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
    EXPECT_THAT(output.str(), HasSubstr(matcher));
  }

  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 1 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 2 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 3 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 4 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 5 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 7 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 8 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 9 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 10 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 12 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 13 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 14 calling")));
  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: sample 15 calling")));

  int occurrences            = 0;
  std::string::size_type pos = 0;
  std::string samplerTestOutput(output.str());
  std::string target("calling child-begin function");
  while ((pos = samplerTestOutput.find(target, pos)) != std::string::npos) {
    ++occurrences;
    pos += target.length();
  }

  EXPECT_EQ(occurrences, 2);

  EXPECT_THAT(output.str(), Not(HasSubstr("KokkosP: FATAL: No child library of "
                                          "sampler utility library to call")));

  EXPECT_THAT(output.str(),
              Not(HasSubstr("KokkosP: FATAL: Kokkos Tools Programming "
                            "Interface's tool-invoked Fence is NULL!")));
}
