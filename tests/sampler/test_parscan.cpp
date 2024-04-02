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
    
    int N = 100;
    int64_t result;
    Kokkos::View<int64_t*> post("postfix_sum", N);
    Kokkos::View<int64_t*> pre("prefix_sum", N);

  for (int iter = 0; iter < 150; iter++) { 
      result = 0;
      Kokkos::parallel_scan("Loop1", N,
      KOKKOS_LAMBDA(int64_t i, int64_t& partial_sum, bool is_final) {
      if(is_final) pre(i) = partial_sum;
      partial_sum += i;
      if(is_final) post(i) = partial_sum;
    }, result);
  } // end timestepping loop
 } // end explicit Tester 

KOKKOS_FUNCTION void operator()(const int) const {}
};

static const std::vector<std::string> matchers {
    "(.*)KokkosP: sample 51 calling child-begin function...(.*)",
    "(.*)KokkosP: sample 51 finished with child-begin function.(.*)",
    "(.*)KokkosP: sample 51 calling child-end function...(.*)",
    "(.*)KokkosP: sample 51 calling child-end function.(.*)",
    "(.*)KokkosP: sample 102 calling child-begin function...(.*)",
    "(.*)KokkosP: sample 102 finished with child-begin function.(.*)",
    "(.*)KokkosP: sample 102 calling child-end function...(.*)",
    "(.*)KokkosP: sample 102 calling child-end function.(.*)"};

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
  // std::cout.flush();
  std::cout.rdbuf(coutbuf);
  std::cout << output.str() << std::endl;

  //! Analyze test output.
  for (const auto& matcher : matchers) {
    EXPECT_THAT(output.str(), ::testing::ContainsRegex(matcher));
  }  // end TEST
}
