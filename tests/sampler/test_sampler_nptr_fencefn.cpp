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
    //! expression to check this. If the tool fence function pointer is NULL,
    //! the program should exit.

    for (int iter = 0; iter < 150; iter++) {
      Kokkos::parallel_for("named kernel",
                           Kokkos::RangePolicy<execution_space>(space, 0, 1),
                           *this);
    }
  }

  KOKKOS_FUNCTION void operator()(const int) const {}
};

/**
 * @test This test checks that the sampling tool does not create a null fence
 function pointer.
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

  EXPECT_THAT(output.str(), testing::Not(testing::HasSubstr(
                                "KokkosP: FATAL: Kokkos Tools Programming "
                                "Interface's tool-invoked Fence is NULL!")));
}
