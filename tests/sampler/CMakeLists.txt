kp_add_executable_and_test(
    TARGET_NAME       test_sampling
    SOURCE_FILE       test_sampler.cpp
    KOKKOS_TOOLS_LIBS kp_kokkos_sampler kp_kernel_logger
    KOKKOS_TOOLS_SAMPLER_VERBOSE 2
    KOKKOS_TOOLS_SAMPLER_SKIP 50
    KOKKOS_TOOLS_GLOBALFENCES 1
)
