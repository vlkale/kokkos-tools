kp_add_executable_and_test(
    TARGET_NAME       test_sampling_parfor
    SOURCE_FILE       test_parfor.cpp
    KOKKOS_TOOLS_LIBS kp_kokkos_sampler kp_kernel_logger
    KOKKOS_TOOLS_SAMPLER_VERBOSE 2
    KOKKOS_TOOLS_SAMPLER_SKIP 50
    KOKKOS_TOOLS_GLOBALFENCES 1
)

kp_add_executable_and_test(
    TARGET_NAME       test_sampling_parscan
    SOURCE_FILE       test_parscan.cpp
    KOKKOS_TOOLS_LIBS kp_kokkos_sampler kp_kernel_logger
    KOKKOS_TOOLS_SAMPLER_VERBOSE 2
    KOKKOS_TOOLS_SAMPLER_SKIP 50
    KOKKOS_TOOLS_GLOBALFENCES 1
)

kp_add_executable_and_test(
    TARGET_NAME       test_sampling_parreduce
    SOURCE_FILE       test_parreduce.cpp
    KOKKOS_TOOLS_LIBS kp_kokkos_sampler kp_kernel_logger
    KOKKOS_TOOLS_SAMPLER_VERBOSE 2
    KOKKOS_TOOLS_SAMPLER_SKIP 50
    KOKKOS_TOOLS_GLOBALFENCES 1
)

kp_add_executable_and_test(
    TARGET_NAME       test_sampling_nonullfence
    SOURCE_FILE       test_sampler_nptr_fencefn.cpp
    KOKKOS_TOOLS_LIBS kp_kokkos_sampler kp_kernel_logger
    KOKKOS_TOOLS_SAMPLER_VERBOSE 2
    KOKKOS_TOOLS_SAMPLER_SKIP 50
    KOKKOS_TOOLS_GLOBALFENCES 1
)
