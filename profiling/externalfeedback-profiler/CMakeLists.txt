# Minimal external feedback plugin (Hello->hash->World)
enable_language(C)
kp_add_library(kp_externalfeedback_profiler
    kp_externalfeedback_profiler.cpp
    externalfeedback.cpp
)

# Small runner to demo the external feedback handshake without requiring Kokkos
kp_add_executable(externalfeedback_demo_runner demo_runner.cpp)
target_link_libraries(externalfeedback_demo_runner PRIVATE kp_externalfeedback_profiler)

# POSIX-only demo runner (pure C, avoids C++ std headers)
kp_add_executable(externalfeedback_demo_runner_posix demo_runner_posix.c)
target_link_libraries(externalfeedback_demo_runner_posix PRIVATE kp_externalfeedback_profiler pthread)
set_target_properties(externalfeedback_demo_runner_posix PROPERTIES LINKER_LANGUAGE CXX)

# Super-minimal runner that just prints "world" (no library linkage)
kp_add_executable(externalfeedback_min_runner demo_min_runner.c)
set_target_properties(externalfeedback_min_runner PROPERTIES LINKER_LANGUAGE C)

# Signal-driven POSIX runner
kp_add_executable(externalfeedback_demo_runner_signal demo_runner_signal_posix.c)
target_link_libraries(externalfeedback_demo_runner_signal PRIVATE kp_externalfeedback_profiler pthread)
set_target_properties(externalfeedback_demo_runner_signal PROPERTIES LINKER_LANGUAGE CXX)

# Pure C version (no C++ std headers needed)
add_library(kp_externalfeedback_c STATIC externalfeedback_c.c)
target_link_libraries(kp_externalfeedback_c PRIVATE pthread)
set_target_properties(kp_externalfeedback_c PROPERTIES LINKER_LANGUAGE C)

kp_add_executable(externalfeedback_demo_runner_signal_c demo_runner_signal_c.c)
target_link_libraries(externalfeedback_demo_runner_signal_c PRIVATE kp_externalfeedback_c pthread)
set_target_properties(externalfeedback_demo_runner_signal_c PROPERTIES LINKER_LANGUAGE C)
