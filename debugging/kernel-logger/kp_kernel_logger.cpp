//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <cstdio>
#include <inttypes.h>
#include <vector>
#include <string>
#include <limits>
#include <cstring>
#include <iostream>

std::vector<std::string> regions;
static uint64_t uniqID;
struct SpaceHandle {
  char name[64];
};

void kokkosp_print_region_stack_indent(const int level) {
  std::cout << "KokkosP: ";

  for (int i = 0; i < level; i++) {
    std::cout << "  ";
  }
}

int kokkosp_print_region_stack() {
  int level = 0;

  for (auto regItr = regions.begin(); regItr != regions.end(); regItr++) {
    kokkosp_print_region_stack_indent(level);
    std::cout << *regItr << '\n';

    level++;
  }

  return level;
}

extern "C" void kokkosp_init_library(const int loadSeq,
                                     const uint64_t interfaceVer,
                                     const uint32_t /*devInfoCount*/,
                                     void* /*deviceInfo*/) {
  std::cout << "KokkosP: Kernel Logger Library Initialized (sequence is "
            << loadSeq << ", version: " << interfaceVer << ")\n";
  uniqID = 0;
}

extern "C" void kokkosp_finalize_library() {
  std::cout << "KokkosP: Kokkos library finalization called.\n";
}

extern "C" void kokkosp_begin_parallel_for(const char* name,
                                           const uint32_t devID,
                                           uint64_t* kID) {
  *kID = uniqID++;

  std::cout << "KokkosP: Executing parallel-for kernel on device " << devID
            << " with unique execution identifier " << *kID << '\n';

  int level = kokkosp_print_region_stack();
  kokkosp_print_region_stack_indent(level);
  std::cout << "    " << name << '\n';
}

extern "C" void kokkosp_end_parallel_for(const uint64_t kID) {
  std::abort();
  std::cout << "KokkosP: Execution of kernel " << kID << " is completed.\n";
}

extern "C" void kokkosp_begin_parallel_scan(const char* name,
                                            const uint32_t devID,
                                            uint64_t* kID) {
  *kID = uniqID++;

  std::cout << "KokkosP: Executing parallel-scan kernel on device " << devID
            << " with unique execution identifier " << *kID << '\n';

  int level = kokkosp_print_region_stack();
  kokkosp_print_region_stack_indent(level);

  std::cout << "    " << name << '\n';
}

extern "C" void kokkospk_end_parallel_scan(const uint64_t kID) {
  std::cout << "KokkosP: Execution of kernel " << kID << " is completed.\n";
}

extern "C" void kokkosp_begin_parallel_reduce(const char* name,
                                              const uint32_t devID,
                                              uint64_t* kID) {
  *kID = uniqID++;

  std::cout << "KokkosP: Executing parallel-reduce kernel on device " << devID
            << " with unique execution identifier " << *kID << '\n';

  int level = kokkosp_print_region_stack();
  kokkosp_print_region_stack_indent(level);

  std::cout << "    " << name << '\n';
}

extern "C" void kokkosp_end_parallel_reduce(const uint64_t kID) {
  std::cout << "KokkosP: Execution of kernel " << kID << " is completed.\n";
}

extern "C" void kokkosp_begin_fence(const char* name, const uint32_t devID,
                                    uint64_t* kID) {
  // filter out fence as this is a duplicate and unneeded (causing the tool to
  // hinder performance of application). We use strstr for checking if the
  // string contains the label of a fence (we assume the user will always have
  // the word fence in the label of the fence).
  if (std::strstr(name, "Kokkos Profile Tool Fence")) {
    // set the dereferenced execution identifier to be the maximum value of
    // uint64_t, which is assumed to never be assigned
    *kID = std::numeric_limits<uint64_t>::max();
  } else {
    *kID = uniqID++;

    std::cout << "KokkosP: Executing fence on device " << devID
              << " with unique execution identifier " << kID << '\n';

    int level = kokkosp_print_region_stack();
    kokkosp_print_region_stack_indent(level);

    std::cout << "    " << name << '\n';
  }
}

extern "C" void kokkosp_end_fence(const uint64_t kID) {
  // if we find the kID to be maximum value uint64_t, then the callback is
  // dealing with the application's fence, which we filtered out in the callback
  // for fences
  if (kID != std::numeric_limits<uint64_t>::max()) {
    std::cout << "KokkosP: Execution of fence %llu is completed.\n";
  }
}

extern "C" void kokkosp_push_profile_region(char* regionName) {
  std::cout << "KokkosP: Entering profiling region: " << regionName << '\n';

  std::string regionNameStr(regionName);
  regions.push_back(regionNameStr);
}

extern "C" void kokkosp_pop_profile_region() {
  if (regions.size() > 0) {
    std::cout << "KokkosP: Exiting profiling region: " << regions.back()
              << '\n';
    regions.pop_back();
  }
}

extern "C" void kokkosp_allocate_data(SpaceHandle handle, const char* name,
                                      void* ptr, uint64_t size) {
  std::cout << "KokkosP: Allocate<" << handle.name << "> name: " << name
            << " pointer: " << ptr << " size: " << size << '\n';
}

extern "C" void kokkosp_deallocate_data(SpaceHandle handle, const char* name,
                                        void* ptr, uint64_t size) {
  std::cout << "KokkosP: Deallocate<" << handle.name << "> name: " << name
            << " pointer: " << ptr << " size: " << size << '\n';
}

extern "C" void kokkosp_begin_deep_copy(SpaceHandle dst_handle,
                                        const char* dst_name,
                                        const void* dst_ptr,
                                        SpaceHandle src_handle,
                                        const char* src_name,
                                        const void* src_ptr, uint64_t size) {
  std::cout << "KokkosP: DeepCopy<" << dst_handle.name << ',' << src_handle.name
            << "> DST(name: " << dst_name << " pointer: " << dst_ptr
            << ") SRC(name: " << src_name << " pointer " << src_ptr
            << ") Size: " << size << '\n';
}
