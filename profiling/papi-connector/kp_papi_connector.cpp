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

#include <stdio.h>
#include <inttypes.h>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <string>
#include <iostream>
#include <stack>
#include <sstream>
#include <map>

#include "kp_papi_connector_domain.h"

typedef void (*initFunction)(const int, const uint64_t, const uint32_t, void*);
typedef void (*finalizeFunction)();
typedef void (*beginFunction)(const char*, const uint32_t, uint64_t*);
typedef void (*endFunction)(uint64_t);

static initFunction initProfileLibrary         = NULL;
static finalizeFunction finalizeProfileLibrary = NULL;
static beginFunction beginForCallee            = NULL;
static beginFunction beginScanCallee           = NULL;
static beginFunction beginReduceCallee         = NULL;
static endFunction endForCallee                = NULL;
static endFunction endScanCallee               = NULL;
static endFunction endReduceCallee             = NULL;



extern "C" void kokkosp_init_library(const int loadSeq,
                                     const uint64_t interfaceVer,
                                     const uint32_t devInfoCount,
                                     void* deviceInfo) {
  printf("-----------------------------------------------------------\n");
  printf("KokkosP: PAPI Connector (sequence is %d, version: %llu)\n", loadSeq,
         interfaceVer);
  printf("-----------------------------------------------------------\n");
  
  char* profileLibrary = getenv("KOKKOS_TOOLS_LIBS");
  if (NULL == profileLibrary) {
    printf(
        "Checking KOKKOS_PROFILE_LIBRARY. WARNING: This is a depreciated "
        "variable. Please use KOKKOS_TOOLS_LIBS\n");
    profileLibrary = getenv("KOKKOS_PROFILE_LIBRARY");
    if (NULL == profileLibrary) {
      std::cout << "KokkosP: FATAL: No library to call in " << profileLibrary
                << "!\n";
      exit(-1);
    }
  }


  
  char* envBuffer = (char*)malloc(sizeof(char) * (strlen(profileLibrary) + 1));
  strcpy(envBuffer, profileLibrary);

  char* nextLibrary = strtok(envBuffer, ";");

  for (int i = 0; i < loadSeq; i++) {
    nextLibrary = strtok(NULL, ";");
  }

  nextLibrary = strtok(NULL, ";");

  if (NULL == nextLibrary) {
    std::cout << "KokkosP: FATAL: No child library of sampler utility library "
                 "to call in "
              << profileLibrary << "!\n";
    exit(-1);
  } else {
    if (tool_verbosity > 0) {
      std::cout << "KokkosP: Next library to call: " << nextLibrary << "\n";
      std::cout << "KokkosP: Loading child library of sampler..\n";
    }

    void* childLibrary = dlopen(nextLibrary, RTLD_NOW | RTLD_GLOBAL);

    if (NULL == childLibrary) {
      fprintf(stderr, "KokkosP: Error: Unable to load: %s (Error=%s)\n",
              nextLibrary, dlerror());
      exit(-1);
    } else {
      beginForCallee =
          (beginFunction)dlsym(childLibrary, "kokkosp_begin_parallel_for");
      beginScanCallee =
          (beginFunction)dlsym(childLibrary, "kokkosp_begin_parallel_scan");
      beginReduceCallee =
          (beginFunction)dlsym(childLibrary, "kokkosp_begin_parallel_reduce");

      endScanCallee =
          (endFunction)dlsym(childLibrary, "kokkosp_end_parallel_scan");
      endForCallee =
          (endFunction)dlsym(childLibrary, "kokkosp_end_parallel_for");
      endReduceCallee =
          (endFunction)dlsym(childLibrary, "kokkosp_end_parallel_reduce");

      initProfileLibrary =
          (initFunction)dlsym(childLibrary, "kokkosp_init_library");
      finalizeProfileLibrary =
          (finalizeFunction)dlsym(childLibrary, "kokkosp_finalize_library");

      if (NULL != initProfileLibrary) {
        (*initProfileLibrary)(loadSeq + 1, interfaceVer, devInfoCount,
                              deviceInfo);
      }

      if (tool_verbosity > 0) {
        std::cout << "KokkosP: Function Status:\n";
        std::cout << "KokkosP: begin-parallel-for:      "
                  << ((beginForCallee == NULL) ? "no" : "yes") << "\n";
        std::cout << "KokkosP: begin-parallel-scan:      "
                  << ((beginScanCallee == NULL) ? "no" : "yes") << "\n";
        std::cout << "KokkosP: begin-parallel-reduce:      "
                  << ((beginReduceCallee == NULL) ? "no" : "yes") << "\n";
        std::cout << "KokkosP: end-parallel-for:      "
                  << ((endForCallee == NULL) ? "no" : "yes") << "\n";
        std::cout << "KokkosP: end-parallel-scan:      "
                  << ((endScanCallee == NULL) ? "no" : "yes") << "\n";
        std::cout << "KokkosP: end-parallel-reduce:      "
                  << ((endReduceCallee == NULL) ? "no" : "yes") << "\n";
      }
    }
  }


  free(envBuffer);
  /* The following advanced functions of PAPI's high-level API are not part
   * of the official release. But they might be introduced in later PAPI
   * releases. PAPI_hl_init is now called from the first PAPI_hl_region_begin
   * call.
   */
  // PAPI_hl_init();
  /* set default values */
  // PAPI_hl_set_events("perf::TASK-CLOCK,PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_FP_OPS");
}

extern "C" void kokkosp_finalize_library() {
  /* The following advanced functions of PAPI's high-level API are not part
   * of the official release. But they might be introduced in later PAPI
   * releases. PAPI_hl_print_output is registered by the "atexit" function and
   * will be called at process termination, see
   * http://man7.org/linux/man-pages/man3/atexit.3.html.
   */
  // PAPI_hl_print_output();
  // PAPI_hl_finalize();

  printf("-----------------------------------------------------------\n");
  printf("KokkosP: Finalization of PAPI Connector. Complete.\n");
  printf("-----------------------------------------------------------\n");
}

extern "C" void kokkosp_begin_parallel_for(const char* name, uint32_t devid,
                                           uint64_t* kernid) {
  // printf("kokkosp_begin_parallel_for: %s %d\n", name, *kernid);
  std::stringstream ss;
  ss << "kokkosp_parallel_for:" << name;
  parallel_for_name.push(ss.str());
  PAPI_hl_region_begin(ss.str().c_str());
}
extern "C" void kokkosp_end_parallel_for(uint64_t kernid) {
  // printf("kokkosp_end_parallel_for: %d\n", kernid);
  if (parallel_for_name.empty() == false) {
    PAPI_hl_region_end(parallel_for_name.top().c_str());
    parallel_for_name.pop();
  } else {
    printf("Begin callback of parallel_for does not exist!\n");
  }
}

extern "C" void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid,
                                              uint64_t* kernid) {
  // printf("kokkosp_begin_parallel_reduce: %s %d\n", name, *kernid);
  std::stringstream ss;
  ss << "kokkosp_parallel_reduce:" << name;
  parallel_reduce_name.push(ss.str());
  PAPI_hl_region_begin(ss.str().c_str());
}
extern "C" void kokkosp_end_parallel_reduce(uint64_t kernid) {
  // printf("kokkosp_end_parallel_reduce: %d\n", kernid);
  if (parallel_reduce_name.empty() == false) {
    PAPI_hl_region_end(parallel_reduce_name.top().c_str());
    parallel_reduce_name.pop();
  } else {
    printf("Begin callback of parallel_reduce does not exist!\n");
  }
}

extern "C" void kokkosp_begin_parallel_scan(const char* name, uint32_t devid,
                                            uint64_t* kernid) {
  // printf("kokkosp_begin_parallel_scan: %s %d\n", name, *kernid);
  std::stringstream ss;
  ss << "kokkosp_parallel_scan:" << name;
  parallel_scan_name.push(ss.str());
  PAPI_hl_region_begin(ss.str().c_str());
}
extern "C" void kokkosp_end_parallel_scan(uint64_t kernid) {
  // printf("kokkosp_end_parallel_scan: %d\n", kernid);
  if (parallel_scan_name.empty() == false) {
    PAPI_hl_region_end(parallel_scan_name.top().c_str());
    parallel_scan_name.pop();
  } else {
    printf("Begin callback of parallel_scan does not exist!\n");
  }
}

extern "C" void kokkosp_push_profile_region(const char* name) {
  std::stringstream ss;
  ss << "kokkosp_profile_region:" << name;
  region_name.push(ss.str());
  PAPI_hl_region_begin(ss.str().c_str());
}

extern "C" void kokkosp_profile_event(const char* name) {
  if (region_name.empty() == false) {
    PAPI_hl_read(region_name.top().c_str());
  }
}

extern "C" void kokkosp_pop_profile_region() {
  if (region_name.empty() == false) {
    PAPI_hl_region_end(region_name.top().c_str());
    region_name.pop();
  } else {
    printf("Region does not exist!\n");
  }
}

extern "C" void kokkosp_begin_deep_copy(SpaceHandle dst_handle,
                                        const char* dst_name,
                                        const void* dst_ptr,
                                        SpaceHandle src_handle,
                                        const char* src_name,
                                        const void* src_ptr, uint64_t size) {
  PAPI_hl_region_begin("kokkosp_deep_copy");
}
extern "C" void kokkosp_end_deep_copy() {
  PAPI_hl_region_end("kokkosp_deep_copy");
}

extern "C" void kokkosp_create_profile_section(const char* name,
                                               uint32_t* sec_id) {
  // printf("kokkosp_create_profile_section: %s %d\n", name, *sec_id);
  profile_section.insert(std::pair<uint32_t, std::string>(*sec_id, name));
}
extern "C" void kokkosp_destroy_profile_section(uint32_t sec_id) {
  // printf("kokkosp_destroy_profile_section: %d\n", sec_id);
  std::map<uint32_t, std::string>::iterator it;
  it = profile_section.find(sec_id);
  if (it != profile_section.end()) profile_section.erase(it);
}

extern "C" void kokkosp_start_profile_section(uint32_t sec_id) {
  // printf("kokkosp_start_profile_section: %d\n", sec_id);
  std::map<uint32_t, std::string>::iterator it;
  it = profile_section.find(sec_id);
  if (it != profile_section.end()) {
    std::stringstream ss;
    ss << it->second << ":" << sec_id;
    region_name.push(ss.str());
    PAPI_hl_region_begin(ss.str().c_str());
  }
}
extern "C" void kokkosp_stop_profile_section(uint32_t sec_id) {
  // printf("kokkosp_stop_profile_section: %d\n", sec_id);
  std::map<uint32_t, std::string>::iterator it;
  it = profile_section.find(sec_id);
  if (it != profile_section.end()) {
    std::stringstream ss;
    ss << it->second << ":" << sec_id;
    region_name.push(ss.str());
    PAPI_hl_region_end(ss.str().c_str());
  }
}
