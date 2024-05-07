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
#include "kp_core.hpp"

static bool tool_globfences;

namespace KokkosTools {
namespace PAPIconnector {

// a hash table mapping kID to nestedkID
static std::unordered_map<uint64_t, uint64_t> infokID;

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

void kokkosp_request_tool_settings(const uint32_t,
                                   Kokkos_Tools_ToolSettings* settings) {
  settings->requires_global_fencing = true;
  if (tool_globfences) {
    settings->requires_global_fencing = true;
  } else {
    settings->requires_global_fencing = false;
  }
}
 void kokkosp_init_library(const int loadSeq,
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

  const char* tool_global_fences = getenv("KOKKOS_TOOLS_GLOBALFENCES");
  if (NULL != tool_global_fences) {
    tool_globfences = (atoi(tool_global_fences) != 0);
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

void kokkosp_finalize_library() {
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

void kokkosp_begin_parallel_for(const char* name, uint32_t devid,
                                           uint64_t* kernid) {
  // printf("kokkosp_begin_parallel_for: %s %d\n", name, *kernid);
  std::stringstream ss;
  ss << "kokkosp_parallel_for:" << name;
  parallel_for_name.push(ss.str());
  PAPI_hl_region_begin(ss.str().c_str());

      if (NULL != beginForCallee) {
      uint64_t nestedkID = 0;
      (*beginForCallee)(name, devID, &nestedkID);
      if (tool_verbosity > 0) {
        std::cout << "KokkosP: PAPI child callee " << *kID
                  << " finished with child-begin function.\n";
      }
      infokID.insert({*kID, nestedkID});
    }
}
void kokkosp_end_parallel_for(uint64_t kernid) {
  // printf("kokkosp_end_parallel_for: %d\n", kernid);
  if (parallel_for_name.empty() == false) {
    PAPI_hl_region_end(parallel_for_name.top().c_str());
    parallel_for_name.pop();
  } else {
    printf("Begin callback of parallel_for does not exist!\n");
  }
  if(NULL!=endForCallee)
  {
  (*endForCallee)(retrievedNestedkID);
  if (tool_verbosity > 0) {
        std::cout << "KokkosP: papi child callee " << kID
                  << " finished with its end function.\n";
     }
    infokID.erase(kID);
  }
}

void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid,
                                              uint64_t* kernid) {
  // printf("kokkosp_begin_parallel_reduce: %s %d\n", name, *kernid);
  std::stringstream ss;
  ss << "kokkosp_parallel_reduce:" << name;
  parallel_reduce_name.push(ss.str());
  PAPI_hl_region_begin(ss.str().c_str());
}
void kokkosp_end_parallel_reduce(uint64_t kernid) {
  // printf("kokkosp_end_parallel_reduce: %d\n", kernid);
  if (parallel_reduce_name.empty() == false) {
    PAPI_hl_region_end(parallel_reduce_name.top().c_str());
    parallel_reduce_name.pop();
  } else {
    printf("Begin callback of parallel_reduce does not exist!\n");
  }
}

 void kokkosp_begin_parallel_scan(const char* name, uint32_t devid,
                                            uint64_t* kernid) {
  // printf("kokkosp_begin_parallel_scan: %s %d\n", name, *kernid);
  std::stringstream ss;
  ss << "kokkosp_parallel_scan:" << name;
  parallel_scan_name.push(ss.str());
  PAPI_hl_region_begin(ss.str().c_str());
}
void kokkosp_end_parallel_scan(uint64_t kernid) {
  // printf("kokkosp_end_parallel_scan: %d\n", kernid);
  if (parallel_scan_name.empty() == false) {
    PAPI_hl_region_end(parallel_scan_name.top().c_str());
    parallel_scan_name.pop();
  } else {
    printf("Begin callback of parallel_scan does not exist!\n");
  }
}

void kokkosp_push_profile_region(const char* name) {
  std::stringstream ss;
  ss << "kokkosp_profile_region:" << name;
  region_name.push(ss.str());
  PAPI_hl_region_begin(ss.str().c_str());
}

void kokkosp_profile_event(const char* name) {
  if (region_name.empty() == false) {
    PAPI_hl_read(region_name.top().c_str());
  }
}

void kokkosp_pop_profile_region() {
  if (region_name.empty() == false) {
    PAPI_hl_region_end(region_name.top().c_str());
    region_name.pop();
  } else {
    printf("Region does not exist!\n");
  }
}

void kokkosp_begin_deep_copy(SpaceHandle dst_handle,
                                        const char* dst_name,
                                        const void* dst_ptr,
                                        SpaceHandle src_handle,
                                        const char* src_name,
                                        const void* src_ptr, uint64_t size) {
  PAPI_hl_region_begin("kokkosp_deep_copy");
}
void kokkosp_end_deep_copy() {
  PAPI_hl_region_end("kokkosp_deep_copy");
}

void kokkosp_create_profile_section(const char* name,
                                               uint32_t* sec_id) {
  // printf("kokkosp_create_profile_section: %s %d\n", name, *sec_id);
  profile_section.insert(std::pair<uint32_t, std::string>(*sec_id, name));
}
void kokkosp_destroy_profile_section(uint32_t sec_id) {
  // printf("kokkosp_destroy_profile_section: %d\n", sec_id);
  std::map<uint32_t, std::string>::iterator it;
  it = profile_section.find(sec_id);
  if (it != profile_section.end()) profile_section.erase(it);
}

void kokkosp_start_profile_section(uint32_t sec_id) {
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
void kokkosp_stop_profile_section(uint32_t sec_id) {
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

Kokkos::Tools::Experimental::EventSet get_event_set() {
  Kokkos::Tools::Experimental::EventSet my_event_set;
  memset(&my_event_set, 0,
         sizeof(my_event_set));  // zero any pointers not set here
  my_event_set.request_tool_settings  = kokkosp_request_tool_settings;
  my_event_set.init                   = kokkosp_init_library;
  my_event_set.finalize               = kokkosp_finalize_library;
  my_event_set.push_region            = kokkosp_push_profile_region;
  my_event_set.pop_region             = kokkosp_pop_profile_region;
  my_event_set.begin_parallel_for     = kokkosp_begin_parallel_for;
  my_event_set.begin_parallel_reduce  = kokkosp_begin_parallel_reduce;
  my_event_set.begin_parallel_scan    = kokkosp_begin_parallel_scan;
  my_event_set.end_parallel_for       = kokkosp_end_parallel_for;
  my_event_set.end_parallel_reduce    = kokkosp_end_parallel_reduce;
  my_event_set.end_parallel_scan      = kokkosp_end_parallel_scan;
  my_event_set.create_profile_section = kokkosp_create_profile_section;
  my_event_set.start_profile_section  = kokkosp_start_profile_section;
  my_event_set.stop_profile_section   = kokkosp_stop_profile_section;
  my_event_set.profile_event          = kokkosp_profile_event;
  return my_event_set;
}

}  // end namespace PAPIconnector
}  // end namespace KokkosTools

extern "C" {

namespace impl = KokkosTools::PAPIConnector;

EXPOSE_TOOL_SETTINGS(impl::kokkosp_request_tool_settings)
EXPOSE_INIT(impl::kokkosp_init_library)
EXPOSE_FINALIZE(impl::kokkosp_finalize_library)
EXPOSE_PUSH_REGION(impl::kokkosp_push_profile_region)
EXPOSE_POP_REGION(impl::kokkosp_pop_profile_region)
EXPOSE_BEGIN_PARALLEL_FOR(impl::kokkosp_begin_parallel_for)
EXPOSE_END_PARALLEL_FOR(impl::kokkosp_end_parallel_for)
EXPOSE_BEGIN_PARALLEL_SCAN(impl::kokkosp_begin_parallel_scan)
EXPOSE_END_PARALLEL_SCAN(impl::kokkosp_end_parallel_scan)
EXPOSE_BEGIN_PARALLEL_REDUCE(impl::kokkosp_begin_parallel_reduce)
EXPOSE_END_PARALLEL_REDUCE(impl::kokkosp_end_parallel_reduce)
EXPOSE_CREATE_PROFILE_SECTION(impl::kokkosp_create_profile_section)
EXPOSE_START_PROFILE_SECTION(impl::kokkosp_start_profile_section)
EXPOSE_STOP_PROFILE_SECTION(impl::kokkosp_stop_profile_section)
EXPOSE_PROFILE_EVENT(impl::kokkosp_profile_event);
}  // extern "C"
