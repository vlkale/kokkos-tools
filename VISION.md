
## tools slide 

### example scenario 
Suppose you are asked to obtain an application execution timeline of Kokkos program using CUDA backend on NVIDIA GPU, and then also obtain that timeline on an AMD gpu so as to do an Apples-to apples comparison. You could use amd to prof and nvprof.  But these are not standards across platforms. The problem of performance portable HPC extends down into the supporting hpC Vendor tools.  

## Need: 
1. -  _portably_ support user’s understanding and fixing of a c++ programs: 

 (A) effectiveness of Kokkos program: 
     Understanding: debugging 
     Fixing: resilience / error correction 

 (B) efficiency of Kokkos program : understanding: profiling 
 Fixing: auto-tuning 

2. Tools themselves should be efficient (eg tool induced fences) and effective (easy build, tool don’t modify state of underlying tools). Can use tool for auto tuning correctness and performance. 

## Existing : note raja does not have tools,

### Vision: hpc software tools set that effectively and efficiently assesses and improves a Kokkos program _as a whole_ (from all angles ). The assessment should be done so as to be portable across platforms (apples-to apples comparisons) and the improvement via a tool should be done to ensure code is tuned across all platforms (e.g., for a application x using dynamic scheduling with chunk size 4-16 seems to work best across many platforms ) and then given a particular platform (chunk size 4 is good on NVIDIA gpu, chunk size 8 is good on AMDGPU). 

### To do this: 
- understand aspect of effectiveness and efficiency of Kokkos/C++ programs together , and optimize/together aspects of effectiveness and efficiency together. 

- Multi-objective tuning for a particular metrics in the dimension of the aspect of efficiency, e.g. reduce thread idle time while also managing cache misses. 


### Goal: 
add fuller set of tools to satisfy 1 and make enchantments such as builds eliminating tool-induced fences to satisfy 2. 

- profile and debug deep copy 
- tool induced Fence 
- 

Apps: arborx , Examinimd 
Platforms: perlmutter, crusher, frontier, kahuna, windows, macosx 

Compilers: clang, gcc, nvcc, 
Works with MPI 


## 
