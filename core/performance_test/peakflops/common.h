/*
  For intel haswell, broadwell: set expected flops/cycle = 32, and veclength=8
  For intel skylake           : set expected flops/cycle = 64, and veclength=16
  Notet that the performance output on skylake can be about twice as the expected flops/cycle. This is because the rdtsp() is off by about a factor of two on this cpu.
  To DO: remove rdtsp() and use flops/sec for measuring performance.
 */
#ifndef PEAKFLOPS_COMMON_H
#define PEAKFLOPS_COMMON_H

#include <gtest/gtest.h>

// NOTE: The user may have to override these for a given platform
#ifndef CABANA_PERFORMANCE_ERROR_MARGIN
#define CABANA_PERFORMANCE_ERROR_MARGIN 0.9
#endif

#ifndef CABANA_PERFORMANCE_EXPECTED_FLOPS
#define CABANA_PERFORMANCE_EXPECTED_FLOPS 32
#endif

#ifndef CABANA_PERFORMANCE_VECLENGTH
#define CABANA_PERFORMANCE_VECLENGTH 8
#endif

static inline unsigned long long rdtscp() {
  unsigned long long u;
  asm volatile ("rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0":"=q"(u)::"%rax", "%rdx", "%rcx");
  return u;
}

#endif // guard
