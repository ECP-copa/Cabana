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

#ifndef CABANA_PERFORMANCE_VECLENTH
#define CABANA_PERFORMANCE_VECLENTH (8)
#endif

static inline unsigned long long rdtscp() {
  unsigned long long u;
  asm volatile ("rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0":"=q"(u)::"%rax", "%rdx", "%rcx");
  return u;
}

#endif // guard
