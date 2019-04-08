#ifndef PEAKFLOPS_COMMON_H
#define PEAKFLOPS_COMMON_H

#include <gtest/gtest.h>

// NOTE: The user may have to override these for a given platform
#ifndef ERROR_MARGIN
#define ERROR_MARGIN 0.9
#endif

#ifndef EXPECTED_FLOPS
#define EXPECTED_FLOPS 32
#endif

#ifndef VECLENTH
#define VECLENTH (8)
#endif

static inline unsigned long long rdtscp() {
  unsigned long long u;
  asm volatile ("rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0":"=q"(u)::"%rax", "%rdx", "%rcx");
  return u;
}

#endif // guard
