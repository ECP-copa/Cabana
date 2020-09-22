/*
  Note that the performance output on skylake can be about twice as the expected flops/cycle. This is because the rdtsp() is off by about a factor of two on this cpu.
  TODO: remove rdtsp() and use flops/sec for measuring performance.
 */
#ifndef PEAKFLOPS_COMMON_H
#define PEAKFLOPS_COMMON_H

#include <gtest/gtest.h>

static inline unsigned long long rdtscp() {
  unsigned long long u;
  asm volatile ("rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0":"=q"(u)::"%rax", "%rdx", "%rcx");
  return u;
}

#endif // guard
