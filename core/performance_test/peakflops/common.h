/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*
  Note that the performance output on skylake can be about twice as the expected
  flops/cycle. This is because the rdtsp() is off by about a factor of two on
  this cpu.
  TODO: remove rdtsp() and use flops/sec for measuring performance.
 */
#ifndef PEAKFLOPS_COMMON_H
#define PEAKFLOPS_COMMON_H

static inline unsigned long long rdtscp()
{
    unsigned long long u;
    asm volatile( "rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0"
                  : "=q"( u )::"%rax", "%rdx", "%rcx" );
    return u;
}

#endif // guard
