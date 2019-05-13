/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Core.hpp>

#include <cstdlib>
#include <iostream>

#ifndef VECLENTH
#define VECLENTH 8
#endif
//clock counter
static inline unsigned long long rdtscp() {
  unsigned long long u;
  asm volatile ("rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0":"=q"(u)::"%rax", "%rdx", "%rcx");
  return u;
}
//---------------------------------------------------------------------------//
// Define particle data.
//---------------------------------------------------------------------------//
// User field enumeration. These will be used to index into the data set. Must
// start at 0 and increment contiguously.
//
// NOTE: Users don't have to make this enum (or some other set of integral
// constants) but it is a nice way to provide meaning to the different data
// types and values assigned to the particles.
//
// NOTE: These enums are also ordered in the same way as the data in the
// template parameters below.
enum UserParticleFields
{
    PositionX = 0,
};

// Designate the types that the particles will hold.
using ParticleDataTypes = Cabana::MemberTypes<float>;

// Declare the memory space.
using MemorySpace = Cabana::HostSpace;

// Set the type for the particle AoSoA.
using ParticleList =
    Cabana::AoSoA<ParticleDataTypes,MemorySpace,VECLENTH>;

// Declare a struct-of-arrays that is identical to the data layout in the
// Cabana AoSoA.
struct data_t
{
    float vec[VECLENTH];
};


//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//
// Move function using the array-of-struct-of-arrays synatx.
void movePx(data_t *__restrict__ a,  data_t *__restrict__ x0,
            data_t *__restrict__ x1, data_t *__restrict__ x2,
            data_t *__restrict__ x3, data_t *__restrict__ x4,
            data_t *__restrict__ x5, data_t *__restrict__ x6,
            data_t *__restrict__ x7, data_t *__restrict__ x8,
            data_t *__restrict__ x9, data_t *__restrict__ c,
            long n, int num_struct )
{
    long i;
    int s;
    int j;

    asm volatile ("# ax+c loop begin");
    for(i = 0; i<n; i++)
    {
        for ( s = 0; s < num_struct; ++s )
        {
            for(j=0; j<VECLENTH; j++)
            {
                x0[s].vec[j] = a[s].vec[j]*x0[s].vec[j]+ c[s].vec[j];
                x1[s].vec[j] = a[s].vec[j]*x1[s].vec[j]+ c[s].vec[j];
                x2[s].vec[j] = a[s].vec[j]*x2[s].vec[j]+ c[s].vec[j];
                x3[s].vec[j] = a[s].vec[j]*x3[s].vec[j]+ c[s].vec[j];
                x4[s].vec[j] = a[s].vec[j]*x4[s].vec[j]+ c[s].vec[j];
                x5[s].vec[j] = a[s].vec[j]*x5[s].vec[j]+ c[s].vec[j];
                x6[s].vec[j] = a[s].vec[j]*x6[s].vec[j]+ c[s].vec[j];
                x7[s].vec[j] = a[s].vec[j]*x7[s].vec[j]+ c[s].vec[j];
                x8[s].vec[j] = a[s].vec[j]*x8[s].vec[j]+ c[s].vec[j];
                x9[s].vec[j] = a[s].vec[j]*x9[s].vec[j]+ c[s].vec[j];
            }
        }
    }
    asm volatile ("# ax+c loop end");
    for ( s = 0; s < num_struct; ++s )
    {
        for(j=0; j<VECLENTH; j++)
        {
            x0[s].vec[j] = x0[s].vec[j]+x1[s].vec[j]+x2[s].vec[j]+x3[s].vec[j]+x4[s].vec[j]+
                           x5[s].vec[j]+x6[s].vec[j]+x7[s].vec[j]+x8[s].vec[j]+x9[s].vec[j];
        }
    }
}

//---------------------------------------------------------------------------//
// Move function using the single particle index and slice syntax.
template<typename SliceType>
void moveSlices(SliceType a,  SliceType x0, SliceType x1, SliceType x2,
               SliceType x3, SliceType x4, SliceType x5, SliceType x6,
               SliceType x7, SliceType x8, SliceType x9, SliceType c,
               long n, int num_struct )
{
    long i;
    int s;
    int j;

    asm volatile ("# ax+c loop begin");
    for(i = 0; i<n; i++)
    {
        for ( s = 0; s < num_struct; ++s )
        {
            for(j=s*VECLENTH; j<(s+1)*VECLENTH; j++)
            {
                x0(j) = a(j)*x0(j)+ c(j);
                x1(j) = a(j)*x1(j)+ c(j);
                x2(j) = a(j)*x2(j)+ c(j);
                x3(j) = a(j)*x3(j)+ c(j);
                x4(j) = a(j)*x4(j)+ c(j);
                x5(j) = a(j)*x5(j)+ c(j);
                x6(j) = a(j)*x6(j)+ c(j);
                x7(j) = a(j)*x7(j)+ c(j);
                x8(j) = a(j)*x8(j)+ c(j);
                x9(j) = a(j)*x9(j)+ c(j);
            }
        }
    }
    asm volatile ("# ax+c loop end");
    for ( s = 0; s < num_struct; ++s )
    {
        for(j=s*VECLENTH; j<(s+1)*VECLENTH; j++)
        {
            x0(j) = x0(j)+x1(j)+x2(j)+x3(j)+x4(j)+x5(j)+x6(j)+x7(j)+x8(j)+x9(j);
        }
    }
}

//---------------------------------------------------------------------------//
// Move function using struct and array indices and slice syntax.
template<typename SliceType>
void moveSlicesWithAccess(SliceType a,  SliceType x0, SliceType x1, SliceType x2,
                         SliceType x3, SliceType x4, SliceType x5, SliceType x6,
                         SliceType x7, SliceType x8, SliceType x9, SliceType c,
                         long n, int num_struct )
{
    long i;
    int s;
    int j;

    asm volatile ("# ax+c loop begin");
    for(i = 0; i<n; i++)
    {
        for ( s = 0; s < num_struct; ++s )
        {
            for(j=0; j<VECLENTH; j++)
            {
                x0.access(s,j) = a.access(s,j)*x0.access(s,j)+ c.access(s,j);
                x1.access(s,j) = a.access(s,j)*x1.access(s,j)+ c.access(s,j);
                x2.access(s,j) = a.access(s,j)*x2.access(s,j)+ c.access(s,j);
                x3.access(s,j) = a.access(s,j)*x3.access(s,j)+ c.access(s,j);
                x4.access(s,j) = a.access(s,j)*x4.access(s,j)+ c.access(s,j);
                x5.access(s,j) = a.access(s,j)*x5.access(s,j)+ c.access(s,j);
                x6.access(s,j) = a.access(s,j)*x6.access(s,j)+ c.access(s,j);
                x7.access(s,j) = a.access(s,j)*x7.access(s,j)+ c.access(s,j);
                x8.access(s,j) = a.access(s,j)*x8.access(s,j)+ c.access(s,j);
                x9.access(s,j) = a.access(s,j)*x9.access(s,j)+ c.access(s,j);
            }
        }
    }
    asm volatile ("# ax+c loop end");
    for ( s = 0; s < num_struct; ++s )
    {
        for(j=0; j<VECLENTH; j++)
        {
            x0.access(s,j) = x0.access(s,j)+x1.access(s,j)+x2.access(s,j)+x3.access(s,j)+
                             x4.access(s,j)+x5.access(s,j)+x6.access(s,j)+x7.access(s,j)+
                             x8.access(s,j)+x9.access(s,j);
        }
    }
}

//---------------------------------------------------------------------------//
// Run the performance test.
void run()
{
    //number of outer loop (e.g. timestepping)
    long n = static_cast<long>(2e4);

    // Declare a number of particles.
    const int array_size = VECLENTH;
    int num_struct = 100;
    int num_particle = num_struct*array_size;

    // Create the particle lists.
    ParticleList a_( "a", num_particle );
    ParticleList x_( "x", num_particle );
    ParticleList c_( "c", num_particle );
    ParticleList x1_( "x1", num_particle );
    ParticleList x2_( "x2", num_particle );
    ParticleList x3_( "x3", num_particle );
    ParticleList x4_( "x4", num_particle );
    ParticleList x5_( "x5", num_particle );
    ParticleList x6_( "x6", num_particle );
    ParticleList x7_( "x7", num_particle );
    ParticleList x8_( "x8", num_particle );
    ParticleList x9_( "x9", num_particle );

    // Get a slice of the x position field from each particle list.
    auto ma = Cabana::slice<PositionX>(a_);
    auto mc = Cabana::slice<PositionX>(c_);
    auto m0 = Cabana::slice<PositionX>(x_);
    auto m1 = Cabana::slice<PositionX>(x1_);
    auto m2 = Cabana::slice<PositionX>(x2_);
    auto m3 = Cabana::slice<PositionX>(x3_);
    auto m4 = Cabana::slice<PositionX>(x4_);
    auto m5 = Cabana::slice<PositionX>(x5_);
    auto m6 = Cabana::slice<PositionX>(x6_);
    auto m7 = Cabana::slice<PositionX>(x7_);
    auto m8 = Cabana::slice<PositionX>(x8_);
    auto m9 = Cabana::slice<PositionX>(x9_);

    // Initialize particle data.
    long seed = 76843802738543;
    unsigned short rg[3] = { static_cast<unsigned short>(seed >> 16),
                             static_cast<unsigned short>(seed >> 8),
                             static_cast<unsigned short>(seed) };
    for (int idx = 0; idx < num_particle; ++idx)
    {
        ma(idx) = erand48(rg);
        m0(idx) = erand48(rg);
        mc(idx) = erand48(rg);
        m1(idx) = erand48(rg);
        m2(idx) = erand48(rg);
        m3(idx) = erand48(rg);
        m4(idx) = erand48(rg);
        m5(idx) = erand48(rg);
        m6(idx) = erand48(rg);
        m7(idx) = erand48(rg);
        m8(idx) = erand48(rg);
        m9(idx) = erand48(rg);
    }

    // Cast particle data to an explicit array-of-struct-of-arrays.
    auto* pa = (data_t*)(a_.data());
    auto* px = (data_t*)(x_.data());
    auto* pc = (data_t*)(c_.data());
    auto* px1 = (data_t*)(x1_.data());
    auto* px2 = (data_t*)(x2_.data());
    auto* px3 = (data_t*)(x3_.data());
    auto* px4 = (data_t*)(x4_.data());
    auto* px5 = (data_t*)(x5_.data());
    auto* px6 = (data_t*)(x6_.data());
    auto* px7 = (data_t*)(x7_.data());
    auto* px8 = (data_t*)(x8_.data());
    auto* px9 = (data_t*)(x9_.data());

    // Print initial conditions.
    for (int idx = 0; idx < array_size; ++idx)
    {
        printf("x_[%d] = %f,%f\n", idx, m0(idx),px->vec[idx]);
    }

    //move particles
    unsigned long long c1 = rdtscp();
    movePx(pa,px,px1,px2,px3,px4,px5,px6,px7,px8,px9,pc,n,num_struct);
    unsigned long long c2 = rdtscp();

    unsigned long long c3 = rdtscp();
    moveSlices(ma,m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,mc,n,num_struct);
    unsigned long long c4 = rdtscp();

    unsigned long long c5 = rdtscp();
    moveSlicesWithAccess(ma,m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,mc,n,num_struct);
    unsigned long long c6 = rdtscp();

    // Calculate times and number of flops.
    unsigned long long dc1 = c2 - c1;
    unsigned long long dc2 = c4 - c3;
    unsigned long long dc3 = c6 - c5;
    double flops = 10*2*num_particle*(double)n;

    // Output results.
    std::cout << std::endl;
    std::cout<<flops<<" flops" << std::endl;;
    std::cout << std::endl;
    std::cout << "AoSoA Cast" << std::endl;
    std::cout<<dc1<<" clocks 1"<<std::endl;
    std::cout<<flops/dc1<<" flops/clock 1\n";
    std::cout << std::endl;
    std::cout << "Slice Single Index" << std::endl;
    std::cout<<dc2<<" clocks 2"<<std::endl;
    std::cout<<flops/dc2<<" flops/clock 2\n";
    std::cout << std::endl;
    std::cout << "Slice Struct/Array Index" << std::endl;
    std::cout<<dc3<<" clocks 3"<<std::endl;
    std::cout<<flops/dc3<<" flops/clock 3\n";
    std::cout << std::endl;

    for (int idx = 0; idx < VECLENTH; idx++)
    {
        printf("x_[%d] = %f\n", idx, m0(idx));
    }
}


//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char **argv )
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    run();

    return 0;
}

//---------------------------------------------------------------------------//
