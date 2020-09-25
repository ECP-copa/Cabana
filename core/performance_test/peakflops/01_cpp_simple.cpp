#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gtest/gtest.h>
/*
For skylake
Model name:            Intel(R) Xeon(R) Gold 6152 CPU @ 2.10GHz
gcc version 7.2.0
g++ -O3  -march=native peakflops.cpp -DVECLENTH=16
640000000.000000 flops
6482486 clocks
98.727556 flops/clock

For knl
Model name:            Intel(R) Xeon Phi(TM) CPU 7250 @ 1.40GHz
g++ version 8.1.0
g++ -O3  -march=native peakflops.cpp -DVECLENTH=16
640000000.000000 flops
10900834 clocks
58.711104 flops/clock

For haswell/broadwell
Model name:            Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz
gcc 7.3.0/intel 17.0.6/clang 5.0.1

$ g++ -O3  -march=native  peakflops.cpp -fopt-info-vec
$ ./a.out
160000000000.000000 flops
4200302774 clocks
38.092492 flops/clock
x_[0] = 2.161056
x_[1] = 11.734885
x_[2] = 4.641315
x_[3] = 160.230576
x_[4] = 1.043288
x_[5] = 6.865434
x_[6] = 14.501436
x_[7] = 1.216459

$ icpc -O3 -march=core-avx2 -fma peakflops.cpp
[gchen@cn151 openmp]$ ./a.out
160000000000.000000 flops
4200379130 clocks
38.091800 flops/clock
x_[0] = 2.161056
x_[1] = 11.734885
x_[2] = 4.641316
x_[3] = 160.230591
x_[4] = 1.043288
x_[5] = 6.865434
x_[6] = 14.501434
x_[7] = 1.216459

$ clang++ -O3 peakflops.cpp -mavx2 -mfma -ffp-contract=fast
$ ./a.out
160000000000.000000 flops
4200978198 clocks
38.086368 flops/clock
x_[0] = 2.161056
x_[1] = 11.734885
x_[2] = 4.641315
x_[3] = 160.230576
x_[4] = 1.043288
x_[5] = 6.865434
x_[6] = 14.501436
x_[7] = 1.216459
-------------------
For Sandy/Ivy Bridge you need to unroll by >=3:

Only FP Add has dependency on the previous iteration of the loop
FP Add can issue every cycle
FP Add takes three cycles to complete
Thus unrolling by 3/1 = 3 completely hides the latency (theoretically)
FP Mul and FP Load do not have a dependency on the previous iteration and you
can rely on the OoO core to issue them in the near-optimal order. These
instructions could affect the unroll factor only if they lowered the throughput
of FP Add (not the case here, FP Load + FP Add + FP Mul can issue every cycle).

For Haswell you need to unroll by 10 (as we do below, using x0..x9):

Only FMA has dependency on the previous iteration of the loop
FMA can double-issue every cycle (i.e. on average independent instructions take
0.5 cycles) FMA has latency of 5 Thus unrolling by 5/0.5 = 10 completely hides
FMA latency The two FP Load microoperations do not have a dependency on the
previous iteration, and can co-issue with 2x FMA, so they don't affect the
unroll factor.
*/

struct data
{
    float vec[CABANA_PERFORMANCE_VECLENGTH];
};

struct data *
axpy_10( struct data *__restrict__ a, struct data *__restrict__ x0,
         struct data *__restrict__ x1, struct data *__restrict__ x2,
         struct data *__restrict__ x3, struct data *__restrict__ x4,
         struct data *__restrict__ x5, struct data *__restrict__ x6,
         struct data *__restrict__ x7, struct data *__restrict__ x8,
         struct data *__restrict__ x9, struct data *__restrict__ c, long n )
{
    long i;
    int j;

    asm volatile( "# ax+c loop begin" );
    for ( i = 0; i < n; i++ )
    {
#pragma omp simd
        for ( j = 0; j < CABANA_PERFORMANCE_VECLENGTH; j++ )
        {
            x0->vec[j] = a->vec[j] * x0->vec[j] + c->vec[j];
            x1->vec[j] = a->vec[j] * x1->vec[j] + c->vec[j];
            x2->vec[j] = a->vec[j] * x2->vec[j] + c->vec[j];
            x3->vec[j] = a->vec[j] * x3->vec[j] + c->vec[j];
            x4->vec[j] = a->vec[j] * x4->vec[j] + c->vec[j];
            x5->vec[j] = a->vec[j] * x5->vec[j] + c->vec[j];
            x6->vec[j] = a->vec[j] * x6->vec[j] + c->vec[j];
            x7->vec[j] = a->vec[j] * x7->vec[j] + c->vec[j];
            x8->vec[j] = a->vec[j] * x8->vec[j] + c->vec[j];
            x9->vec[j] = a->vec[j] * x9->vec[j] + c->vec[j];
        }
    }
    asm volatile( "# ax+c loop end" );

    for ( j = 0; j < CABANA_PERFORMANCE_VECLENGTH; j++ )
    {
        x0->vec[j] = x0->vec[j] + x1->vec[j] + x2->vec[j] + x3->vec[j] +
                     x4->vec[j] + x5->vec[j] + x6->vec[j] + x7->vec[j] +
                     x8->vec[j] + x9->vec[j] + (float)n;
    }
    return x0;
}

TEST( cpp, simple )
{
    long n = static_cast<long>( CABANA_PERFORMANCE_ITERATIONS );
    long seed = CABANA_PERFORMANCE_SEED;

    data *a_ = new data();
    data *x_ = new data();
    data *x1_ = new data();
    data *x2_ = new data();
    data *x3_ = new data();
    data *x4_ = new data();
    data *x5_ = new data();
    data *x6_ = new data();
    data *x7_ = new data();
    data *x8_ = new data();
    data *x9_ = new data();
    data *c_ = new data();

    long i;
    unsigned short rg[3] = { static_cast<unsigned short>( seed >> 16 ),
                             static_cast<unsigned short>( seed >> 8 ),
                             static_cast<unsigned short>( seed ) };

    for ( i = 0; i < CABANA_PERFORMANCE_VECLENGTH; i++ )
    {
        a_->vec[i] = erand48( rg );
        x_->vec[i] = erand48( rg );
        c_->vec[i] = erand48( rg );
        x1_->vec[i] = erand48( rg );
        x2_->vec[i] = erand48( rg );
        x3_->vec[i] = erand48( rg );
        x4_->vec[i] = erand48( rg );
        x5_->vec[i] = erand48( rg );
        x6_->vec[i] = erand48( rg );
        x7_->vec[i] = erand48( rg );
        x8_->vec[i] = erand48( rg );
        x9_->vec[i] = erand48( rg );
    }

    unsigned long long c0 = rdtscp();

    x_ = axpy_10( a_, x_, x1_, x2_, x3_, x4_, x5_, x6_, x7_, x8_, x9_, c_, n );

    unsigned long long c1 = rdtscp();

    unsigned long long dc = c1 - c0;
    double flops = 10 * 2 * CABANA_PERFORMANCE_VECLENGTH * (double)n;
    printf( "Outer loop n=%lu\n", n );
    printf( "Inner loop VECLENTH=%d\n", CABANA_PERFORMANCE_VECLENGTH );
    printf( "%f flops\n", flops );
    printf( "%llu clocks\n", dc );

    double flops_clock = flops / dc;
    printf( "%f flops/clock\n", flops_clock );

    for ( i = 0; i < CABANA_PERFORMANCE_VECLENGTH; i++ )
    {
        printf( "x_[%ld] = %f\n", i, x_->vec[i] );
    }

    delete a_;
    delete x_;
    delete c_;
    delete x1_;
    delete x2_;
    delete x3_;
    delete x4_;
    delete x5_;
    delete x6_;
    delete x7_;
    delete x8_;
    delete x9_;

    bool acceptable_fraction = false;
    double expected_flops_clock = CABANA_PERFORMANCE_EXPECTED_FLOPS;

    printf( "Expected %f \n", expected_flops_clock );
    printf( "(with margin %f )\n",
            expected_flops_clock * CABANA_PERFORMANCE_ERROR_MARGIN );

    if ( flops_clock > expected_flops_clock * CABANA_PERFORMANCE_ERROR_MARGIN )
    {
        acceptable_fraction = true;
    }

    EXPECT_TRUE( acceptable_fraction );
}
