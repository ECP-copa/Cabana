#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "rdtsc.h"

#ifndef VECLENTH
#define VECLENTH (8)
#endif

typedef Kokkos::View<float[VECLENTH],Kokkos::MemoryTraits<Kokkos::Restrict> > view_type;

/*
void __attribute__ ((noinline)) inita(const view_type & __restrict__  a,const view_type & __restrict__  x0,const view_type & __restrict__   x1,const view_type & __restrict__   x2,const view_type & __restrict__  c){
    asm volatile ("# ax+c loop begin");
    for(int l=1;l<10000;l++){
#pragma omp simd
        for(int j=0; j<8; j++){
            x1(j) =a(j)*x1(j) +c(j);
            x2(j) =a(j)*x2(j) +c(j);
            x0(j) =a(j)*x0(j) +c(j);
        }
    }
    asm volatile ("# ax+c loop end");
}

*/

// TODO: move to shared file
static inline unsigned long long rdtscp() {
    unsigned long long u;
    asm volatile ("rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0":"=q"(u)::"%rax", "%rdx", "%rcx");
    return u;
}


struct data{
    float vec[VECLENTH];
};

//struct data * axpy_10(struct data *restrict a, struct data *restrict x0, struct data *restrict x1, struct data *restrict x2, struct data *restrict x3, struct data *restrict x4, struct data *restrict x5, struct data *restrict x6, struct data *restrict x7, struct data *restrict x8, struct data *restrict x9,struct data *restrict c, long n) {
//struct data * axpy_10(struct data *__restrict__ a, struct data *__restrict__ x0, struct data *__restrict__ x1, struct data *__restrict__ x2, struct data *__restrict__ x3, struct data *__restrict__ x4, struct data *__restrict__ x5, struct data *__restrict__ x6, struct data *__restrict__ x7, struct data *__restrict__ x8, struct data *__restrict__ x9,struct data *__restrict__ c, long n) {

view_type //__attribute__ ((noinline))
axpy_10(
        const view_type & __restrict__  a,
        const view_type & __restrict__  x0,
        const view_type & __restrict__  x1,
        const view_type & __restrict__  x2,
        const view_type & __restrict__  x3,
        const view_type & __restrict__  x4,
        const view_type & __restrict__  x5,
        const view_type & __restrict__  x6,
        const view_type & __restrict__  x7,
        const view_type & __restrict__  x8,
        const view_type & __restrict__  x9,
        const view_type & __restrict__  c,
        long n
)
{
    long i;
    int j;

    asm volatile ("# ax+c loop begin");
    for(i = 0; i<n; i++)
    {
#pragma omp simd
        for(j=0; j<VECLENTH; j++)
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
    asm volatile ("# ax+c loop end");

    for(j=0; j<VECLENTH; j++){
        x0(j) = x0(j)+x1(j)+x2(j)+x3(j)+x4(j)+x5(j)+x6(j)+x7(j)+x8(j)+x9(j)+(float)n;
    }
    return x0;
}


int main(int argc, char ** argv) {

    long n = static_cast<long>(2e6); //1000000000; //MAXBYTES/sizeof(float);
    /*long seed = (argc > 2 ? atol(argv[2]) : 76843802738543);*/
    long seed = 76843802738543;

    const int N = 1;
    Kokkos::initialize (argc, argv);

    { // Scoped block for Kokkos finalize

    view_type a_("a",N);
    view_type x0_("x0",N);
    view_type x1_("x1",N);
    view_type x2_("x2",N);
    view_type x3_("x3",N);
    view_type x4_("x4",N);
    view_type x5_("x5",N);
    view_type x6_("x6",N);
    view_type x7_("x7",N);
    view_type x8_("x8",N);
    view_type x9_("x9",N);
    view_type c_("c",N);

    /*
    data * a_=new data;
    data * x_=new data;
    data * x1_=new data;
    data * x2_=new data;
    data * x3_=new data;
    data * x4_=new data;
    data * x5_=new data;
    data * x6_=new data;
    data * x7_=new data;
    data * x8_=new data;
    data * x9_=new data;
    data * c_=new data;
    */

    long i,j;
    unsigned short rg[3] = { static_cast<unsigned short>(seed >> 16), static_cast<unsigned short>(seed >> 8), static_cast<unsigned short>(seed) };

    for (i = 0; i < VECLENTH; i++)
    {
        a_(i)  = erand48(rg);
        c_(i)  = erand48(rg);
        x0_(i)  = erand48(rg);
        x1_(i) = erand48(rg);
        x2_(i) = erand48(rg);
        x3_(i) = erand48(rg);
        x4_(i) = erand48(rg);
        x5_(i) = erand48(rg);
        x6_(i) = erand48(rg);
        x7_(i) = erand48(rg);
        x8_(i) = erand48(rg);
        x9_(i) = erand48(rg);
    }
    // for (i = 0; i < VECLENTH; i++) {
    //   printf("x_[%ld] = %f\n", i, x8_->vec[i]);
    // }
    unsigned long long c0 = rdtscp();

    x0_ = axpy_10(a_,x0_,x1_,x2_,x3_,x4_,x5_,x6_,x7_,x8_,x9_,c_,n);

    unsigned long long c1 = rdtscp();

    unsigned long long dc = c1 - c0;
    double flops = 10*2*VECLENTH*(double)n;
    printf("Outer loop n=%lu\n", n);
    printf("Inner loop VECLENTH=%d\n", VECLENTH);
    printf("%f flops\n", flops);
    printf("%llu clocks\n", dc);
    printf("%f flops/clock\n", flops / dc);


    for (i = 0; i < VECLENTH; i++) {
        printf("x0_[%ld] = %f\n", i, x0_(i) );
    }

    /*
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
    */

    }
    Kokkos::finalize ();
    return 0;
}


