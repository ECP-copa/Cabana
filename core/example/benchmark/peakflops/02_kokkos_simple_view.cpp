#include <Kokkos_Core.hpp>

typedef Kokkos::View<float[8],Kokkos::MemoryTraits<Kokkos::Restrict> > view_type;

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
