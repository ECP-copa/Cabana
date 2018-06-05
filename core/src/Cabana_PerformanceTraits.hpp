#ifndef CABANA_PERFORMANCETRAITS_HPP
#define CABANA_PERFORMANCETRAITS_HPP

#include <Cabana_InnerArrayLayout.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class PerformanceTraits
  \brief Default settings for execution spaces.
*/
template<class ExecutionSpace>
class PerformanceTraits;

//---------------------------------------------------------------------------//
// Serial specialization.
#if defined( KOKKOS_ENABLE_SERIAL )
template<>
class PerformanceTraits<Kokkos::Serial>
{
  public:
    static constexpr int array_size = 1;
    using kokkos_layout = Kokkos::LayoutRight;
    using inner_array_layout = InnerArrayLayout<array_size,kokkos_layout>;
    using parallel_for_tag = StructParallelTag;
};
#endif

//---------------------------------------------------------------------------//
// OpenMP specialization.
#if defined( KOKKOS_ENABLE_OPENMP )
template<>
class PerformanceTraits<Kokkos::OpenMP>
{
  public:
    static constexpr int array_size = 64;
    using kokkos_layout = Kokkos::LayoutRight;
    using inner_array_layout = InnerArrayLayout<array_size,kokkos_layout>;
    using parallel_for_tag = StructParallelTag;
};
#endif

//---------------------------------------------------------------------------//
// Cuda specialization. Use the warp traits.
#if defined( KOKKOS_ENABLE_CUDA )
template<>
class PerformanceTraits<Kokkos::Cuda>
{
  public:
    static constexpr int array_size = Kokkos::Impl::CudaTraits::WarpSize;
    using kokkos_layout = Kokkos::LayoutLeft;
    using inner_array_layout =
        InnerArrayLayout<array_size,kokkos_layout>;
    using parallel_for_tag = StructAndArrayParallelTag;
};
#endif

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PERFORMANCETRAITS_HPP
