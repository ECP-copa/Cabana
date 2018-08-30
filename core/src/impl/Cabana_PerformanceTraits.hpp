#ifndef CABANA_PERFORMANCETRAITS_HPP
#define CABANA_PERFORMANCETRAITS_HPP

#include <Cabana_Types.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

namespace Cabana
{
namespace Impl
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
    static constexpr int vector_length = 8;
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
    static constexpr int vector_length = 8;
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
    static constexpr int vector_length = Kokkos::Impl::CudaTraits::WarpSize;
    using parallel_for_tag = IndexParallelTag;
};
#endif

//---------------------------------------------------------------------------//

} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_PERFORMANCETRAITS_HPP
