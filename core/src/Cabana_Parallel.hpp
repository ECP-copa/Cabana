#ifndef CABANA_PARALLEL_HPP
#define CABANA_PARALLEL_HPP

#include <Cabana_Index.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel.hpp>
#include <KokkosExp_MDRangePolicy.hpp>

#include <cstdlib>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Algorithm tags.

//! 1D parallelism over structs.
class StructParallel {};

//! 1D parallelism over inner arrays.
class ArrayParallel {};

//! 2D parallelism over structs and inner arrays.
class StructAndArrayParallel {};

//---------------------------------------------------------------------------//
// Parallel-for 1D struct parallel specialization.
template<class ExecutionSpace, class FunctorType>
inline void parallel_for( const Index& begin,
                          const Index& end,
                          const FunctorType& functor,
                          const StructParallel&,
                          const std::string& str = "" )
{
    // Create a range policy over the structs. If the end is not at a struct
    // boundary we need to add an extra struct so we loop through the last
    // unfilled struct.
    std::size_t s_begin = begin.s();
    std::size_t s_end = (0 == end.i()) ? end.s() : end.s() + 1;
    Kokkos::RangePolicy<ExecutionSpace> exec_policy( s_begin, s_end );

    // Create a wrapper for the functor. Each struct is given a thread and
    // each thread loops over the inner arrays.
    std::size_t array_size = begin.a();
    auto functor_wrapper =
        KOKKOS_LAMBDA( std::size_t s )
        {
            std::size_t i_begin = (s == s_begin) ? begin.i() : 0;
            std::size_t i_end = ((s == s_end - 1) && (end.i() != 0))
            ? end.i() : array_size;
            for ( std::size_t i = i_begin; i < i_end; ++i )
            {
                Index idx( array_size, s, i );
                functor( idx );
            }
        };

    // Execute the functor.
    Kokkos::parallel_for( str, exec_policy, functor_wrapper );

    // Fence.
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// Parallel-for 1D inner array parallel specialization.
template<class ExecutionSpace, class FunctorType>
inline void parallel_for( const Index& begin,
                          const Index& end,
                          const FunctorType& functor,
                          const ArrayParallel&,
                          const std::string& str = "" )
{
    // Loop over structs. If the end is not at a struct
    // boundary we need to add an extra struct so we loop through the last
    // unfilled struct.
    std::size_t array_size = begin.a();
    std::size_t s_begin = begin.s();
    std::size_t s_end = (0 == end.i()) ? end.s() : end.s() + 1;
    for ( std::size_t s = s_begin; s < s_end; ++s )
    {
        // Create a range policy over the array.
        std::size_t i_begin = (s == s_begin) ? begin.i() : 0;
        std::size_t i_end = ((s == s_end - 1) && (end.i() != 0))
                            ? end.i() : array_size;
        Kokkos::RangePolicy<ExecutionSpace> exec_policy( i_begin, i_end );

        // Create a wrapper for the functor. Each struct is given a thread and
        // each thread loops over the inner arrays.
        auto functor_wrapper =
            KOKKOS_LAMBDA( std::size_t i )
            {
                Index idx( array_size, s, i );
                functor( idx );
            };

        // Execute the functor.
        Kokkos::parallel_for( str, exec_policy, functor_wrapper );
    }

    // Fence.
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// Parallel-for2D parallel over structs and inner arrays specialization.
template<class ExecutionSpace, class FunctorType>
inline void parallel_for( const Index& begin,
                          const Index& end,
                          const FunctorType& functor,
                          const StructAndArrayParallel&,
                          const std::string& str = "" )
{
    // Type aliases.
    using kokkos_policy =
        Kokkos::MDRangePolicy<ExecutionSpace,
                              Kokkos::Rank<2>,
                              Kokkos::IndexType<std::size_t> >;
    using point_type = typename kokkos_policy::point_type;

    // Make a 2D execution policy.
    std::size_t array_size = begin.a();
    std::size_t s_begin = begin.s();
    std::size_t s_end = (0 == end.i()) ? end.s() : end.s() + 1;
    point_type lower = {s_begin,0};
    point_type upper = {s_end,array_size};
    kokkos_policy exec_policy( lower, upper );

    // Create a wrapper for the functor.
    auto functor_wrapper =
        KOKKOS_LAMBDA( std::size_t s, std::size_t i )
        {
            Index idx( array_size, s, i );
            if ( idx >= begin || idx < end ) functor( idx );
        };

    // Execute the functor.
    Kokkos::parallel_for( str, exec_policy, functor_wrapper );

    // Fence.
    Kokkos::fence();
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PARALLEL_HPP
