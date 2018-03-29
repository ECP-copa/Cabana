#ifndef CABANA_PARALLEL_HPP
#define CABANA_PARALLEL_HPP

#include <Cabana_ExecutionPolicy.hpp>
#include <Cabana_Index.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <Kokkos_Pair.hpp>

#include <cstdlib>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Forward declaration of performance traits.
template<class ExecutionSpace>
class PerformanceTraits;

//---------------------------------------------------------------------------//
// Algorithm tags.

//! 1D parallelism over structs.
class StructParallelTag {};

//! 1D parallelism over inner arrays.
class ArrayParallelTag {};

//! 2D parallelism over structs and inner arrays.
class StructAndArrayParallelTag {};

//---------------------------------------------------------------------------//
// Helper functions
//---------------------------------------------------------------------------//
// Given a begin and end index get the starting and ending struct indices.
KOKKOS_INLINE_FUNCTION
Kokkos::pair<std::size_t,std::size_t>
getStructBounds( const Index& begin, const Index& end )
{
    Kokkos::pair<std::size_t,std::size_t> struct_bounds;

    // The first struct is the struct index of the beginning.
    struct_bounds.first = begin.s();

    // If the end is also at the front of an array that means the struct index
    // of end is also the ending struct index. If not, we are not iterating
    // all the way through the arrays of the last struct. In this case we add
    // 1 to ensure that the loop over structs loops through all structs with
    // data.
    struct_bounds.second = (0 == end.i()) ? end.s() : end.s() + 1;

    return struct_bounds;
}

//---------------------------------------------------------------------------//
// Given a begin and end index, struct bounds, and a struct index get the
// starting and ending array indices.
KOKKOS_INLINE_FUNCTION
Kokkos::pair<std::size_t,std::size_t>
getArrayBounds( const Index& begin,
                const Index& end,
                const Kokkos::pair<std::size_t,std::size_t>& struct_bounds,
                const std::size_t s )
{
    Kokkos::pair<std::size_t,std::size_t> array_bounds;

    // If the given struct index is also the index of the struct index in
    // begin, use the starting array index. If not, that means we have passed
    // the first struct and all subsequent structs start at array index 0.
    array_bounds.first = (s == struct_bounds.first) ? begin.i() : 0;

    // If we are in the last struct unfilled struct then use the array index
    // of end. If not, we are looping through the current array all the way to
    // the end so use the array size.
    array_bounds.second =
        ((s == struct_bounds.second - 1) && (end.i() != 0))
        ? end.i() : begin.a();

    return array_bounds;
}

//---------------------------------------------------------------------------//
// Parallel for
//---------------------------------------------------------------------------//
/*!
  \brief Execute \c functor in parallel according to the execution \c policy.

  \tparam ExecutionPolicy The execution over which to execute the functor.

  \tparam FunctorType The functor type to execute.

  \param exec_policy The policy over which to execute the functor.

  \param functor The functor to execute in parallel

  \param str An optional name for the functor. Will be forwarded to the
  Kokkos::parallel_for called by this code and can be used for identification
  and profiling purposes.

  A "functor" is a class containing the function to execute in parallel, data
  needed for that execution, and an optional \c execution_space typedef.  Here
  is an example functor for parallel_for:

  \code
  class FunctorType {
  public:
  typedef  ...  execution_space ;
  void operator() ( Index idx ) const ;
  };
  \endcode

  In the above example, \c Index is a Cabana index to a given AoSoA element.
  Its <tt>operator()</tt> method defines the operation to parallelize, over
  the range of indices <tt>idx=[begin,end]</tt>.  This compares to a single
  iteration \c idx of a \c for loop.
*/
template<class ExecutionPolicy, class FunctorType>
inline void parallel_for( const ExecutionPolicy& exec_policy,
                          const FunctorType& functor,
                          const std::string& str = "" )
{
    using exec_space = typename ExecutionPolicy::execution_space;
    parallel_for( exec_policy,
                  functor,
                  typename PerformanceTraits<exec_space>::parallel_for_tag(),
                  str );
}

//---------------------------------------------------------------------------//
/*!
  \brief Parallel-for 1D struct parallel specialization.

  Takes an instance of the \c StructParallelTag to indicate specialization.

  Creates 1D parallel over structs with internal array loops in serial on each
  thread:
  \code
  AoSoA aosoa;
  Index begin, end;
  parallel_for( s : num_structs )
  {
      for( i : array_size(s) )
      {
          Index idx( array_size, s, i );
          functor( idx );
      }
  }
  \endcode
*/
template<class ExecutionPolicy, class FunctorType>
inline void parallel_for( const ExecutionPolicy& exec_policy,
                          const FunctorType& functor,
                          const StructParallelTag&,
                          const std::string& str = "" )
{
    // Kokkos execution policy type alias.
    using kokkos_policy =
        Kokkos::RangePolicy<typename ExecutionPolicy::execution_space>;

    // Create a range policy over the structs. If the end is not at a struct
    // boundary we need to add an extra struct so we loop through the last
    // unfilled struct.
    auto begin = exec_policy.begin();
    auto end = exec_policy.end();
    auto struct_bounds = getStructBounds( begin, end );
    kokkos_policy k_policy( struct_bounds.first, struct_bounds.second );

    // Create a wrapper for the functor. Each struct is given a thread and
    // each thread loops over the inner arrays.
    std::size_t array_size = begin.a();
    auto functor_wrapper =
        KOKKOS_LAMBDA( const std::size_t s )
        {
            auto array_bounds = getArrayBounds( begin, end, struct_bounds, s );
            for ( std::size_t i = array_bounds.first;
                  i < array_bounds.second;
                  ++i )
            {
                Index idx( array_size, s, i );
                functor( idx );
            }
        };

    // Execute the functor.
    Kokkos::parallel_for( str, k_policy, functor_wrapper );

    // Fence.
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
/*!
  \brief Parallel-for 1D inner array parallel specialization.

  Takes an instance of the \c ArrayParallelTag to indicate specialization.

  Creates a serial outer loop over structs and 1D parallel loops over inner
  arrays:
  \code
  AoSoA aosoa;
  Index begin, end;
  for( s : num_structs )
  {
      parallel_for( i : array_size(s) )
      {
          Index idx( array_size, s, i );
          functor( idx );
      }
  }
  \endcode
*/
template<class ExecutionPolicy, class FunctorType>
inline void parallel_for( const ExecutionPolicy& exec_policy,
                          const FunctorType& functor,
                          const ArrayParallelTag&,
                          const std::string& str = "" )
{
    // Kokkos execution policy type alias.
    using kokkos_policy =
        Kokkos::RangePolicy<typename ExecutionPolicy::execution_space>;

    // Loop over structs. If the end is not at a struct boundary we need to
    // add an extra struct so we loop through the last unfilled struct.
    auto begin = exec_policy.begin();
    auto end = exec_policy.end();
    auto struct_bounds = getStructBounds( begin, end );
    std::size_t array_size = begin.a();
    for ( std::size_t s = struct_bounds.first;
          s < struct_bounds.second;
          ++s )
    {
        // Create a range policy over the array.
        auto array_bounds = getArrayBounds( begin, end, struct_bounds, s );
        kokkos_policy k_policy( array_bounds.first, array_bounds.second );

        // Create a wrapper for the functor. Each struct is given a thread and
        // each thread loops over the inner arrays.
        auto functor_wrapper =
            KOKKOS_LAMBDA( const std::size_t i )
            {
                Index idx( array_size, s, i );
                functor( idx );
            };

        // Execute the functor.
        Kokkos::parallel_for( str, k_policy, functor_wrapper );
    }

    // Fence.
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
/*!
  \brief Parallel-for 2D struct and inner array parallel specialization.

  Takes an instance of the \c StructAndArrayParallelTag to indicate specialization.

  Creates a 2D parallel loop over structs and their inner arrays:
  \code
  AoSoA aosoa;
  Index begin, end;
  parallel_for( s : num_structs )
  {
      parallel_for( i : array_size(s) )
      {
          Index idx( array_size, s, i );
          functor( idx );
      }
  }
  \endcode
*/
template<class ExecutionPolicy, class FunctorType>
inline void parallel_for( const ExecutionPolicy& exec_policy,
                          const FunctorType& functor,
                          const StructAndArrayParallelTag&,
                          const std::string& str = "" )
{
    // Type aliases.
    constexpr auto kokkos_iterate= Kokkos::Iterate::Right;
    using kokkos_policy =
        Kokkos::MDRangePolicy<typename ExecutionPolicy::execution_space,
                              Kokkos::Rank<2,kokkos_iterate,kokkos_iterate>,
                              Kokkos::IndexType<std::size_t> >;
    using point_type = typename kokkos_policy::point_type;

    // Make a 2D execution policy.
    auto begin = exec_policy.begin();
    auto end = exec_policy.end();
    std::size_t array_size = begin.a();
    auto struct_bounds = getStructBounds( begin, end );
    point_type lower = { struct_bounds.first, 0 };
    point_type upper = { struct_bounds.second, array_size };
    kokkos_policy k_policy( lower, upper );

    // Create a wrapper for the functor.
    auto functor_wrapper =
        KOKKOS_LAMBDA( const std::size_t s, const std::size_t i )
        {
            Index idx( array_size, s, i );
            if ( begin <= idx && idx < end ) functor( idx );
        };

    // Execute the functor.
    Kokkos::parallel_for( str, k_policy, functor_wrapper );

    // Fence.
    Kokkos::fence();
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PARALLEL_HPP
