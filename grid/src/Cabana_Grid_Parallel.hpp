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

/*!
  \file Cabana_Grid_Parallel.hpp
  \brief Logical grid extension of Kokkos parallel iteration
*/
#ifndef CABANA_GRID_PARALLEL_HPP
#define CABANA_GRID_PARALLEL_HPP

#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <string>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
// Grid Parallel For
//---------------------------------------------------------------------------//
/*!
  \brief Execute a functor in parallel with a multidimensional execution
  policy specified by the given index space.

  \tparam FunctorType The functor type to execute.

  \tparam ExecutionSpace The execution space type.

  \tparam N The dimension of the index space.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param index_space The index space over which to loop.

  \param functor The functor to execute.
 */
template <class FunctorType, class ExecutionSpace, long N>
inline void grid_parallel_for( const std::string& label,
                               const ExecutionSpace& exec_space,
                               const IndexSpace<N>& index_space,
                               const FunctorType& functor )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::grid_parallel_for" );
    Kokkos::parallel_for(
        label, createExecutionPolicy( index_space, exec_space ), functor );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a functor with a work tag in parallel with a multidimensional
  execution policy specified by the given index space.

  \tparam FunctorType The functor type to execute.

  \tparam WorkTag The functor execution tag.

  \tparam ExecutionSpace The execution space type.

  \tparam N The dimension of the index space.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param index_space The index space over which to loop.

  \param work_tag The functor execution tag.

  \param functor The functor to execute.
 */
template <class FunctorType, class WorkTag, class ExecutionSpace, long N>
inline void
grid_parallel_for( const std::string& label, const ExecutionSpace& exec_space,
                   const IndexSpace<N>& index_space, const WorkTag& work_tag,
                   const FunctorType& functor )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::grid_parallel_for" );
    Kokkos::parallel_for(
        label, createExecutionPolicy( index_space, exec_space, work_tag ),
        functor );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a functor in parallel with a multidimensional execution
  policy specified by the given local grid, decomposition, and entity
  type. The loop indices are local.

  \tparam FunctorType The functor type to execute.

  \tparam ExecutionSpace The execution space type.

  \tparam MeshType The mesh type of the local grid.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param local_grid The local grid to iterate over.

  \param decomposition The decomposition type of the entities (own,ghost).

  \param entity_type The entity type over which to loop.

  \param functor The functor to execute.
 */
template <class FunctorType, class ExecutionSpace, class MeshType,
          class DecompositionType, class EntityType>
inline void
grid_parallel_for( const std::string& label, const ExecutionSpace& exec_space,
                   const LocalGrid<MeshType>& local_grid,
                   const DecompositionType& decomposition,
                   const EntityType& entity_type, const FunctorType& functor )
{
    auto index_space =
        local_grid.indexSpace( decomposition, entity_type, Local() );
    grid_parallel_for( label, exec_space, index_space, functor );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a functor with a work tag in parallel with a multidimensional
  execution policy specified by the given local grid, decomposition, and entity
  type. The loop indices are local.

  \tparam FunctorType The functor type to execute.

  \tparam WorkTag The functor work tag.

  \tparam ExecutionSpace The execution space type.

  \tparam MeshType The mesh type of the local grid.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param local_grid The local grid to iterate over.

  \param decomposition The decomposition type of the entities (own,ghost).

  \param entity_type The entity type over which to loop.

  \param work_tag The functor execution tag.

  \param functor The functor to execute.
 */
template <class FunctorType, class WorkTag, class ExecutionSpace,
          class MeshType, class DecompositionType, class EntityType>
inline void
grid_parallel_for( const std::string& label, const ExecutionSpace& exec_space,
                   const LocalGrid<MeshType>& local_grid,
                   const DecompositionType& decomposition,
                   const EntityType& entity_type, const WorkTag& work_tag,
                   const FunctorType& functor )
{
    auto index_space =
        local_grid.indexSpace( decomposition, entity_type, Local() );
    grid_parallel_for( label, exec_space, index_space, work_tag, functor );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a functor in parallel with a linear execution
  policy specified by the set of given index spaces. 4D specialization.

  \tparam FunctorType The functor type to execute.

  \tparam ExecutionSpace The execution space type.

  \tparam NumSpace The number of index spaces.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param index_spaces The set of index spaces over which to loop.

  \param functor The functor to execute. Signature is f(space_id,i,j,k,l)
  space_id is the index of the index space in index_spaces.
 */
template <class FunctorType, class ExecutionSpace, std::size_t NumSpace>
inline void
grid_parallel_for( const std::string& label, const ExecutionSpace& exec_space,
                   const Kokkos::Array<IndexSpace<4>, NumSpace>& index_spaces,
                   const FunctorType& functor )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::grid_parallel_for" );

    // Compute the total number of threads needed and the index space offsets
    // via inclusive scan.
    int size = index_spaces[0].size();
    Kokkos::Array<int, NumSpace> exclusive_offsets;
    Kokkos::Array<int, NumSpace> inclusive_offsets;
    exclusive_offsets[0] = 0;
    inclusive_offsets[0] = size;
    for ( std::size_t sp = 1; sp < NumSpace; ++sp )
    {
        size += index_spaces[sp].size();
        exclusive_offsets[sp] =
            exclusive_offsets[sp - 1] + index_spaces[sp - 1].size();
        inclusive_offsets[sp] =
            inclusive_offsets[sp - 1] + index_spaces[sp].size();
    }

    // Unroll the index spaces into a linear parallel for.
    Kokkos::parallel_for(
        label, Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, size ),
        KOKKOS_LAMBDA( const int n ) {
            // Get the index space id.
            int s = -1;
            for ( std::size_t sp = 0; sp < NumSpace; ++sp )
            {
                if ( n < inclusive_offsets[sp] )
                {
                    s = sp;
                    break;
                }
            }

            // Linear id in index space.
            int linear_id = n - exclusive_offsets[s];

            // Compute entity id.
            int extent_l = index_spaces[s].extent( 3 );
            int extent_kl = extent_l * index_spaces[s].extent( Dim::K );
            int extent_jkl = extent_kl * index_spaces[s].extent( Dim::J );
            int i_base = linear_id / extent_jkl;
            int stride_j = extent_jkl * i_base;
            int j_base = ( linear_id - stride_j ) / extent_kl;
            int stride_k = ( stride_j + extent_kl * j_base );
            int k_base = ( linear_id - stride_k ) / extent_l;
            int l_base = linear_id - stride_k - k_base * extent_l;

            // Execute the user functor. Provide the index space id so they
            // can discern which space they are in within the functor.
            functor( s, i_base + index_spaces[s].min( Dim::I ),
                     j_base + index_spaces[s].min( Dim::J ),
                     k_base + index_spaces[s].min( Dim::K ),
                     l_base + index_spaces[s].min( 3 ) );
        } );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a functor in parallel with a linear execution
  policy specified by the set of given index spaces. 3D specialization.

  \tparam FunctorType The functor type to execute.

  \tparam ExecutionSpace The execution space type.

  \tparam NumSpace The number of index spaces.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param index_spaces The set of index spaces over which to loop.

  \param functor The functor to execute. Signature is f(space_id,i,j,k)
  space_id is the index of the index space in index_spaces.
 */
template <class FunctorType, class ExecutionSpace, std::size_t NumSpace>
inline void
grid_parallel_for( const std::string& label, const ExecutionSpace& exec_space,
                   const Kokkos::Array<IndexSpace<3>, NumSpace>& index_spaces,
                   const FunctorType& functor )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::grid_parallel_for" );

    // Compute the total number of threads needed and the index space offsets
    // via inclusive scan.
    int size = index_spaces[0].size();
    Kokkos::Array<int, NumSpace> exclusive_offsets;
    Kokkos::Array<int, NumSpace> inclusive_offsets;
    exclusive_offsets[0] = 0;
    inclusive_offsets[0] = size;
    for ( std::size_t sp = 1; sp < NumSpace; ++sp )
    {
        size += index_spaces[sp].size();
        exclusive_offsets[sp] =
            exclusive_offsets[sp - 1] + index_spaces[sp - 1].size();
        inclusive_offsets[sp] =
            inclusive_offsets[sp - 1] + index_spaces[sp].size();
    }

    // Unroll the index spaces into a linear parallel for.
    Kokkos::parallel_for(
        label, Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, size ),
        KOKKOS_LAMBDA( const int n ) {
            // Get the index space id.
            int s = -1;
            for ( std::size_t sp = 0; sp < NumSpace; ++sp )
            {
                if ( n < inclusive_offsets[sp] )
                {
                    s = sp;
                    break;
                }
            }

            // Linear id in index space.
            int linear_id = n - exclusive_offsets[s];

            // Compute entity id.
            int extent_k = index_spaces[s].extent( Dim::K );
            int extent_jk = extent_k * index_spaces[s].extent( Dim::J );
            int i_base = linear_id / extent_jk;
            int stride_j = extent_jk * i_base;
            int j_base = ( linear_id - stride_j ) / extent_k;
            int k_base = linear_id - extent_k * j_base - stride_j;

            // Execute the user functor. Provide the index space id so they
            // can discern which space they are in within the functor.
            functor( s, i_base + index_spaces[s].min( Dim::I ),
                     j_base + index_spaces[s].min( Dim::J ),
                     k_base + index_spaces[s].min( Dim::K ) );
        } );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a functor in parallel with a linear execution
  policy specified by the set of given index spaces. 2D specialization.

  \tparam FunctorType The functor type to execute.

  \tparam ExecutionSpace The execution space type.

  \tparam NumSpace The number of index spaces.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param index_spaces The set of index spaces over which to loop.

  \param functor The functor to execute. Signature is f(space_id,i,j)
  space_id is the index of the index space in index_spaces.
 */
template <class FunctorType, class ExecutionSpace, std::size_t NumSpace>
inline void
grid_parallel_for( const std::string& label, const ExecutionSpace& exec_space,
                   const Kokkos::Array<IndexSpace<2>, NumSpace>& index_spaces,
                   const FunctorType& functor )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::grid_parallel_for" );

    // Compute the total number of threads needed and the index space offsets
    // via inclusive scan.
    int size = index_spaces[0].size();
    Kokkos::Array<int, NumSpace> exclusive_offsets;
    Kokkos::Array<int, NumSpace> inclusive_offsets;
    exclusive_offsets[0] = 0;
    inclusive_offsets[0] = size;
    for ( std::size_t sp = 1; sp < NumSpace; ++sp )
    {
        size += index_spaces[sp].size();
        exclusive_offsets[sp] =
            exclusive_offsets[sp - 1] + index_spaces[sp - 1].size();
        inclusive_offsets[sp] =
            inclusive_offsets[sp - 1] + index_spaces[sp].size();
    }

    // Unroll the index spaces into a linear parallel for.
    Kokkos::parallel_for(
        label, Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, size ),
        KOKKOS_LAMBDA( const int n ) {
            // Get the index space id.
            int s = -1;
            for ( std::size_t sp = 0; sp < NumSpace; ++sp )
            {
                if ( n < inclusive_offsets[sp] )
                {
                    s = sp;
                    break;
                }
            }

            // Linear id in index space.
            int linear_id = n - exclusive_offsets[s];

            // Compute entity id.
            int extent_j = index_spaces[s].extent( Dim::J );
            int i_base = linear_id / extent_j;
            int j_base = linear_id - i_base * extent_j;

            // Execute the user functor. Provide the index space id so they
            // can discern which space they are in within the functor.
            functor( s, i_base + index_spaces[s].min( Dim::I ),
                     j_base + index_spaces[s].min( Dim::J ) );
        } );
}

//---------------------------------------------------------------------------//
// Grid Parallel Reduce
//---------------------------------------------------------------------------//
/*!
  \brief Execute a reduction functor in parallel with a multidimensional
  execution policy specified by the given index space.

  \tparam FunctorType The functor type to execute.

  \tparam ExecutionSpace The execution space type.

  \tparam N The dimension of the index space.

  \tparam ReduceType The reduction type.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param index_space The index space over which to loop.

  \param functor The functor to execute.

  \param reducer The parallel reduce result.
 */
template <class FunctorType, class ExecutionSpace, long N, class ReduceType>
inline void grid_parallel_reduce( const std::string& label,
                                  const ExecutionSpace& exec_space,
                                  const IndexSpace<N>& index_space,
                                  const FunctorType& functor,
                                  ReduceType& reducer )
{
    Kokkos::Profiling::ScopedRegion region(
        "Cabana::Grid::grid_parallel_reduce" );
    Kokkos::parallel_reduce( label,
                             createExecutionPolicy( index_space, exec_space ),
                             functor, reducer );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a reduction functor with a work tag in parallel with a
  multidimensional execution policy specified by the given index space.

  \tparam FunctorType The functor type to execute.

  \tparam WorkTag The functor execution tag.

  \tparam ExecutionSpace The execution space type.

  \tparam N The dimension of the index space.

  \tparam ReduceType The reduction type.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param index_space The index space over which to loop.

  \param work_tag The functor execution tag.

  \param functor The functor to execute.

  \param reducer The parallel reduce result.
 */
template <class FunctorType, class WorkTag, class ExecutionSpace, long N,
          class ReduceType>
inline void
grid_parallel_reduce( const std::string& label,
                      const ExecutionSpace& exec_space,
                      const IndexSpace<N>& index_space, const WorkTag& work_tag,
                      const FunctorType& functor, ReduceType& reducer )
{
    Kokkos::Profiling::ScopedRegion region(
        "Cabana::Grid::grid_parallel_reduce" );
    Kokkos::parallel_reduce(
        label, createExecutionPolicy( index_space, exec_space, work_tag ),
        functor, reducer );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a reduction functor in parallel with a multidimensional
  execution policy specified by the given local grid, decomposition, and
  entity type. The loop indices are local.

  \tparam FunctorType The functor type to execute.

  \tparam ExecutionSpace The execution space type.

  \tparam MeshType The mesh type of the local grid.

  \tparam ReduceType The reduction type.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param local_grid The local grid to iterate over.

  \param decomposition The decomposition type of the entities (own,ghost).

  \param entity_type The entity type over which to loop.

  \param functor The functor to execute.

  \param reducer The parallel reduce result.
 */
template <class FunctorType, class ExecutionSpace, class MeshType,
          class DecompositionType, class EntityType, class ReduceType>
inline void grid_parallel_reduce( const std::string& label,
                                  const ExecutionSpace& exec_space,
                                  const LocalGrid<MeshType>& local_grid,
                                  const DecompositionType& decomposition,
                                  const EntityType& entity_type,
                                  const FunctorType& functor,
                                  ReduceType& reducer )
{
    auto index_space =
        local_grid.indexSpace( decomposition, entity_type, Local() );
    grid_parallel_reduce( label, exec_space, index_space, functor, reducer );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a reduction functor with a work tag in parallel with a
  multidimensional execution policy specified by the given local grid,
  decomposition, and entity type. The loop indices are local.

  \tparam FunctorType The functor type to execute.

  \tparam WorkTag The functor work tag.

  \tparam ExecutionSpace The execution space type.

  \tparam MeshType The mesh type of the local grid.

  \tparam ReduceType The reduction type.

  \param label Parallel region label.

  \param exec_space An execution space instance.

  \param decomposition The decomposition type of the entities (own,ghost).

  \param local_grid The local grid to iterate over.

  \param entity_type The entity type over which to loop.

  \param work_tag The functor execution tag.

  \param functor The functor to execute.

  \param reducer The parallel reduce result.
 */
template <class FunctorType, class WorkTag, class ExecutionSpace,
          class MeshType, class DecompositionType, class EntityType,
          class ReduceType>
inline void grid_parallel_reduce(
    const std::string& label, const ExecutionSpace& exec_space,
    const LocalGrid<MeshType>& local_grid,
    const DecompositionType& decomposition, const EntityType& entity_type,
    const WorkTag& work_tag, const FunctorType& functor, ReduceType& reducer )
{
    auto index_space =
        local_grid.indexSpace( decomposition, entity_type, Local() );
    grid_parallel_reduce( label, exec_space, index_space, work_tag, functor,
                          reducer );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

namespace Cajita
{
//! \cond Deprecated
template <class... Args>
CAJITA_DEPRECATED void grid_parallel_for( Args&&... args )
{
    return Cabana::Grid::grid_parallel_for( std::forward<Args>( args )... );
}

template <class... Args>
CAJITA_DEPRECATED void grid_parallel_reduce( Args&&... args )
{
    return Cabana::Grid::grid_parallel_reduce( std::forward<Args>( args )... );
}
//! \endcond
} // namespace Cajita

#endif // end CABANA_GRID_PARALLEL_HPP
