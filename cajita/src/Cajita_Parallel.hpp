/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_PARALLEL_HPP
#define CAJITA_PARALLEL_HPP

#include <Cajita_IndexSpace.hpp>
#include <Cajita_LocalGrid.hpp>

#include <string>

namespace Cajita
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
inline void grid_parallel_for( const std::string & label,
                               const ExecutionSpace & exec_space,
                               const IndexSpace<N> & index_space,
                               const FunctorType & functor )
{
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

  \param tag The functor execution tag.

  \param functor The functor to execute.
 */
template <class FunctorType, class WorkTag, class ExecutionSpace, long N>
inline void
grid_parallel_for( const std::string & label, const ExecutionSpace & exec_space,
                   const IndexSpace<N> & index_space, const WorkTag & work_tag,
                   const FunctorType & functor )
{
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

  \param decomposition The decomposition type of the entities (own,ghost).

  \param entity_type The entity type over which to loop.

  \param functor The functor to execute.
 */
template <class FunctorType, class ExecutionSpace, class MeshType,
          class DecompositionType, class EntityType>
inline void
grid_parallel_for( const std::string & label, const ExecutionSpace & exec_space,
                   const LocalGrid<MeshType> & local_grid,
                   const DecompositionType & decomposition,
                   const EntityType & entity_type, const FunctorType & functor )
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

  \param decomposition The decomposition type of the entities (own,ghost).

  \param entity_type The entity type over which to loop.

  \param functor The functor to execute.
 */
template <class FunctorType, class WorkTag, class ExecutionSpace,
          class MeshType, class DecompositionType, class EntityType>
inline void
grid_parallel_for( const std::string & label, const ExecutionSpace & exec_space,
                   const LocalGrid<MeshType> & local_grid,
                   const DecompositionType & decomposition,
                   const EntityType & entity_type, const WorkTag & work_tag,
                   const FunctorType & functor )
{
    auto index_space =
        local_grid.indexSpace( decomposition, entity_type, Local() );
    grid_parallel_for( label, exec_space, index_space, work_tag, functor );
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
inline void grid_parallel_reduce( const std::string & label,
                                  const ExecutionSpace & exec_space,
                                  const IndexSpace<N> & index_space,
                                  const FunctorType & functor,
                                  ReduceType & reducer )
{
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

  \param tag The functor execution tag.

  \param functor The functor to execute.

  \param reducer The parallel reduce result.
 */
template <class FunctorType, class WorkTag, class ExecutionSpace, long N,
          class ReduceType>
inline void grid_parallel_reduce( const std::string & label,
                                  const ExecutionSpace & exec_space,
                                  const IndexSpace<N> & index_space,
                                  const WorkTag & work_tag,
                                  const FunctorType & functor,
                                  ReduceType & reducer )
{
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

  \param decomposition The decomposition type of the entities (own,ghost).

  \param entity_type The entity type over which to loop.

  \param functor The functor to execute.

  \param reducer The parallel reduce result.
 */
template <class FunctorType, class ExecutionSpace, class MeshType,
          class DecompositionType, class EntityType, class ReduceType>
inline void grid_parallel_reduce( const std::string & label,
                                  const ExecutionSpace & exec_space,
                                  const LocalGrid<MeshType> & local_grid,
                                  const DecompositionType & decomposition,
                                  const EntityType & entity_type,
                                  const FunctorType & functor,
                                  ReduceType & reducer )
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

  \param entity_type The entity type over which to loop.

  \param functor The functor to execute.

  \param reducer The parallel reduce result.
 */
template <class FunctorType, class WorkTag, class ExecutionSpace,
          class MeshType, class DecompositionType, class EntityType,
          class ReduceType>
inline void
grid_parallel_reduce( const std::string & label,
                      const ExecutionSpace & exec_space,
                      const LocalGrid<MeshType> & local_grid,
                      const DecompositionType & decomposition,
                      const EntityType & entity_type, const WorkTag & work_tag,
                      const FunctorType & functor, ReduceType & reducer )
{
    auto index_space =
        local_grid.indexSpace( decomposition, entity_type, Local() );
    grid_parallel_reduce( label, exec_space, index_space, work_tag, functor,
                          reducer );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_PARALLEL_HPP
