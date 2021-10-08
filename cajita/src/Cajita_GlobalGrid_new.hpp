/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cajita_GlobalGrid.hpp
  \brief Global grid
*/
#ifndef CAJITA_GLOBALGRID_NEW_HPP
#define CAJITA_GLOBALGRID_NEW_HPP

#include <Cajita_GlobalMesh.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <Cajita_Partitioner.hpp>
#include <Cajita_SparseDimPartitioner.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <array>
#include <memory>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  \brief Global logical grid base, store global mesh, rank related information.
  \tparam MeshType Mesh type (uniform, non-uniform, sparse)
*/
template <class MeshType>
class GlobalGridBase
{
  public:
  private:
    MPI_Comm _cart_comm;
    std::shared_ptr<GlobalMesh<MeshType>> _global_mesh;
    std::array<bool, num_space_dim> _periodic;
    std::array<int, num_space_dim> _ranks_per_dim;
    std::array<int, num_space_dim> _cart_rank;
    std::array<bool, num_space_dim> _boundary_lo;
    std::array<bool, num_space_dim> _boundary_hi;
};

//---------------------------------------------------------------------------//
/*!
  \brief Global logical grid for uniform and non-uniform grids
  \tparam MeshType Mesh type (uniform, non-uniform)
*/
template <class MeshType>
class GlobalGrid : GlobalGridBase<MeshType>
{
  public:
  private:
    std::array<int, num_space_dim> _owned_num_cell;
    std::array<int, num_space_dim> _global_cell_offset;
    std::shared_ptr<BlockPartitioner<num_space_dim>> _partitioner;
};

//---------------------------------------------------------------------------//
/*!
  \brief Global logical grid, specialization for sparse grids
  \tparam Scalar Mesh floating point type.
  \tparam NumSpaceDim Spatial dimension
*/
template <typename Device, class Scalar, unsigned long long CellPerTileDim = 4,
          std::size_t NumSpaceDim = 3>
class GlobalGrid<SparseMesh<Scalar, NumSpaceDim>>
    : GlobalGridBase<SparseMesh<Scalar, NumSpaceDim>>
{
  public:
    //! Kokkos device type.
    using device_type = Device;
    //! Number of bits (per dimension) needed to index the cells inside a tile
    static constexpr unsigned long long cell_bits_per_tile_dim =
        bitCount( CellPerTileDim );
    //! Number of cells inside each tile (per dimension), tile size reset to
    //! power of 2
    static constexpr unsigned long long cell_num_per_tile_dim =
        1 << cell_bits_per_tile_dim;

  private:
    std::array<int, num_space_dim> _owned_num_cell;
    std::array<int, num_space_dim> _global_cell_offset;
    std::shared_ptr<SparseDimPartitioner<device_type, cell_num_per_tile_dim>>
        _partitioner;
};

} // end namespace Cajita

#endif // !CAJITA_GLOBALGRID_NEW_HPP