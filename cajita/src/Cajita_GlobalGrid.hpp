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
    //! Mesh type.
    using mesh_type = MeshType;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    /*!
     \brief Constructor.
     \param comm The communicator over which to define the grid.
     \param global_mesh The global mesh data.
     \param periodic Whether each logical dimension is periodic.
    */
    GlobalGridBase( MPI_Comm comm,
                    const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
                    const std::array<bool, num_space_dim>& periodic,
                    const BlockPartitioner<num_space_dim>& partitioner );

    // Destructor.
    ~GlobalGridBase();

    //! \brief Get the communicator. This communicator was generated with a
    //! Cartesian topology.
    MPI_Comm comm() const;

    //! \brief Get the global mesh data.
    const GlobalMesh<MeshType>& globalMesh() const;

    //! \brief Get whether a given dimension is periodic.
    bool isPeriodic( const int dim ) const;

    //! \brief Determine if this block is on a low boundary in this dimension.
    //! \param dim Spatial dimension.
    bool onLowBoundary( const int dim ) const;

    //! \brief Determine if this block is on a high boundary in this dimension.
    //! \param dim Spatial dimension.
    bool onHighBoundary( const int dim ) const;

    //! \brief Get the number of blocks in each dimension in the global mesh.
    //! \param dim Spatial dimension.
    int dimNumBlock( const int dim ) const;

    //! \brief Get the total number of blocks.
    int totalNumBlock() const;

    //! \brief Get the id of this block in a given dimension.
    //! \param dim Spatial dimension.
    int dimBlockId( const int dim ) const;

    //! \brief Get the id of this block.
    int blockId() const;

    /*!
      \brief Get the MPI rank of a block with the given indices. If the rank is
      out of bounds and the boundary is not periodic, return -1 to indicate an
      invalid rank.
      \param ijk %Array of block indices.
    */
    int blockRank( const std::array<int, num_space_dim>& ijk ) const;

    /*!
      \brief Get the MPI rank of a block with the given indices. If the rank is
      out of bounds and the boundary is not periodic, return -1 to indicate an
      invalid rank.
      \param i,j,k Block index.
    */
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> blockRank( const int i, const int j,
                                               const int k ) const;

    /*!
      \brief Get the MPI rank of a block with the given indices. If the rank is
      out of bounds and the boundary is not periodic, return -1 to indicate an
      invalid rank.
      \param i,j Block index.
    */
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, int> blockRank( const int i, const int j ) const;

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumEntity( Cell, const int dim ) const;
    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumEntity( Node, const int dim ) const;
    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumEntity( Face<Dim::I>, const int dim ) const;
    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumEntity( Face<Dim::J>, const int dim ) const;

    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> globalNumEntity( Face<Dim::K>,
                                                     const int dim ) const;
    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> globalNumEntity( Edge<Dim::I>,
                                                     const int dim ) const;
    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> globalNumEntity( Edge<Dim::J>,
                                                     const int dim ) const;
    //! \brief Get the global number of entities in a given dimension.
    //! \param dim Spatial dimension.
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int> globalNumEntity( Edge<Dim::K>,
                                                     const int dim ) const;

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
// template <class MeshType, class PartitionerType>
template <class MeshType>
class GlobalGrid : public GlobalGridBase<MeshType>
{
  public:
    //! Mesh type.
    using mesh_type = MeshType;

    //! Spatial dimension
    using GlobalGridBase<mesh_type>::num_space_dim;

    //! partitioner type
    // using partitioner_type = PartitionerType;

    /*!
       \brief Constructor.
       \param comm The communicator over which to define the grid.
       \param global_mesh The global mesh data.
       \param periodic Whether each logical dimension is periodic.
       \param partitioner The grid partitioner.
    */
    GlobalGrid( MPI_Comm comm,
                const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
                const std::array<bool, num_space_dim>& periodic,
                const BlockPartitioner<num_space_dim>& partitioner );
    // const PartitionerType& partitioner );

    //! \brief Get the owned number of cells in a given dimension of this
    //! block. \param dim Spatial dimension.
    int ownedNumCell( const int dim ) const;

    //! \brief Get the global offset in a given dimension. This is where our
    //! block starts in the global indexing scheme.
    //! \param dim Spatial dimension.
    int globalOffset( const int dim ) const;

    //! \brief Set number of cells and offset of local part of the grid. Make
    //! sure these are consistent across all ranks.
    //! \param num_cell New number of owned cells for all dimensions.
    //! \param offset New global offset for all dimensions.
    void setNumCellAndOffset( const std::array<int, num_space_dim>& num_cell,
                              const std::array<int, num_space_dim>& offset );

  private:
    std::array<int, num_space_dim> _owned_num_cell;
    std::array<int, num_space_dim> _global_cell_offset;
    // const BlockPartitioner<num_space_dim> _partitioner;
    // std::shared_ptr<PartitionerType> _partitioner;
};

//---------------------------------------------------------------------------//
/*!
  \brief Global logical grid, specialization for sparse grids
  \tparam Scalar Mesh floating point type: default = 4
  \tparam NumSpaceDim Spatial dimension: default = 3
*/
template <typename Device, class Scalar, unsigned long long CellPerTileDim = 4,
          std::size_t NumSpaceDim = 3>
class GlobalSparseGrid : GlobalGridBase<SparseMesh<Scalar, NumSpaceDim>>
{
  public:
    //! Mesh type.
    using mesh_type = SparseMesh<Scalar, NumSpaceDim>;

    //! Spatial dimension
    using GlobalGridBase<mesh_type>::num_space_dim;

    //! Kokkos device type.
    using device_type = Device;

    //! Number of bits (per dimension) needed to index the cells inside a tile
    static constexpr unsigned long long cell_bits_per_tile_dim =
        bitCount( CellPerTileDim );

    //! Number of cells inside each tile (per dimension), tile size reset to
    //! power of 2
    static constexpr unsigned long long cell_num_per_tile_dim =
        1 << cell_bits_per_tile_dim;

    GlobalSparseGrid( MPI_Comm comm,
                      const std::shared_ptr<GlobalMesh<mesh_type>>& global_mesh,
                      const std::array<bool, num_space_dim>& periodic,
                      std::shared_ptr<SparseDimPartitioner<
                          device_type, cell_num_per_tile_dim>>& partitioner );

  private:
    std::array<int, num_space_dim> _owned_num_cell;
    std::array<int, num_space_dim> _global_cell_offset;
    SparseDimPartitioner<device_type, cell_num_per_tile_dim> _partitioner;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a global grid.
  \param comm The communicator over which to define the grid.
  \param global_mesh The global mesh data.
  \param periodic Whether each logical dimension is periodic.
  \param partitioner The grid partitioner.
*/
// template <class MeshType, class PartitionerType>
// std::shared_ptr<GlobalGrid<MeshType, PartitionerType>> createGlobalGrid(
//     MPI_Comm comm, const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
//     const std::array<bool, MeshType::num_space_dim>& periodic,
//     const DimBlockPartitioner<MeshType::num_space_dim>& partitioner )
// {
//     return std::make_shared<
//         GlobalGrid<MeshType, DimBlockPartitioner<MeshType::num_space_dim>>>(
//         comm, global_mesh, periodic, partitioner );
// }

// template <class MeshType, class PartitionerType>
// std::shared_ptr<GlobalGrid<MeshType, PartitionerType>> createGlobalGrid(
//     MPI_Comm comm, const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
//     const std::array<bool, MeshType::num_space_dim>& periodic,
//     const ManualBlockPartitioner<MeshType::num_space_dim>& partitioner )
// {
//     return std::make_shared<
//         GlobalGrid<MeshType,
//         ManualBlockPartitioner<MeshType::num_space_dim>>>( comm, global_mesh,
//         periodic, partitioner );
// }
template <class MeshType>
std::shared_ptr<GlobalGrid<MeshType>>
createGlobalGrid( MPI_Comm comm,
                  const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
                  const std::array<bool, MeshType::num_space_dim>& periodic,
                  const BlockPartitioner<MeshType::num_space_dim>& partitioner )
{
    return std::make_shared<GlobalGrid<MeshType>>( comm, global_mesh, periodic,
                                                   partitioner );
}

//---------------------------------------------------------------------------/

} // end namespace Cajita

//---------------------------------------------------------------------------//
// Template implementation
//---------------------------------------------------------------------------//

#include <Cajita_GlobalGrid_impl.hpp>

//---------------------------------------------------------------------------//u

#endif // !CAJITA_GLOBALGRID_NEW_HPP