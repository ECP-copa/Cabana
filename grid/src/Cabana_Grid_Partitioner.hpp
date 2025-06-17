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
  \file Cabana_Grid_Partitioner.hpp
  \brief Multi-node grid partitioner
*/
#ifndef CABANA_GRID_PARTITIONER_HPP
#define CABANA_GRID_PARTITIONER_HPP

#include <array>
#include <stdexcept>

#include <mpi.h>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
/*!
  \brief Block partitioner base class.
  \tparam NumSpaceDim Spatial dimension.
  Given global mesh parameters, the block partitioner computes how many MPI
  ranks are assigned to each logical dimension.
*/
template <std::size_t NumSpaceDim>
class BlockPartitioner
{
  public:
    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    ~BlockPartitioner() = default;

    /*!
      \brief Get the number of MPI ranks in each dimension of the grid.
      \param comm The communicator to use for the partitioning.
      \param global_cells_per_dim The number of global cells in each dimension.
      \return The number of MPI ranks in each dimension of the grid.
    */
    virtual std::array<int, num_space_dim> ranksPerDimension(
        MPI_Comm comm,
        const std::array<int, num_space_dim>& global_cells_per_dim ) const = 0;

    /*!
      \brief Get the owned number of cells and global cell offset of the current
      MPI rank.
      \param cart_comm The MPI Cartesian communicator for the partitioning.
      \param global_cells_per_dim The number of global cells in each dimension.
      \param owned_num_cell (Return) The owned number of cells of the current
      MPI rank in each dimension.
      \param global_cell_offset (Return) The global cell offset of the current
      MPI rank in each dimension
    */
    virtual void ownedCellInfo(
        MPI_Comm cart_comm,
        const std::array<int, num_space_dim>& global_cells_per_dim,
        std::array<int, num_space_dim>& owned_num_cell,
        std::array<int, num_space_dim>& global_cell_offset ) const = 0;
};

//---------------------------------------------------------------------------//
/*!
  \brief Manual block partitioner.
  \tparam NumSpaceDim Spatial dimension.
  Assign MPI blocks from a fixed user input.
*/
template <std::size_t NumSpaceDim>
class ManualBlockPartitioner : public BlockPartitioner<NumSpaceDim>
{
  public:
    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    /*!
      \brief Constructor
      \param ranks_per_dim MPI ranks per dimension.
    */
    ManualBlockPartitioner( const std::array<int, NumSpaceDim>& ranks_per_dim )
        : _ranks_per_dim( ranks_per_dim )
    {
    }

    /*!
      \brief Get the MPI ranks per dimension.
      \param comm MPI communicator.
    */
    std::array<int, NumSpaceDim>
    ranksPerDimension( MPI_Comm comm,
                       const std::array<int, NumSpaceDim>& ) const override
    {
        int comm_size;
        MPI_Comm_size( comm, &comm_size );
        int nrank = 1;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            nrank *= _ranks_per_dim[d];
        if ( comm_size != nrank )
            throw std::runtime_error(
                "Cabana::Grid::ManualBlockPartitioner::ranksPerDimension: "
                "ManualBlockPartitioner ranks do not match comm size" );
        return _ranks_per_dim;
    }

    /*!
      \brief Get the owned number of cells of the current MPI rank.
      \param cart_comm The MPI Cartesian communicator for the partitioning.
      \param global_cells_per_dim The number of global cells and the global
      cell offset in each dimension.
      \param owned_num_cell (Return) The owned number of cells of the current
      MPI rank in each dimension.
      \param global_cell_offset (Return) The global cell offset of the current
      MPI rank in each dimension
    */
    void ownedCellInfo(
        MPI_Comm cart_comm,
        const std::array<int, num_space_dim>& global_cells_per_dim,
        std::array<int, num_space_dim>& owned_num_cell,
        std::array<int, num_space_dim>& global_cell_offset ) const override
    {
        // Get the cells per dimension and the remainder.
        std::array<int, num_space_dim> cells_per_dim;
        std::array<int, num_space_dim> dim_remainder;
        std::array<int, num_space_dim> cart_rank;
        averageCellInfo( cart_comm, global_cells_per_dim, cart_rank,
                         cells_per_dim, dim_remainder );

        // Compute the global cell offset and the local low corner on this rank
        // by computing the starting global cell index via exclusive scan.
        global_cell_offset =
            globalCellOffsetHelper( cart_rank, cells_per_dim, dim_remainder );

        // Compute the number of local cells in this rank in each dimension.
        owned_num_cell =
            ownedCellsHelper( cart_rank, cells_per_dim, dim_remainder );
    }

    /*!
      \brief Get the owned number of cells of the
      current MPI rank.
      \param cart_comm The MPI Cartesian communicator for the partitioning.
      \param global_cells_per_dim The number of global cells in each dimension.
      \return The owned number of cells per dimension of the current MPI rank.
    */
    std::array<int, num_space_dim> ownedCellsPerDimension(
        MPI_Comm cart_comm,
        const std::array<int, num_space_dim>& global_cells_per_dim ) const
    {
        // Get the cells per dimension and the remainder.
        std::array<int, num_space_dim> cells_per_dim;
        std::array<int, num_space_dim> dim_remainder;
        std::array<int, num_space_dim> cart_rank;
        averageCellInfo( cart_comm, global_cells_per_dim, cart_rank,
                         cells_per_dim, dim_remainder );

        // Compute the number of local cells in this rank in each dimension.
        return ownedCellsHelper( cart_rank, cells_per_dim, dim_remainder );
    }

  private:
    /*!
    \brief Get the average owned number of cells and the remainder.
    \param cart_comm The MPI Cartesian communicator for the partitioning.
    \param global_cells_per_dim The number of global cells in each dimension.
    \param cart_rank MPI Cartesian rank index
    \param cells_per_dim (Return) The average owned number of cells in each
    dimension.
    \param dim_remainder (Return) The cell remainder after averagely assign
    cells to MPI ranks in each dimension.
    */
    inline void
    averageCellInfo( MPI_Comm cart_comm,
                     const std::array<int, num_space_dim>& global_cells_per_dim,
                     std::array<int, num_space_dim>& cart_rank,
                     std::array<int, num_space_dim>& cells_per_dim,
                     std::array<int, num_space_dim>& dim_remainder ) const
    {
        // Get the Cartesian size and topology index of this rank.
        int linear_rank;
        MPI_Comm_rank( cart_comm, &linear_rank );
        MPI_Cart_coords( cart_comm, linear_rank, num_space_dim,
                         cart_rank.data() );
        // Get the cells per dimension and the remainder.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            cells_per_dim[d] = global_cells_per_dim[d] / _ranks_per_dim[d];
            dim_remainder[d] = global_cells_per_dim[d] % _ranks_per_dim[d];
        }
    }

    /*!
      \brief Get the owned number of cells in this rank in each dimension.
      \param cart_rank MPI Cartesian rank index.
      \param cells_per_dim The average owned number of cells in each
      dimension.
      \param dim_remainder The cell remainder after averagely assign
      cells to MPI ranks in each dimension.
      \return Owned cell number in this rank in each dimension
    */
    inline std::array<int, num_space_dim> ownedCellsHelper(
        const std::array<int, num_space_dim>& cart_rank,
        const std::array<int, num_space_dim>& cells_per_dim,
        const std::array<int, num_space_dim>& dim_remainder ) const
    {
        std::array<int, num_space_dim> owned_num_cell;
        // Compute the number of local cells in this rank in each dimension.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            owned_num_cell[d] = cells_per_dim[d];
            if ( dim_remainder[d] > cart_rank[d] )
                ++owned_num_cell[d];
        }
        return owned_num_cell;
    }

    /*!
      \brief Get the global cell offset in this rank ineach dimension.
      \param cart_rank MPI Cartesian rank index.
      \param cells_per_dim (Return) The average owned number of cells in each
      dimension.
      \param dim_remainder (Return) The cell remainder after averagely assign
      cells to MPI ranks in each dimension.
      \return Global cell offset in this rank in each dimension.
    */
    inline std::array<int, num_space_dim> globalCellOffsetHelper(
        const std::array<int, num_space_dim>& cart_rank,
        const std::array<int, num_space_dim>& cells_per_dim,
        const std::array<int, num_space_dim>& dim_remainder ) const
    {
        std::array<int, num_space_dim> global_cell_offset;
        // Compute the global cell offset and the local low corner on this rank
        // by computing the starting global cell index via exclusive scan.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            global_cell_offset[d] = 0;
            for ( int r = 0; r < cart_rank[d]; ++r )
            {
                global_cell_offset[d] += cells_per_dim[d];
                if ( dim_remainder[d] > r )
                    ++global_cell_offset[d];
            }
        }
        return global_cell_offset;
    }

  private:
    std::array<int, NumSpaceDim> _ranks_per_dim;
};

//---------------------------------------------------------------------------//
/*!
  \brief Dimension-only partitioner.
  \tparam NumSpaceDim Spatial dimension.
  Use MPI to compute the most uniform block distribution possible (i.e. the one
  that has the minimal number of neighbor communication messages in halo
  exchange). This distribution is independent of mesh parameters - only the size
  of the communicator is considered. Depending on the problem, this may not be
  the optimal partitioning depending on cell-counts and workloads as the reduced
  number of MPI messages may be overshadowed by the load imbalance during
  computation.
*/
template <std::size_t NumSpaceDim>
class DimBlockPartitioner : public BlockPartitioner<NumSpaceDim>
{
  public:
    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! Default constructor (automatically partitioned in all dimensions).
    DimBlockPartitioner()
    {
        // Initialize so that all ranks will be modified when requested.
        initUnpartitionedDims();
    };

    /*!
      \brief Constructor for NumSpaceDim-1 (1d for 2d system).
      \param dim Dimension to ignore for MPI decomposition.
    */
    DimBlockPartitioner( const int dim )
    {
        initUnpartitionedDims();

        // Only partition in the other dimensions.
        _unpartitioned_dim[dim] = 1;
    };

    /*!
      \brief Constructor for 1d decomposition (3d systems only).
      \param dim_1 First dimension to ignore for MPI decomposition.
      \param dim_2 Second dimension to ignore for MPI decomposition.
    */
    DimBlockPartitioner( const int dim_1, const int dim_2 )
    {
        static_assert(
            NumSpaceDim > 2,
            "Cannot partition 2d system with 2 unpartitioned dimensions." );

        initUnpartitionedDims();

        // Only partition in the third dimension.
        _unpartitioned_dim[dim_1] = 1;
        _unpartitioned_dim[dim_2] = 1;
    };

    /*!
      \brief Get the MPI ranks per dimension.
      \param comm MPI communicator.
    */
    std::array<int, NumSpaceDim>
    ranksPerDimension( MPI_Comm comm,
                       const std::array<int, NumSpaceDim>& ) const override
    {
        int comm_size;
        MPI_Comm_size( comm, &comm_size );
        auto ranks_per_dim = _unpartitioned_dim;
        MPI_Dims_create( comm_size, NumSpaceDim, ranks_per_dim.data() );

        return ranks_per_dim;
    }

    /*!
      \brief Get the owned number of cells and the global cell offset of the
      current MPI rank.
      \param cart_comm The MPI Cartesian communicator for the partitioning.
      \param global_cells_per_dim The number of global cells in each dimension.
      \param owned_num_cell (Return) The owned number of cells of the current
      MPI rank in each dimension.
      \param global_cell_offset (Return) The global cell offset of the current
      MPI rank in each dimension
    */
    void ownedCellInfo(
        MPI_Comm cart_comm,
        const std::array<int, num_space_dim>& global_cells_per_dim,
        std::array<int, num_space_dim>& owned_num_cell,
        std::array<int, num_space_dim>& global_cell_offset ) const override
    {
        // Get the cells per dimension and the remainder.
        std::array<int, num_space_dim> cells_per_dim;
        std::array<int, num_space_dim> dim_remainder;
        std::array<int, num_space_dim> cart_rank;
        averageCellInfo( cart_comm, global_cells_per_dim, cart_rank,
                         cells_per_dim, dim_remainder );

        // Compute the global cell offset and the local low corner on this rank
        // by computing the starting global cell index via exclusive scan.
        global_cell_offset =
            globalCellOffsetHelper( cart_rank, cells_per_dim, dim_remainder );

        // Compute the number of local cells in this rank in each dimension.
        owned_num_cell =
            ownedCellsHelper( cart_rank, cells_per_dim, dim_remainder );
    }

    /*!
      \brief Get the owned number of cells of the
      current MPI rank.
      \param cart_comm The MPI Cartesian communicator for the partitioning.
      \param global_cells_per_dim The number of global cells in each dimension.
      \return The owned number of cells of the current
      MPI rank in each dimension.
    */
    std::array<int, num_space_dim> ownedCellsPerDimension(
        MPI_Comm cart_comm,
        const std::array<int, num_space_dim>& global_cells_per_dim ) const
    {
        // Get the cells per dimension and the remainder.
        std::array<int, num_space_dim> cells_per_dim;
        std::array<int, num_space_dim> dim_remainder;
        std::array<int, num_space_dim> cart_rank;
        averageCellInfo( cart_comm, global_cells_per_dim, cart_rank,
                         cells_per_dim, dim_remainder );

        // Compute the number of local cells in this rank in each dimension.
        return ownedCellsHelper( cart_rank, cells_per_dim, dim_remainder );
    }

  private:
    std::array<int, NumSpaceDim> _unpartitioned_dim;

    //! Initialize all partition dimensions to zero (allow them to be modified).
    void initUnpartitionedDims()
    {
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            _unpartitioned_dim[d] = 0;
    }

    /*!
      \brief Get the average owned number of cells and the remainder.
      \param cart_comm The MPI Cartesian communicator for the partitioning.
      \param global_cells_per_dim The number of global cells in each dimension.
      \param cart_rank MPI Cartesian rank index
      \param cells_per_dim (Return) The average owned number of cells in each
      dimension.
      \param dim_remainder (Return) The cell remainder after averagely assign
      cells to MPI ranks in each dimension.
    */
    inline void
    averageCellInfo( MPI_Comm cart_comm,
                     const std::array<int, num_space_dim>& global_cells_per_dim,
                     std::array<int, num_space_dim>& cart_rank,
                     std::array<int, num_space_dim>& cells_per_dim,
                     std::array<int, num_space_dim>& dim_remainder ) const
    {
        // Get the Cartesian size and topology index of this rank.
        std::array<int, num_space_dim> ranks_per_dim;
        std::array<int, num_space_dim> cart_period;
        MPI_Cart_get( cart_comm, num_space_dim, ranks_per_dim.data(),
                      cart_period.data(), cart_rank.data() );

        // Get the cells per dimension and the remainder.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            cells_per_dim[d] = global_cells_per_dim[d] / ranks_per_dim[d];
            dim_remainder[d] = global_cells_per_dim[d] % ranks_per_dim[d];
        }
    }

    /*!
      \brief Get the owned number of cells in this rank in each dimension.
      \param cart_rank MPI Cartesian rank index.
      \param cells_per_dim The average owned number of cells in each
      dimension.
      \param dim_remainder The cell remainder after averagely assign
      cells to MPI ranks in each dimension.
      \return Owned cell number in this rank in each dimension
    */
    inline std::array<int, num_space_dim> ownedCellsHelper(
        const std::array<int, num_space_dim>& cart_rank,
        const std::array<int, num_space_dim>& cells_per_dim,
        const std::array<int, num_space_dim>& dim_remainder ) const
    {
        std::array<int, num_space_dim> owned_num_cell;
        // Compute the number of local cells in this rank in each dimension.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            owned_num_cell[d] = cells_per_dim[d];
            if ( dim_remainder[d] > cart_rank[d] )
                ++owned_num_cell[d];
        }
        return owned_num_cell;
    }

    /*!
      \brief Get the global cell offset in this rank ineach dimension.
      \param cart_rank MPI Cartesian rank index.
      \param cells_per_dim (Return) The average owned number of cells in each
      dimension.
      \param dim_remainder (Return) The cell remainder after averagely assign
      cells to MPI ranks in each dimension.
      \return Global cell offset in this rank in each dimension.
    */
    inline std::array<int, num_space_dim> globalCellOffsetHelper(
        const std::array<int, num_space_dim>& cart_rank,
        const std::array<int, num_space_dim>& cells_per_dim,
        const std::array<int, num_space_dim>& dim_remainder ) const
    {
        std::array<int, num_space_dim> global_cell_offset;
        // Compute the global cell offset and the local low corner on this rank
        // by computing the starting global cell index via exclusive scan.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            global_cell_offset[d] = 0;
            for ( int r = 0; r < cart_rank[d]; ++r )
            {
                global_cell_offset[d] += cells_per_dim[d];
                if ( dim_remainder[d] > r )
                    ++global_cell_offset[d];
            }
        }
        return global_cell_offset;
    }
};

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_PARTITIONER_HPP
