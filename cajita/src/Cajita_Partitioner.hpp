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

#ifndef CAJITA_PARTITIONER_HPP
#define CAJITA_PARTITIONER_HPP

#include <array>
#include <stdexcept>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Block partitioner base class.
// Given global mesh parameters, the block partitioner computes how many MPI
// ranks are assigned to each logical dimension.
template <std::size_t NumSpaceDim>
class BlockPartitioner
{
  public:
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    ~BlockPartitioner() = default;

    /*!
      \brief Get the number of MPI ranks in each dimension of the grid.
      \param comm The communicator to use for the partitioning.
      \param global_cells_per_dim The number of global cells in each dimension.
      \return The number of MPI ranks in each dimension of the grid.
    */
    virtual std::array<int, NumSpaceDim> ranksPerDimension(
        MPI_Comm comm,
        const std::array<int, NumSpaceDim>& global_cells_per_dim ) const = 0;
};

//---------------------------------------------------------------------------//
// Manual block partitioner.
//
// Assign MPI blocks from a fixed user input.
template <std::size_t NumSpaceDim>
class ManualBlockPartitioner : public BlockPartitioner<NumSpaceDim>
{
  public:
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    ManualBlockPartitioner( const std::array<int, NumSpaceDim>& ranks_per_dim )
        : _ranks_per_dim( ranks_per_dim )
    {
    }

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
                "ManualPartitioner ranks do not match comm size" );
        return _ranks_per_dim;
    }

  private:
    std::array<int, NumSpaceDim> _ranks_per_dim;
};

//---------------------------------------------------------------------------//
// Dimension-only partitioner.
//
// Use MPI to compute the most uniform block distribution possible (i.e. the
// one that has the minimal number of neighbor communication messages in halo
// exchange). This distribution is independent of mesh parameters - only the
// size of the communicator is considered. Depending on the problem, this may
// not be the optimal partitioning depending on cell-counts and workloads as
// the reduced number of MPI messages may be overshadowed by the load
// imbalance during computation.
template <std::size_t NumSpaceDim>
class DimBlockPartitioner : public BlockPartitioner<NumSpaceDim>
{
  public:
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    std::array<int, NumSpaceDim>
    ranksPerDimension( MPI_Comm comm,
                       const std::array<int, NumSpaceDim>& ) const override
    {
        int comm_size;
        MPI_Comm_size( comm, &comm_size );

        std::array<int, NumSpaceDim> ranks_per_dim;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            ranks_per_dim[d] = 0;
        MPI_Dims_create( comm_size, NumSpaceDim, ranks_per_dim.data() );

        return ranks_per_dim;
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_PARTITIONER_HPP
