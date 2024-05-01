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
  \file Cabana_Grid_LoadBalancer.hpp
  \brief Load Balancer
*/
#ifndef CABANA_GRID_LOADBALANCER_HPP
#define CABANA_GRID_LOADBALANCER_HPP

#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_Types.hpp>
#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <ALL.hpp>

#include <mpi.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace Cabana
{
namespace Grid
{
namespace Experimental
{
//---------------------------------------------------------------------------//
/*!
  \brief Load Balancer for global grid.
  \tparam MeshType Mesh type (uniform, non-uniform, sparse)
*/
template <class MeshType>
class LoadBalancer;

/*!
  \brief Load Balancer for global grid.
  \tparam Scalar Mesh floating point type.
  \tparam NumSpaceDim Spatial dimension.
*/
template <class Scalar, std::size_t NumSpaceDim>
class LoadBalancer<UniformMesh<Scalar, NumSpaceDim>>
{
  public:
    //! Mesh type.
    using mesh_type = UniformMesh<Scalar, NumSpaceDim>;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    /*!
      \brief Constructor if domains may be arbitrarily small.
      \param comm MPI communicator to use for load balancing communication.
      \param global_grid The initial global grid.
    */
    LoadBalancer( MPI_Comm comm,
                  const std::shared_ptr<GlobalGrid<mesh_type>>& global_grid )
        : _global_grid( global_grid )
        , _comm( comm )
    {
        // todo(sschulz): We don't need the partitioner except for creating the
        // global grid again. It would suffice to retrieve the partitioner from
        // the global grid, but it isn't saved there as well.
        setupLibrary();
    }

    /*!
      \brief Constructor if domains have a minimum size.
      \param comm MPI communicator to use for load balancing communication.
      \param global_grid The initial global grid.
      \param min_domain_size The minimal domain size in any dimension.
    */
    LoadBalancer( MPI_Comm comm,
                  const std::shared_ptr<GlobalGrid<mesh_type>>& global_grid,
                  const double min_domain_size )
        : _global_grid( global_grid )
        , _comm( comm )
    {
        setupLibrary();
        std::vector<double> vec_min_domain_size( NumSpaceDim, min_domain_size );
        _liball->setMinDomainSize( vec_min_domain_size );
    }

    /*!
      \brief Constructor if domains have a minimum size.
      \param comm MPI communicator to use for load balancing communication.
      \param global_grid The initial global grid.
      \param min_domain_size The minimal domain size in each dimension.
    */
    LoadBalancer( MPI_Comm comm,
                  const std::shared_ptr<GlobalGrid<mesh_type>>& global_grid,
                  const std::array<double, NumSpaceDim> min_domain_size )
        : _global_grid( global_grid )
        , _comm( comm )
    {
        setupLibrary();
        std::vector<double> vec_min_domain_size( min_domain_size.begin(),
                                                 min_domain_size.end() );
        _liball->setMinDomainSize( vec_min_domain_size );
    }

    /*!
      \brief Create a new, balanced global grid and return that.
      \param global_mesh The global mesh data.
      \param partitioner The grid partitioner.
      \param local_work Local amount of work that is balanced.
    */
    std::shared_ptr<GlobalGrid<mesh_type>> createBalancedGlobalGrid(
        const std::shared_ptr<GlobalMesh<mesh_type>>& global_mesh,
        const BlockPartitioner<NumSpaceDim>& partitioner,
        const double local_work )
    {
        Kokkos::Profiling::ScopedRegion region(
            "Cabana::Grid::LoadBalancer::balance" );

        // Create new decomposition
        _liball->setWork( local_work );
        _liball->balance();
        // Calculate new local cell offset and local extent
        std::vector<ALL::Point<double>> updated_vertices =
            _liball->getVertices();
        std::array<double, NumSpaceDim> cell_size;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            cell_size[d] = _global_grid->globalMesh().cellSize( d );
        std::array<int, NumSpaceDim> cell_index_lo, cell_index_hi;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            cell_index_lo[d] =
                std::rint( updated_vertices.at( 0 )[d] / cell_size[d] );
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            cell_index_hi[d] =
                std::rint( updated_vertices.at( 1 )[d] / cell_size[d] );
        std::array<int, NumSpaceDim> num_cell;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            num_cell[d] = cell_index_hi[d] - cell_index_lo[d];
        // Create new global grid
        // todo(sschulz): Can GlobalGrid be constructed with an already
        // cartesian communicator? MPI_Cart_Create is called with the given
        // comm.
        std::array<bool, NumSpaceDim> periodic;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            periodic[d] = _global_grid->isPeriodic( d );
        std::shared_ptr<GlobalGrid<mesh_type>> global_grid =
            createGlobalGrid( _comm, global_mesh, periodic, partitioner );
        global_grid->setNumCellAndOffset( num_cell, cell_index_lo );
        _global_grid = global_grid;

        return _global_grid;
    }

    //! \brief Return array of low and high corner of current internal domain.
    //!        This is not aligned to the mesh!
    const std::array<double, NumSpaceDim * 2> getInternalVertices() const
    {
        std::array<double, NumSpaceDim * 2> internal_vertices;
        std::vector<ALL::Point<double>> lb_vertices = _liball->getVertices();
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            internal_vertices[d] = lb_vertices.at( 0 )[d];
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            internal_vertices[d + NumSpaceDim] = lb_vertices.at( 1 )[d];
        return internal_vertices;
        // todo(sschulz): Is this ok to pass arrays?
    }

    //! \brief Return array of low and high corner of current domain.
    //!        Represents the actual domain layout.
    const std::array<double, NumSpaceDim * 2> getVertices() const
    {
        std::array<double, NumSpaceDim> cell_size;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            cell_size[d] = _global_grid->globalMesh().cellSize( d );
        std::array<double, NumSpaceDim * 2> vertices;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            vertices[d] = _global_grid->globalOffset( d ) * cell_size[d];
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            vertices[d + NumSpaceDim] =
                vertices[d] + _global_grid->ownedNumCell( d ) * cell_size[d];
        return vertices;
        // todo(sschulz): Is this ok to pass arrays?
    }

    //! \brief Return current load imbalance (wmax-wmin)/(wmax+wmin).
    //!        Must be called by all ranks.
    double getImbalance() const
    {
        const double local_work = _liball->getWork();
        double max_work, min_work;
        MPI_Allreduce( &local_work, &max_work, 1, MPI_DOUBLE, MPI_MAX, _comm );
        MPI_Allreduce( &local_work, &min_work, 1, MPI_DOUBLE, MPI_MIN, _comm );
        return ( max_work - min_work ) / ( max_work + min_work );
    }

    // todo(sschulz): Methods to access single values from the vertices, as in
    // the other classes.

    // todo(sschulz): Should also be relatively straight forward to extend to
    // support non uniform mesh, since only the calculation of the vertices and
    // inverse need to be changed. And those would be an exclusive scan of the
    // edge coordinates. Likewise the calculation of cells is rounding to the
    // nearest sum.

  private:
    //! \brief Necessary setup for the library common to all constructors.
    void setupLibrary()
    {
        // todo(sschulz): Investigate, why usage of num_space_dim instead of
        // NumSpaceDim causes a linker error.
        _liball = std::make_shared<ALL::ALL<double, double>>( ALL::TENSOR,
                                                              NumSpaceDim, 0 );
        std::vector<int> loc( NumSpaceDim, 0 );
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            loc[d] = _global_grid->dimBlockId( d );
        std::vector<int> size( NumSpaceDim, 0 );
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            size[d] = _global_grid->dimNumBlock( d );
        _liball->setProcGridParams( loc, size );
        _liball->setCommunicator( _comm );
        int rank;
        MPI_Comm_rank( _global_grid->comm(), &rank );
        _liball->setProcTag( rank );
        _liball->setup();
        // Set initial vertices
        std::array<double, NumSpaceDim> cell_size;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            cell_size[d] = _global_grid->globalMesh().cellSize( d );
        std::vector<ALL::Point<double>> lb_vertices(
            2, ALL::Point<double>( NumSpaceDim ) );
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            lb_vertices.at( 0 )[d] =
                _global_grid->globalOffset( d ) * cell_size[d];
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            lb_vertices.at( 1 )[d] =
                lb_vertices.at( 0 )[d] +
                _global_grid->ownedNumCell( d ) * cell_size[d];
        _liball->setVertices( lb_vertices );
    }

    std::shared_ptr<ALL::ALL<double, double>> _liball;
    std::shared_ptr<GlobalGrid<mesh_type>> _global_grid;
    MPI_Comm _comm;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a load balancer
  \param comm MPI communicator to use for load balancing communication.
  \param global_grid The initial global grid.
*/
template <class Scalar, std::size_t NumSpaceDim>
std::shared_ptr<LoadBalancer<UniformMesh<Scalar, NumSpaceDim>>>
createLoadBalancer(
    MPI_Comm comm,
    const std::shared_ptr<GlobalGrid<UniformMesh<Scalar, NumSpaceDim>>>&
        global_grid )
{
    return std::make_shared<LoadBalancer<UniformMesh<Scalar, NumSpaceDim>>>(
        comm, global_grid );
}
/*!
  \brief Create a load balancer
  \param comm MPI communicator to use for load balancing communication.
  \param global_grid The initial global grid.
  \param min_domain_size The minimal domain size in any dimension.
*/
template <class Scalar, std::size_t NumSpaceDim>
std::shared_ptr<LoadBalancer<UniformMesh<Scalar, NumSpaceDim>>>
createLoadBalancer(
    MPI_Comm comm,
    const std::shared_ptr<GlobalGrid<UniformMesh<Scalar, NumSpaceDim>>>&
        global_grid,
    const double min_domain_size )
{
    return std::make_shared<LoadBalancer<UniformMesh<Scalar, NumSpaceDim>>>(
        comm, global_grid, min_domain_size );
}
/*!
  \brief Create a load balancer
  \param comm MPI communicator to use for load balancing communication.
  \param global_grid The initial global grid.
  \param min_domain_size The minimal domain size in each dimension.
  \return Shared pointer to a LoadBalancer.
*/
template <class Scalar, std::size_t NumSpaceDim>
std::shared_ptr<LoadBalancer<UniformMesh<Scalar, NumSpaceDim>>>
createLoadBalancer(
    MPI_Comm comm,
    const std::shared_ptr<GlobalGrid<UniformMesh<Scalar, NumSpaceDim>>>&
        global_grid,
    const std::array<double, NumSpaceDim> min_domain_size )
{
    return std::make_shared<LoadBalancer<UniformMesh<Scalar, NumSpaceDim>>>(
        comm, global_grid, min_domain_size );
}

} // end namespace Experimental
} // namespace Grid
} // namespace Cabana

//---------------------------------------------------------------------------//

namespace Cajita
{
//! \cond Deprecated
namespace Experimental
{
template <class MeshType>
using LoadBalancer CAJITA_DEPRECATED =
    Cabana::Grid::Experimental::LoadBalancer<MeshType>;

template <class... Args>
CAJITA_DEPRECATED auto createLoadBalancer( Args&&... args )
{
    return Cabana::Grid::Experimental::createLoadBalancer(
        std::forward<Args>( args )... );
}
//! \endcond
} // namespace Experimental
} // namespace Cajita

#endif // end CABANA_GRID_LOADBALANCER_HPP
