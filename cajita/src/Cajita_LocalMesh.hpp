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
  \file Cajita_LocalMesh.hpp
  \brief Local mesh
*/
#ifndef CAJTIA_LOCALMESH_HPP
#define CAJTIA_LOCALMESH_HPP

#include <Cajita_LocalGrid.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Forward decalaration of local mesh.
template <class Device, class MeshType>
class LocalMesh;

//---------------------------------------------------------------------------//
//! Local mesh partial specialization for uniform mesh.
template <class Scalar, class Device, std::size_t NumSpaceDim>
class LocalMesh<Device, UniformMesh<Scalar, NumSpaceDim>>
{
  public:
    //! Mesh type.
    using mesh_type = UniformMesh<Scalar, NumSpaceDim>;

    //! Scalar type for geometric operations.
    using scalar_type = Scalar;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! Kokkos device type.
    using device_type = Device;
    //! Kokkos memory space.
    using memory_space = typename Device::memory_space;
    //! Kokkos execution space.
    using execution_space = typename Device::execution_space;

    //! Default constructor.
    LocalMesh() = default;

    //! Constructor.
    LocalMesh( const LocalGrid<UniformMesh<Scalar, num_space_dim>>& local_grid )
    {
        const auto& global_grid = local_grid.globalGrid();
        const auto& global_mesh = global_grid.globalMesh();

        // Get the cell size.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _cell_size[d] = global_mesh.cellSize( d );

        // Compute face area.
        if ( 3 == num_space_dim )
        {
            _face_area[Dim::I] = _cell_size[Dim::J] * _cell_size[Dim::K];
            _face_area[Dim::J] = _cell_size[Dim::I] * _cell_size[Dim::K];
            _face_area[Dim::K] = _cell_size[Dim::I] * _cell_size[Dim::J];
        }
        else if ( 2 == num_space_dim )
        {
            _face_area[Dim::I] = _cell_size[Dim::J];
            _face_area[Dim::J] = _cell_size[Dim::I];
        }

        // Compute cell volume.
        _cell_volume = 1.0;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _cell_volume *= _cell_size[d];

        // Compute the owned low corner
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _own_low_corner[d] = global_mesh.lowCorner( d ) +
                                 _cell_size[d] * global_grid.globalOffset( d );

        // Compute the owned high corner
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _own_high_corner[d] =
                global_mesh.lowCorner( d ) +
                _cell_size[d] * ( global_grid.globalOffset( d ) +
                                  global_grid.ownedNumCell( d ) );

        // Compute the ghosted low corner
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _ghost_low_corner[d] = lowCorner( Own(), d ) -
                                   local_grid.haloCellWidth() * _cell_size[d];

        // Compute the ghosted high corner
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _ghost_high_corner[d] = highCorner( Own(), d ) +
                                    local_grid.haloCellWidth() * _cell_size[d];

        // Periodicity
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _periodic[d] = global_grid.isPeriodic( d );

        // Determine if a block is on the low or high boundaries.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            _boundary_lo[d] = global_grid.onLowBoundary( d );
            _boundary_hi[d] = global_grid.onHighBoundary( d );
        }
    }

    //! Determine if the mesh is periodic in the given dimension.
    KOKKOS_INLINE_FUNCTION
    bool isPeriodic( const int dim ) const { return _periodic[dim]; }

    //! Determine if this block is on a low boundary in the given dimension.
    KOKKOS_INLINE_FUNCTION
    bool onLowBoundary( const int dim ) const { return _boundary_lo[dim]; }

    //! Determine if this block is on a high boundary in the given dimension.
    KOKKOS_INLINE_FUNCTION
    bool onHighBoundary( const int dim ) const { return _boundary_hi[dim]; }

    //! Get the physical coordinate of the low corner of the owned local grid.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    Scalar lowCorner( Own, const int dim ) const
    {
        return _own_low_corner[dim];
    }

    //! Get the physical coordinate of the low corner of the local grid
    //! including ghosts.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    Scalar lowCorner( Ghost, const int dim ) const
    {
        return _ghost_low_corner[dim];
    }

    //! Get the physical coordinate of the high corner of the owned local grid.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    Scalar highCorner( Own, const int dim ) const
    {
        return _own_high_corner[dim];
    }

    //! Get the physical coordinate of the high corner of the local grid
    //! including ghosts.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    Scalar highCorner( Ghost, const int dim ) const
    {
        return _ghost_high_corner[dim];
    }

    //! Get the extent of a given dimension.
    template <typename Decomposition>
    KOKKOS_FUNCTION Scalar extent( Decomposition d, const int dim ) const
    {
        return highCorner( d, dim ) - lowCorner( d, dim );
    }

    //! Get the coordinates of a Cell given the local index.
    //! Local indexing is relative to the ghosted decomposition of the mesh
    //! block, which correlates directly to local index spaces associated with
    //! the block.
    KOKKOS_INLINE_FUNCTION
    void coordinates( Cell, const int index[num_space_dim],
                      Scalar x[num_space_dim] ) const
    {
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            x[d] = _ghost_low_corner[d] +
                   ( Scalar( index[d] ) + Scalar( 0.5 ) ) * _cell_size[d];
    }

    //! Get the coordinates of a Node given the local index.
    //! Local indexing is relative to the ghosted decomposition of the mesh
    //! block, which correlates directly to local index spaces associated with
    //! the block.
    KOKKOS_INLINE_FUNCTION
    void coordinates( Node, const int index[num_space_dim],
                      Scalar x[num_space_dim] ) const
    {
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            x[d] = _ghost_low_corner[d] + Scalar( index[d] ) * _cell_size[d];
    }

    //! Get the coordinates of a Face given the local index.
    //! Local indexing is relative to the ghosted decomposition of the mesh
    //! block, which correlates directly to local index spaces associated with
    //! the block.
    template <int Dir>
    KOKKOS_INLINE_FUNCTION void coordinates( Face<Dir>,
                                             const int index[num_space_dim],
                                             Scalar x[num_space_dim] ) const
    {
        static_assert( Dir < num_space_dim, "Face dimension out of bounds" );
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            if ( Dir == d )
                x[d] =
                    _ghost_low_corner[d] + Scalar( index[d] ) * _cell_size[d];
            else
                x[d] = _ghost_low_corner[d] +
                       ( Scalar( index[d] ) + Scalar( 0.5 ) ) * _cell_size[d];
    }

    //! Get the coordinates of an Edge given the local index.
    //! Local indexing is relative to the ghosted decomposition of the mesh
    //! block, which correlates directly to local index spaces associated with
    //! the block.
    template <int Dir, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    coordinates( Edge<Dir>, const int index[3], Scalar x[3] ) const
    {
        for ( std::size_t d = 0; d < 3; ++d )
            if ( Dir == d )
                x[d] = _ghost_low_corner[d] +
                       ( Scalar( index[d] ) + Scalar( 0.5 ) ) * _cell_size[d];
            else
                x[d] =
                    _ghost_low_corner[d] + Scalar( index[d] ) * _cell_size[d];
    }

    //! Get the measure of a Node at the given index.
    //! Local indexing is relative to the ghosted decomposition of the mesh
    //! block and correlates directly to local index spaces associated with the
    //! block.
    KOKKOS_INLINE_FUNCTION
    Scalar measure( Node, const int[num_space_dim] ) const { return 0.0; }

    //! Get the measure of an Edge at the given index.
    //! Local indexing is relative to the ghosted decomposition of the mesh
    //! block and correlates directly to local index spaces associated with the
    //! block.
    template <int Dir, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, Scalar>
    measure( Edge<Dir>, const int[3] ) const
    {
        return _cell_size[Dir];
    }

    //! Get the measure of a Face at the given index.
    //! Local indexing is relative to the ghosted decomposition of the mesh
    //! block and correlates directly to local index spaces associated with the
    //! block.
    template <int Dir>
    KOKKOS_INLINE_FUNCTION Scalar measure( Face<Dir>,
                                           const int[num_space_dim] ) const
    {
        static_assert( Dir < num_space_dim, "Face dimension out of bounds" );
        return _face_area[Dir];
    }

    //! Get the measure of a Cell at the given index.
    //! Local indexing is relative to the ghosted decomposition of the mesh
    //! block and correlates directly to local index spaces associated with the
    //! block.
    KOKKOS_INLINE_FUNCTION
    Scalar measure( Cell, const int[num_space_dim] ) const
    {
        return _cell_volume;
    }

  private:
    Kokkos::Array<Scalar, num_space_dim> _cell_size;
    Kokkos::Array<Scalar, num_space_dim> _face_area;
    Scalar _cell_volume;
    Kokkos::Array<Scalar, num_space_dim> _own_low_corner;
    Kokkos::Array<Scalar, num_space_dim> _own_high_corner;
    Kokkos::Array<Scalar, num_space_dim> _ghost_low_corner;
    Kokkos::Array<Scalar, num_space_dim> _ghost_high_corner;
    Kokkos::Array<bool, num_space_dim> _periodic;
    Kokkos::Array<bool, num_space_dim> _boundary_lo;
    Kokkos::Array<bool, num_space_dim> _boundary_hi;
};

//---------------------------------------------------------------------------//
//! Global mesh partial specialization for non-uniform mesh.
template <class Scalar, class Device, std::size_t NumSpaceDim>
class LocalMesh<Device, NonUniformMesh<Scalar, NumSpaceDim>>
{
  public:
    //! Mesh type.
    using mesh_type = NonUniformMesh<Scalar, NumSpaceDim>;

    //! Scalar type for geometric operations.
    using scalar_type = Scalar;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! Device type.
    using device_type = Device;
    //! Kokkos memory space.
    using memory_space = typename Device::memory_space;
    //! Kokkos execution space.
    using execution_space = typename Device::execution_space;

    //! Constructor.
    LocalMesh(
        const LocalGrid<NonUniformMesh<Scalar, num_space_dim>>& local_grid )
    {
        const auto& global_grid = local_grid.globalGrid();
        const auto& global_mesh = global_grid.globalMesh();

        // Compute the owned low corner.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _own_low_corner[d] =
                global_mesh.nonUniformEdge( d )[global_grid.globalOffset( d )];

        // Compute the owned high corner
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _own_high_corner[d] =
                global_mesh.nonUniformEdge( d )[global_grid.globalOffset( d ) +
                                                global_grid.ownedNumCell( d )];

        // Compute the ghosted low corner
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            // Interior domain case.
            if ( !global_grid.onLowBoundary( d ) )
            {
                _ghost_low_corner[d] = global_mesh.nonUniformEdge(
                    d )[global_grid.globalOffset( d ) -
                        local_grid.haloCellWidth()];
            }

            // Periodic boundary. Use cells on other side of boundary to
            // generate geometry.
            else if ( global_grid.isPeriodic( d ) )
            {
                int nedge = global_mesh.nonUniformEdge( d ).size();
                _ghost_low_corner[d] =
                    global_mesh.nonUniformEdge( d ).front() -
                    ( global_mesh.nonUniformEdge( d ).back() -
                      global_mesh.nonUniformEdge(
                          d )[nedge - local_grid.haloCellWidth() - 1] );
            }

            // In the non-periodic boundary case we extrapolate halo cells to
            // have the same width as the boundary cell.
            else
            {
                Scalar dx = global_mesh.nonUniformEdge( d )[1] -
                            global_mesh.nonUniformEdge( d )[0];
                _ghost_low_corner[d] = global_mesh.nonUniformEdge( d ).front() -
                                       dx * local_grid.haloCellWidth();
            }
        }

        // Compute the ghosted high corner
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            // Interior domain case.
            if ( !global_grid.onHighBoundary( d ) )
            {
                _ghost_high_corner[d] = global_mesh.nonUniformEdge(
                    d )[global_grid.globalOffset( d ) +
                        global_grid.ownedNumCell( d ) +
                        local_grid.haloCellWidth()];
            }

            // Periodic boundary. Use cells on other side of boundary to
            // generate geometry.
            else if ( global_grid.isPeriodic( d ) )
            {
                _ghost_high_corner[d] =
                    global_mesh.nonUniformEdge( d ).back() +
                    ( global_mesh.nonUniformEdge(
                          d )[local_grid.haloCellWidth()] -
                      global_mesh.nonUniformEdge( d ).front() );
            }

            // In the non-periodic boundary case we extrapolate halo cells to
            // have the same width as the boundary cell.
            else
            {
                int nedge = global_mesh.nonUniformEdge( d ).size();
                Scalar dx = global_mesh.nonUniformEdge( d )[nedge - 1] -
                            global_mesh.nonUniformEdge( d )[nedge - 2];
                _ghost_high_corner[d] = global_mesh.nonUniformEdge( d ).back() +
                                        dx * local_grid.haloCellWidth();
            }
        }

        // Get the node index spaces.
        auto owned_nodes_local =
            local_grid.indexSpace( Own(), Node(), Local() );
        auto ghosted_nodes_local =
            local_grid.indexSpace( Ghost(), Node(), Local() );
        auto owned_nodes_global =
            local_grid.indexSpace( Own(), Node(), Global() );
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            // Allocate edges on the device for this dimension.
            const auto& global_edge = global_mesh.nonUniformEdge( d );
            int nedge = ghosted_nodes_local.extent( d );
            int nedge_global = global_edge.size();
            _local_edges[d] = Kokkos::View<Scalar*, Device>(
                Kokkos::ViewAllocateWithoutInitializing( "local_edges" ),
                nedge );

            // Compute edges on the host.
            auto edge_mirror = Kokkos::create_mirror_view( Kokkos::HostSpace(),
                                                           _local_edges[d] );

            // Compute the owned edges.
            for ( int n = owned_nodes_local.min( d );
                  n < owned_nodes_local.max( d ); ++n )
            {
                edge_mirror( n ) = global_edge[owned_nodes_global.min( d ) + n -
                                               owned_nodes_local.min( d )];
            }

            // Compute the lower boundary edges.
            if ( !global_grid.onLowBoundary( d ) )
            {
                // Interior block gets edges from neighbors.
                for ( int n = 0; n < owned_nodes_local.min( d ); ++n )
                {
                    edge_mirror( n ) =
                        global_edge[owned_nodes_global.min( d ) + n -
                                    owned_nodes_local.min( d )];
                }
            }
            else if ( global_grid.isPeriodic( d ) )
            {
                // Periodic boundary block gets edges from neighbor on
                // opposite side of boundary.
                for ( int n = 0; n < owned_nodes_local.min( d ); ++n )
                {
                    edge_mirror( n ) =
                        global_edge.front() - global_edge.back() +
                        global_edge[global_edge.size() - 1 -
                                    local_grid.haloCellWidth() + n];
                }
            }
            else
            {
                // Non-periodic boundary block extrapolates edges using
                // boundary cell width.
                for ( int n = 0; n < owned_nodes_local.min( d ); ++n )
                {
                    Scalar dx = global_edge[1] - global_edge[0];
                    edge_mirror( n ) = global_edge.front() -
                                       ( owned_nodes_local.min( d ) - n ) * dx;
                }
            }

            // Compute the upper boundary edges.
            if ( !global_grid.onHighBoundary( d ) )
            {
                // Interior block gets edges from neighbors.
                for ( int n = owned_nodes_local.max( d );
                      n < ghosted_nodes_local.max( d ); ++n )
                {
                    edge_mirror( n ) =
                        global_edge[owned_nodes_global.min( d ) + n -
                                    owned_nodes_local.min( d )];
                }
            }
            else if ( global_grid.isPeriodic( d ) )
            {
                // Periodic boundary block gets edges from neighbor on
                // opposite side of boundary.
                for ( int n = 0; n < ghosted_nodes_local.max( d ) -
                                         owned_nodes_local.max( d );
                      ++n )
                {
                    edge_mirror( owned_nodes_local.max( d ) + n ) =
                        global_edge.back() + global_edge[n] -
                        global_edge.front();
                }
            }
            else
            {
                // Non-periodic boundary block extrapolates edges using
                // boundary cell width.
                for ( int n = owned_nodes_local.max( d );
                      n < ghosted_nodes_local.max( d ); ++n )
                {
                    Scalar dx = global_edge[nedge_global - 1] -
                                global_edge[nedge_global - 2];
                    edge_mirror( n ) =
                        global_edge.back() +
                        ( n - owned_nodes_local.max( d ) + 1 ) * dx;
                }
            }

            // Copy edges to the device.
            Kokkos::deep_copy( _local_edges[d], edge_mirror );
        }

        // Periodicity
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _periodic[d] = global_grid.isPeriodic( d );

        // Determine if a block is on the low or high boundaries.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            _boundary_lo[d] = global_grid.onLowBoundary( d );
            _boundary_hi[d] = global_grid.onHighBoundary( d );
        }
    }

    //! Determine if the mesh is periodic in the given dimension.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    bool isPeriodic( const int dim ) const { return _periodic[dim]; }

    //! Determine if this block is on a low boundary in the given dimension.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    bool onLowBoundary( const int dim ) const { return _boundary_lo[dim]; }

    //! Determine if this block is on a high boundary in the given dimension.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    bool onHighBoundary( const int dim ) const { return _boundary_hi[dim]; }

    //! Get the physical coordinate of the low corner of the owned local grid.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    Scalar lowCorner( Own, const int dim ) const
    {
        return _own_low_corner[dim];
    }

    //! Get the physical coordinate of the low corner of the local grid
    //! including ghosts.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    Scalar lowCorner( Ghost, const int dim ) const
    {
        return _ghost_low_corner[dim];
    }

    //! Get the physical coordinate of the high corner of the owned local grid.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    Scalar highCorner( Own, const int dim ) const
    {
        return _own_high_corner[dim];
    }

    //! Get the physical coordinate of the high corner of the local grid
    //! including ghosts.
    //! \param dim Spatial dimension.
    KOKKOS_INLINE_FUNCTION
    Scalar highCorner( Ghost, const int dim ) const
    {
        return _ghost_high_corner[dim];
    }

    //! Get the physical length of the local grid of the given decomposition.
    //! \param d Decomposition: Own or Ghost
    //! \param dim Spatial dimension.
    template <typename Decomposition>
    KOKKOS_FUNCTION Scalar extent( Decomposition d, const int dim ) const
    {
        return highCorner( d, dim ) - lowCorner( d, dim );
    }

    /*!
      Get the coordinates of a Cell given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block, which correlates directly to
      local index spaces associated with the block.
      \param x Calculated Cell position
    */
    KOKKOS_INLINE_FUNCTION
    void coordinates( Cell, const int index[num_space_dim],
                      Scalar x[num_space_dim] ) const
    {
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            x[d] = ( _local_edges[d]( index[d] + 1 ) +
                     _local_edges[d]( index[d] ) ) /
                   Scalar( 2.0 );
    }

    /*!
      Get the coordinates of a Node given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block.
      \param x Calculated Node position
    */
    KOKKOS_INLINE_FUNCTION
    void coordinates( Node, const int index[num_space_dim],
                      Scalar x[num_space_dim] ) const
    {
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            x[d] = _local_edges[d]( index[d] );
    }

    /*!
      Get the coordinates of a Face given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block.
      \param x Calculated Face position
    */
    template <int Dir>
    KOKKOS_INLINE_FUNCTION void coordinates( Face<Dir>,
                                             const int index[num_space_dim],
                                             Scalar x[num_space_dim] ) const
    {
        static_assert( Dir < num_space_dim, "Face dimension out of bounds" );

        for ( std::size_t d = 0; d < num_space_dim; ++d )
            if ( Dir == d )
                x[d] = _local_edges[d]( index[d] );
            else
                x[d] = ( _local_edges[d]( index[d] + 1 ) +
                         _local_edges[d]( index[d] ) ) /
                       Scalar( 2.0 );
    }

    /*!
      Get the coordinates of a Edge given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block.
      \param x Calculated Edge position
    */
    template <int Dir, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    coordinates( Edge<Dir>, const int index[3], Scalar x[3] ) const
    {
        for ( std::size_t d = 0; d < 3; ++d )
            if ( Dir == d )
                x[d] = ( _local_edges[d]( index[d] + 1 ) +
                         _local_edges[d]( index[d] ) ) /
                       Scalar( 2.0 );
            else
                x[d] = _local_edges[d]( index[d] );
    }

    /*!
      Get the measure of a Node.
    */
    KOKKOS_INLINE_FUNCTION
    Scalar measure( Node, const int[num_space_dim] ) const { return 0.0; }

    /*!
      Get the measure of a 3d Edge given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block
    */
    template <int Dir, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, Scalar>
    measure( Edge<Dir>, const int index[3] ) const
    {
        return _local_edges[Dir][index[Dir] + 1] -
               _local_edges[Dir][index[Dir]];
    }

    /*!
      Get the measure of an 3d I-Face given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block
    */
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, Scalar>
    measure( Face<Dim::I>, const int index[3] ) const
    {
        return measure( Edge<Dim::J>(), index ) *
               measure( Edge<Dim::K>(), index );
    }

    /*!
      Get the measure of a 3d J-Face given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block
    */
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, Scalar>
    measure( Face<Dim::J>, const int index[3] ) const
    {
        return measure( Edge<Dim::I>(), index ) *
               measure( Edge<Dim::K>(), index );
    }

    /*!
      Get the measure of a 3d K-Face given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block
    */
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, Scalar>
    measure( Face<Dim::K>, const int index[3] ) const
    {
        return measure( Edge<Dim::I>(), index ) *
               measure( Edge<Dim::J>(), index );
    }

    /*!
      Get the measure of a 2d I-Face given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block
    */
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, Scalar>
    measure( Face<Dim::I>, const int index[2] ) const
    {
        return _local_edges[Dim::J][index[Dim::J] + 1] -
               _local_edges[Dim::J][index[Dim::J]];
    }

    /*!
      Get the measure of a 2d J-Face given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block
    */
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, Scalar>
    measure( Face<Dim::J>, const int index[2] ) const
    {
        return _local_edges[Dim::I][index[Dim::I] + 1] -
               _local_edges[Dim::I][index[Dim::I]];
    }

    /*!
      Get the measure of a Cell given the local index.
      \param index %Array of local indices relative to the
      ghosted decomposition of the mesh block
    */
    KOKKOS_INLINE_FUNCTION
    Scalar measure( Cell, const int index[num_space_dim] ) const
    {
        Scalar m = 1.0;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            m *= _local_edges[d][index[d] + 1] - _local_edges[d][index[d]];
        return m;
    }

  private:
    Kokkos::Array<Scalar, num_space_dim> _own_low_corner;
    Kokkos::Array<Scalar, num_space_dim> _own_high_corner;
    Kokkos::Array<Scalar, num_space_dim> _ghost_low_corner;
    Kokkos::Array<Scalar, num_space_dim> _ghost_high_corner;
    Kokkos::Array<Kokkos::View<Scalar*, Device>, num_space_dim> _local_edges;
    Kokkos::Array<bool, num_space_dim> _periodic;
    Kokkos::Array<bool, num_space_dim> _boundary_lo;
    Kokkos::Array<bool, num_space_dim> _boundary_hi;
};

//---------------------------------------------------------------------------//
//! Creation function for local mesh.
template <class Device, class MeshType>
LocalMesh<Device, MeshType>
createLocalMesh( const LocalGrid<MeshType>& local_grid )
{
    return LocalMesh<Device, MeshType>( local_grid );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJTIA_LOCALMESH_HPP
