/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJTIA_LOCALMESH_HPP
#define CAJTIA_LOCALMESH_HPP

#include <Cajita_Block.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Forward decalaration of local mesh.
template <class Device, class MeshType>
class LocalMesh;

//---------------------------------------------------------------------------//
// Local mesh partial specialization for uniform mesh.
template <class Scalar, class Device>
class LocalMesh<Device, UniformMesh<Scalar>>
{
  public:
    // Mesh type.
    using mesh_type = UniformMesh<Scalar>;

    // Scalar type for geometric operations.
    using scalar_type = Scalar;

    // Device type.
    using device_type = Device;
    using memory_space = typename Device::memory_space;
    using execution_space = typename Device::execution_space;

    // Constructor.
    LocalMesh( const Block<UniformMesh<Scalar>> &block )
    {
        const auto &global_grid = block.globalGrid();
        const auto &global_mesh = global_grid.globalMesh();

        _cell_size = global_mesh.uniformCellSize();

        // Compute the owned low corner
        for ( int d = 0; d < 3; ++d )
            _own_low_corner[d] = global_mesh.lowCorner( d ) +
                                 _cell_size * global_grid.globalOffset( d );

        // Compute the owned high corner
        for ( int d = 0; d < 3; ++d )
            _own_high_corner[d] =
                global_mesh.lowCorner( d ) +
                _cell_size * ( global_grid.globalOffset( d ) +
                               global_grid.ownedNumCell( d ) );

        // Compute the ghosted low corner
        for ( int d = 0; d < 3; ++d )
            _ghost_low_corner[d] =
                ( global_grid.isPeriodic( d ) ||
                  global_grid.dimBlockId( d ) > 0 )
                    ? lowCorner( Own(), d ) - block.haloCellWidth() * _cell_size
                    : lowCorner( Own(), d );

        // Compute the ghosted high corner
        for ( int d = 0; d < 3; ++d )
            _ghost_high_corner[d] = ( global_grid.isPeriodic( d ) ||
                                      global_grid.dimBlockId( d ) <
                                          global_grid.dimNumBlock( d ) - 1 )
                                        ? highCorner( Own(), d ) +
                                              block.haloCellWidth() * _cell_size
                                        : highCorner( Own(), d );
    }

    // Get the physical coordinate of the low corner of the block in a given
    // dimension in the given decomposition.
    KOKKOS_INLINE_FUNCTION
    Scalar lowCorner( Own, const int dim ) const
    {
        return _own_low_corner[dim];
    }

    KOKKOS_INLINE_FUNCTION
    Scalar lowCorner( Ghost, const int dim ) const
    {
        return _ghost_low_corner[dim];
    }

    // Get the physical coordinate of the high corner of the block in a given
    // dimension in the given decomposition.
    KOKKOS_INLINE_FUNCTION
    Scalar highCorner( Own, const int dim ) const
    {
        return _own_high_corner[dim];
    }

    KOKKOS_INLINE_FUNCTION
    Scalar highCorner( Ghost, const int dim ) const
    {
        return _ghost_high_corner[dim];
    }

    // Get the size of a cell in a given dimension given the local index of
    // the cell in that dimension. The local indexing is relative to the
    // ghosted decomposition of the mesh block and correlates directly to
    // local index spaces associated with the block.
    KOKKOS_INLINE_FUNCTION
    Scalar cellSize( const int, const int ) const { return _cell_size; }

    // Get the coordinate of an entity of the given type in the given
    // dimension at given the local index of the entity in that dimension. The
    // local indexing is relative to the ghosted decomposition of the mesh
    // block and correlates directly to local index spaces associated with the
    // block.
    KOKKOS_INLINE_FUNCTION
    Scalar coordinate( Cell, const int i, const int dim ) const
    {
        return _ghost_low_corner[dim] +
               ( Scalar( i ) + Scalar( 0.5 ) ) * _cell_size;
    }

    KOKKOS_INLINE_FUNCTION
    Scalar coordinate( Node, const int i, const int dim ) const
    {
        return _ghost_low_corner[dim] + Scalar( i ) * _cell_size;
    }

    template <int Dir>
    KOKKOS_INLINE_FUNCTION Scalar coordinate( Face<Dir>, const int i,
                                              const int dim ) const
    {
        return ( Dir != dim ) ? coordinate( Cell(), i, dim )
                              : coordinate( Node(), i, dim );
    }

    template <int Dir>
    KOKKOS_INLINE_FUNCTION Scalar coordinate( Edge<Dir>, const int i,
                                              const int dim ) const
    {
        return ( Dir == dim ) ? coordinate( Cell(), i, dim )
                              : coordinate( Node(), i, dim );
    }

  private:
    Scalar _cell_size;
    Kokkos::Array<Scalar, 3> _own_low_corner;
    Kokkos::Array<Scalar, 3> _own_high_corner;
    Kokkos::Array<Scalar, 3> _ghost_low_corner;
    Kokkos::Array<Scalar, 3> _ghost_high_corner;
};

//---------------------------------------------------------------------------//
// Global mesh partial specialization for non-uniform mesh.
template <class Scalar, class Device>
class LocalMesh<Device, NonUniformMesh<Scalar>>
{
  public:
    // Mesh type.
    using mesh_type = NonUniformMesh<Scalar>;

    // Scalar type for geometric operations.
    using scalar_type = Scalar;

    // Device type.
    using device_type = Device;
    using memory_space = typename Device::memory_space;
    using execution_space = typename Device::execution_space;

    // Constructor.
    LocalMesh( const Block<NonUniformMesh<Scalar>> &block )
    {
        const auto &global_grid = block.globalGrid();
        const auto &global_mesh = global_grid.globalMesh();

        // Compute the owned low corner.
        for ( int d = 0; d < 3; ++d )
            _own_low_corner[d] =
                global_mesh.nonUniformEdge( d )[global_grid.globalOffset( d )];

        // Compute the owned high corner
        for ( int d = 0; d < 3; ++d )
            _own_high_corner[d] =
                global_mesh.nonUniformEdge( d )[global_grid.globalOffset( d ) +
                                                global_grid.ownedNumCell( d )];

        // Compute the ghosted low corner
        for ( int d = 0; d < 3; ++d )
        {
            if ( global_grid.dimBlockId( d ) > 0 )
            {
                _ghost_low_corner[d] = global_mesh.nonUniformEdge(
                    d )[global_grid.globalOffset( d ) - block.haloCellWidth()];
            }
            else if ( global_grid.isPeriodic( d ) )
            {
                int nedge = global_mesh.nonUniformEdge( d ).size();
                _ghost_low_corner[d] =
                    global_mesh.nonUniformEdge( d ).front() -
                    ( global_mesh.nonUniformEdge( d ).back() -
                      global_mesh.nonUniformEdge(
                          d )[nedge - block.haloCellWidth() - 1] );
            }
            else
            {
                _ghost_low_corner[d] = global_mesh.nonUniformEdge( d ).front();
            }
        }

        // Compute the ghosted high corner
        for ( int d = 0; d < 3; ++d )
        {
            if ( global_grid.dimBlockId( d ) <
                 global_grid.dimNumBlock( d ) - 1 )
            {
                _ghost_high_corner[d] = global_mesh.nonUniformEdge(
                    d )[global_grid.globalOffset( d ) +
                        global_grid.ownedNumCell( d ) + block.haloCellWidth()];
            }
            else if ( global_grid.isPeriodic( d ) )
            {
                _ghost_high_corner[d] =
                    global_mesh.nonUniformEdge( d ).back() +
                    ( global_mesh.nonUniformEdge( d )[block.haloCellWidth()] -
                      global_mesh.nonUniformEdge( d ).front() );
            }
            else
            {
                _ghost_high_corner[d] = global_mesh.nonUniformEdge( d ).back();
            }
        }

        // Get the node index spaces.
        auto owned_nodes_local = block.indexSpace( Own(), Node(), Local() );
        auto ghosted_nodes_local = block.indexSpace( Ghost(), Node(), Local() );
        auto ghosted_nodes_global =
            block.indexSpace( Ghost(), Node(), Global() );
        for ( int d = 0; d < 3; ++d )
        {
            // Allocate edges on the device for this dimension.
            const auto &global_edge = global_mesh.nonUniformEdge( d );
            int nedge = ghosted_nodes_global.extent( d );
            _local_edges[d] = Kokkos::View<Scalar *, Device>(
                Kokkos::ViewAllocateWithoutInitializing( "local_edges" ),
                nedge );

            // Compute edges on the host.
            auto edge_mirror = Kokkos::create_mirror_view( Kokkos::HostSpace(),
                                                           _local_edges[d] );

            // Compute the owned edges.
            for ( int n = owned_nodes_local.min( d );
                  n < owned_nodes_local.max( d ); ++n )
                edge_mirror( n ) =
                    global_edge[ghosted_nodes_global.min( d ) + n];

            // Compute the lower boundary edges.
            if ( global_grid.dimBlockId( d ) > 0 )
                for ( int n = 0; n < owned_nodes_local.min( d ); ++n )
                    edge_mirror( n ) =
                        global_edge[ghosted_nodes_global.min( d ) + n];
            else if ( global_grid.isPeriodic( d ) )
                for ( int n = 0; n < owned_nodes_local.min( d ); ++n )
                    edge_mirror( n ) = global_edge.front() -
                                       global_edge.back() +
                                       global_edge[global_edge.size() - 1 -
                                                   block.haloCellWidth() + n];

            // Compute the upper boundary edges.
            if ( global_grid.dimBlockId( d ) <
                 global_grid.dimNumBlock( d ) - 1 )
                for ( int n = owned_nodes_local.max( d );
                      n < ghosted_nodes_local.max( d ); ++n )
                    edge_mirror( n ) =
                        global_edge[ghosted_nodes_global.min( d ) + n];
            else if ( global_grid.isPeriodic( d ) )
                for ( int n = 0; n < ghosted_nodes_local.max( d ) -
                                         owned_nodes_local.max( d );
                      ++n )
                    edge_mirror( owned_nodes_local.max( d ) + n ) =
                        global_edge.back() + global_edge[n] -
                        global_edge.front();

            // Copy edges to the device.
            Kokkos::deep_copy( _local_edges[d], edge_mirror );
        }
    }

    // Get the physical coordinate of the low corner of the block in a given
    // dimension in the given decomposition.
    KOKKOS_INLINE_FUNCTION
    Scalar lowCorner( Own, const int dim ) const
    {
        return _own_low_corner[dim];
    }

    KOKKOS_INLINE_FUNCTION
    Scalar lowCorner( Ghost, const int dim ) const
    {
        return _ghost_low_corner[dim];
    }

    // Get the physical coordinate of the high corner of the block in a given
    // dimension in the given decomposition.
    KOKKOS_INLINE_FUNCTION
    Scalar highCorner( Own, const int dim ) const
    {
        return _own_high_corner[dim];
    }

    KOKKOS_INLINE_FUNCTION
    Scalar highCorner( Ghost, const int dim ) const
    {
        return _ghost_high_corner[dim];
    }

    // Get the size of a cell in a given dimension given the local index of the
    // cell in that dimension. The local indexing is relative to the ghosted
    // decomposition of the mesh block and correlates directly to local index
    // spaces associated with the block.
    KOKKOS_INLINE_FUNCTION
    Scalar cellSize( const int i, const int dim ) const
    {
        return _local_edges[dim]( i + 1 ) - _local_edges[dim]( i );
    }

    // Get the coordinate of an entity of the given type in the given
    // dimension at given the local index of the entity in that dimension. The
    // local indexing is relative to the ghosted decomposition of the mesh
    // block and correlates directly to local index spaces associated with the
    // block.
    KOKKOS_INLINE_FUNCTION
    Scalar coordinate( Cell, const int i, const int dim ) const
    {
        return ( _local_edges[dim]( i + 1 ) + _local_edges[dim]( i ) ) /
               Scalar( 2.0 );
    }

    KOKKOS_INLINE_FUNCTION
    Scalar coordinate( Node, const int i, const int dim ) const
    {
        return _local_edges[dim]( i );
    }

    template <int Dir>
    KOKKOS_INLINE_FUNCTION Scalar coordinate( Face<Dir>, const int i,
                                              const int dim ) const
    {
        return ( Dir != dim ) ? coordinate( Cell(), i, dim )
                              : coordinate( Node(), i, dim );
    }

    template <int Dir>
    KOKKOS_INLINE_FUNCTION Scalar coordinate( Edge<Dir>, const int i,
                                              const int dim ) const
    {
        return ( Dir == dim ) ? coordinate( Cell(), i, dim )
                              : coordinate( Node(), i, dim );
    }

  private:
    Kokkos::Array<Scalar, 3> _own_low_corner;
    Kokkos::Array<Scalar, 3> _own_high_corner;
    Kokkos::Array<Scalar, 3> _ghost_low_corner;
    Kokkos::Array<Scalar, 3> _ghost_high_corner;
    Kokkos::Array<Kokkos::View<Scalar *, Device>, 3> _local_edges;
};

//---------------------------------------------------------------------------//
// Creation function.
template <class Device, class MeshType>
LocalMesh<Device, MeshType> createLocalMesh( const Block<MeshType> &block )
{
    return LocalMesh<Device, MeshType>( block );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJTIA_LOCALMESH_HPP
