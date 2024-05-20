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
  \file Cabana_Grid_SparseArray.hpp
  \brief Sparse grid fields arrays using AoSoA
*/
#ifndef CABANA_GRID_SPARSE_ARRAY_HPP
#define CABANA_GRID_SPARSE_ARRAY_HPP

#include <Cabana_Grid_SparseIndexSpace.hpp>
#include <Cabana_Grid_SparseLocalGrid.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Cabana_AoSoA.hpp>
#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <Kokkos_Core.hpp>

#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

#include <mpi.h>

namespace Cabana
{
namespace Grid
{

namespace Experimental
{
//---------------------------------------------------------------------------//
/*!
  \brief Entity layout for sparse array data on the local sparse mesh.

  \tparam DataTypes Array member types (Cabana::MemberTypes)
  \tparam EntityType Array entity type: Cell, Node, Edge, or Face
  \tparam MeshType Mesh type: SparseMesh
  \tparam SparseMapType: sparse map type
*/
template <class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
class SparseArrayLayout
{
  public:
    //! Mesh Type, should be SparseMesh
    using mesh_type = MeshType;
    //! Scalar Type
    using scalar_type = typename mesh_type::scalar_type;
    // check if mesh_type is SparseMesh
    static_assert( isSparseMesh<MeshType>::value,
                   "[SparesArrayLayout] Support only SparseMesh" );

    //! Entity Type
    using entity_type = EntityType;
    //! Array member types, such as Cabana::MemberTypes<double, float[3]>
    using member_types = DataTypes;

    //! Abbreviation for Sparse Map Type
    using sparse_map_type = SparseMapType;
    //! value type in sparse map, i.e., the tile ID (array ID) type
    using value_type = typename sparse_map_type::value_type;
    //! key type in sparse map, i.e., the tile key type
    using key_type = typename sparse_map_type::key_type;
    //! least bit number required to represent local cell ids inside a tile per
    //! dimension
    static constexpr unsigned long long cell_bits_per_tile_dim =
        sparse_map_type::cell_bits_per_tile_dim;
    //! cell number inside each tile per dimension
    static constexpr unsigned long long cell_num_per_tile_dim =
        sparse_map_type::cell_num_per_tile_dim;
    //! dimension number
    static constexpr std::size_t num_space_dim = sparse_map_type::rank;
    //! Memory space, the same memory space in sparse map
    using memory_space = typename sparse_map_type::memory_space;

    /*!
    \brief (Host) Constructor
    \param local_grid Shared pointer to local grid
    \param sparse_map Reference to sparse map
    \param over_allocation  Factor to increase reserved size for Edge and Face
    entity
    */
    SparseArrayLayout( const std::shared_ptr<LocalGrid<MeshType>>& local_grid,
                       SparseMapType& sparse_map, const float over_allocation )
        : _over_allocation( over_allocation )
        , _local_grid( local_grid )
        , _map( std::forward<sparse_map_type>( sparse_map ) )
    {
        auto sparse_mesh = _local_grid->globalGrid().globalMesh();
        _cell_size[0] = sparse_mesh.cellSize( 0 );
        _cell_size[1] = sparse_mesh.cellSize( 1 );
        _cell_size[2] = sparse_mesh.cellSize( 2 );
        _global_low_corner[0] = sparse_mesh.lowCorner( 0 );
        _global_low_corner[1] = sparse_mesh.lowCorner( 1 );
        _global_low_corner[2] = sparse_mesh.lowCorner( 2 );
    }

    //! get reference of sparse map
    SparseMapType& sparseMap() { return _map; }

    //! Get the local grid over which this layout is defined.
    const std::shared_ptr<LocalGrid<MeshType>> localGrid() const
    {
        return _local_grid;
    }

    //! array size in cell
    inline uint64_t sizeCell() const { return _map.sizeCell(); }

    //! array size in tile
    inline uint64_t sizeTile() const { return _map.sizeTile(); }

    /*!
      \brief array reservation size in cell
      \param factor scale up the real size as reserved space
    */
    inline uint64_t reservedCellSize( float factor ) const
    {
        return _map.reservedCellSize( factor );
    }

    //! Array size in cell (default size measurse: cell)
    inline uint64_t arraySize() const { return sizeCell(); }

    //! clear valid info inside array layout; i.e. clear sparse map
    inline void clear() { _map.clear(); }

    /*!
      \brief Register valid grids in sparse map according to input particle
      positions.
      \param positions Input particle positions.
      \param particle_num Number of valid particles inside positions
      \param p2g_radius The half range of grids that will be influenced by each
      particle, depending on the interpolation kernel
    */
    template <class ExecSpace, class PositionSliceType>
    void registerSparseMap( PositionSliceType& positions,
                            const int particle_num, const int p2g_radius = 1 )
    {
        // get references
        Kokkos::Array<scalar_type, 3> dx_inv = {
            (scalar_type)1.0 / _cell_size[0], (scalar_type)1.0 / _cell_size[1],
            (scalar_type)1.0 / _cell_size[2] };
        auto& map = _map;
        auto& low_corner = _global_low_corner;
        // register sparse map in sparse array layout
        Kokkos::parallel_for(
            "register sparse map in sparse array layout",
            Kokkos::RangePolicy<ExecSpace>( 0, particle_num ),
            KOKKOS_LAMBDA( const int pid ) {
                scalar_type pos[3] = { positions( pid, 0 ) - low_corner[0],
                                       positions( pid, 1 ) - low_corner[1],
                                       positions( pid, 2 ) - low_corner[2] };
                int grid_base[3] = {
                    static_cast<int>( std::lround( pos[0] * dx_inv[0] ) -
                                      p2g_radius ),
                    static_cast<int>( std::lround( pos[1] * dx_inv[1] ) -
                                      p2g_radius ),
                    static_cast<int>( std::lround( pos[2] * dx_inv[2] ) -
                                      p2g_radius ) };
                // register grids that will have data transfer with the particle
                const int p2g_size = p2g_radius * 2;
                for ( int i = 0; i <= p2g_size; ++i )
                    for ( int j = 0; j <= p2g_size; ++j )
                        for ( int k = 0; k <= p2g_size; ++k )
                        {
                            int cell_id[3] = { grid_base[0] + i,
                                               grid_base[1] + j,
                                               grid_base[2] + k };
                            map.insertCell( cell_id[0], cell_id[1],
                                            cell_id[2] );
                        }
            } );
    }

    /*!
      \brief (Device) Query the 1D cell ID from the 3D cell ijk
      \param cell_i, cell_j, cell_k Cell ID in each dimension
    */
    KOKKOS_FORCEINLINE_FUNCTION
    value_type queryCell( const int cell_i, const int cell_j,
                          const int cell_k ) const
    {
        return _map.queryCell( cell_i, cell_j, cell_k );
    }

    /*!
      \brief (Device) Query the 1D tile ID from the 3D tile ijk
      \param cell_i, cell_j, cell_k Cell ID in each dimension
    */
    KOKKOS_FORCEINLINE_FUNCTION
    value_type queryTile( const int cell_i, const int cell_j,
                          const int cell_k ) const
    {
        return _map.queryTile( cell_i, cell_j, cell_k );
    }

    /*!
      \brief (Device) Query the 1D tile key from the 3D tile ijk
      \param tile_i, tile_j, tile_k Tile ID in each dimension
    */
    KOKKOS_FORCEINLINE_FUNCTION
    value_type queryTileFromTileId( const int tile_i, const int tile_j,
                                    const int tile_k ) const
    {
        return _map.queryTileFromTileId( tile_i, tile_j, tile_k );
    }

    /*!
      \brief (Device) Query the 1D tile key from the 1D tile key
      \param tile_key 1D tile key
    */
    KOKKOS_FORCEINLINE_FUNCTION
    value_type queryTileFromTileKey( const key_type tile_key ) const
    {
        return _map.queryTileFromTileKey( tile_key );
    }

    /*!
      \brief (Device) Get local cell ID from cell IJK
      \param cell_i, cell_j, cell_k Cell ID in each dimension (both local and
      global IJK work)
    */
    KOKKOS_FORCEINLINE_FUNCTION
    value_type cellLocalId( const int cell_i, const int cell_j,
                            const int cell_k ) const
    {
        return _map.cell_local_id( cell_i, cell_j, cell_k );
    }

  private:
    //! factor to increase array size for special grid entities
    float _over_allocation;
    //! cell size
    Kokkos::Array<scalar_type, 3> _cell_size;
    //! global low corner
    Kokkos::Array<scalar_type, 3> _global_low_corner;
    //! Sparse local grid
    std::shared_ptr<LocalGrid<MeshType>> _local_grid;
    //！ sparse map
    sparse_map_type _map;
}; // end class SparseArrayLayout

//! Sparse array layout static type checker.
template <class>
struct is_sparse_array_layout : public std::false_type
{
};

//! Sparse array layout static type checker.
template <class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
struct is_sparse_array_layout<
    SparseArrayLayout<DataTypes, EntityType, MeshType, SparseMapType>>
    : public std::true_type
{
};

//! Sparse array layout static type checker.
template <class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
struct is_sparse_array_layout<
    const SparseArrayLayout<DataTypes, EntityType, MeshType, SparseMapType>>
    : public std::true_type
{
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create sparse array layout over the entities of a sparse local grid.
  \param local_grid The sparse local grid over which to create the layout.
  \param sparse_map The reference to a pre-created sparse map.
  \param over_allocation  Factor used to increase allocation size for special
  entities.

  \note EntityType The entity: Cell, Node, Face, or Edge.
*/
template <class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
auto createSparseArrayLayout(
    const std::shared_ptr<LocalGrid<MeshType>>& local_grid,
    SparseMapType& sparse_map, EntityType, const float over_allocation = 1.01f )
{
    return std::make_shared<
        SparseArrayLayout<DataTypes, EntityType, MeshType, SparseMapType>>(
        local_grid, sparse_map, over_allocation );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sparse array of field data on the local sparse mesh; Array data is
  stored in AoSoA manner, with each tile being the SoA element

  \tparam DataTypes Data types (Cabana::MemberTypes).
  \tparam MemorySpace Kokkos memory space.
  \tparam EntityType Array entity type (node, cell, face, edge).
  \tparam MeshType Mesh type (sparse mesh).
  \tparam SparseMapType Sparse map type.
*/
template <class DataTypes, class MemorySpace, class EntityType, class MeshType,
          class SparseMapType>
class SparseArray
{
  public:
    //! self type
    using sparse_array_type = SparseArray<DataTypes, MemorySpace, EntityType,
                                          MeshType, SparseMapType>;
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    static_assert( Kokkos::is_memory_space<MemorySpace>() );
    //! Default execution space.
    using execution_space = typename memory_space::execution_space;
    //! Memory space size type
    using size_type = typename memory_space::size_type;
    //! Array entity type (node, cell, face, edge).
    using entity_type = EntityType;
    //! Mesh type
    using mesh_type = MeshType;
    //! Sparse map type
    using sparse_map_type = SparseMapType;
    //! Dimension number
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;
    //! Least bits required to represent all cells inside a tile
    static constexpr unsigned long long cell_bits_per_tile =
        sparse_map_type::cell_bits_per_tile;
    //! Cell ID mask inside a tile
    static constexpr unsigned long long cell_mask_per_tile =
        sparse_map_type::cell_mask_per_tile;

    // AoSoA related types
    //! DataTypes Data types (Cabana::MemberTypes).
    using member_types = DataTypes;
    //! AoSoA vector length = cell number in each tile
    static constexpr int vector_length = sparse_map_type::cell_num_per_tile;
    //! AosoA type
    using aosoa_type = Cabana::AoSoA<member_types, memory_space, vector_length>;
    //! SoA Type
    using soa_type = Cabana::SoA<member_types, vector_length>;
    //! AoSoA tuple type
    using tuple_type = Cabana::Tuple<member_types>;

    //! Sparse array layout type
    using array_layout = SparseArrayLayout<member_types, entity_type, mesh_type,
                                           sparse_map_type>;
    /*!
      \brief (Host) Constructor
      \param label array description
      \param layout sparse array layout
    */
    SparseArray( const std::string label, array_layout layout )
        : _layout( layout )
        , _data( label )
    {
    }

    // ------------------------------------------------------------------------
    // AoSoA-related interfaces
    //! Get AoSoA reference
    aosoa_type& aosoa() { return _data; }
    /*!
      \brief Get array label (description)
      \return Array label (description)
    */
    std::string label() const { return _data.label(); }
    //! Get reference to the array layout (device accessible)
    array_layout& layout() { return _layout; }
    //! Get const reference to the array layout (device accessible)
    const array_layout& layout() const { return _layout; }

    /*!
      \brief Resize the AoSoA array according to the input.
      \param n Target size.
    */
    inline void resize( const size_type n ) { _data.resize( n ); }

    /*!
      \brief Reserve the AoSoA array according to the sparse map info in layout.
    */
    inline void resize() { resize( _layout.sizeCell() ); }

    /*!
      \brief Reserve the AoSoA array according to the input.
      \param n Target reserved size.
    */
    inline void reserve( const size_type n ) { _data.reserve( n ); }

    /*!
      \brief Reserve the AoSoA array according to the sparse map info in layout.
      \param factor scale up the real size as reserved space
    */
    inline void reserveFromMap( const double factor = 1.2 )
    {
        reserve( _layout.reservedCellSize( factor ) );
    }

    //! Shrink allocation to fit the valid size
    inline void shrinkToFit() { _data.shrinkToFit(); }
    //! Clear sparse array, including resize valid AoSoA size to 0 and clear
    //! sparse layout
    inline void clear()
    {
        resize( 0 );
        _layout.clear();
    };

    //! Get AoSoA capacity
    KOKKOS_FUNCTION
    size_type capacity() { return _data.capacity(); }
    //! Get AoSoA size (valid number of elements)
    KOKKOS_FUNCTION
    size_type size() const { return _data.size(); }
    //! Test if the AoSoA array is empty
    KOKKOS_FUNCTION
    bool empty() const { return ( size() == 0 ); }
    //! Get the number of SoA inside an AoSoA structure
    KOKKOS_FORCEINLINE_FUNCTION
    size_type numSoA() const { return _data.numSoA(); }
    //! Get data array size at a given struct member index
    KOKKOS_FORCEINLINE_FUNCTION
    size_type arraySize( const size_type s ) const
    {
        return _data.arraySize( s );
    }

    /*!
      \brief Register valid grids in sparse map according to input particle
      positions.
      \param positions Input particle positions.
      \param particle_num Number of valid particles inside positions
      \param p2g_radius The half range of grids that will be influenced by each
      particle, depending on the interpolation kernel
    */
    template <class PositionSliceType>
    void registerSparseGrid( PositionSliceType& positions, int particle_num,
                             const int p2g_radius = 1 )
    {
        _layout.template registerSparseMap<execution_space, PositionSliceType>(
            positions, particle_num, p2g_radius );
        this->resize( _layout.sparseMap().sizeCell() );
    }

    // ------------------------------------------------------------------------
    /*!
      \brief (Device) Access tile SoA from tile-related information
      \param tile_i, tile_j, tile_k Tile index in each dimension
    */
    KOKKOS_FORCEINLINE_FUNCTION
    soa_type& accessTile( const int tile_i, const int tile_j,
                          const int tile_k ) const
    {
        auto tile_id = _layout.queryTileFromTileId( tile_i, tile_j, tile_k );
        return _data.access( tile_id );
    }

    /*!
      \brief (Device) Access tile SoA from tile-related information
      \param tile_id 1D Tile ID (registered ID in sparse map, which is also the
      real allocation Id inside the AoS)
    */
    template <typename Value>
    KOKKOS_FORCEINLINE_FUNCTION soa_type&
    accessTile( const Value tile_id ) const
    {
        return _data.access( tile_id );
    }

    /*!
      \brief (Device) Access tile SoA from cell-related information
      \param cell_i, cell_j, cell_k Cell index in each dimension
    */
    KOKKOS_FORCEINLINE_FUNCTION
    soa_type& accessTileFromCell( const int cell_i, const int cell_j,
                                  const int cell_k ) const
    {
        auto tile_id = _layout.queryTile( cell_i, cell_j, cell_k );
        return _data.access( tile_id );
    }

    // ------------------------------------------------------------------------
    /*!
      \brief Access AoSoA tuple from tile key and local cell id
      \param tile_key Tile Key inside sparse map
      \param cell_local_id local Cell ID inside the tile
    */
    template <typename Key>
    KOKKOS_FORCEINLINE_FUNCTION tuple_type
    getTuple( const Key tile_key, const int cell_local_id ) const
    {
        auto tile_id = _layout.queryTileFromTileKey( tile_key );
        return _data.getTuple( ( tile_id << cell_bits_per_tile ) |
                               ( cell_local_id & cell_mask_per_tile ) );
    }

    // ------------------------------------------------------------------------
    /*!
      \brief Access element from cell IJK, access corresponding element's
      channels with extra indices
      \param cell_ijk Cell ID in each dimension
      \param ids Ids to access channels inside a data member/element
    */
    template <std::size_t M, typename... Indices>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( const Kokkos::Array<int, 3> cell_ijk, Indices&&... ids ) const
    {
        auto& soa = accessTileFromCell( cell_ijk[0], cell_ijk[1], cell_ijk[2] );
        auto array_index =
            _layout.cellLocalId( cell_ijk[0], cell_ijk[1], cell_ijk[2] );
        return Cabana::get<M>( soa, array_index, ids... );
    }

    /*!
      \brief Access element from cell IJK
      \param cell_ijk Cell ID in each dimension
    */
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( const Kokkos::Array<int, 3> cell_ijk ) const
    {
        auto& soa = accessTileFromCell( cell_ijk[0], cell_ijk[1], cell_ijk[2] );
        auto array_index =
            _layout.cellLocalId( cell_ijk[0], cell_ijk[1], cell_ijk[2] );
        return Cabana::get<M>( soa, array_index );
    }

    /*!
      \brief Access element in a hierarchical manner, from tile IJK and then
      local cell IJK, access corresponding element's channels with extra indices
      \param tile_ijk Tile ID in each dimension
      \param local_cell_ijk Local Cell ID in each dimension
      \param ids Ids to access channels inside a data member/element
    */
    template <std::size_t M, typename... Indices>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( const Kokkos::Array<int, 3> tile_ijk,
             const Kokkos::Array<int, 3> local_cell_ijk,
             Indices&&... ids ) const
    {
        auto& soa = accessTile( tile_ijk[0], tile_ijk[1], tile_ijk[2] );
        auto array_index = _layout.cellLocalId(
            local_cell_ijk[0], local_cell_ijk[1], local_cell_ijk[2] );
        return Cabana::get<M>( soa, array_index, ids... );
    }

    /*!
      \brief Access element in a hierarchical manner, from tile IJK and then
      local cell IJK
      \param tile_ijk Tile ID in each dimension
      \param local_cell_ijk Local Cell ID in each dimension
    */
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( const Kokkos::Array<int, 3> tile_ijk,
             const Kokkos::Array<int, 3> local_cell_ijk ) const
    {
        auto& soa = accessTile( tile_ijk[0], tile_ijk[1], tile_ijk[2] );
        auto array_index = _layout.cellLocalId(
            local_cell_ijk[0], local_cell_ijk[1], local_cell_ijk[2] );
        return Cabana::get<M>( soa, array_index );
    }

    /*!
      \brief Access element in a hierarchical manner, from 1D tile ID and then
      local cell IJK, access corresponding element's channels with extra indices
      \param tile_id the 1D Tile ID
      \param local_cell_ijk Local Cell ID in each dimension
      \param ids Ids to access channels inside a data member/element
    */
    template <std::size_t M, typename... Indices>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( const int tile_id, const Kokkos::Array<int, 3> local_cell_ijk,
             Indices&&... ids ) const
    {
        auto& soa = _data.access( tile_id );
        auto array_index = _layout.cellLocalId(
            local_cell_ijk[0], local_cell_ijk[1], local_cell_ijk[2] );
        return Cabana::get<M>( soa, array_index, ids... );
    }

    /*!
      \brief Access element in a hierarchical manner, from 1D tile ID and then
      local cell IJK
      \param tile_id the 1D Tile ID
      \param local_cell_ijk Local Cell ID in each dimension
    */
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( const int tile_id,
             const Kokkos::Array<int, 3> local_cell_ijk ) const
    {
        auto& soa = _data.access( tile_id );
        auto array_index = _layout.cellLocalId(
            local_cell_ijk[0], local_cell_ijk[1], local_cell_ijk[2] );
        return Cabana::get<M>( soa, array_index );
    }

    /*!
      \brief Access element in a hierarchical manner, from 1D tile ID and then
      1D local cell ID, access corresponding element's channels with extra
      indices
      \param tile_id the 1D Tile ID
      \param cell_id the 1D Local Cell ID
      \param ids Ids to access channels inside a data member/element
    */
    template <std::size_t M, typename... Indices>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( const int tile_id, const int cell_id, Indices&&... ids ) const
    {
        auto& soa = _data.access( tile_id );
        return Cabana::get<M>( soa, cell_id, ids... );
    }

    /*!
      \brief Access element in a hierarchical manner, from 1D tile ID and then
      1D local cell ID
      \param tile_id the 1D Tile ID
      \param cell_id the 1D Local Cell ID
    */
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( const int tile_id, const int cell_id ) const
    {
        auto& soa = _data.access( tile_id );
        return Cabana::get<M>( soa, cell_id );
    }

  private:
    //! Sparse array layout
    array_layout _layout;
    //! AoSoA sparse grid data
    aosoa_type _data;

}; // end class SparseArray

//---------------------------------------------------------------------------//
// Scatic type checker.
//---------------------------------------------------------------------------//
//! Sparse array static type checker.
template <class>
struct is_sparse_array : public std::false_type
{
};

//! Sparse array static type checker.
template <class DataTypes, class MemorySpace, class EntityType, class MeshType,
          class SparseMapType>
struct is_sparse_array<
    SparseArray<DataTypes, MemorySpace, EntityType, MeshType, SparseMapType>>
    : public std::true_type
{
};

//! Sparse array static type checker.
template <class DataTypes, class MemorySpace, class EntityType, class MeshType,
          class SparseMapType>
struct is_sparse_array<const SparseArray<DataTypes, MemorySpace, EntityType,
                                         MeshType, SparseMapType>>
    : public std::true_type
{
};

//---------------------------------------------------------------------------//
// Array creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create sparse array based on the sparse array layout
  \param label The sparse array data description.
  \param layout The sparse array layout.
  \return SparseArray
*/
template <class MemorySpace, class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
auto createSparseArray(
    const std::string label,
    SparseArrayLayout<DataTypes, EntityType, MeshType, SparseMapType>& layout )
{
    return std::make_shared<SparseArray<DataTypes, MemorySpace, EntityType,
                                        MeshType, SparseMapType>>( label,
                                                                   layout );
}

} // namespace Experimental
} // namespace Grid
} // namespace Cabana

namespace Cajita
{
namespace Experimental
{
//! \cond Deprecated
template <class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
using SparseArrayLayout CAJITA_DEPRECATED =
    Cabana::Grid::Experimental::SparseArrayLayout<DataTypes, EntityType,
                                                  MeshType, SparseMapType>;

template <class... Args>
CAJITA_DEPRECATED auto createSparseArrayLayout( Args&&... args )
{
    return Cabana::Grid::Experimental::createSparseArrayLayout(
        std::forward<Args>( args )... );
}

template <class T>
using is_sparse_array_layout CAJITA_DEPRECATED =
    Cabana::Grid::Experimental::is_sparse_array_layout<T>;

template <class DataTypes, class MemorySpace, class EntityType, class MeshType,
          class SparseMapType>
using SparseArray CAJITA_DEPRECATED =
    Cabana::Grid::Experimental::SparseArray<DataTypes, MemorySpace, EntityType,
                                            MeshType, SparseMapType>;

template <class T>
using is_sparse_array CAJITA_DEPRECATED =
    Cabana::Grid::Experimental::is_sparse_array<T>;

template <class MemorySpace, class... Args>
CAJITA_DEPRECATED auto createSparseArray( Args&&... args )
{
    return Cabana::Grid::Experimental::createSparseArray<MemorySpace>(
        std::forward<Args>( args )... );
}
//! \endcond
} // namespace Experimental
} // end namespace Cajita

#endif // CABANA_GRID_SPARSE_ARRAY_HPP
