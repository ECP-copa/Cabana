#include <Cajita_MpiTraits.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <Cabana_AoSoA.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

#include <mpi.h>

namespace Cajita
{

namespace Experimental
{
//---------------------------------------------------------------------------//
// Indexing type tags
//---------------------------------------------------------------------------//
// H - Hierachical
// T_C: indexing manner: tile ijk - cell ijk
struct Index_H_T_C
{
};

// C: indexing matter: cell ijk
struct Index_H_C
{
};

// ID: real array id
struct Index_H_ID_C
{
};

struct Index_ID
{
};

//---------------------------------------------------------------------------//
template <class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
class SparseArrayLayout
{
  public:
    using mesh_type = MeshType;
    using entity_type = EntityType;
    using member_types = DataTypes;

    static_assert( isSparseMesh<MeshType>::value,
                   "[SparesArrayLayout] Support only SparseMesh" );

    using sparse_map_type = SparseMapType;
    using value_type = typename sparse_map_type::value_type;
    using key_type = typename sparse_map_type::key_type;
    static constexpr unsigned long long cell_bits_per_tile_dim =
        sparse_map_type::cell_bits_per_tile_dim;
    static constexpr unsigned long long cell_num_per_tile_dim =
        sparse_map_type::cell_num_per_tile_dim;
    static constexpr std::size_t num_space_dim = sparse_map_type::rank;

    using memory_space = typename sparse_map_type::memory_space;

    // [0] shared_owned_num [1] shared_ghost_num
    // using indices_view = Kokkos::View<value_type* [2], memory_space>;
    // [0] shared_owned_num [1] shared_ghost_num
    // using counting_view = Kokkos::View<int[2], memory_space,
    //    Kokkos::MemoryTraits<Kokkos::Atomic>>;

    SparseArrayLayout( const std::shared_ptr<LocalGrid<MeshType>>& local_grid,
                       SparseMapType& sparse_map, const float bc_factor )
        : _bc_factor( bc_factor )
        , _local_grid_ptr( local_grid )
        , _map( std::forward<sparse_map_type>( sparse_map ) )
    {
    }

    const std::shared_ptr<LocalGrid<MeshType>> localGrid() const
    {
        return _local_grid_ptr;
    }

    SparseMapType& sparseMap() { return _map; }

    // size
    inline uint64_t arraySizeCell() const
    {
        entity_type t;
        return arraySizeCellImpl( t );
    }
    inline uint64_t arraySizeTile() const
    {
        entity_type t;
        return arraySizeTileImpl( t );
    }

    inline void clear() { _map.clear(); }

    template <class ExecSpace, class PositionSliceType>
    void register_sparse_map( PositionSliceType& poses )
    {
        using scalar_type = typename mesh_type::scalar_type;

        auto& sparse_mesh = _local_grid_ptr->globalGrid().globalMesh();
        Kokkos::Array<scalar_type, 3> dx_inv = {
            1.0 / sparse_mesh.cellSize( 0 ), 1.0 / sparse_mesh.cellSize( 1 ),
            1.0 / sparse_mesh.cellSize( 2 ) };
        auto& map = _map;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecSpace>( 0, poses.size() ),
            KOKKOS_LAMBDA( const int pid ) {
                scalar_type pos[3] = { poses( pid, 0 ), poses( pid, 1 ),
                                       poses( pid, 2 ) };
                int grid_base[3] = {
                    static_cast<int>( std::lround( pos[0] * dx_inv[0] ) - 1 ),
                    static_cast<int>( std::lround( pos[1] * dx_inv[1] ) - 1 ),
                    static_cast<int>( std::lround( pos[2] * dx_inv[2] ) - 1 ) };
                for ( int i = 0; i <= 2; ++i )
                    for ( int j = 0; j <= 2; ++j )
                        for ( int k = 0; k <= 2; ++k )
                        {
                            int cell_id[3] = { grid_base[0] + i,
                                               grid_base[1] + j,
                                               grid_base[2] + k };
                            map.insertCell( cell_id[0], cell_id[1],
                                            cell_id[2] );
                        }
            } );
    }

    // query index
    KOKKOS_FORCEINLINE_FUNCTION
    value_type queryCell( const int i, const int j, const int k ) const
    {
        return _map.queryCell( i, j, k );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    value_type queryTile( const int i, const int j, const int k ) const
    {
        return _map.queryTile( i, j, k );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    value_type queryTile_FT( const int i, const int j, const int k ) const
    {
        return _map.queryTile_FT( i, j, k );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    value_type queryTileFromTileKey( const key_type tile_key ) const
    {
        return _map.queryTileFromTileKey( tile_key );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    value_type cell_local_id( const int cell_i, const int cell_j,
                              const int cell_k ) const
    {
        return _map.cell_local_id( cell_i, cell_j, cell_k );
    }

    // Default unit: cell
    inline uint64_t arraySize() const { return arraySizeCell(); }

  private:
    //-------------------------------------------------------------------//
    inline uint64_t arraySizeCellImpl( Cell ) const
    {
        return static_cast<uint64_t>( _map.reservedCellSize() );
    }
    inline uint64_t arraySizeCellImpl( Node ) const
    {
        return static_cast<uint64_t>( _map.reservedCellSize() * _bc_factor );
    }
    template <int dim>
    inline uint64_t arraySizeCellImpl( Face<dim> ) const
    {
        return static_cast<uint64_t>( _map.reservedCellSize() * _bc_factor );
    }
    template <int dim>
    inline uint64_t arraySizeCellImpl( Edge<dim> ) const
    {
        return static_cast<uint64_t>( _map.reservedCellSize() * _bc_factor );
    }

    //-------------------------------------------------------------------//
    inline uint64_t arraySizeTileImpl( Cell ) const
    {
        return static_cast<uint64_t>( _map.reservedTileSize() );
    }
    inline uint64_t arraySizeTileImpl( Node ) const
    {
        return static_cast<uint64_t>( _map.reservedTileSize() * _bc_factor );
    }
    template <int dim>
    inline uint64_t arraySizeTileImpl( Face<dim> ) const
    {
        return static_cast<uint64_t>( _map.reservedTileSize() * _bc_factor );
    }
    template <int dim>
    inline uint64_t arraySizeTileImpl( Edge<dim> ) const
    {
        return static_cast<uint64_t>( _map.reservedTileSize() * _bc_factor );
    }

  private:
    float _bc_factor;
    std::shared_ptr<LocalGrid<MeshType>> _local_grid_ptr;
    sparse_map_type _map;
}; // end class SparseArrayLayout

//! Array static type checker.
template <class>
struct is_sparse_array_layout : public std::false_type
{
};

//! Array static type checker.
template <class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
struct is_sparse_array_layout<
    SparseArrayLayout<DataTypes, EntityType, MeshType, SparseMapType>>
    : public std::true_type
{
};

//! Array static type checker.
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
template <class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
SparseArrayLayout<DataTypes, EntityType, MeshType, SparseMapType>
createSparseArrayLayout( const std::shared_ptr<LocalGrid<MeshType>>& local_grid,
                         SparseMapType& sparse_map, EntityType,
                         const float bc_factor = 1.01f )
{
    return SparseArrayLayout<DataTypes, EntityType, MeshType, SparseMapType>(
        local_grid, sparse_map, bc_factor );
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

template <class DataTypes, class DeviceType, class EntityType, class MeshType,
          class SparseMapType>
class SparseArray
{
  public:
    using sparse_array_type =
        SparseArray<DataTypes, DeviceType, EntityType, MeshType, SparseMapType>;
    // Device Types
    using device_type = DeviceType;
    using memory_space = typename device_type::memory_space;
    using execution_space = typename device_type::execution_space;
    using size_type = typename memory_space::size_type;
    // Other types
    using entity_type = EntityType;
    using mesh_type = MeshType;
    using sparse_map_type = SparseMapType;
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;
    static constexpr unsigned long long cell_bits_per_tile =
        sparse_map_type::cell_bits_per_tile;
    static constexpr unsigned long long cell_mask_per_tile =
        sparse_map_type::cell_mask_per_tile;

    // AoSoA related types
    using member_types = DataTypes;
    static constexpr int vector_length = sparse_map_type::cell_num_per_tile;
    using aosoa_type = Cabana::AoSoA<member_types, memory_space, vector_length>;
    using soa_type = Cabana::SoA<member_types, vector_length>;
    using tuple_type = Cabana::Tuple<member_types>;

    using array_layout = SparseArrayLayout<member_types, entity_type, mesh_type,
                                           sparse_map_type>;
    // Constructor
    SparseArray( const std::string label, array_layout& layout )
        : _layout( std::forward<array_layout>( layout ) )
        , _data( label )
    {
        reserve( _layout.arraySizeCell() );
    }

    // ------------------------------------------------------------------------
    // AoSoA interfaces
    aosoa_type& aosoa() { return _data; }
    std::string label() const { return _data.label(); }
    array_layout& layout() { return _layout; }

    inline void resize( const size_type n ) { _data.resize( n ); }
    inline void reserve( const size_type n ) { _data.reserve( n ); }
    inline void shrinkToFit() { _data.shrinkToFit(); }
    inline void clear()
    {
        resize( 0 );
        _layout.clear();
    };

    template <class PositionSliceType>
    void register_sparse_grid( PositionSliceType& poses )
    {
        _layout
            .template register_sparse_map<execution_space, PositionSliceType>(
                poses );
        this->resize( _layout.sparseMap().sizeCell() );
    }

    KOKKOS_FUNCTION
    size_type capacity() { return _data.capacity(); }
    KOKKOS_FUNCTION
    size_type size() const { return _data.size(); }
    KOKKOS_FUNCTION
    bool empty() const { return ( size() == 0 ); }

    KOKKOS_FORCEINLINE_FUNCTION
    size_type numSoA() const { return _data.numSoA(); }

    KOKKOS_FORCEINLINE_FUNCTION
    size_type arraySize( const size_type s ) const
    {
        return _data.arraySize( s );
    }

    // ------------------------------------------------------------------------
    // data access
    // access soa
    KOKKOS_FORCEINLINE_FUNCTION
    soa_type& access_tile_FT( const int tile_i, const int tile_j,
                              const int tile_k ) const
    {
        auto tile_id = _layout.queryTile_FT( tile_i, tile_j, tile_k );
        return _data.access( tile_id );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    soa_type& access_tile( const int cell_i, const int cell_j,
                           const int cell_k ) const
    {
        auto tile_id = _layout.queryTile( cell_i, cell_j, cell_k );
        return _data.access( tile_id );
    }

    template <typename Value>
    KOKKOS_FORCEINLINE_FUNCTION soa_type&
    access_tile( const Value tile_id ) const
    {
        return _data.access( tile_id );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    tuple_type getTuple( const int cell_i, const int cell_j,
                         const int cell_k ) const
    {
        auto cell_id = _layout.queryCell( cell_i, cell_j, cell_k );
        return _data.getTuple( cell_id );
    }

    template <typename Key>
    KOKKOS_FORCEINLINE_FUNCTION tuple_type
    getTuple( const Key tile_key, const int cell_local_id ) const
    {
        auto tile_id = _layout.queryTileFromTileKey( tile_key );
        return _data.getTuple( ( tile_id << cell_bits_per_tile ) |
                               ( cell_local_id & cell_mask_per_tile ) );
    }

    template <std::size_t M, typename... Indices>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( Index_H_C, const int cell_i, const int cell_j, const int cell_k,
             Indices&&... ids ) const
    {
        auto& soa = access_tile( cell_i, cell_j, cell_k );
        auto array_index = _layout.cell_local_id( cell_i, cell_j, cell_k );
        return Cabana::get<M>( soa, array_index, ids... );
    }

    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( Index_H_C, const int cell_i, const int cell_j,
             const int cell_k ) const
    {
        auto& soa = access_tile( cell_i, cell_j, cell_k );
        auto array_index = _layout.cell_local_id( cell_i, cell_j, cell_k );
        return Cabana::get<M>( soa, array_index );
    }

    template <std::size_t M, typename... Indices>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( Index_H_T_C, const int tile_i, const int tile_j, const int tile_k,
             const int local_cell_i, const int local_cell_j,
             const int local_cell_k, Indices&&... ids ) const
    {
        auto& soa = access_tile_FT( tile_i, tile_j, tile_k );
        auto array_index =
            _layout.cell_local_id( local_cell_i, local_cell_j, local_cell_k );
        return Cabana::get<M>( soa, array_index, ids... );
    }

    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( Index_H_T_C, const int tile_i, const int tile_j, const int tile_k,
             const int local_cell_i, const int local_cell_j,
             const int local_cell_k ) const
    {
        auto& soa = access_tile_FT( tile_i, tile_j, tile_k );
        auto array_index =
            _layout.cell_local_id( local_cell_i, local_cell_j, local_cell_k );
        return Cabana::get<M>( soa, array_index );
    }

    // template <std::size_t M>
    // KOKKOS_INLINE_FUNCTION typename soa_type::template
    // member_reference_type<M> get( Index_H_T_C, const int tile_i, const int
    // tile_j, const int tile_k,
    //      const int local_cell_i, const int local_cell_j,
    //      const int local_cell_k ) const
    // {
    //     auto& soa = access_tile_FT( tile_i, tile_j, tile_k );
    //     auto array_index = _layout.sparseMap().cell_local_id(
    //         local_cell_i, local_cell_j, local_cell_k );
    //     return Cabana::get<M>( soa, array_index );
    // }

    template <std::size_t M, typename... Indices>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( Index_H_ID_C, const int array_id, const int local_cell_i,
             const int local_cell_j, const int local_cell_k,
             Indices&&... ids ) const
    {
        auto& soa = _data.access( array_id );
        auto cell_id =
            _layout.cell_local_id( local_cell_i, local_cell_j, local_cell_k );
        return Cabana::get<M>( soa, cell_id, ids... );
    }

    // template <std::size_t M>
    // KOKKOS_FORCEINLINE_FUNCTION
    //     typename soa_type::template member_reference_type<M>
    //     get( Index_H_ID_C, const int array_id, const int local_cell_i,
    //          const int local_cell_j, const int local_cell_k ) const
    // {
    //     auto& soa = _data.access( array_id );
    //     auto cell_id =
    //         _layout.cell_local_id( local_cell_i, local_cell_j, local_cell_k
    //         );
    //     return Cabana::get<M>( soa, cell_id );
    // }

    template <std::size_t M, typename... Indices>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( Index_ID, const int array_id, const int cell_id,
             Indices&&... ids ) const
    {
        auto& soa = _data.access( array_id );
        return Cabana::get<M>( soa, cell_id, ids... );
    }

    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
        typename soa_type::template member_reference_type<M>
        get( Index_ID, const int array_id, const int cell_id ) const
    {
        auto& soa = _data.access( array_id );
        return Cabana::get<M>( soa, cell_id );
    }
    // should add more data access interfaces
    // such as slices, or index based data accessing

  private:
    array_layout _layout;
    aosoa_type _data;

}; // end class SparseArray

//---------------------------------------------------------------------------//
// Scatic type checker.
//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_sparse_array : public std::false_type
{
};

template <class DataTypes, class DeviceType, class EntityType, class MeshType,
          class SparseMapType>
struct is_sparse_array<
    SparseArray<DataTypes, DeviceType, EntityType, MeshType, SparseMapType>>
    : public std::true_type
{
};

template <class DataTypes, class DeviceType, class EntityType, class MeshType,
          class SparseMapType>
struct is_sparse_array<const SparseArray<DataTypes, DeviceType, EntityType,
                                         MeshType, SparseMapType>>
    : public std::true_type
{
};

//---------------------------------------------------------------------------//
// Array creation.
//---------------------------------------------------------------------------//
template <class DeviceType, class DataTypes, class EntityType, class MeshType,
          class SparseMapType>
SparseArray<DataTypes, DeviceType, EntityType, MeshType, SparseMapType>
createSparseArray(
    const std::string label,
    SparseArrayLayout<DataTypes, EntityType, MeshType, SparseMapType>& layout )
{
    return SparseArray<DataTypes, DeviceType, EntityType, MeshType,
                       SparseMapType>( label, layout );
}

namespace SparseArrayOp
{

}; // end namespace SparseArrayOp

} // namespace Experimental
} // end namespace Cajita

#endif