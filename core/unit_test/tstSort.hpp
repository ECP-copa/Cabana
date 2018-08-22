#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberSlice.hpp>
#include <Cabana_Sort.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
void testSortByKey()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberDataTypes<float[dim_1],
                                              int,
                                              double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create a Kokkos view for the keys.
    using KeyViewType =
        Kokkos::View<int*,typename AoSoA_t::memory_space::kokkos_memory_space>;
    KeyViewType keys( "keys", num_data );

    // Create the AoSoA data and keys. Create the data in reverse order so we
    // can see that it is sorted.
    auto v0 = aosoa.view( Cabana::MemberTag<0>() );
    auto v1 = aosoa.view( Cabana::MemberTag<1>() );
    auto v2 = aosoa.view( Cabana::MemberTag<2>() );
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;

        keys( p ) = reverse_index;
    }

    // Sort the aosoa by keys.
    auto permute_vector = Cabana::sortByKey( aosoa, keys, false );

    // Check the result of the sort.
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), p + i );

        EXPECT_EQ( v1( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), p + i + j );

        EXPECT_EQ( permute_vector(p), (unsigned) reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testBinByKey()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberDataTypes<float[dim_1],
                                              int,
                                              double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    using size_type =
        typename AoSoA_t::memory_space::kokkos_memory_space::size_type;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create a Kokkos view for the keys.
    using KeyViewType =
        Kokkos::View<int*,typename AoSoA_t::memory_space::kokkos_memory_space>;
    KeyViewType keys( "keys", num_data );

    // Create the AoSoA data and keys. Create the data in reverse order so we
    // can see that it is sorted.
    auto v0 = aosoa.view( Cabana::MemberTag<0>() );
    auto v1 = aosoa.view( Cabana::MemberTag<1>() );
    auto v2 = aosoa.view( Cabana::MemberTag<2>() );
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;

        keys( p ) = reverse_index;
    }

    // Bin the aosoa by keys. Use one bin per data point to effectively make
    // this a sort.
    auto bin_data = Cabana::binByKey( aosoa, keys, num_data-1, false );

    // Check the result of the sort.
    EXPECT_EQ( bin_data.numBin(), num_data );
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), p + i );

        EXPECT_EQ( v1( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), p + i + j );

        EXPECT_EQ( bin_data.binSize(p), 1 );
        EXPECT_EQ( bin_data.binOffset(p), size_type(p) );
        EXPECT_EQ( bin_data.permutation(p), reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testSortByMember()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberDataTypes<float[dim_1],
                                              int,
                                              double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = aosoa.view( Cabana::MemberTag<0>() );
    auto v1 = aosoa.view( Cabana::MemberTag<1>() );
    auto v2 = aosoa.view( Cabana::MemberTag<2>() );
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;
    }

    // Sort the aosoa by the 1D member.
    auto permute_vector =
        Cabana::sortByMember( aosoa, Cabana::MemberTag<1>(), false );

    // Check the result of the sort.
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), p + i );

        EXPECT_EQ( v1( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), p + i + j );

        EXPECT_EQ( permute_vector(p), (unsigned) reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testSortByMemberDataOnly()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberDataTypes<float[dim_1],
                                              int,
                                              double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = aosoa.view( Cabana::MemberTag<0>() );
    auto v1 = aosoa.view( Cabana::MemberTag<1>() );
    auto v2 = aosoa.view( Cabana::MemberTag<2>() );
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;
    }

    // Sort the aosoa by the 1D member.
    auto permute_vector =
        Cabana::sortByMember( aosoa, Cabana::MemberTag<1>(), true );

    // Check that the data didn't get sorted and the permutation vector is
    // corrector.
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), reverse_index + i );

        EXPECT_EQ( v1( p ), reverse_index );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), reverse_index + i + j );

        EXPECT_EQ( permute_vector(p), (unsigned) reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testBinByMember()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberDataTypes<float[dim_1],
                                              int,
                                              double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    using size_type =
        typename AoSoA_t::memory_space::kokkos_memory_space::size_type;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = aosoa.view( Cabana::MemberTag<0>() );
    auto v1 = aosoa.view( Cabana::MemberTag<1>() );
    auto v2 = aosoa.view( Cabana::MemberTag<2>() );
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;
    }

    // Bin the aosoa by the 1D member. Use one bin per data point to
    // effectively make this a sort.
    auto bin_data =
        Cabana::binByMember( aosoa, Cabana::MemberTag<1>(), num_data-1, false );

    // Check the result of the sort.
    EXPECT_EQ( bin_data.numBin(), num_data );
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), p + i );

        EXPECT_EQ( v1( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), p + i + j );

        EXPECT_EQ( bin_data.binSize(p), 1 );
        EXPECT_EQ( bin_data.binOffset(p), size_type(p) );
        EXPECT_EQ( bin_data.permutation(p), reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testBinByMemberDataOnly()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberDataTypes<float[dim_1],
                                              int,
                                              double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    using size_type =
        typename AoSoA_t::memory_space::kokkos_memory_space::size_type;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = aosoa.view( Cabana::MemberTag<0>() );
    auto v1 = aosoa.view( Cabana::MemberTag<1>() );
    auto v2 = aosoa.view( Cabana::MemberTag<2>() );
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;
    }

    // Bin the aosoa by the 1D member. Use one bin per data point to
    // effectively make this a sort. Don't actually move the particle data
    // though - just create the binning data.
    auto bin_data =
        Cabana::binByMember( aosoa, Cabana::MemberTag<1>(), num_data-1, true );

    // Check the result of the sort. Make sure nothing moved execpt the
    // binning data.
    EXPECT_EQ( bin_data.numBin(), num_data );
    for ( int p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), reverse_index + i );

        EXPECT_EQ( v1( p ), reverse_index );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), reverse_index + i + j );

        EXPECT_EQ( bin_data.binSize(p), 1 );
        EXPECT_EQ( bin_data.binOffset(p), size_type(p) );
        EXPECT_EQ( bin_data.permutation(p), reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testGridBin3d()
{
    // Make an AoSoA with positions and ijk cell ids.
    enum MyFields { Position = 0, CellId = 1 };
    using DataTypes = Cabana::MemberDataTypes<double[3],int[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    using size_type =
        typename AoSoA_t::memory_space::kokkos_memory_space::size_type;
    int num_p = 1000;
    AoSoA_t aosoa( num_p );

    // Set the problem so each particle lives in the center of a cell on a
    // regular grid of cell size 1 and total size 10x10x10. We are making them
    // in the reverse order we expect the sort to happen. The sort binary
    // operator should order by i first and k last.
    int nx = 10;
    double dx = 1.0;
    double x_min = 0.0;
    double x_max = x_min + nx * dx;
    auto pos = aosoa.view( Cabana::MemberTag<Position>() );
    auto cell_id = aosoa.view( Cabana::MemberTag<CellId>() );
    int particle_id = 0;
    for ( int k = 0; k < nx; ++k )
    {
        for ( int j = 0; j < nx; ++j )
        {
            for ( int i = 0; i < nx; ++i, ++particle_id )
            {
                cell_id( particle_id, 0 ) = i;
                cell_id( particle_id, 1 ) = j;
                cell_id( particle_id, 2 ) = k;

                pos( particle_id, 0 ) = x_min + (i + 0.5) * dx;
                pos( particle_id, 1 ) = x_min + (j + 0.5) * dx;
                pos( particle_id, 2 ) = x_min + (k + 0.5) * dx;
            }
        }
    }

    // Bin the particles in the grid.
    double grid_delta[3] = {dx,dx,dx};
    double grid_min[3] = {x_min,x_min,x_min};
    double grid_max[3] = {x_max,x_max,x_max};
    auto bin_data =
        Cabana::binByCartesianGrid3d( aosoa, Cabana::MemberTag<Position>(),
                                      false, grid_delta, grid_min, grid_max );

    // Checking the binning. The order should be reversed with the i index
    // moving the slowest.
    EXPECT_EQ( bin_data.totalBins(), nx*nx*nx );
    EXPECT_EQ( bin_data.numBin(0), nx );
    EXPECT_EQ( bin_data.numBin(1), nx );
    EXPECT_EQ( bin_data.numBin(2), nx );
    particle_id = 0;
    for ( int i = 0; i < nx; ++i )
    {
        for ( int j = 0; j < nx; ++j )
        {
            for ( int k = 0; k < nx; ++k, ++particle_id )
            {
                EXPECT_EQ( cell_id( particle_id, 0 ), i );
                EXPECT_EQ( cell_id( particle_id, 1 ), j );
                EXPECT_EQ( cell_id( particle_id, 2 ), k );
                EXPECT_EQ( bin_data.cardinalBinIndex(i,j,k), particle_id );
                EXPECT_EQ( bin_data.binSize(i,j,k), 1 );
                EXPECT_EQ( bin_data.binOffset(i,j,k),
                           size_type(particle_id) );
            }
        }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, sort_by_key_test )
{
    testSortByKey();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, bin_by_key_test )
{
    testBinByKey();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, sort_by_member_test )
{
    testSortByMember();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, sort_by_member_data_only_test )
{
    testSortByMemberDataOnly();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, bin_by_member_test )
{
    testBinByMember();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, bin_by_member_data_only_test )
{
    testBinByMemberDataOnly();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, grid_bin_3d_test )
{
    testGridBin3d();
}

//---------------------------------------------------------------------------//

} // end namespace Test
