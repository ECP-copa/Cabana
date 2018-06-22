#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberSlice.hpp>
#include <Cabana_Sort.hpp>

#include <boost/test/unit_test.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( sort_by_key_test )
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
    using KeyViewType = Kokkos::View<int*,TEST_MEMSPACE>;
    KeyViewType keys( "keys", num_data );

    // Create the AoSoA data and keys. Create the data in reverse order so we
    // can see that it is sorted.
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
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
    Cabana::sortByKey( aosoa, keys );

    // Check the result of the sort.
    for ( int p = 0; p < aosoa.size(); ++p )
    {
       for ( int i = 0; i < dim_1; ++i )
           BOOST_CHECK( v0( p, i ) == p + i );

       BOOST_CHECK( v1( p ) == p );

       for ( int i = 0; i < dim_1; ++i )
           for ( int j = 0; j < dim_2; ++j )
               BOOST_CHECK( v2( p, i, j ) == p + i + j );
    }
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( bin_by_key_test )
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
    using KeyViewType = Kokkos::View<int*,TEST_MEMSPACE>;
    KeyViewType keys( "keys", num_data );

    // Create the AoSoA data and keys. Create the data in reverse order so we
    // can see that it is sorted.
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
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
    Cabana::binByKey( aosoa, keys, num_data );

    // Check the result of the sort.
    for ( int p = 0; p < aosoa.size(); ++p )
    {
       for ( int i = 0; i < dim_1; ++i )
           BOOST_CHECK( v0( p, i ) == p + i );

       BOOST_CHECK( v1( p ) == p );

       for ( int i = 0; i < dim_1; ++i )
           for ( int j = 0; j < dim_2; ++j )
               BOOST_CHECK( v2( p, i, j ) == p + i + j );
    }
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( sort_by_member_test )
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
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
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
    Cabana::sortByMember<1>( aosoa );

    // Check the result of the sort.
    for ( int p = 0; p < aosoa.size(); ++p )
    {
       for ( int i = 0; i < dim_1; ++i )
           BOOST_CHECK( v0( p, i ) == p + i );

       BOOST_CHECK( v1( p ) == p );

       for ( int i = 0; i < dim_1; ++i )
           for ( int j = 0; j < dim_2; ++j )
               BOOST_CHECK( v2( p, i, j ) == p + i + j );
    }
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( bin_by_member_test )
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
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
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
    Cabana::binByMember<1>( aosoa, num_data );

    // Check the result of the sort.
    for ( int p = 0; p < aosoa.size(); ++p )
    {
       for ( int i = 0; i < dim_1; ++i )
           BOOST_CHECK( v0( p, i ) == p + i );

       BOOST_CHECK( v1( p ) == p );

       for ( int i = 0; i < dim_1; ++i )
           for ( int j = 0; j < dim_2; ++j )
               BOOST_CHECK( v2( p, i, j ) == p + i + j );
    }
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( grid_bin_3d_test )
{
    // Make an AoSoA with positions and ijk cell ids.
    enum MyFields { Position = 0, CellId = 1 };
    using DataTypes = Cabana::MemberDataTypes<double[3],int[3]>;
    int num_p = 1000;
    Cabana::AoSoA<DataTypes,TEST_MEMSPACE> aosoa( num_p );

    // Set the problem so each particle lives in the center of a cell on a
    // regular grid of cell size 1 and total size 10x10x10. We are making them
    // in the reverse order we expect the sort to happen. The sort binary
    // operator should order by i first and k last.
    int nx = 10;
    double dx = 1.0;
    double x_min = 0.0;
    double x_max = x_min + nx * dx;
    auto pos = Cabana::slice<Position>( aosoa );
    auto cell_id = Cabana::slice<CellId>( aosoa );
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
    Cabana::binByRegularGrid3d<Position>( aosoa,
                                          dx, dx, dx,
                                          x_min, x_min, x_min,
                                          x_max, x_max, x_max );

    // Checking the binning. The order should be reversed with the i index
    // moving the slowest.
    particle_id = 0;
    for ( int i = 0; i < nx; ++i )
    {
        for ( int j = 0; j < nx; ++j )
        {
            for ( int k = 0; k < nx; ++k, ++particle_id )
            {
                BOOST_CHECK( cell_id( particle_id, 0 ) == i );
                BOOST_CHECK( cell_id( particle_id, 1 ) == j );
                BOOST_CHECK( cell_id( particle_id, 2 ) == k );
            }
        }
    }
}

//---------------------------------------------------------------------------//
