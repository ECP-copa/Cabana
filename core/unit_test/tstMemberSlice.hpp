#include <Cabana_MemberSlice.hpp>
#include <Cabana_AoSoA.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template<class aosoa_type>
void checkDataMembers(
    aosoa_type aosoa,
    const float fval, const double dval, const int ival,
    const int dim_1, const int dim_2,
    const int dim_3, const int dim_4 )
{
    auto view_0 = aosoa.template view<0>();
    auto view_1 = aosoa.template view<1>();
    auto view_2 = aosoa.template view<2>();
    auto view_3 = aosoa.template view<3>();
    auto view_4 = aosoa.template view<4>();

    for ( auto idx = 0; idx != aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    BOOST_CHECK( view_0( idx, i, j, k ) ==
                                         fval * (i+j+k) );

        // Member 1.
        BOOST_CHECK( view_1( idx ) == ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        BOOST_CHECK( view_2( idx, i, j, k, l ) ==
                                             fval * (i+j+k+l) );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            BOOST_CHECK( view_3( idx, i ) == dval * i );

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                BOOST_CHECK( view_4( idx, i, j ) == dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( slice_serial_api_test )
{
    // Manually set the inner array size. Select a layout by default.
    using inner_array_layout = Cabana::InnerArrayLayout<16>;

    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;
    const int dim_4 = 3;

    // Declare data types.
    using DataTypes =
        Cabana::MemberDataTypes<float[dim_1][dim_2][dim_3],
                                int,
                                float[dim_1][dim_2][dim_3][dim_4],
                                double[dim_1],
                                double[dim_1][dim_2]
                                >;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,inner_array_layout,TEST_MEMSPACE>;

    // Make sure that it is actually an AoSoA.
    BOOST_CHECK( Cabana::is_aosoa<AoSoA_t>::value );

    // Create an AoSoA.
    int num_data = 35;
    AoSoA_t aosoa( num_data );

    // Create some slices.
    auto slice_0 = aosoa.view<0>();
    auto slice_1 = aosoa.view<1>();
    auto slice_2 = aosoa.view<2>();
    auto slice_3 = aosoa.view<3>();
    auto slice_4 = aosoa.view<4>();

    // Check that they are slices.
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_0)>::value );
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_1)>::value );
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_2)>::value );
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_3)>::value );
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_4)>::value );

    // Check sizes.
    BOOST_CHECK( slice_0.size() == int(35) );
    BOOST_CHECK( slice_0.numSoA() == int(3) );

    BOOST_CHECK( slice_0.arraySize(0) == int(16) );
    BOOST_CHECK( slice_0.arraySize(1) == int(16) );
    BOOST_CHECK( slice_0.arraySize(3) == int(3) );

    BOOST_CHECK( slice_0.rank() == int(3) );
    int e00 = slice_0.extent(0);
    BOOST_CHECK( e00 == dim_1 );
    int e01 = slice_0.extent(1);
    BOOST_CHECK( e01 == dim_2 );
    int e02 = slice_0.extent(2);
    BOOST_CHECK( e02 == dim_3 );
    int e03 = slice_0.extent(3);
    BOOST_CHECK( e03 == int(0) );

    BOOST_CHECK( slice_1.rank() == int(0) );
    int e10 = slice_1.extent(0);
    BOOST_CHECK( e10 == int(0) );
    int e11 = slice_1.extent(1);
    BOOST_CHECK( e11 == int(0) );
    int e12 = slice_1.extent(2);
    BOOST_CHECK( e12 == int(0) );
    int e13 = slice_1.extent(3);
    BOOST_CHECK( e13 == int(0) );

    BOOST_CHECK( slice_2.rank() == int(4) );
    int e20 = slice_2.extent(0);
    BOOST_CHECK( e20 == dim_1 );
    int e21 = slice_2.extent(1);
    BOOST_CHECK( e21 == dim_2 );
    int e22 = slice_2.extent(2);
    BOOST_CHECK( e22 == dim_3 );
    int e23 = slice_2.extent(3);
    BOOST_CHECK( e23 == dim_4 );

    BOOST_CHECK( slice_3.rank() == int(1) );
    int e30 = slice_3.extent(0);
    BOOST_CHECK( e30 == dim_1 );
    int e31 = slice_3.extent(1);
    BOOST_CHECK( e31 == int(0) );
    int e32 = slice_3.extent(2);
    BOOST_CHECK( e32 == int(0) );
    int e33 = slice_3.extent(3);
    BOOST_CHECK( e33 == int(0) );

    BOOST_CHECK( slice_4.rank() == int(2) );
    int e40 = slice_4.extent(0);
    BOOST_CHECK( e40 == dim_1 );
    int e41 = slice_4.extent(1);
    BOOST_CHECK( e41 == dim_2 );
    int e42 = slice_4.extent(2);
    BOOST_CHECK( e42 == int(0) );
    int e43 = slice_4.extent(3);
    BOOST_CHECK( e43 == int(0) );

    // Initialize data with the rank accessors.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( auto idx = 0; idx != aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    slice_0( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        slice_1( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        slice_2( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            slice_3( idx, i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                slice_4( idx, i, j ) = dval * (i+j);
    }

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}
