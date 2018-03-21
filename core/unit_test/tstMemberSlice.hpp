#include <Cabana_MemberSlice.hpp>
#include <Cabana_AoSoA.hpp>

#include <boost/test/unit_test.hpp>

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template<class aosoa_type>
void checkDataMembers(
    aosoa_type aosoa,
    const float fval, const double dval, const int ival,
    const std::size_t dim_1, const std::size_t dim_2,
    const std::size_t dim_3, const std::size_t dim_4 )
{
    for ( auto idx = aosoa.begin(); idx != aosoa.end(); ++idx )
    {
        // Member 0.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    BOOST_CHECK( aosoa.template get<0>( idx, i, j, k ) ==
                                         fval * (i+j+k) );

        // Member 1.
        BOOST_CHECK( aosoa.template get<1>( idx ) == ival );

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    for ( std::size_t l = 0; l < dim_4; ++l )
                        BOOST_CHECK( aosoa.template get<2>( idx, i, j, k, l ) ==
                                             fval * (i+j+k+l) );

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            BOOST_CHECK( aosoa.template get<3>( idx, i ) == dval * i );

        // Member 4.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                BOOST_CHECK( aosoa.template get<4>( idx, i, j ) == dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( slice_serial_api_test )
{
    // Manually set the inner array size.
    using inner_array_size = Cabana::InnerArraySize<10>;

    // Data dimensions.
    const std::size_t dim_1 = 3;
    const std::size_t dim_2 = 2;
    const std::size_t dim_3 = 4;
    const std::size_t dim_4 = 3;

    // Declare data types.
    using DataTypes =
        Cabana::MemberDataTypes<float[dim_1][dim_2][dim_3],
                                int,
                                float[dim_1][dim_2][dim_3][dim_4],
                                double[dim_1],
                                double[dim_1][dim_2]
                                >;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,inner_array_size,TEST_MEMSPACE>;

    // Make sure that it is actually an AoSoA.
    BOOST_CHECK( Cabana::is_aosoa<AoSoA_t>::value );

    // Create an AoSoA.
    std::size_t num_data = 35;
    AoSoA_t aosoa( num_data );

    // Create some slices.
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );
    auto slice_4 = Cabana::slice<4>( aosoa );

    // Check that they are slices.
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_0)>::value );
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_1)>::value );
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_2)>::value );
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_3)>::value );
    BOOST_CHECK( Cabana::is_member_slice<decltype(slice_4)>::value );

    // Check sizes.
    BOOST_CHECK( slice_0.size() == std::size_t(35) );
    BOOST_CHECK( slice_0.numSoA() == std::size_t(4) );

    BOOST_CHECK( slice_0.arraySize(0) == std::size_t(10) );
    BOOST_CHECK( slice_0.arraySize(1) == std::size_t(10) );
    BOOST_CHECK( slice_0.arraySize(2) == std::size_t(10) );
    BOOST_CHECK( slice_0.arraySize(3) == std::size_t(5) );

    BOOST_CHECK( slice_0.rank() == std::size_t(3) );
    std::size_t e00 = slice_0.extent(0);
    BOOST_CHECK( e00 == dim_1 );
    std::size_t e01 = slice_0.extent(1);
    BOOST_CHECK( e01 == dim_2 );
    std::size_t e02 = slice_0.extent(2);
    BOOST_CHECK( e02 == dim_3 );
    std::size_t e03 = slice_0.extent(3);
    BOOST_CHECK( e03 == std::size_t(0) );

    BOOST_CHECK( slice_1.rank() == std::size_t(0) );
    std::size_t e10 = slice_1.extent(0);
    BOOST_CHECK( e10 == std::size_t(0) );
    std::size_t e11 = slice_1.extent(1);
    BOOST_CHECK( e11 == std::size_t(0) );
    std::size_t e12 = slice_1.extent(2);
    BOOST_CHECK( e12 == std::size_t(0) );
    std::size_t e13 = slice_1.extent(3);
    BOOST_CHECK( e13 == std::size_t(0) );

    BOOST_CHECK( slice_2.rank() == std::size_t(4) );
    std::size_t e20 = slice_2.extent(0);
    BOOST_CHECK( e20 == dim_1 );
    std::size_t e21 = slice_2.extent(1);
    BOOST_CHECK( e21 == dim_2 );
    std::size_t e22 = slice_2.extent(2);
    BOOST_CHECK( e22 == dim_3 );
    std::size_t e23 = slice_2.extent(3);
    BOOST_CHECK( e23 == dim_4 );

    BOOST_CHECK( slice_3.rank() == std::size_t(1) );
    std::size_t e30 = slice_3.extent(0);
    BOOST_CHECK( e30 == dim_1 );
    std::size_t e31 = slice_3.extent(1);
    BOOST_CHECK( e31 == std::size_t(0) );
    std::size_t e32 = slice_3.extent(2);
    BOOST_CHECK( e32 == std::size_t(0) );
    std::size_t e33 = slice_3.extent(3);
    BOOST_CHECK( e33 == std::size_t(0) );

    BOOST_CHECK( slice_4.rank() == std::size_t(2) );
    std::size_t e40 = slice_4.extent(0);
    BOOST_CHECK( e40 == dim_1 );
    std::size_t e41 = slice_4.extent(1);
    BOOST_CHECK( e41 == dim_2 );
    std::size_t e42 = slice_4.extent(2);
    BOOST_CHECK( e42 == std::size_t(0) );
    std::size_t e43 = slice_4.extent(3);
    BOOST_CHECK( e43 == std::size_t(0) );

    // Initialize data with the rank accessors.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( auto idx = aosoa.begin(); idx != aosoa.end(); ++idx )
    {
        // Member 0.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    slice_0( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        slice_1( idx ) = ival;

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    for ( std::size_t l = 0; l < dim_4; ++l )
                        slice_2( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            slice_3( idx, i ) = dval * i;

        // Member 4.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                slice_4( idx, i, j ) = dval * (i+j);
    }

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}
