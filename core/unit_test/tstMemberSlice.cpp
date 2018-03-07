#include <Cabana_MemberSlice.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Serial.hpp>

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
                    BOOST_TEST( aosoa.template get<0>( idx, i, j, k ) ==
                                         fval * (i+j+k) );

        // Member 1.
        BOOST_TEST( aosoa.template get<1>( idx ) == ival );

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    for ( std::size_t l = 0; l < dim_4; ++l )
                        BOOST_TEST( aosoa.template get<2>( idx, i, j, k, l ) ==
                                             fval * (i+j+k+l) );

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            BOOST_TEST( aosoa.template get<3>( idx, i ) == dval * i );

        // Member 4.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                BOOST_TEST( aosoa.template get<4>( idx, i, j ) == dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( slice_serial_api_test )
{
    // Inner array size.
    const std::size_t array_size = 10;

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
    using AoSoA_t = Cabana::AoSoA<DataTypes,Cabana::Serial,array_size>;

    // Make sure that it is actually an AoSoA.
    BOOST_TEST( Cabana::is_aosoa<AoSoA_t>::value );

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
    BOOST_TEST( Cabana::is_member_slice<decltype(slice_0)>::value );
    BOOST_TEST( Cabana::is_member_slice<decltype(slice_1)>::value );
    BOOST_TEST( Cabana::is_member_slice<decltype(slice_2)>::value );
    BOOST_TEST( Cabana::is_member_slice<decltype(slice_3)>::value );
    BOOST_TEST( Cabana::is_member_slice<decltype(slice_4)>::value );

    // Check sizes.
    BOOST_TEST( slice_0.size() == std::size_t(35) );
    BOOST_TEST( slice_0.numSoA() == std::size_t(4) );

    BOOST_TEST( slice_0.arraySize(0) == std::size_t(10) );
    BOOST_TEST( slice_0.arraySize(1) == std::size_t(10) );
    BOOST_TEST( slice_0.arraySize(2) == std::size_t(10) );
    BOOST_TEST( slice_0.arraySize(3) == std::size_t(5) );

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

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( slice_serial_pointer_stride_test )
{
    // Inner array size.
    const std::size_t array_size = 103;

    // Declare data types. Note that this test only uses rank-0 data.
    using DataTypes =
        Cabana::MemberDataTypes<float,
                                int,
                                double,
                                int,
                                double
                                >;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,Cabana::Serial,array_size>;

    // Create an AoSoA.
    std::size_t num_data = 350;
    AoSoA_t aosoa( num_data );

    // Create some slices.
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );
    auto slice_4 = Cabana::slice<4>( aosoa );

    // Get pointers to the data.
    float* p0 = slice_0.pointer();
    int* p1 = slice_1.pointer();
    double* p2 = slice_2.pointer();
    int* p3 = slice_3.pointer();
    double* p4 = slice_4.pointer();

    // Get the strides between the member arrays.
    std::size_t st0 = slice_0.stride();
    std::size_t st1 = slice_1.stride();
    std::size_t st2 = slice_2.stride();
    std::size_t st3 = slice_3.stride();
    std::size_t st4 = slice_4.stride();

    // Initialize the data with raw pointer/stride access. Start by looping
    // over the structs. Each struct has a group of contiguous arrays of size
    // array_size for each member.
    std::size_t num_soa = slice_0.numSoA();
    for ( std::size_t s = 0; s < num_soa; ++s )
    {
        // Loop over the array in each struct and set the values.
        std::size_t local_array_size = slice_0.arraySize( s );
        for ( std::size_t i = 0; i < local_array_size; ++i )
        {
            p0[ s * st0 + i ] = (s + i) * 1.0;
            p1[ s * st1 + i ] = (s + i) * 2;
            p2[ s * st2 + i ] = (s + i) * 3.0;
            p3[ s * st3 + i ] = (s + i) * 4;
            p4[ s * st4 + i ] = (s + i) * 5.0;
        }
    }

    // Check the results.
    for ( auto idx = aosoa.begin(); idx < aosoa.end(); ++idx )
    {
        std::size_t s = idx.s();
        std::size_t i = idx.i();
        BOOST_TEST( aosoa.get<0>(idx) == (s+i)*1.0 );
        BOOST_TEST( aosoa.get<1>(idx) == int((s+i)*2) );
        BOOST_TEST( aosoa.get<2>(idx) == (s+i)*3.0 );
        BOOST_TEST( aosoa.get<3>(idx) == int((s+i)*4) );
        BOOST_TEST( aosoa.get<4>(idx) == (s+i)*5.0 );
    }
}
