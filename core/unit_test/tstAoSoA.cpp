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
BOOST_AUTO_TEST_CASE( aosoa_serial_api_test )
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
    using AoSoA_t = Cabana::AoSoA<DataTypes,inner_array_size,Kokkos::Serial>;

    // Make sure that it is actually an AoSoA.
    BOOST_TEST( Cabana::is_aosoa<AoSoA_t>::value );

    // Create an AoSoA.
    AoSoA_t aosoa;

    // Check sizes.
    BOOST_TEST( aosoa.size() == std::size_t(0) );
    BOOST_TEST( aosoa.capacity() == std::size_t(0) );
    BOOST_TEST( aosoa.numSoA() == std::size_t(0) );

    // Check member type properties.
    BOOST_TEST( aosoa.rank(0) == std::size_t(3) );
    std::size_t e00 = aosoa.extent(0,0);
    BOOST_TEST( e00 == dim_1 );
    std::size_t e01 = aosoa.extent(0,1);
    BOOST_TEST( e01 == dim_2 );
    std::size_t e02 = aosoa.extent(0,2);
    BOOST_TEST( e02 == dim_3 );
    std::size_t e03 = aosoa.extent(0,3);
    BOOST_TEST( e03 == std::size_t(0) );

    BOOST_TEST( aosoa.rank(1) == std::size_t(0) );
    std::size_t e10 = aosoa.extent(1,0);
    BOOST_TEST( e10 == std::size_t(0) );
    std::size_t e11 = aosoa.extent(1,1);
    BOOST_TEST( e11 == std::size_t(0) );
    std::size_t e12 = aosoa.extent(1,2);
    BOOST_TEST( e12 == std::size_t(0) );
    std::size_t e13 = aosoa.extent(1,3);
    BOOST_TEST( e13 == std::size_t(0) );

    BOOST_TEST( aosoa.rank(2) == std::size_t(4) );
    std::size_t e20 = aosoa.extent(2,0);
    BOOST_TEST( e20 == dim_1 );
    std::size_t e21 = aosoa.extent(2,1);
    BOOST_TEST( e21 == dim_2 );
    std::size_t e22 = aosoa.extent(2,2);
    BOOST_TEST( e22 == dim_3 );
    std::size_t e23 = aosoa.extent(2,3);
    BOOST_TEST( e23 == dim_4 );

    BOOST_TEST( aosoa.rank(3) == std::size_t(1) );
    std::size_t e30 = aosoa.extent(3,0);
    BOOST_TEST( e30 == dim_1 );
    std::size_t e31 = aosoa.extent(3,1);
    BOOST_TEST( e31 == std::size_t(0) );
    std::size_t e32 = aosoa.extent(3,2);
    BOOST_TEST( e32 == std::size_t(0) );
    std::size_t e33 = aosoa.extent(3,3);
    BOOST_TEST( e33 == std::size_t(0) );

    BOOST_TEST( aosoa.rank(4) == std::size_t(2) );
    std::size_t e40 = aosoa.extent(4,0);
    BOOST_TEST( e40 == dim_1 );
    std::size_t e41 = aosoa.extent(4,1);
    BOOST_TEST( e41 == dim_2 );
    std::size_t e42 = aosoa.extent(4,2);
    BOOST_TEST( e42 == std::size_t(0) );
    std::size_t e43 = aosoa.extent(4,3);
    BOOST_TEST( e43 == std::size_t(0) );

    // Resize
    std::size_t num_data = 35;
    aosoa.resize( num_data );

    // Check sizes for the new allocation/size.
    BOOST_TEST( aosoa.size() == std::size_t(35) );
    BOOST_TEST( aosoa.capacity() == std::size_t(40) );
    BOOST_TEST( aosoa.numSoA() == std::size_t(4) );

    BOOST_TEST( aosoa.arraySize(0) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(1) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(2) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(3) == std::size_t(5) );

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
                    aosoa.get<0>( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        aosoa.get<1>( idx ) = ival;

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    for ( std::size_t l = 0; l < dim_4; ++l )
                        aosoa.get<2>( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            aosoa.get<3>( idx, i ) = dval * i;

        // Member 4.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                aosoa.get<4>( idx, i, j ) = dval * (i+j);
    }

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now extend the capacity of the container. First make the capacity
    // smaller - this wont actually do anything because we never decrease the
    // allocation of the container.
    aosoa.reserve( 1 );

    // Make sure nothing changed.
    BOOST_TEST( aosoa.size() == std::size_t(35) );
    BOOST_TEST( aosoa.capacity() == std::size_t(40) );
    BOOST_TEST( aosoa.numSoA() == std::size_t(4) );
    BOOST_TEST( aosoa.arraySize(0) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(1) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(2) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(3) == std::size_t(5) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now reserve a bunch of space.
    aosoa.reserve( 1000 );

    // Make sure capacity changed but sizes and data did not.
    BOOST_TEST( aosoa.size() == std::size_t(35) );
    BOOST_TEST( aosoa.capacity() == std::size_t(1000) );
    BOOST_TEST( aosoa.numSoA() == std::size_t(4) );
    BOOST_TEST( aosoa.arraySize(0) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(1) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(2) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(3) == std::size_t(5) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now decrease the size of the container.
    aosoa.resize( 29 );

    // Make sure sizes and data changed but the capacity did not.
    BOOST_TEST( aosoa.size() == std::size_t(29) );
    BOOST_TEST( aosoa.capacity() == std::size_t(1000) );
    BOOST_TEST( aosoa.numSoA() == std::size_t(3) );
    BOOST_TEST( aosoa.arraySize(0) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(1) == std::size_t(10) );
    BOOST_TEST( aosoa.arraySize(2) == std::size_t(9) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( aosoa_raw_data_test )
{
    // Manually set the inner array size.
    using inner_array_size = Cabana::InnerArraySize<103>;

    // Multi dimensional member sizes.
    const std::size_t dim_1 = 3;
    const std::size_t dim_2 = 5;

    // Declare data types. Note that this test only uses rank-0 data.
    using DataTypes =
        Cabana::MemberDataTypes<float,
                                int,
                                double[dim_1][dim_2],
                                int,
                                double
                                >;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,inner_array_size,Kokkos::Serial>;

    // Create an AoSoA using the default constructor.
    std::size_t num_data = 350;
    AoSoA_t aosoa( num_data );

    // Get raw pointers to the data as one would in a C interface (no templates).
    float* p0 = (float*) aosoa.data(0);
    int* p1 = (int*) aosoa.data(1);
    double* p2 = (double*) aosoa.data(2);
    int* p3 = (int*) aosoa.data(3);
    double* p4 = (double*) aosoa.data(4);

    // Get the strides between the member arrays.
    std::size_t st0 = aosoa.stride(0);
    std::size_t st1 = aosoa.stride(1);
    std::size_t st2 = aosoa.stride(2);
    std::size_t st3 = aosoa.stride(3);
    std::size_t st4 = aosoa.stride(4);

    // Member 2 is multidimensional so get its extents.
    std::size_t m2e0 = aosoa.extent(2,0);
    std::size_t m2e1 = aosoa.extent(2,1);
    BOOST_TEST( m2e0 == dim_1 );
    BOOST_TEST( m2e1 == dim_2 );

    // Initialize the data with raw pointer/stride access. Start by looping
    // over the structs. Each struct has a group of contiguous arrays of size
    // array_size for each member.
    std::size_t num_soa = aosoa.numSoA();
    for ( std::size_t s = 0; s < num_soa; ++s )
    {
        // Loop over the array in each struct and set the values.
        std::size_t local_array_size = aosoa.arraySize( s );
        for ( std::size_t i = 0; i < local_array_size; ++i )
        {
            p0[ s * st0 + i ] = (s + i) * 1.0;
            p1[ s * st1 + i ] = (s + i) * 2;
            p3[ s * st3 + i ] = (s + i) * 4;
            p4[ s * st4 + i ] = (s + i) * 5.0;

            // Member 2 has some extra dimensions so add those to the
            // indexing.
            for ( std::size_t j = 0; j < m2e0; ++j )
                for ( std::size_t k = 0; k < m2e1; ++k )
                    p2[ s * st2 + i * m2e0 * m2e1 + j * m2e1 + k ] =
                        (s + i + j + k) * 3.0;
        }
    }

    // Check the results.
    for ( auto idx = aosoa.begin(); idx < aosoa.end(); ++idx )
    {
        std::size_t s = idx.s();
        std::size_t i = idx.i();
        BOOST_TEST( aosoa.get<0>(idx) == (s+i)*1.0 );
        BOOST_TEST( aosoa.get<1>(idx) == int((s+i)*2) );
        BOOST_TEST( aosoa.get<3>(idx) == int((s+i)*4) );
        BOOST_TEST( aosoa.get<4>(idx) == (s+i)*5.0 );

        // Member 2 has some extra dimensions so check those too.
        for ( std::size_t j = 0; j < dim_1; ++j )
            for ( std::size_t k = 0; k < dim_2; ++k )
                BOOST_TEST( aosoa.get<2>(idx,j,k) == (s+i+j+k)*3.0 );
    }
}
