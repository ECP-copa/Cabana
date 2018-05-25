#include <Cabana_AoSoA.hpp>
#include <Cabana_Index.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

//---------------------------------------------------------------------------//
// Check the data given a set of values in an aosoa.
template<class aosoa_type>
void checkDataMembers(
    aosoa_type aosoa,
    const float fval, const double dval, const int ival,
    const int dim_1, const int dim_2,
    const int dim_3, const int dim_4 )
{
    for ( auto idx = 0; idx < aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    BOOST_CHECK( aosoa.template get<0>( idx, i, j, k ) ==
                                fval * (i+j+k) );

        // Member 1.
        BOOST_CHECK( aosoa.template get<1>( idx ) == ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        BOOST_CHECK( aosoa.template get<2>( idx, i, j, k, l ) ==
                                    fval * (i+j+k+l) );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            BOOST_CHECK( aosoa.template get<3>( idx, i ) == dval * i );

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                BOOST_CHECK( aosoa.template get<4>( idx, i, j ) == dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( aosoa_serial_api_test )
{
    // Manually set the inner array size.
    using inner_array_size = Cabana::InnerArraySize<16>;

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
    using AoSoA_t = Cabana::AoSoA<DataTypes,inner_array_size,TEST_MEMSPACE>;

    // Make sure that it is actually an AoSoA.
    BOOST_CHECK( Cabana::is_aosoa<AoSoA_t>::value );

    // Create an AoSoA.
    AoSoA_t aosoa;

    // Check sizes.
    BOOST_CHECK( aosoa.size() == int(0) );
    BOOST_CHECK( aosoa.capacity() == int(0) );
    BOOST_CHECK( aosoa.numSoA() == int(0) );

    // Check member type properties.
    BOOST_CHECK( aosoa.rank(0) == int(3) );
    int e00 = aosoa.extent(0,0);
    BOOST_CHECK( e00 == dim_1 );
    int e01 = aosoa.extent(0,1);
    BOOST_CHECK( e01 == dim_2 );
    int e02 = aosoa.extent(0,2);
    BOOST_CHECK( e02 == dim_3 );
    int e03 = aosoa.extent(0,3);
    BOOST_CHECK( e03 == int(0) );

    BOOST_CHECK( aosoa.rank(1) == int(0) );
    int e10 = aosoa.extent(1,0);
    BOOST_CHECK( e10 == int(0) );
    int e11 = aosoa.extent(1,1);
    BOOST_CHECK( e11 == int(0) );
    int e12 = aosoa.extent(1,2);
    BOOST_CHECK( e12 == int(0) );
    int e13 = aosoa.extent(1,3);
    BOOST_CHECK( e13 == int(0) );

    BOOST_CHECK( aosoa.rank(2) == int(4) );
    int e20 = aosoa.extent(2,0);
    BOOST_CHECK( e20 == dim_1 );
    int e21 = aosoa.extent(2,1);
    BOOST_CHECK( e21 == dim_2 );
    int e22 = aosoa.extent(2,2);
    BOOST_CHECK( e22 == dim_3 );
    int e23 = aosoa.extent(2,3);
    BOOST_CHECK( e23 == dim_4 );

    BOOST_CHECK( aosoa.rank(3) == int(1) );
    int e30 = aosoa.extent(3,0);
    BOOST_CHECK( e30 == dim_1 );
    int e31 = aosoa.extent(3,1);
    BOOST_CHECK( e31 == int(0) );
    int e32 = aosoa.extent(3,2);
    BOOST_CHECK( e32 == int(0) );
    int e33 = aosoa.extent(3,3);
    BOOST_CHECK( e33 == int(0) );

    BOOST_CHECK( aosoa.rank(4) == int(2) );
    int e40 = aosoa.extent(4,0);
    BOOST_CHECK( e40 == dim_1 );
    int e41 = aosoa.extent(4,1);
    BOOST_CHECK( e41 == dim_2 );
    int e42 = aosoa.extent(4,2);
    BOOST_CHECK( e42 == int(0) );
    int e43 = aosoa.extent(4,3);
    BOOST_CHECK( e43 == int(0) );

    // Resize
    int num_data = 35;
    aosoa.resize( num_data );

    // Check sizes for the new allocation/size.
    BOOST_CHECK( aosoa.size() == int(35) );
    BOOST_CHECK( aosoa.capacity() == int(48) );
    BOOST_CHECK( aosoa.numSoA() == int(3) );

    BOOST_CHECK( aosoa.arraySize(0) == int(16) );
    BOOST_CHECK( aosoa.arraySize(1) == int(16) );
    BOOST_CHECK( aosoa.arraySize(2) == int(3) );

    // Test bounds.
    auto end = aosoa.size();
    auto end_si = Cabana::Impl::Index<inner_array_size::value>::aosoa(end);
    BOOST_CHECK( end_si.second == int(3) );
    BOOST_CHECK( end_si.first == int(2) );

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
                    aosoa.get<0>( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        aosoa.get<1>( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        aosoa.get<2>( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            aosoa.get<3>( idx, i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                aosoa.get<4>( idx, i, j ) = dval * (i+j);
    }

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now extend the capacity of the container. First make the capacity
    // smaller - this wont actually do anything because we never decrease the
    // allocation of the container.
    aosoa.reserve( 1 );

    // Make sure nothing changed.
    BOOST_CHECK( aosoa.size() == int(35) );
    BOOST_CHECK( aosoa.capacity() == int(48) );
    BOOST_CHECK( aosoa.numSoA() == int(3) );
    BOOST_CHECK( aosoa.arraySize(0) == int(16) );
    BOOST_CHECK( aosoa.arraySize(1) == int(16) );
    BOOST_CHECK( aosoa.arraySize(2) == int(3) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now reserve a bunch of space.
    aosoa.reserve( 1024 );

    // Make sure capacity changed but sizes and data did not.
    BOOST_CHECK( aosoa.size() == int(35) );
    BOOST_CHECK( aosoa.capacity() == int(1024) );
    BOOST_CHECK( aosoa.numSoA() == int(3) );
    BOOST_CHECK( aosoa.arraySize(0) == int(16) );
    BOOST_CHECK( aosoa.arraySize(1) == int(16) );
    BOOST_CHECK( aosoa.arraySize(2) == int(3) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now decrease the size of the container.
    aosoa.resize( 29 );

    // Make sure sizes and data changed but the capacity did not.
    BOOST_CHECK( aosoa.size() == int(29) );
    BOOST_CHECK( aosoa.capacity() == int(1024) );
    BOOST_CHECK( aosoa.numSoA() == int(2) );
    BOOST_CHECK( aosoa.arraySize(0) == int(16) );
    BOOST_CHECK( aosoa.arraySize(1) == int(13) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( aosoa_raw_data_test )
{
    // Manually set the inner array size.
    using inner_array_size = Cabana::InnerArraySize<32>;

    // Multi dimensional member sizes.
    const int dim_1 = 3;
    const int dim_2 = 5;

    // Declare data types. Note that this test only uses rank-0 data.
    using DataTypes =
        Cabana::MemberDataTypes<float,
                                int,
                                double[dim_1][dim_2],
                                int,
                                double
                                >;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,inner_array_size,TEST_MEMSPACE>;

    // Create an AoSoA using the default constructor.
    int num_data = 350;
    AoSoA_t aosoa( num_data );

    // Get raw pointers to the data as one would in a C interface (no templates).
    float* p0 = (float*) aosoa.data(0);
    int* p1 = (int*) aosoa.data(1);
    double* p2 = (double*) aosoa.data(2);
    int* p3 = (int*) aosoa.data(3);
    double* p4 = (double*) aosoa.data(4);

    // Get the strides between the member arrays.
    int st0 = aosoa.stride(0);
    int st1 = aosoa.stride(1);
    int st2 = aosoa.stride(2);
    int st3 = aosoa.stride(3);
    int st4 = aosoa.stride(4);

    // Member 2 is multidimensional so get its extents.
    int m2e0 = aosoa.extent(2,0);
    int m2e1 = aosoa.extent(2,1);
    BOOST_CHECK( m2e0 == dim_1 );
    BOOST_CHECK( m2e1 == dim_2 );

    // Initialize the data with raw pointer/stride access. Start by looping
    // over the structs. Each struct has a group of contiguous arrays of size
    // array_size for each member.
    int num_soa = aosoa.numSoA();
    for ( int s = 0; s < num_soa; ++s )
    {
        // Loop over the array in each struct and set the values.
        int local_array_size = aosoa.arraySize( s );
        for ( int i = 0; i < local_array_size; ++i )
        {
            p0[ s * st0 + i ] = (s + i) * 1.0;
            p1[ s * st1 + i ] = (s + i) * 2;
            p3[ s * st3 + i ] = (s + i) * 4;
            p4[ s * st4 + i ] = (s + i) * 5.0;

            // Member 2 has some extra dimensions so add those to the
            // indexing.
            for ( int j = 0; j < m2e0; ++j )
                for ( int k = 0; k < m2e1; ++k )
                    p2[ s * st2 + i * m2e0 * m2e1 + j * m2e1 + k ] =
                        (s + i + j + k) * 3.0;
        }
    }

    // Check the results.
    for ( auto idx = 0; idx < aosoa.size(); ++idx )
    {
        auto aosoa_indices = Cabana::Impl::Index<inner_array_size::value>::aosoa( idx );
        int s = aosoa_indices.first;
        int i = aosoa_indices.second;

        BOOST_CHECK( aosoa.get<0>(idx) == (s+i)*1.0 );
        BOOST_CHECK( aosoa.get<1>(idx) == int((s+i)*2) );
        BOOST_CHECK( aosoa.get<3>(idx) == int((s+i)*4) );
        BOOST_CHECK( aosoa.get<4>(idx) == (s+i)*5.0 );

        // Member 2 has some extra dimensions so check those too.
        for ( int j = 0; j < dim_1; ++j )
            for ( int k = 0; k < dim_2; ++k )
                BOOST_CHECK( aosoa.get<2>(idx,j,k) == (s+i+j+k)*3.0 );
    }
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( aosoa_particle_test )
{
    // Data dimensions.
    const std::size_t dim_1 = 3;
    const std::size_t dim_2 = 2;
    const std::size_t dim_3 = 4;
    const std::size_t dim_4 = 3;

    // Declare member types.
    using T0 = float[dim_1][dim_2][dim_3];
    using T1 = int;
    using T2 = float[dim_1][dim_2][dim_3][dim_4];
    using T3 = double[dim_1];
    using T4 = double[dim_1][dim_2];

    // Declare data types.
    using DataTypes = Cabana::MemberDataTypes<T0,T1,T2,T3,T4>;

    // Declare the particle type.
    using Particle_t = Cabana::Particle<DataTypes>;

    // Create a view of particles.
    std::size_t num_data = 453;
    Kokkos::View<Particle_t*,TEST_MEMSPACE> particles( "particles", num_data );

    // Create a compatible particle AoSoA.
    using inner_array_size = Cabana::InnerArraySize<128>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,inner_array_size,TEST_MEMSPACE>;
    AoSoA_t aosoa( num_data );

    // Initialize aosoa data.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( auto idx = 0; idx != aosoa.size(); ++idx )
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

    // Assign the AoSoA data to the particles.
    for ( auto idx = 0; idx < aosoa.size(); ++idx )
         particles( idx ) = aosoa.getParticle( idx );

    // Change the particle data.
    fval = 2.1;
    dval = 9.21;
    ival = 3;
    for ( std::size_t idx = 0; idx < num_data; ++idx )
    {
        // Member 0.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    particles( idx ).get<0>( i, j, k ) = fval * (i+j+k);

        // Member 1.
        particles( idx ).get<1>() = ival;

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    for ( std::size_t l = 0; l < dim_4; ++l )
                        particles( idx ).get<2>( i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            particles( idx ).get<3>( i ) = dval * i;

        // Member 4.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                particles( idx ).get<4>( i, j ) = dval * (i+j);
    }

    // Assign the particle data back to the AoSoA.
    for ( auto idx = 0; idx < aosoa.size(); ++idx )
        aosoa.setParticle( idx, particles(idx) );

    // Check the results.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}
