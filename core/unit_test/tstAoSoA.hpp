#include <Cabana_AoSoA.hpp>
#include <impl/Cabana_Index.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// Check the data given a set of values in an aosoa.
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

    for ( auto idx = 0; idx < aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    EXPECT_EQ( view_0( idx, i, j, k ),
                                fval * (i+j+k) );

        // Member 1.
        EXPECT_EQ( view_1( idx ), ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        EXPECT_EQ( view_2( idx, i, j, k, l ),
                                    fval * (i+j+k+l) );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( view_3( idx, i ), dval * i );

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( view_4( idx, i, j ), dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// Test an AoSoA with a given layout.
template<class Layout>
void testAoSoA()
{
    // Manually set the inner array size.
    using inner_array_layout = Cabana::InnerArrayLayout<16,Layout>;

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
    EXPECT_TRUE( Cabana::is_aosoa<AoSoA_t>::value );

    // Create an AoSoA.
    AoSoA_t aosoa;

    // Get field views.
    auto view_0 = aosoa.template view<0>();
    auto view_1 = aosoa.template view<1>();
    auto view_2 = aosoa.template view<2>();
    auto view_3 = aosoa.template view<3>();
    auto view_4 = aosoa.template view<4>();

    // Check sizes.
    EXPECT_EQ( aosoa.size(), int(0) );
    EXPECT_EQ( aosoa.capacity(), int(0) );
    EXPECT_EQ( aosoa.numSoA(), int(0) );

    // Resize
    int num_data = 35;
    aosoa.resize( num_data );

    // Check sizes for the new allocation/size.
    EXPECT_EQ( aosoa.size(), int(35) );
    EXPECT_EQ( aosoa.capacity(), int(48) );
    EXPECT_EQ( aosoa.numSoA(), int(3) );

    EXPECT_EQ( aosoa.arraySize(0), int(16) );
    EXPECT_EQ( aosoa.arraySize(1), int(16) );
    EXPECT_EQ( aosoa.arraySize(2), int(3) );

    // Test bounds.
    auto end = aosoa.size();
    int end_s = Cabana::Impl::Index<16>::s(end);
    int end_i = Cabana::Impl::Index<16>::i(end);
    EXPECT_EQ( end_s, 2 );
    EXPECT_EQ( end_i, 3 );

    // Get field views again. We invalidated the pointers by resizing the
    // views.
    view_0 = aosoa.template view<0>();
    view_1 = aosoa.template view<1>();
    view_2 = aosoa.template view<2>();
    view_3 = aosoa.template view<3>();
    view_4 = aosoa.template view<4>();

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
                    view_0( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        view_1( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        view_2( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            view_3( idx, i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                view_4( idx, i, j ) = dval * (i+j);
    }

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now extend the capacity of the container. First make the capacity
    // smaller - this wont actually do anything because we never decrease the
    // allocation of the container.
    aosoa.reserve( 1 );

    // Make sure nothing changed.
    EXPECT_EQ( aosoa.size(), int(35) );
    EXPECT_EQ( aosoa.capacity(), int(48) );
    EXPECT_EQ( aosoa.numSoA(), int(3) );
    EXPECT_EQ( aosoa.arraySize(0), int(16) );
    EXPECT_EQ( aosoa.arraySize(1), int(16) );
    EXPECT_EQ( aosoa.arraySize(2), int(3) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now reserve a bunch of space.
    aosoa.reserve( 1024 );

    // Make sure capacity changed but sizes and data did not.
    EXPECT_EQ( aosoa.size(), int(35) );
    EXPECT_EQ( aosoa.capacity(), int(1024) );
    EXPECT_EQ( aosoa.numSoA(), int(3) );
    EXPECT_EQ( aosoa.arraySize(0), int(16) );
    EXPECT_EQ( aosoa.arraySize(1), int(16) );
    EXPECT_EQ( aosoa.arraySize(2), int(3) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now decrease the size of the container.
    aosoa.resize( 29 );

    // Make sure sizes and data changed but the capacity did not.
    EXPECT_EQ( aosoa.size(), int(29) );
    EXPECT_EQ( aosoa.capacity(), int(1024) );
    EXPECT_EQ( aosoa.numSoA(), int(2) );
    EXPECT_EQ( aosoa.arraySize(0), int(16) );
    EXPECT_EQ( aosoa.arraySize(1), int(13) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
// Raw data layout right test.
void testRawDataLayoutRight()
{
    // Manually set the inner array size.
    using inner_array_layout = Cabana::InnerArrayLayout<32,Kokkos::LayoutRight>;

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
    using AoSoA_t = Cabana::AoSoA<DataTypes,inner_array_layout,TEST_MEMSPACE>;

    // Create an AoSoA using the default constructor.
    int num_data = 350;
    AoSoA_t aosoa( num_data );

    // Get views of fields.
    auto view_0 = aosoa.view<0>();
    auto view_1 = aosoa.view<1>();
    auto view_2 = aosoa.view<2>();
    auto view_3 = aosoa.view<3>();
    auto view_4 = aosoa.view<4>();

    // Get raw pointers to the data as one would in a C interface (no templates).
    float* p0 = view_0.data();
    int* p1 = view_1.data();
    double* p2 = view_2.data();
    int* p3 = view_3.data();
    double* p4 = view_4.data();

    // Get the strides between the member arrays.
    int st0 = view_0.stride();
    int st1 = view_1.stride();
    int st2 = view_2.stride();
    int st3 = view_3.stride();
    int st4 = view_4.stride();

    // Member 2 is multidimensional so get its extents.
    int m2e0 = view_2.extent(1);
    int m2e1 = view_2.extent(2);
    EXPECT_EQ( m2e0, dim_1 );
    EXPECT_EQ( m2e1, dim_2 );

    // Initialize the data with raw pointer/stride access. Start by looping
    // over the structs. Each struct has a group of contiguous arrays of size
    // array_size for each member.
    int num_soa = view_0.numSoA();
    for ( int s = 0; s < num_soa; ++s )
    {
        // Loop over the array in each struct and set the values.
        int local_array_size = view_0.arraySize( s );
        for ( int i = 0; i < local_array_size; ++i )
        {
            p0[ s * st0 + i ] = (s + i) * 1.0;
            p1[ s * st1 + i ] = (s + i) * 2;
            p3[ s * st3 + i ] = (s + i) * 4;
            p4[ s * st4 + i ] = (s + i) * 5.0;

            // Member 2 has some extra dimensions so add those to the
            // indexing. Note this is layout right.
            for ( int j = 0; j < m2e0; ++j )
                for ( int k = 0; k < m2e1; ++k )
                    p2[ s * st2 + i * m2e0 * m2e1 + j * m2e1 + k ] =
                        (s + i + j + k) * 3.0;
        }
    }

    // Check the results.
    for ( auto idx = 0; idx < aosoa.size(); ++idx )
    {
        int s = Cabana::Impl::Index<32>::s( idx );
        int i = Cabana::Impl::Index<32>::i( idx );

        EXPECT_EQ( view_0(idx), (s+i)*1.0 );
        EXPECT_EQ( view_1(idx), int((s+i)*2) );
        EXPECT_EQ( view_3(idx), int((s+i)*4) );
        EXPECT_EQ( view_4(idx), (s+i)*5.0 );

        // Member 2 has some extra dimensions so check those too.
        for ( int j = 0; j < dim_1; ++j )
            for ( int k = 0; k < dim_2; ++k )
                EXPECT_EQ( view_2(idx,j,k), (s+i+j+k)*3.0 );
    }
}

//---------------------------------------------------------------------------//
// Raw data layout left test.
void testRawDataLayoutLeft()
{
    // Manually set the inner array size.
    using inner_array_layout = Cabana::InnerArrayLayout<16,Kokkos::LayoutLeft>;

    // Multi dimensional member sizes.
    const int dim_1 = 3;
    const int dim_2 = 5;

    // Declare data types. Note that this test only uses rank-0 data.
    using DataTypes =
        Cabana::MemberDataTypes<float,
                                int,
                                double[dim_1][dim_2],
                                int,
                                double>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,inner_array_layout,TEST_MEMSPACE>;

    // Create an AoSoA using the default constructor.
    int num_data = 350;
    AoSoA_t aosoa( num_data );

    // Get views of fields.
    auto view_0 = aosoa.view<0>();
    auto view_1 = aosoa.view<1>();
    auto view_2 = aosoa.view<2>();
    auto view_3 = aosoa.view<3>();
    auto view_4 = aosoa.view<4>();

    // Get raw pointers to the data as one would in a C interface (no templates).
    float* p0 = view_0.data();
    int* p1 = view_1.data();
    double* p2 = view_2.data();
    int* p3 = view_3.data();
    double* p4 = view_4.data();

    // Get the strides between the member arrays.
    int st0 = view_0.stride();
    int st1 = view_1.stride();
    int st2 = view_2.stride();
    int st3 = view_3.stride();
    int st4 = view_4.stride();

    // Member 2 is multidimensional so get its extents.
    int m2e0 = view_2.extent(1);
    int m2e1 = view_2.extent(2);
    EXPECT_EQ( m2e0, dim_1 );
    EXPECT_EQ( m2e1, dim_2 );

    // Initialize the data with raw pointer/stride access. Start by looping
    // over the structs. Each struct has a group of contiguous arrays of size
    // array_size for each member.
    int num_soa = view_0.numSoA();
    for ( int s = 0; s < num_soa; ++s )
    {
        // Loop over the array in each struct and set the values.
        int local_array_size = view_0.arraySize( s );
        for ( int i = 0; i < local_array_size; ++i )
        {
            p0[ s * st0 + i ] = (s + i) * 1.0;
            p1[ s * st1 + i ] = (s + i) * 2;
            p3[ s * st3 + i ] = (s + i) * 4;
            p4[ s * st4 + i ] = (s + i) * 5.0;

            // Member 2 has some extra dimensions so add those to the
            // indexing. Note this is layout left.
            for ( int j = 0; j < m2e0; ++j )
                for ( int k = 0; k < m2e1; ++k )
                    p2[ s * st2 + k * 16 * m2e0 + j * 16 + i ] =
                        (s + i + j + k) * 3.0;
        }
    }

    // Check the results.
    for ( int idx = 0; idx < aosoa.size(); ++idx )
    {
        int s = Cabana::Impl::Index<16>::s( idx );
        int i = Cabana::Impl::Index<16>::i( idx );

        EXPECT_EQ( view_0(idx), (s+i)*1.0 );
        EXPECT_EQ( view_1(idx), int((s+i)*2) );
        EXPECT_EQ( view_3(idx), int((s+i)*4) );
        EXPECT_EQ( view_4(idx), (s+i)*5.0 );

        // Member 2 has some extra dimensions so check those too.
        for ( int j = 0; j < dim_1; ++j )
            for ( int k = 0; k < dim_2; ++k )
                EXPECT_EQ( view_2(idx,j,k), (s+i+j+k)*3.0 );
    }
}

//---------------------------------------------------------------------------//
// Particle test.
void testParticle()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;
    const int dim_4 = 3;

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
    int num_data = 453;
    Kokkos::View<Particle_t*,TEST_MEMSPACE> particles( "particles", num_data );

    // Create a compatible particle AoSoA.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    AoSoA_t aosoa( num_data );

    // Initialize aosoa data.
    auto view_0 = aosoa.view<0>();
    auto view_1 = aosoa.view<1>();
    auto view_2 = aosoa.view<2>();
    auto view_3 = aosoa.view<3>();
    auto view_4 = aosoa.view<4>();
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( auto idx = 0; idx != aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    view_0( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        view_1( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        view_2( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            view_3( idx, i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                view_4( idx, i, j ) = dval * (i+j);
    }

    // Assign the AoSoA data to the particles.
    for ( auto idx = 0; idx < aosoa.size(); ++idx )
         particles( idx ) = aosoa.getParticle( idx );

    // Change the particle data.
    fval = 2.1;
    dval = 9.21;
    ival = 3;
    for ( int idx = 0; idx < num_data; ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    particles( idx ).get<0>( i, j, k ) = fval * (i+j+k);

        // Member 1.
        particles( idx ).get<1>() = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        particles( idx ).get<2>( i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            particles( idx ).get<3>( i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                particles( idx ).get<4>( i, j ) = dval * (i+j);
    }

    // Assign the particle data back to the AoSoA.
    for ( auto idx = 0; idx < aosoa.size(); ++idx )
        aosoa.setParticle( idx, particles(idx) );

    // Check the results.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_layout_right_test )
{
    testAoSoA<Kokkos::LayoutRight>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_layout_left_test )
{
    testAoSoA<Kokkos::LayoutLeft>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_raw_data_layout_right_test )
{
    testRawDataLayoutRight();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_raw_data_layout_left_test )
{
    testRawDataLayoutLeft();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_particle_test )
{
    testParticle();
}

//---------------------------------------------------------------------------//

} // end namespace Test
