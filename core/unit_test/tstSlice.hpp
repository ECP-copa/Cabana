#include <Cabana_AoSoA.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// Initialize data members
template<class aosoa_type>
void initializeDataMembers(
    aosoa_type aosoa,
    const float fval, const double dval, const int ival,
    const int dim_1, const int dim_2,
    const int dim_3, const int dim_4 )
{
    auto slice_0 = aosoa.slice( Cabana::MemberTag<0>() );
    auto slice_1 = aosoa.slice( Cabana::MemberTag<1>() );
    auto slice_2 = aosoa.slice( Cabana::MemberTag<2>() );
    auto slice_3 = aosoa.slice( Cabana::MemberTag<3>() );
    auto slice_4 = aosoa.slice( Cabana::MemberTag<4>() );

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
}

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template<class aosoa_type>
void checkDataMembers(
    aosoa_type aosoa,
    const float fval, const double dval, const int ival,
    const int dim_1, const int dim_2,
    const int dim_3, const int dim_4 )
{
    auto slice_0 = aosoa.slice( Cabana::MemberTag<0>() );
    auto slice_1 = aosoa.slice( Cabana::MemberTag<1>() );
    auto slice_2 = aosoa.slice( Cabana::MemberTag<2>() );
    auto slice_3 = aosoa.slice( Cabana::MemberTag<3>() );
    auto slice_4 = aosoa.slice( Cabana::MemberTag<4>() );

    for ( auto idx = 0; idx != aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    EXPECT_EQ( slice_0( idx, i, j, k ),
                               fval * (i+j+k) );

        // Member 1.
        EXPECT_EQ( slice_1( idx ), ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        EXPECT_EQ( slice_2( idx, i, j, k, l ),
                                   fval * (i+j+k+l) );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( slice_3( idx, i ), dval * i );

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( slice_4( idx, i, j ), dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// API test function
void apiTest()
{
    // Manually set the inner array size with the test layout.
    const int vector_length = 16;

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

    // Create an AoSoA.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE,vector_length>;
    int num_data = 35;
    AoSoA_t aosoa( num_data );

    // Create some slices.
    auto slice_0 = aosoa.slice( Cabana::MemberTag<0>() );
    auto slice_1 = aosoa.slice( Cabana::MemberTag<1>() );
    auto slice_2 = aosoa.slice( Cabana::MemberTag<2>() );
    auto slice_3 = aosoa.slice( Cabana::MemberTag<3>() );
    auto slice_4 = aosoa.slice( Cabana::MemberTag<4>() );

    // Check that they are slices.
    EXPECT_TRUE( Cabana::is_slice<decltype(slice_0)>::value );
    EXPECT_TRUE( Cabana::is_slice<decltype(slice_1)>::value );
    EXPECT_TRUE( Cabana::is_slice<decltype(slice_2)>::value );
    EXPECT_TRUE( Cabana::is_slice<decltype(slice_3)>::value );
    EXPECT_TRUE( Cabana::is_slice<decltype(slice_4)>::value );

    // Check field sizes.
    EXPECT_EQ( slice_0.size(), 35 );
    EXPECT_EQ( slice_0.numSoA(), 3 );
    EXPECT_EQ( slice_0.fieldRank(), 3 );
    int e01 = slice_0.fieldExtent(0);
    EXPECT_EQ( e01, dim_1 );
    int e02 = slice_0.fieldExtent(1);
    EXPECT_EQ( e02, dim_2 );
    int e03 = slice_0.fieldExtent(2);
    EXPECT_EQ( e03, dim_3 );

    EXPECT_EQ( slice_1.size(), 35 );
    EXPECT_EQ( slice_1.numSoA(), 3 );
    EXPECT_EQ( slice_1.fieldRank(), 0 );

    EXPECT_EQ( slice_2.size(), 35 );
    EXPECT_EQ( slice_2.numSoA(), 3 );
    EXPECT_EQ( slice_2.fieldRank(), 4 );
    int e21 = slice_2.fieldExtent(0);
    EXPECT_EQ( e21, dim_1 );
    int e22 = slice_2.fieldExtent(1);
    EXPECT_EQ( e22, dim_2 );
    int e23 = slice_2.fieldExtent(2);
    EXPECT_EQ( e23, dim_3 );
    int e24 = slice_2.fieldExtent(3);
    EXPECT_EQ( e24, dim_4 );

    EXPECT_EQ( slice_3.size(), 35 );
    EXPECT_EQ( slice_3.numSoA(), 3 );
    EXPECT_EQ( slice_3.fieldRank(), 1 );
    int e31 = slice_3.fieldExtent(0);
    EXPECT_EQ( e31, dim_1 );

    EXPECT_EQ( slice_4.size(), 35 );
    EXPECT_EQ( slice_4.numSoA(), 3 );
    EXPECT_EQ( slice_4.fieldRank(), 2 );
    int e41 = slice_4.fieldExtent(0);
    EXPECT_EQ( e41, dim_1 );
    int e42 = slice_4.fieldExtent(1);
    EXPECT_EQ( e42, dim_2 );

    // Initialize data with the () operator. The implementation of operator()
    // calls access() and therefore tests that as well.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    initializeDataMembers(
        aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Check the raw pointer interface sizes.
    EXPECT_EQ( slice_0.rank(), 5 );
    EXPECT_EQ( slice_0.extent(0), 3 );
    EXPECT_EQ( slice_0.extent(1), 16 );
    EXPECT_EQ( slice_0.extent(2), dim_1 );
    EXPECT_EQ( slice_0.extent(3), dim_2 );
    EXPECT_EQ( slice_0.extent(4), dim_3 );

    EXPECT_EQ( slice_1.rank(), 2 );
    EXPECT_EQ( slice_1.extent(0), 3 );
    EXPECT_EQ( slice_1.extent(1), 16 );

    EXPECT_EQ( slice_2.rank(), 6 );
    EXPECT_EQ( slice_2.extent(0), 3 );
    EXPECT_EQ( slice_2.extent(1), 16 );
    EXPECT_EQ( slice_2.extent(2), dim_1 );
    EXPECT_EQ( slice_2.extent(3), dim_2 );
    EXPECT_EQ( slice_2.extent(4), dim_3 );
    EXPECT_EQ( slice_2.extent(5), dim_4 );

    EXPECT_EQ( slice_3.rank(), 3 );
    EXPECT_EQ( slice_3.extent(0), 3 );
    EXPECT_EQ( slice_3.extent(1), 16 );
    EXPECT_EQ( slice_3.extent(2), dim_1 );

    EXPECT_EQ( slice_4.rank(), 4 );
    EXPECT_EQ( slice_4.extent(0), 3 );
    EXPECT_EQ( slice_4.extent(1), 16 );
    EXPECT_EQ( slice_4.extent(2), dim_1 );
    EXPECT_EQ( slice_4.extent(3), dim_2 );

    // Now manipulate the data with the raw pointer interface.
    fval = 9.22;
    dval = 5.67;
    ival = 12;
    auto p0 = slice_0.data();
    auto p1 = slice_1.data();
    auto p2 = slice_2.data();
    auto p3 = slice_3.data();
    auto p4 = slice_4.data();
    for ( int s = 0; s < slice_0.numSoA(); ++s )
        for ( int a = 0; a < slice_0.arraySize(s); ++a )
        {
            // Member 0.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    for ( int k = 0; k < dim_3; ++k )
                        p0[ s*slice_0.stride(0) +
                            a*slice_0.stride(1) +
                            i*slice_0.stride(2) +
                            j*slice_0.stride(3) +
                            k*slice_0.stride(4) ] = fval * (i+j+k);

            // Member 1.
            p1[ s*slice_1.stride(0) +
                a*slice_1.stride(1) ] = ival;

            // Member 2.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    for ( int k = 0; k < dim_3; ++k )
                        for ( int l = 0; l < dim_4; ++l )
                            p2[ s*slice_2.stride(0) +
                                a*slice_2.stride(1) +
                                i*slice_2.stride(2) +
                                j*slice_2.stride(3) +
                                k*slice_2.stride(4) +
                                l*slice_2.stride(5) ] = fval * (i+j+k+l);

            // Member 3.
            for ( int i = 0; i < dim_1; ++i )
                p3[ s*slice_3.stride(0) +
                    a*slice_3.stride(1) +
                    i*slice_3.stride(2) ] = dval * i;

            // Member 4.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    p4[ s*slice_4.stride(0) +
                        a*slice_4.stride(1) +
                        i*slice_4.stride(2) +
                        j*slice_4.stride(3) ] = dval * (i+j);
        }

    // Check the result of pointer manipulation
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
// Random access function
void randomAccessTest()
{
    // Manually set the inner array size with the test layout.
    const int vector_length = 16;

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

    // Create an AoSoA.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE,vector_length>;
    int num_data = 35;
    AoSoA_t aosoa( num_data );

    // Initialize data.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    initializeDataMembers(
        aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Create read-only random access slices.
    auto ra_slice_0 = aosoa.slice( Cabana::MemberTag<0>(),
                                   Cabana::RandomAccessMemory() );
    auto ra_slice_1 = aosoa.slice( Cabana::MemberTag<1>(),
                                   Cabana::RandomAccessMemory() );
    auto ra_slice_2 = aosoa.slice( Cabana::MemberTag<2>(),
                                   Cabana::RandomAccessMemory() );
    auto ra_slice_3 = aosoa.slice( Cabana::MemberTag<3>(),
                                   Cabana::RandomAccessMemory() );
    auto ra_slice_4 = aosoa.slice( Cabana::MemberTag<4>(),
                                   Cabana::RandomAccessMemory() );

    // Create a second aosoa.
    AoSoA_t aosoa_2( num_data );

    // Get normal slices of the data.
    auto slice_0 = aosoa_2.slice( Cabana::MemberTag<0>(),
                                  Cabana::DefaultAccessMemory() );
    auto slice_1 = aosoa_2.slice( Cabana::MemberTag<1>(),
                                  Cabana::DefaultAccessMemory() );
    auto slice_2 = aosoa_2.slice( Cabana::MemberTag<2>(),
                                  Cabana::DefaultAccessMemory() );
    auto slice_3 = aosoa_2.slice( Cabana::MemberTag<3>(),
                                  Cabana::DefaultAccessMemory() );
    auto slice_4 = aosoa_2.slice( Cabana::MemberTag<4>(),
                                  Cabana::DefaultAccessMemory() );

    // Assign the read-only data to the new aosoa.
    for ( auto idx = 0; idx != aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    slice_0( idx, i, j, k ) = ra_slice_0( idx, i, j, k );

        // Member 1.
        slice_1( idx ) = ra_slice_1( idx );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        slice_2( idx, i, j, k, l ) =
                            ra_slice_2( idx, i, j, k, l );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            slice_3( idx, i ) = ra_slice_3( idx, i );

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                slice_4( idx, i, j ) = ra_slice_4( idx, i, j );
    }

    // Check data members for proper assignment.
    checkDataMembers( aosoa_2, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
// Random access function
void atomicAccessTest()
{
    // Manually set the inner array size with the test layout.
    const int vector_length = 16;

    // Declare data types.
    using DataTypes = Cabana::MemberDataTypes<int>;

    // Create an AoSoA.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE,vector_length>;
    int num_data = 35;
    AoSoA_t aosoa( num_data );

    // Get a slice of the data.
    auto slice = aosoa.slice( Cabana::MemberTag<0>() );

    // Set to 0.
    for ( int i = 0; i < num_data; ++i ) slice( i ) = 0;

    // Get an atomic slice of the data.
    auto atomic_slice = aosoa.slice( Cabana::MemberTag<0>(),
                                     Cabana::AtomicAccessMemory() );

    // Have every thread increment all elements of the slice. This should
    // create contention in parallel without the atomic.
    auto increment_op =
        KOKKOS_LAMBDA( const int i )
        {
            for ( int j = 0; j < num_data; ++j ) atomic_slice( j ) += 1;
        };
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, num_data );
    Kokkos::parallel_for( exec_policy, increment_op );
    Kokkos::fence();

    // Check the results of the atomic increment.
    for ( int i = 0; i < num_data; ++i ) EXPECT_EQ( slice(i), num_data );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, api_test )
{
    apiTest();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, random_access_test )
{
    randomAccessTest();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, atomic_access_test )
{
    atomicAccessTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
