#include <Cabana_DeepCopy.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberSlice.hpp>

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
// Perform a deep copy test.
template<class DstMemorySpace, class SrcMemorySpace,
         class DstInnerArrayLayout, class SrcInnerArrayLayout>
void testDeepCopy()
{
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

    // Declare the AoSoA types.
    using DstAoSoA_t = Cabana::AoSoA<DataTypes,DstInnerArrayLayout,DstMemorySpace>;
    using SrcAoSoA_t = Cabana::AoSoA<DataTypes,SrcInnerArrayLayout,SrcMemorySpace>;

    // Create AoSoAs.
    int num_data = 357;
    DstAoSoA_t dst_aosoa( num_data );
    SrcAoSoA_t src_aosoa( num_data );

    // Initialize data with the rank accessors.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( auto idx = 0; idx < src_aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    src_aosoa.template get<0>( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        src_aosoa.template get<1>( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        src_aosoa.template get<2>( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            src_aosoa.template get<3>( idx, i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                src_aosoa.template get<4>( idx, i, j ) = dval * (i+j);
    }

    for ( auto idx = 0; idx < dst_aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    dst_aosoa.template get<0>( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        dst_aosoa.template get<1>( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        dst_aosoa.template get<2>( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            dst_aosoa.template get<3>( idx, i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                dst_aosoa.template get<4>( idx, i, j ) = dval * (i+j);
    }

    // Deep copy
    Cabana::deep_copy( dst_aosoa, src_aosoa );

    // Check values.
    checkDataMembers( dst_aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( deep_copy_to_host_same_layout_test )
{
    testDeepCopy<Kokkos::HostSpace,
                 TEST_MEMSPACE,
                 Cabana::InnerArrayLayout<16,Kokkos::LayoutRight>,
                 Cabana::InnerArrayLayout<16,Kokkos::LayoutRight> >();
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( deep_copy_from_host_same_layout_test )
{
    testDeepCopy<TEST_MEMSPACE,
                 Kokkos::HostSpace,
                 Cabana::InnerArrayLayout<16,Kokkos::LayoutLeft>,
                 Cabana::InnerArrayLayout<16,Kokkos::LayoutLeft> >();
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( deep_copy_to_host_different_layout_test )
{
    testDeepCopy<Kokkos::HostSpace,
                 TEST_MEMSPACE,
                 Cabana::InnerArrayLayout<16,Kokkos::LayoutLeft>,
                 Cabana::InnerArrayLayout<32,Kokkos::LayoutRight> >();
    testDeepCopy<Kokkos::HostSpace,
                 TEST_MEMSPACE,
                 Cabana::InnerArrayLayout<64,Kokkos::LayoutRight>,
                 Cabana::InnerArrayLayout<8,Kokkos::LayoutLeft> >();
}

//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( deep_copy_from_host_different_layout_test )
{
    testDeepCopy<TEST_MEMSPACE,
                 Kokkos::HostSpace,
                 Cabana::InnerArrayLayout<64,Kokkos::LayoutLeft>,
                 Cabana::InnerArrayLayout<8,Kokkos::LayoutRight> >();
    testDeepCopy<TEST_MEMSPACE,
                 Kokkos::HostSpace,
                 Cabana::InnerArrayLayout<16,Kokkos::LayoutRight>,
                 Cabana::InnerArrayLayout<32,Kokkos::LayoutLeft> >();
}

//---------------------------------------------------------------------------//
