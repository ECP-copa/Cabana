#include <impl/Cabana_SoA.hpp>
#include <Cabana_MemberDataTypes.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

#include <gtest/gtest.h>

namespace Test
{
class cabana_soa : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

//---------------------------------------------------------------------------//
// Struct for size comparison.
struct FooData
{
    double _d0[4];
    int _d1[4];
    float _d2[4];
    double _d3[4][2][3];
    unsigned _d4[4][5];
    float _d5[4][3][2][2];
    double _d6[4][4][2][3][2];
};

//---------------------------------------------------------------------------//
// Layout right test.
void testLayoutRight()
{
    // Declare an array layout.
    using array_layout = Cabana::InnerArrayLayout<4,Kokkos::LayoutRight>;

    // Declare an soa type.
    using member_types = Cabana::MemberDataTypes<double,
                                                 int,
                                                 float,
                                                 double[2][3],
                                                 unsigned[5],
                                                 float[3][2][2],
                                                 double[4][2][3][2]>;
    using soa_type = Cabana::Impl::SoA<array_layout,member_types>;

    // Check that the data in the soa is contiguous.
    EXPECT_TRUE( std::is_trivial<soa_type>::value );

    // Check that the soa is the same size as the struct (i.e. they are
    // equivalent).
    EXPECT_EQ( sizeof(FooData), sizeof(soa_type) );

    // Create an soa.
    soa_type soa;

    // Set some data with the soa.
    double v1 = 0.3343;
    auto d0 = Cabana::Impl::getStructMember<0>( soa );
    d0[3] = v1;

    double v2 = 0.992;
    auto d6 = Cabana::Impl::getStructMember<6>( soa );
    d6[2][1][1][1][1] = v2;

    // Check the data.
    EXPECT_EQ( Cabana::Impl::getStructMember<0>( soa )[3], v1 );
    EXPECT_EQ( Cabana::Impl::getStructMember<6>( soa )[2][1][1][1][1], v2 );
}

//---------------------------------------------------------------------------//
// Layout left test.
void testLayoutLeft()
{
    // Declare an array layout.
    using array_layout = Cabana::InnerArrayLayout<4,Kokkos::LayoutLeft>;

    // Declare an soa type.
    using member_types = Cabana::MemberDataTypes<double,
                                                 int,
                                                 float,
                                                 double[2][3],
                                                 unsigned[5],
                                                 float[3][2][2],
                                                 double[4][2][3][2]>;
    using soa_type = Cabana::Impl::SoA<array_layout,member_types>;

    // Check that the data in the soa is contiguous.
    EXPECT_TRUE( std::is_trivial<soa_type>::value );

    // Check that the soa is the same size as the struct (i.e. they are
    // equivalent).
    EXPECT_EQ( sizeof(FooData), sizeof(soa_type) );

    // Create an soa.
    soa_type soa;

    // Set some data with the soa.
    double v1 = 0.3343;
    auto d0 = Cabana::Impl::getStructMember<0>( soa );
    d0[3] = v1;

    double v2 = 0.992;
    auto d6 = Cabana::Impl::getStructMember<6>( soa );
    d6[1][1][1][1][2] = v2;

    // Check the data.
    EXPECT_EQ( Cabana::Impl::getStructMember<0>( soa )[3], v1 );
    EXPECT_EQ( Cabana::Impl::getStructMember<6>( soa )[1][1][1][1][2], v2 );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F( cabana_soa, soa_layout_right_test )
{
    testLayoutRight();
}

//---------------------------------------------------------------------------//
TEST_F( cabana_soa, soa_layout_left_test )
{
    testLayoutLeft();
}

//---------------------------------------------------------------------------//

} // end namespace Test
