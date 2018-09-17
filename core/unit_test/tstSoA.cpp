#include <Cabana_SoA.hpp>
#include <Cabana_MemberTypes.hpp>
#include <Cabana_Types.hpp>

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
// SoA test
void testSoA()
{
    // Declare an array layout.
    const int vector_length = 4;

    // Declare an soa type.
    using member_types = Cabana::MemberTypes<double,
                                             int,
                                             float,
                                             double[2][3],
                                             unsigned[5],
                                             float[3][2][2],
                                             double[4][2][3][2]>;
    using soa_type = Cabana::SoA<vector_length,member_types>;

    // Check that the data in the soa is contiguous.
    EXPECT_TRUE( std::is_trivial<soa_type>::value );

    // Check that the soa is the same size as the struct (i.e. they are
    // equivalent).
    EXPECT_EQ( sizeof(FooData), sizeof(soa_type) );

    // Create an soa.
    soa_type soa;

    // Set some data with the soa.
    double v1 = 0.3343;
    soa.get<0>( 3 ) = v1;

    double v2 = 0.992;
    soa.get<6>( 2, 1, 1, 1, 1 ) = v2;

    // Check the data.
    EXPECT_EQ( soa.get<0>(3), v1 );
    EXPECT_EQ( soa.get<6>(2,1,1,1,1), v2 );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F( cabana_soa, soa_test )
{
    testSoA();
}

//---------------------------------------------------------------------------//

} // end namespace Test
