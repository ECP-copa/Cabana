#include <impl/Cabana_Index.hpp>

#include <gtest/gtest.h>

#include <cstdlib>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test {

class cabana_index : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

TEST_F( cabana_index, index_test )
{
    auto aosoa_idx_s = Cabana::Impl::Index<16>::s(40);
    auto aosoa_idx_i = Cabana::Impl::Index<16>::i(40);
    auto particle_idx = Cabana::Impl::Index<16>::p( 2, 8 );
    EXPECT_EQ( aosoa_idx_s, 2 );
    EXPECT_EQ( aosoa_idx_i, 8 );
    EXPECT_EQ( particle_idx, 40 );

    aosoa_idx_s = Cabana::Impl::Index<64>::s(64);
    aosoa_idx_i = Cabana::Impl::Index<64>::i(64);
    particle_idx = Cabana::Impl::Index<64>::p( 1, 0 );
    EXPECT_EQ( aosoa_idx_s, 1 );
    EXPECT_EQ( aosoa_idx_i, 0 );
    EXPECT_EQ( particle_idx, 64 );
}

} // end namespace Test
