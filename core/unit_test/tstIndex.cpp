#include <impl/Cabana_Index.hpp>

#include <boost/test/unit_test.hpp>

#include <cstdlib>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( index_test )
{
    auto aosoa_idx_s = Cabana::Impl::Index<16>::s(40);
    auto aosoa_idx_i = Cabana::Impl::Index<16>::i(40);
    auto particle_idx = Cabana::Impl::Index<16>::p( 2, 8 );
    BOOST_CHECK( aosoa_idx_s == 2 );
    BOOST_CHECK( aosoa_idx_i == 8 );
    BOOST_CHECK( particle_idx == 40 );

    aosoa_idx_s = Cabana::Impl::Index<64>::s(64);
    aosoa_idx_i = Cabana::Impl::Index<64>::i(64);
    particle_idx = Cabana::Impl::Index<64>::p( 1, 0 );
    BOOST_CHECK( aosoa_idx_s == 1 );
    BOOST_CHECK( aosoa_idx_i == 0 );
    BOOST_CHECK( particle_idx == 64 );
}
