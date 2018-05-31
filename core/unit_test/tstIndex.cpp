#include <Cabana_Index.hpp>

#include <boost/test/unit_test.hpp>

#include <cstdlib>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( index_test )
{
    auto aosoa_idx = Cabana::Impl::Index<16>::aosoa(40);
    auto particle_idx = Cabana::Impl::Index<16>::particle( 2, 8 );
    BOOST_CHECK( aosoa_idx.first == 2 );
    BOOST_CHECK( aosoa_idx.second == 8 );
    BOOST_CHECK( particle_idx == 40 );

    aosoa_idx = Cabana::Impl::Index<64>::aosoa(64);
    particle_idx = Cabana::Impl::Index<64>::particle( 1, 0 );
    BOOST_CHECK( aosoa_idx.first == 1 );
    BOOST_CHECK( aosoa_idx.second == 0 );
    BOOST_CHECK( particle_idx == 64 );
}
