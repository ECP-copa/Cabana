#include <Cabana_Index.hpp>

#include <boost/test/unit_test.hpp>

#include <cstdlib>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( index_test )
{
    Cabana::Index i1( 12, 3, 4 );
    BOOST_TEST( i1.a() == std::size_t(12) );
    BOOST_TEST( i1.s() == std::size_t(3) );
    BOOST_TEST( i1.i() == std::size_t(4) );

    Cabana::Index i2( 12, 3, 4 );
    BOOST_TEST( bool(i1 == i2) );
    BOOST_TEST( bool(i1 <= i2) );
    BOOST_TEST( bool(i1 >= i2) );

    ++i1;
    BOOST_TEST( i1.a() == std::size_t(12) );
    BOOST_TEST( i1.s() == std::size_t(3) );
    BOOST_TEST( i1.i() == std::size_t(5) );
    BOOST_TEST( bool(i1 >= i2) );
    BOOST_TEST( bool(i1 > i2) );

    Cabana::Index i3( 12, 4, 4 );
    BOOST_TEST( bool(i3 != i2) );

    Cabana::Index i4( 12, 4, 5 );
    BOOST_TEST( bool(i3 != i4) );
    BOOST_TEST( bool(i4 > i3) );
    BOOST_TEST( bool(i4 >= i3) );

    Cabana::Index i5( 12, 4, 3 );
    BOOST_TEST( bool(i5 != i3) );
    BOOST_TEST( bool(i5 < i3) );
    BOOST_TEST( bool(i5 <= i3) );

    Cabana::Index i6( 9, 3, 8 );
    ++i6;
    BOOST_TEST( i6.a() == std::size_t(9) );
    BOOST_TEST( i6.s() == std::size_t(4) );
    BOOST_TEST( i6.i() == std::size_t(0) );
}
