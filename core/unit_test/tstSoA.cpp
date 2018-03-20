#include <Cabana_SoA.hpp>

#include <type_traits>

#include <boost/test/unit_test.hpp>

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
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( soa_test )
{
    // Declare an array size.
    const std::size_t array_size = 4;

    // Declare an soa type.
    using soa_type = Cabana::SoA<array_size,
                                 double,
                                 int,
                                 float,
                                 double[2][3],
                                 unsigned[5],
                                 float[3][2][2],
                                 double[4][2][3][2]>;

    // Check that the data in the soa is contiguous.
    BOOST_CHECK( std::is_trivial<soa_type>::value );

    // Check that the soa is the same size as the struct (i.e. they are
    // equivalent).
    BOOST_CHECK( sizeof(FooData) == sizeof(soa_type) );

    // Create an soa.
    soa_type soa;

    // Set some data with the soa.
    double v1 = 0.3343;
    auto d0 = Cabana::getStructMember<0>( soa );
    d0[3] = v1;

    double v2 = 0.992;
    auto d6 = Cabana::getStructMember<6>( soa );
    d6[2][1][1][1][1] = v2;

    // Check the data.
    BOOST_CHECK( Cabana::getStructMember<0>( soa )[3] == v1 );
    BOOST_CHECK(
        Cabana::getStructMember<6>( soa )[2][1][1][1][1] == v2 );
}
