#include <Cabana_Version.hpp>

#include <iostream>

#include <boost/test/unit_test.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( version_test )
{
    auto const cabana_version = Cabana::version();
    BOOST_CHECK( !cabana_version.empty() );
    std::cout << "Cabana version " << cabana_version << std::endl;

    auto const cabana_commit_hash = Cabana::git_commit_hash();
    BOOST_CHECK( !cabana_commit_hash.empty() );
    std::cout << "Cabana commit hash " << cabana_commit_hash << std::endl;
}
