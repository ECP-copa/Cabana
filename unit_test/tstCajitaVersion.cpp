#include <Cajita_Version.hpp>

#include <iostream>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test {

TEST( version, version_test )
{
    auto const version_id = Cajita::version();
    std::cout << "Cajita version " << version_id << std::endl;

    auto const commit_hash = Cajita::gitCommitHash();
    std::cout << "Cajita commit hash " << commit_hash << std::endl;
}

} // end namespace Test
