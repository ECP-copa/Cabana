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
    EXPECT_TRUE( !version_id.empty() );
    std::cout << "Cajita version " << version_id << std::endl;

    auto const commit_hash = Cajita::git_commit_hash();
    EXPECT_TRUE( !commit_hash.empty() );
    std::cout << "Cajita commit hash " << commit_hash << std::endl;
}

} // end namespace Test
