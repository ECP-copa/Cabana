#include <Cabana_Version.hpp>

#include <iostream>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test {

class cabana_version : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

TEST_F( cabana_version, version_test )
{
    auto const cabana_version = Cabana::version();
    EXPECT_TRUE( !cabana_version.empty() );
    std::cout << "Cabana version " << cabana_version << std::endl;

    auto const cabana_commit_hash = Cabana::git_commit_hash();
    EXPECT_TRUE( !cabana_commit_hash.empty() );
    std::cout << "Cabana commit hash " << cabana_commit_hash << std::endl;
}

} // end namespace Test
