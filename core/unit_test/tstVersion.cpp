/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

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
