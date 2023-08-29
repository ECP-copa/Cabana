/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
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
namespace Test
{

TEST( cabana_version, version_test )
{
    auto const version_id = Cabana::version();
    EXPECT_TRUE( !version_id.empty() );
    std::cout << "Cabana version " << version_id << std::endl;

    auto const commit_hash = Cabana::git_commit_hash();
    EXPECT_TRUE( !commit_hash.empty() );
    std::cout << "Cabana commit hash " << commit_hash << std::endl;
}

} // end namespace Test
