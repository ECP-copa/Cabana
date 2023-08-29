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

#include <impl/Cabana_Index.hpp>

#include <gtest/gtest.h>

#include <cstdlib>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test
{

TEST( cabana_index, index_test )
{
    auto aosoa_idx_s = Cabana::Impl::Index<16>::s( 40 );
    auto aosoa_idx_i = Cabana::Impl::Index<16>::a( 40 );
    auto tuple_idx = Cabana::Impl::Index<16>::i( 2, 8 );
    EXPECT_EQ( aosoa_idx_s, 2 );
    EXPECT_EQ( aosoa_idx_i, 8 );
    EXPECT_EQ( tuple_idx, 40 );

    aosoa_idx_s = Cabana::Impl::Index<64>::s( 64 );
    aosoa_idx_i = Cabana::Impl::Index<64>::a( 64 );
    tuple_idx = Cabana::Impl::Index<64>::i( 1, 0 );
    EXPECT_EQ( aosoa_idx_s, 1 );
    EXPECT_EQ( aosoa_idx_i, 0 );
    EXPECT_EQ( tuple_idx, 64 );
}

} // end namespace Test
