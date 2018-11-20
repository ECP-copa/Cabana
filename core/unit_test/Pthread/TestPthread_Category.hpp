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

#ifndef CABANA_TEST_PTHREAD_CATEGORY_HPP
#define CABANA_TEST_PTHREAD_CATEGORY_HPP

#include <Cabana_Types.hpp>

#include <Kokkos_Threads.hpp>

#include <gtest/gtest.h>

namespace Test {

class pthread : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

} // namespace Test

#define TEST_CATEGORY pthread
#define TEST_EXECSPACE Kokkos::Threads
#define TEST_MEMSPACE Cabana::HostSpace

#endif // end CABANA_TEST_PTHREAD_CATEGORY_HPP
