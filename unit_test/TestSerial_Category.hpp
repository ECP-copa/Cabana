/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_TEST_SERIAL_CATEGORY_HPP
#define CAJITA_TEST_SERIAL_CATEGORY_HPP

#define TEST_CATEGORY serial
#define TEST_EXECSPACE Kokkos::Serial
#define TEST_MEMSPACE Kokkos::HostSpace
#define TEST_DEVICE Kokkos::Device<Kokkos::Serial,Kokkos::HostSpace>

#endif // end CAJITA_TEST_SERIAL_CATEGORY_HPP
