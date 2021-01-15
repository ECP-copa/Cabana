/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_TEST_SYCL_CATEGORY_HPP
#define CABANA_TEST_SYCL_CATEGORY_HPP

#define TEST_CATEGORY sycl
#define TEST_EXECSPACE Kokkos::Experimental::SYCL
#define TEST_MEMSPACE Kokkos::Experimental::SYCLDeviceUSMSpace
#define TEST_DEVICE                                                            \
    Kokkos::Device<Kokkos::Experimental::SYCL,                                 \
                   Kokkos::Experimental::SYCLDeviceUSMSpace>

#endif
