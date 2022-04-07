/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_TEST_OPENMP_CATEGORY_HPP
#define CABANA_TEST_OPENMP_CATEGORY_HPP

#define TEST_CATEGORY openmp
#define TEST_EXECSPACE Kokkos::OpenMP
#define TEST_MEMSPACE Kokkos::HostSpace
#define TEST_DEVICE Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>

#endif // end CABANA_TEST_OPENMP_CATEGORY_HPP
