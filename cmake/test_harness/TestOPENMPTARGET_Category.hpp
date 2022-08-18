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

#ifndef CABANA_TEST_OPENMPTARGET_CATEGORY_HPP
#define CABANA_TEST_OPENMPTARGET_CATEGORY_HPP

#define TEST_CATEGORY openmptarget
#define TEST_EXECSPACE Kokkos::Experimental::OpenMPTarget
#define TEST_MEMSPACE Kokkos::Experimental::OpenMPTargetSpace
#define TEST_DEVICE                                                            \
    Kokkos::Device<Kokkos::Experimental::OpenMPTarget,                         \
                   Kokkos::Experimental::OpenMPTargetSpace>

#endif // end CABANA_TEST_OPENMP_CATEGORY_HPP