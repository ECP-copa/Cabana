/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_TEST_CUDA_CATEGORY_HPP
#define CABANA_TEST_CUDA_CATEGORY_HPP

#include <Cabana_Types.hpp>

#include <Kokkos_Cuda.hpp>

#include <gtest/gtest.h>

#define TEST_CATEGORY cuda_uvm
#define TEST_EXECSPACE Kokkos::Cuda
#define TEST_MEMSPACE Cabana::CudaUVMSpace

#endif // end CABANA_TEST_CUDA_CATEGORY_HPP
