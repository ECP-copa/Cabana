############################################################################
# Copyright (c) 2018-2021 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

include(${CMAKE_SOURCE_DIR}/utils/test_harness/test_harness.cmake)

Cabana_add_tests_nobackend(NAMES Version PACKAGE utils)

Cabana_add_tests(NAMES ParameterPack PACKAGE utils)
