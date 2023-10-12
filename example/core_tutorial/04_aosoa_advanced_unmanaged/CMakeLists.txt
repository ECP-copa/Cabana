############################################################################
# Copyright (c) 2018-2023 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

add_executable(AdvancedUnmanagedAoSoA advanced_aosoa_unmanaged.cpp)
target_link_libraries(AdvancedUnmanagedAoSoA Cabana::Core)
add_test(NAME Core_tutorial_04_unmanaged COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:AdvancedUnmanagedAoSoA>)
