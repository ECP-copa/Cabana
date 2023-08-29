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

add_subdirectory(src)

if(Cabana_ENABLE_TESTING)
  add_subdirectory(unit_test)
endif()

if(Cabana_ENABLE_PERFORMANCE_TESTING)
  add_subdirectory(performance_test)
endif()
