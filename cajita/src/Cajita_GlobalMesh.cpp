/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_GlobalMesh.hpp>

namespace Cajita
{

#define CAJITA_INST_GLOBALMESH( MESH, FP, NSD )                                \
    template class GlobalMesh<MESH<FP, NSD>>;

#define CAJITA_INST_GLOBALMESH_NSD( MESH, FP )                                 \
    CAJITA_INST_GLOBALMESH( MESH, FP, 3 )

#define CAJITA_INST_GLOBALMESH_FP( MESH )                                      \
    CAJITA_INST_GLOBALMESH_NSD( MESH, float )                                  \
    CAJITA_INST_GLOBALMESH_NSD( MESH, double )

CAJITA_INST_GLOBALMESH_FP( UniformMesh )
CAJITA_INST_GLOBALMESH_FP( NonUniformMesh )
CAJITA_INST_GLOBALMESH_FP( SparseMesh )

} // end namespace Cajita
