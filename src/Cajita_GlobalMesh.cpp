/****************************************************************************
 * Copyright (c) 2019-2020 by the Cajita authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_GlobalMesh.hpp>

namespace Cajita
{

template class GlobalMesh<UniformMesh<float>>;
template class GlobalMesh<UniformMesh<double>>;

template class GlobalMesh<NonUniformMesh<float>>;
template class GlobalMesh<NonUniformMesh<double>>;

} // end namespace Cajita
