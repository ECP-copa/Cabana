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

#ifndef CAJITA_MPITRAITS_HPP
#define CAJITA_MPITRAITS_HPP

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace Cajita
{

//---------------------------------------------------------------------------//
// Type traits
template <typename T>
struct MpiTraits;

template <>
struct MpiTraits<char>
{
    static MPI_Datatype type() { return MPI_CHAR; }
};

template <>
struct MpiTraits<int>
{
    static MPI_Datatype type() { return MPI_INT; }
};

template <>
struct MpiTraits<long>
{
    static MPI_Datatype type() { return MPI_LONG; }
};

template <>
struct MpiTraits<float>
{
    static MPI_Datatype type() { return MPI_FLOAT; }
};

template <>
struct MpiTraits<double>
{
    static MPI_Datatype type() { return MPI_DOUBLE; }
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_MPITRAITS_HPP
