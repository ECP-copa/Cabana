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

#ifndef CAJITA_MPITRAITS_HPP
#define CAJITA_MPITRAITS_HPP

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace Cajita
{

//---------------------------------------------------------------------------//
//! MPI type trait
//! tparam T Data type.
template <typename T>
struct MpiTraits;

//! MPI type trait
template <>
struct MpiTraits<char>
{
    //! MPI char type
    static MPI_Datatype type() { return MPI_CHAR; }
};

//! MPI type trait
template <>
struct MpiTraits<int>
{
    //! MPI int type
    static MPI_Datatype type() { return MPI_INT; }
};

//! MPI type trait
template <>
struct MpiTraits<long>
{
    //! MPI long type
    static MPI_Datatype type() { return MPI_LONG; }
};

//! MPI type trait
template <>
struct MpiTraits<float>
{
    //! MPI float type
    static MPI_Datatype type() { return MPI_FLOAT; }
};

//! MPI type trait
template <>
struct MpiTraits<double>
{
    //! MPI double type
    static MPI_Datatype type() { return MPI_DOUBLE; }
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_MPITRAITS_HPP
