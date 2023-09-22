/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Grid_MpiTraits.hpp
  \brief MPI traits
*/
#ifndef CABANA_GRID_MPITRAITS_HPP
#define CABANA_GRID_MPITRAITS_HPP

#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace Cabana
{
namespace Grid
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

} // namespace Grid
} // namespace Cabana

namespace Cajita
{
//! \cond Deprecated
template <class T>
using MpiTraits CAJITA_DEPRECATED = Cabana::Grid::MpiTraits<T>;
//! \endcond
} // namespace Cajita

#endif // end CABANA_GRID_MPITRAITS_HPP
