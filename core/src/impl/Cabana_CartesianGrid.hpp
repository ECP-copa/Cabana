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

#ifndef CABANA_CARTESIANGRID_HPP
#define CABANA_CARTESIANGRID_HPP

#include <Kokkos_Core.hpp>

#include <limits>
#include <type_traits>

namespace Cabana
{
namespace Impl
{
//! \cond Impl

template <class Real, typename std::enable_if<
                          std::is_floating_point<Real>::value, int>::type = 0>
class CartesianGrid
{
  public:
    using real_type = Real;

    Real _min_x;
    Real _min_y;
    Real _min_z;
    Real _max_x;
    Real _max_y;
    Real _max_z;
    Real _dx;
    Real _dy;
    Real _dz;
    Real _rdx;
    Real _rdy;
    Real _rdz;
    int _nx;
    int _ny;
    int _nz;

    CartesianGrid() = default;

    CartesianGrid( const Real min_x, const Real min_y, const Real min_z,
                   const Real max_x, const Real max_y, const Real max_z,
                   const Real delta_x, const Real delta_y, const Real delta_z )
        : _min_x( min_x )
        , _min_y( min_y )
        , _min_z( min_z )
        , _max_x( max_x )
        , _max_y( max_y )
        , _max_z( max_z )
    {
        _nx = cellsBetween( max_x, min_x, 1.0 / delta_x );
        _ny = cellsBetween( max_y, min_y, 1.0 / delta_y );
        _nz = cellsBetween( max_z, min_z, 1.0 / delta_z );

        _dx = ( max_x - min_x ) / _nx;
        _dy = ( max_y - min_y ) / _ny;
        _dz = ( max_z - min_z ) / _nz;

        _rdx = 1.0 / _dx;
        _rdy = 1.0 / _dy;
        _rdz = 1.0 / _dz;
    }

    // Get the total number of cells.
    KOKKOS_INLINE_FUNCTION
    std::size_t totalNumCells() const { return _nx * _ny * _nz; }

    // Get the number of cells in each direction.
    KOKKOS_INLINE_FUNCTION
    void numCells( int& num_x, int& num_y, int& num_z )
    {
        num_x = _nx;
        num_y = _ny;
        num_z = _nz;
    }

    // Get the number of cells in a given direction.
    KOKKOS_INLINE_FUNCTION
    int numBin( const int dim ) const
    {
        if ( 0 == dim )
            return _nx;
        else if ( 1 == dim )
            return _ny;
        else if ( 2 == dim )
            return _nz;
        else
            return -1;
    }

    // Given a position get the ijk indices of the cell in which
    KOKKOS_INLINE_FUNCTION
    void locatePoint( const Real xp, const Real yp, const Real zp, int& ic,
                      int& jc, int& kc ) const
    {
        // Since we use a floor function a point on the outer boundary
        // will be found in the next cell, causing an out of bounds error
        ic = cellsBetween( xp, _min_x, _rdx );
        ic = ( ic == _nx ) ? ic - 1 : ic;
        jc = cellsBetween( yp, _min_y, _rdy );
        jc = ( jc == _ny ) ? jc - 1 : jc;
        kc = cellsBetween( zp, _min_z, _rdz );
        kc = ( kc == _nz ) ? kc - 1 : kc;
    }

    // Given a position and a cell index get square of the minimum distance to
    // that point to any point in the cell. If the point is in the cell the
    // returned distance is zero.
    KOKKOS_INLINE_FUNCTION
    Real minDistanceToPoint( const Real xp, const Real yp, const Real zp,
                             const int ic, const int jc, const int kc ) const
    {
        Real xc = _min_x + ( ic + 0.5 ) * _dx;
        Real yc = _min_y + ( jc + 0.5 ) * _dy;
        Real zc = _min_z + ( kc + 0.5 ) * _dz;

        Real rx = fabs( xp - xc ) - 0.5 * _dx;
        Real ry = fabs( yp - yc ) - 0.5 * _dy;
        Real rz = fabs( zp - zc ) - 0.5 * _dz;

        rx = ( rx > 0.0 ) ? rx : 0.0;
        ry = ( ry > 0.0 ) ? ry : 0.0;
        rz = ( rz > 0.0 ) ? rz : 0.0;

        return rx * rx + ry * ry + rz * rz;
    }

    // Given the ijk index of a cell get its cardinal index.
    KOKKOS_INLINE_FUNCTION
    int cardinalCellIndex( const int i, const int j, const int k ) const
    {
        return ( i * _ny + j ) * _nz + k;
    }

    KOKKOS_INLINE_FUNCTION
    void ijkBinIndex( const int cardinal, int& i, int& j, int& k ) const
    {
        i = cardinal / ( _ny * _nz );
        j = ( cardinal / _nz ) % _ny;
        k = cardinal % _nz;
    }

    // Calculate the number of full cells between 2 points.
    KOKKOS_INLINE_FUNCTION
    int cellsBetween( const Real max, const Real min, const Real rdelta ) const
    {
        return Kokkos::floor( ( max - min ) * rdelta );
    }
};

//! \endcond
} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_CARTESIANGRID_HPP
