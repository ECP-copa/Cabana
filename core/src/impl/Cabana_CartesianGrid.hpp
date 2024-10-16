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

template <class Real, std::size_t NumSpaceDim = 3>
class CartesianGrid
{
    static_assert( std::is_floating_point<Real>::value,
                   "Scalar type must be floating point type." );

  public:
    using real_type = Real;
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    Kokkos::Array<real_type, num_space_dim> _min;
    Kokkos::Array<real_type, num_space_dim> _max;
    Kokkos::Array<real_type, num_space_dim> _dx;
    Kokkos::Array<real_type, num_space_dim> _rdx;
    Kokkos::Array<int, num_space_dim> _nx;

    CartesianGrid() {}

    CartesianGrid( const Real min[num_space_dim], const Real max[num_space_dim],
                   const Real delta[num_space_dim] )
    {
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            _min[d] = min[d];
            _max[d] = max[d];
            _nx[d] = cellsBetween( max[d], min[d], 1.0 / delta[d] );
            _dx[d] = ( max[d] - min[d] ) / _nx[d];
            _rdx[d] = 1.0 / _dx[d];
        }
    }
    template <std::size_t NSD = num_space_dim>
    CartesianGrid( const Real min_x, const Real min_y, const Real min_z,
                   const Real max_x, const Real max_y, const Real max_z,
                   const Real delta_x, const Real delta_y, const Real delta_z,
                   typename std::enable_if<NSD == 3, int>::type* = 0 )
        : _min( { min_x, min_y, min_z } )
        , _max( { max_x, max_y, max_z } )
    {
        _nx[0] = cellsBetween( max_x, min_x, 1.0 / delta_x );
        _nx[1] = cellsBetween( max_y, min_y, 1.0 / delta_y );
        _nx[2] = cellsBetween( max_z, min_z, 1.0 / delta_z );

        _dx[0] = ( max_x - min_x ) / _nx[0];
        _dx[1] = ( max_y - min_y ) / _nx[1];
        _dx[2] = ( max_z - min_z ) / _nx[2];

        _rdx[0] = 1.0 / _dx[0];
        _rdx[1] = 1.0 / _dx[1];
        _rdx[2] = 1.0 / _dx[2];
    }

    // Get the total number of cells.
    KOKKOS_INLINE_FUNCTION
    std::size_t totalNumCells() const
    {
        std::size_t total = 1;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            total *= _nx[d];
        return total;
    }

    // Get the number of cells in each direction.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    numCells( int& num_x, int& num_y, int& num_z )
    {
        num_x = _nx[0];
        num_y = _nx[1];
        num_z = _nx[2];
    }

    // Get the number of cells in each direction.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    numCells( int& num_x, int& num_y )
    {
        num_x = _nx[0];
        num_y = _nx[1];
    }

    // Get the number of cells in a given direction.
    KOKKOS_INLINE_FUNCTION
    int numBin( const int dim ) const
    {
        assert( static_cast<std::size_t>( dim ) < num_space_dim );

        if ( 0 == dim )
            return _nx[0];
        else if ( 1 == dim )
            return _nx[1];
        else if ( 2 == dim )
            return _nx[2];
        else
            return -1;
    }

    // Given a position get the ijk indices of the cell in which
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    locatePoint( const Real xp, const Real yp, const Real zp, int& ic, int& jc,
                 int& kc ) const
    {
        // Since we use a floor function a point on the outer boundary
        // will be found in the next cell, causing an out of bounds error
        ic = cellsBetween( xp, _min[0], _rdx[0] );
        ic = ( ic == _nx[0] ) ? ic - 1 : ic;
        jc = cellsBetween( yp, _min[1], _rdx[1] );
        jc = ( jc == _nx[1] ) ? jc - 1 : jc;
        kc = cellsBetween( zp, _min[2], _rdx[2] );
        kc = ( kc == _nx[2] ) ? kc - 1 : kc;
    }

    // Given a position get the ijk indices of the cell in which
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    locatePoint( const Real xp, const Real yp, int& ic, int& jc ) const
    {
        // Since we use a floor function a point on the outer boundary
        // will be found in the next cell, causing an out of bounds error
        ic = cellsBetween( xp, _min[0], _rdx[0] );
        ic = ( ic == _nx[0] ) ? ic - 1 : ic;
        jc = cellsBetween( yp, _min[1], _rdx[1] );
        jc = ( jc == _nx[1] ) ? jc - 1 : jc;
    }

    // Given a position get the ijk indices of the cell in which
    KOKKOS_INLINE_FUNCTION void
    locatePoint( const Kokkos::Array<Real, num_space_dim> p,
                 Kokkos::Array<int, num_space_dim>& c ) const
    {
        // Since we use a floor function a point on the outer boundary
        // will be found in the next cell, causing an out of bounds error
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            c[d] = cellsBetween( p[d], _min[d], _rdx[d] );
            c[d] = ( c[d] == _nx[d] ) ? c[d] - 1 : c[d];
        }
    }

    // Given a position and a cell index get square of the minimum distance to
    // that point to any point in the cell. If the point is in the cell the
    // returned distance is zero.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, Real>
    minDistanceToPoint( const Real xp, const Real yp, const Real zp,
                        const int ic, const int jc, const int kc ) const
    {
        Kokkos::Array<Real, num_space_dim> x;
        x[0] = xp;
        x[1] = yp;
        x[2] = zp;
        Kokkos::Array<int, num_space_dim> c;
        c[0] = ic;
        c[1] = jc;
        c[2] = kc;

        return minDistanceToPoint( x, c );
    }

    // Given a position and a cell index get square of the minimum distance to
    // that point to any point in the cell. If the point is in the cell the
    // returned distance is zero.
    KOKKOS_INLINE_FUNCTION Real
    minDistanceToPoint( const Kokkos::Array<Real, num_space_dim> x,
                        const Kokkos::Array<int, num_space_dim> c ) const
    {
        Real rsqr = 0.0;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            Real xc = _min[d] + ( c[d] + 0.5 ) * _dx[d];
            Real rx = fabs( x[d] - xc ) - 0.5 * _dx[d];

            rx = ( rx > 0.0 ) ? rx : 0.0;

            rsqr += rx * rx;
        }

        return rsqr;
    }

    // Given the ijk index of a cell get its cardinal index.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, int>
    cardinalCellIndex( const int i, const int j, const int k ) const
    {
        return ( i * _nx[1] + j ) * _nx[2] + k;
    }

    // Given the ij index of a cell get its cardinal index.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, int>
    cardinalCellIndex( const int i, const int j ) const
    {
        return i * _nx[1] + j;
    }

    // Given the ij index of a cell get its cardinal index.
    KOKKOS_INLINE_FUNCTION int
    cardinalCellIndex( const Kokkos::Array<int, num_space_dim> ijk ) const
    {
        if constexpr ( num_space_dim == 3 )
            return cardinalCellIndex( ijk[0], ijk[1], ijk[2] );
        else
            return cardinalCellIndex( ijk[0], ijk[1] );
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    ijkBinIndex( const int cardinal, int& i, int& j, int& k ) const
    {
        i = cardinal / ( _nx[1] * _nx[2] );
        j = ( cardinal / _nx[2] ) % _nx[1];
        k = cardinal % _nx[2];
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    ijkBinIndex( const int cardinal, int& i, int& j ) const
    {
        i = cardinal / ( _nx[1] );
        j = cardinal % _nx[1];
    }

    KOKKOS_INLINE_FUNCTION void
    ijkBinIndex( const int cardinal,
                 Kokkos::Array<int, num_space_dim>& ijk ) const
    {
        if constexpr ( num_space_dim == 3 )
            return ijkBinIndex( cardinal, ijk[0], ijk[1], ijk[2] );
        else
            return ijkBinIndex( cardinal, ijk[0], ijk[1] );
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
