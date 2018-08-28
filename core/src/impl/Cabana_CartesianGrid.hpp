#ifndef CABANA_CARTESIANGRID_HPP
#define CABANA_CARTESIANGRID_HPP

#include <Kokkos_Core.hpp>

#include <limits>

namespace Cabana
{
namespace Impl
{
template<typename Scalar>
class CartesianGrid
{
  public:

    Scalar _min_x;
    Scalar _min_y;
    Scalar _min_z;
    Scalar _max_x;
    Scalar _max_y;
    Scalar _max_z;
    Scalar _dx;
    Scalar _dy;
    Scalar _dz;
    Scalar _rdx;
    Scalar _rdy;
    Scalar _rdz;
    int _nx;
    int _ny;
    int _nz;

    CartesianGrid() {}

    CartesianGrid( const Scalar min[3],
                   const Scalar max[3],
                   const Scalar delta[3] )
        : _min_x( min[0] )
        , _min_y( min[1] )
        , _min_z( min[2] )
        , _max_x( max[0] )
        , _max_y( max[1] )
        , _max_z( max[2] )
    {
        _nx = cellsBetween( max[0], min[0], 1.0 / delta[0] );
        _ny = cellsBetween( max[1], min[1], 1.0 / delta[1] );
        _nz = cellsBetween( max[2], min[2], 1.0 / delta[2] );

        _dx = (max[0]-min[0]) / _nx;
        _dy = (max[1]-min[1]) / _ny;
        _dz = (max[2]-min[2]) / _nz;

        _rdx = 1.0 / _dx;
        _rdy = 1.0 / _dy;
        _rdz = 1.0 / _dz;
    }

    // Get the number of cells in each direction.
    KOKKOS_INLINE_FUNCTION
    void numCells( int& num_x, int& num_y, int& num_z )
    {
        num_x = _nx;
        num_y = _ny;
        num_z = _nz;
    }

    // Given a position get the ijk indices of the cell in which
    KOKKOS_INLINE_FUNCTION
    void locatePoint( const Scalar xp,
                      const Scalar yp,
                      const Scalar zp,
                      int& ic,
                      int& jc,
                      int& kc ) const
    {
        ic = cellsBetween( xp, _min_x, _rdx );
        jc = cellsBetween( yp, _min_y, _rdy );
        kc = cellsBetween( zp, _min_z, _rdz );
    }

    // Given a position and a cell index get square of the minimum distance to
    // that point to any point in the cell. If the point is in the cell the
    // returned distance is zero.
    KOKKOS_INLINE_FUNCTION
    Scalar minDistanceToPoint( const Scalar xp,
                               const Scalar yp,
                               const Scalar zp,
                               const int ic,
                               const int jc,
                               const int kc ) const
    {
        Scalar xc = _min_x + (ic+0.5)*_dx;
        Scalar yc = _min_y + (jc+0.5)*_dy;
        Scalar zc = _min_z + (kc+0.5)*_dz;

        Scalar rx = abs(xp-xc) - 0.5*_dx;
        Scalar ry = abs(yp-yc) - 0.5*_dy;
        Scalar rz = abs(zp-zc) - 0.5*_dz;

        rx = ( rx > 0.0 ) ? rx : 0.0;
        ry = ( ry > 0.0 ) ? ry : 0.0;
        rz = ( rz > 0.0 ) ? rz : 0.0;

        return rx*rx + ry*ry + rz*rz;
    }

    // Given the ijk index of a cell get its cardinal index.
    KOKKOS_INLINE_FUNCTION
    int cardinalCellIndex( const int i, const int j, const int k ) const
    { return (i * _ny + j) * _nz + k; }

    KOKKOS_INLINE_FUNCTION
    void ijkBinIndex( const int cardinal, int& i, int& j, int& k ) const
    {
        i = cardinal / (_ny*_nz);
        j = ( cardinal / _nz ) % _ny;
        k = cardinal % _nz;
    }

    // Calculate the number of full cells between 2 points.
    KOKKOS_INLINE_FUNCTION
    int cellsBetween( const Scalar max, const Scalar min, const Scalar rdelta ) const
    { return std::floor( (max-min) * rdelta ); }
};

} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_CARTESIANGRID_HPP
