#ifndef CABANA_INDEX_HPP
#define CABANA_INDEX_HPP

#include <Cabana_Macros.hpp>

namespace Cabana
{

//---------------------------------------------------------------------------//
/*
  \class Index

  \brief An index for indexing into Arrays-of-Structs-of-Arrays.

  This index creates the appearance of a 1-dimensional outer set of indices
  where there are actually two, allowing for the composition of loops that
  appear 1-dimensional to the user and accessor functions that are of a single
  dimension.
*/
class Index
{
  private:

    // The inner array size.
    std::size_t _a;

    // The struct index.
    std::size_t _s;

    // Array offset index in the struct.
    std::size_t _i;

  public:

    // Constructor.
    CABANA_FUNCTION
    Index( const std::size_t array_size,
           const std::size_t struct_id,
           const std::size_t offset )
        : _a( array_size )
        , _s( struct_id )
        , _i( offset )
    {};

    // Get the struct index.
    CABANA_INLINE_FUNCTION
    std::size_t s() const
    { return _s; }

    // Get the array offset in the struct.
    CABANA_INLINE_FUNCTION
    std::size_t i() const
    { return _i; }

    // Prefix increment operator.
    CABANA_INLINE_FUNCTION
    Index& operator++()
    {
        _i = ( _a - 1 == _i ) ? 0 : _i + 1;
        _s = ( 0 == _i ) ? _s + 1: _s;
        return *this;
    };

    // Postfix increment operator.
    CABANA_INLINE_FUNCTION
    Index operator++(int)
    {
        Index temp = *this;
        ++*this;
        return temp;
    };

    // Equality comparator.
    CABANA_INLINE_FUNCTION
    bool operator==( const Index& rhs ) const
    {
        return (_s == rhs._s) && (_i == rhs._i);
    }

    // Inequality comparator.
    CABANA_INLINE_FUNCTION
    bool operator!=( const Index& rhs ) const
    {
        return (_s != rhs._s) || (_i != rhs._i);
    }

    // Less-than operator.
    CABANA_INLINE_FUNCTION
    bool operator<( const Index& rhs ) const
    {
        return (_s < rhs._s) || ((_s == rhs._s) && (_i < rhs._i));
    }

    // Greater-than operator.
    CABANA_INLINE_FUNCTION
    bool operator>( const Index& rhs ) const
    {
        return (_s > rhs._s) || ((_s == rhs._s) && (_i > rhs._i));
    }
};


//---------------------------------------------------------------------------//

}

#endif // CABANA_INDEX_HPP
