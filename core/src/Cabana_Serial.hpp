#ifndef CABANA_SERIAL_HPP
#define CABANA_SERIAL_HPP

#include <type_traits>
#include <cstdlib>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Serial tag.
struct Serial {};

//---------------------------------------------------------------------------//
/*!
 * \brief Memory policy for serial computations.
 */
template<>
struct MemoryPolicy<Serial>
{
    //! Allocate array of a number of objects of type T. This will only work
    //! if T is of trivial type (trivially copyable and contiguous).
    template<class T>
    static
    typename std::enable_if<std::is_trivial<T>::value,void>::type
    allocate( T*& ptr, const std::size_t n )
    {
        ptr = (T*) malloc( n * sizeof(T) );
    }

    //! Dellocate an array.
    template<class T>
    static void deallocate( T* ptr )
    {
        free( ptr );
    }

    //! Copy from one address in the memory space to another in the same
    //! memory space.
    template<class T>
    static void copy( T* destination, const T* source, const std::size_t count )
    {
        std::copy( source, source + count, destination );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_SERIAL_HPP
