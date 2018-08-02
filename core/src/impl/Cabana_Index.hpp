#ifndef CABANA_INDEX_HPP
#define CABANA_INDEX_HPP

#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
namespace Impl
{

//---------------------------------------------------------------------------//
/*!
  \class Index

  \brief Class for converting between integral particle indices and AoSoA
  indices.

  \tparam N The inner array size of the AoSoA. Must be a power of 2.
*/
template<int N,
         typename std::enable_if<(Impl::IsPowerOfTwo<N>::value),int>::type = 0>
class Index
{
  public:

    // Inner array size.
    static constexpr int array_size = N;

    // Array size offset.
    static constexpr int array_size_offset = (array_size - 1);

    // Number of binary bits needed to hold the array size.
    static constexpr int array_size_binary_bits =
        Impl::LogBase2<array_size>::value;

    /*!
      \brief Given a particle index get the AoSoA struct index.

      \param particle_index The particle index.

      \return The index of the struct in which the particle is located.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr int s( const int particle_index )
    {
        return (particle_index - (particle_index & array_size_offset)) >> array_size_binary_bits;
    }

    /*!
      \brief Given a particle index get the AoSoA array index.

      \param particle_index The particle index.

      \return The index of the array index in the struct in which the particle
      is located.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr int i( const int particle_index )
    {
        return particle_index & array_size_offset;
    }

    /*!
      \brief Given a struct index and array index in an AoSoA get the particle
      index.

      \param struct_index The struct index.

      \param array_index The array index.

      \return The particle index.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr int p( const int struct_index, const int array_index )
    {
        return (struct_index << array_size_binary_bits) + array_index;
    }
};

//---------------------------------------------------------------------------//

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_INDEX_HPP
