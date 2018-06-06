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
      \brief Given a particle index get the AoSoA array and struct indices.

      \param particle_index The particle index.

      \return The indices of the struct and the array index in that struct in
      which the particle is located.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static Kokkos::pair<int,int> aosoa( const int particle_index )
    {
        Kokkos::pair<int,int> indices;

        // Array index.
        indices.second = particle_index & array_size_offset;

        // Struct index
        indices.first = (particle_index - indices.second) >> array_size_binary_bits;

        return indices;
    }

    /*!
      \brief Given a struct index and array index in an AoSoA get the particle
      index.

      \param struct_index The struct index.

      \param array_index The array index.

      \return The particle index.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static int particle( const int struct_index, const int array_index )
    {
        return (struct_index << array_size_binary_bits) + array_index;
    }
};

//---------------------------------------------------------------------------//

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_INDEX_HPP
