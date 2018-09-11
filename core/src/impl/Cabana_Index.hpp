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

  \brief Class for converting between 1d and 2d aosoa indices.

  \tparam VectorLength The inner array size of the AoSoA.
*/
template<int VectorLength,
         typename std::enable_if<(Impl::IsVectorLengthValid<VectorLength>::value),
                                 int>::type = 0>
class Index
{
  public:

    // Inner array size.
    static constexpr int vector_length = VectorLength;

    // Array size offset.
    static constexpr int vector_length_offset = (vector_length - 1);

    // Number of binary bits needed to hold the array size.
    static constexpr int vector_length_binary_bits =
        Impl::LogBase2<vector_length>::value;

    /*!
      \brief Given a tuple index get the AoSoA struct index.

      \param tuple_index The tuple index.

      \return The index of the struct in which the tuple is located.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr int s( const int tuple_index )
    {
        return (tuple_index - (tuple_index & vector_length_offset)) >>
            vector_length_binary_bits;
    }

    /*!
      \brief Given a tuple index get the AoSoA array index.

      \param tuple_index The tuple index.

      \return The index of the array index in the struct in which the tuple
      is located.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr int i( const int tuple_index )
    {
        return tuple_index & vector_length_offset;
    }

    /*!
      \brief Given a struct index and array index in an AoSoA get the tuple
      index.

      \param struct_index The struct index.

      \param array_index The array index.

      \return The tuple index.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr int p( const int struct_index, const int array_index )
    {
        return (struct_index << vector_length_binary_bits) + array_index;
    }
};

//---------------------------------------------------------------------------//

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_INDEX_HPP
