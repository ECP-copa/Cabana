#ifndef CABANA_MEMBERDATATYPES_HPP
#define CABANA_MEMBERDATATYPES_HPP

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
 \class MemberDataTypes
 \brief General sequence of types for SoA and AoSoA member data.
*/
template<typename... Types>
struct MemberDataTypes
{
    static constexpr std::size_t size = sizeof...(Types);
};

//---------------------------------------------------------------------------//
/*!
  \class MemberDataTypeAtIndex
  \brief Get the type of the member at a given index.
*/
template<std::size_t I, typename T, typename... Types>
struct MemberDataTypeAtIndex;

template<typename T, typename... Types>
struct MemberDataTypeAtIndex<0,T,Types...>
{
    using type = T;
};

template<std::size_t I, typename T, typename... Types>
struct MemberDataTypeAtIndex
{
    using type = typename MemberDataTypeAtIndex<I-1,Types...>::type;
};

//---------------------------------------------------------------------------//
namespace Impl
{

//---------------------------------------------------------------------------//
// Implementation detail to compute the product of the extents of the type
// starting at the given dimension.
template<typename T, std::size_t D>
struct ExtentsProduct;

template<typename T>
struct ExtentsProduct<T,0>
{
    static constexpr std::size_t value = std::extent<T,0>::value;
};

template<typename T, std::size_t D>
struct ExtentsProduct
{
    static constexpr std::size_t value =
        std::extent<T,D>::value * ExtentsProduct<T,D-1>::value;
};

//---------------------------------------------------------------------------//
// Implementation detail for getting the number of values for a given type.
template<typename T, std::size_t Rank>
struct MemberNumberOfValuesImpl;

template<typename T>
struct MemberNumberOfValuesImpl<T,0>
{
    // The total number of values. Rank-0 types have 1 value.
    static constexpr std::size_t value = 1;
};

template<typename T, std::size_t Rank>
struct MemberNumberOfValuesImpl
{
    // The total number of values. Types with rank greater than 0 have a value
    // equal to the product of their dimension extents.
    static constexpr std::size_t value = ExtentsProduct<T,Rank-1>::value;
};

//---------------------------------------------------------------------------//
/*!
  \class MemberNumberOfValues

  \brief Get the total number of values in a member type (product of all
  dimension extents).
*/
template<typename T>
struct MemberNumberOfValues
{
    static constexpr std::size_t value =
        Impl::MemberNumberOfValuesImpl<T,std::rank<T>::value>::value;
};

//---------------------------------------------------------------------------//

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_MEMBERDATATYPES_HPP
