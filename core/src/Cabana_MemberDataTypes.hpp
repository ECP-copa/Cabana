#ifndef CABANA_MEMBERDATATYPES_HPP
#define CABANA_MEMBERDATATYPES_HPP

#include <cstdlib>

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
// Get the type at a given index.
template<std::size_t I, typename T, typename... Types>
struct MemberDataTypeAtIndex
{
    using type = typename MemberDataTypeAtIndex<I-1,Types...>::type;
};

template<typename T, typename... Types>
struct MemberDataTypeAtIndex<0,T,Types...>
{
    using type = T;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_MEMBERDATATYPES_HPP
