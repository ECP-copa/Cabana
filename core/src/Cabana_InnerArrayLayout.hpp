#ifndef CABANA_INNERARRAYLAYOUT_HPP
#define CABANA_INNERARRAYLAYOUT_HPP

#include <Cabana_Types.hpp>
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
namespace Impl
{

//---------------------------------------------------------------------------//
// Given an array layout and a potentially multi dimensional type T along with
// its rank, compose the inner array type.
template<typename T, std::size_t Rank, int VectorLength>
struct InnerArrayTypeImpl;

// rank-0 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,0,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    using type = value_type[VectorLength];
};

// rank-1 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,1,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    using type = value_type[D0][VectorLength];
};

// rank-2 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,2,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    using type = value_type[D1][D0][VectorLength];
};

// rank-3 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,3,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    using type = value_type[D2][D1][D0][VectorLength];
};

// rank-4 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,4,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    static constexpr std::size_t D3 = std::extent<T,3>::value;
    using type = value_type[D3][D2][D1][D0][VectorLength];
};

//---------------------------------------------------------------------------//
// Inner array type.
template<typename T,int VectorLength>
struct InnerArrayType
{
    using type =
        typename InnerArrayTypeImpl<T,std::rank<T>::value,VectorLength>::type;
};

//---------------------------------------------------------------------------//

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_INNERARRAYLAYOUT_HPP
