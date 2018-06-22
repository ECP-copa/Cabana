#ifndef CABANA_TYPETRAITS_HPP
#define CABANA_TYPETRAITS_HPP

#include <type_traits>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
// Checks if an integer is a power of two. N must be greater than 0.
template<int N, typename std::enable_if<(N > 0),int>::type = 0>
struct IsPowerOfTwo
{
    static constexpr bool value = ( (N & (N - 1)) == 0 );
};

//---------------------------------------------------------------------------//
// Calculate the base-2 logarithm of an integer which must be a power of 2 and
// greater than 0.
template<int N,
         typename std::enable_if<(IsPowerOfTwo<N>::value),int>::type = 0>
struct LogBase2
{
    static constexpr int value = 1 + LogBase2<(N>>1U)>::value;
};

template<>
struct LogBase2<1>
{
    static constexpr int value = 0;
};

//---------------------------------------------------------------------------//
// Given a type T of the given rank give the Kokkos data type.
template<typename T, std::size_t Rank>
struct KokkosDataTypeImpl;

template<typename T>
struct KokkosDataTypeImpl<T,0>
{
    using value_type = typename std::remove_all_extents<T>::type;
    using type = value_type*;
};

template<typename T>
struct KokkosDataTypeImpl<T,1>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    using type = value_type*[D0];
};

template<typename T>
struct KokkosDataTypeImpl<T,2>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    using type = value_type*[D0][D1];
};

template<typename T>
struct KokkosDataTypeImpl<T,3>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    using type = value_type*[D0][D1][D2];
};

template<typename T>
struct KokkosDataTypeImpl<T,4>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    static constexpr std::size_t D3 = std::extent<T,3>::value;
    using type = value_type*[D0][D1][D2][D3];
};

template<typename T>
struct KokkosDataType
{
    using type = KokkosDataTypeImpl<T,std::rank<T>::value>;
};

//---------------------------------------------------------------------------//

} // end namespace Impl
} // end namespace Cabana

#endif // CABANA_TYPETRAITS_HPP
