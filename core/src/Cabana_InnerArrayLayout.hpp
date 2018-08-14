#ifndef CABANA_INNERARRAYLAYOUT_HPP
#define CABANA_INNERARRAYLAYOUT_HPP

#include <Cabana_Types.hpp>
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class InnerArrayLayout
  \brief Static inner array layout definition. This is the layout of the arrays in
  the struct-of-arrays and the layout of the potentially multidimensional data
  in those arrays. The inner array size must be a power of 2. Default to
  layout right.
*/
template<std::size_t N,
         typename ArrayLayout = LayoutRight,
         typename std::enable_if<
             (Impl::IsPowerOfTwo<N>::value && N > 0),int>::type = 0>
struct InnerArrayLayout
{
    static constexpr std::size_t size = N;
    using layout = ArrayLayout;
};

//---------------------------------------------------------------------------//
// Static type checker.
template<class >
struct is_inner_array_layout : public std::false_type {};

template<std::size_t N, typename ArrayLayout>
struct is_inner_array_layout<InnerArrayLayout<N,ArrayLayout> >
    : public std::true_type {};

template<std::size_t N, typename ArrayLayout>
struct is_inner_array_layout<const InnerArrayLayout<N,ArrayLayout> >
    : public std::true_type {};

//---------------------------------------------------------------------------//
namespace Impl
{

//---------------------------------------------------------------------------//
// Given an array layout and a potentially multi dimensional type T along with
// its rank, compose the inner array type.
template<typename T, std::size_t Rank, typename InnerArrayLayout_t>
struct InnerArrayTypeImpl;

// Layout-right specialization. User input for member data types is in
// C-ordering and therefore we can directly use the input type.
template<typename T, std::size_t Rank, std::size_t N>
struct InnerArrayTypeImpl<T,Rank,InnerArrayLayout<N,LayoutRight> >
{
    using type = T[N];
};

// Layout-left rank-0 specialization.
template<typename T, std::size_t N>
struct InnerArrayTypeImpl<T,0,InnerArrayLayout<N,LayoutLeft> >
{
    using value_type = typename std::remove_all_extents<T>::type;
    using type = value_type[N];
};

// Layout-left rank-1 specialization.
template<typename T, std::size_t N>
struct InnerArrayTypeImpl<T,1,InnerArrayLayout<N,LayoutLeft> >
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    using type = value_type[D0][N];
};

// Layout-left rank-2 specialization.
template<typename T, std::size_t N>
struct InnerArrayTypeImpl<T,2,InnerArrayLayout<N,LayoutLeft> >
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    using type = value_type[D1][D0][N];
};

// Layout-left rank-3 specialization.
template<typename T, std::size_t N>
struct InnerArrayTypeImpl<T,3,InnerArrayLayout<N,LayoutLeft> >
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    using type = value_type[D2][D1][D0][N];
};

// Layout-left rank-4 specialization.
template<typename T, std::size_t N>
struct InnerArrayTypeImpl<T,4,InnerArrayLayout<N,LayoutLeft> >
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    static constexpr std::size_t D3 = std::extent<T,3>::value;
    using type = value_type[D3][D2][D1][D0][N];
};

//---------------------------------------------------------------------------//
// Inner array type.
template<typename T,typename InnerArrayLayout_t>
struct InnerArrayType
{
    using type =
        typename InnerArrayTypeImpl<
        T,std::rank<T>::value,InnerArrayLayout_t>::type;
};

//---------------------------------------------------------------------------//

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_INNERARRAYLAYOUT_HPP
