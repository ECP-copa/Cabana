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

} // end namespace Impl
} // end namespace Cabana

#endif // CABANA_TYPETRAITS_HPP
