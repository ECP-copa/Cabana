#ifndef CABANA_INDEXSEQUENCE_HPP
#define CABANA_INDEXSEQUENCE_HPP

#include <cstdlib>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
/*!
  \class IndexSequence
  \brief General sequence of indices.
 */
template<std::size_t... Indices>
struct IndexSequence
{
    static constexpr std::size_t size = sizeof...(Indices);
    using type = IndexSequence<Indices...>;
};

//---------------------------------------------------------------------------//
/*!
 * \brief Append an index to a sequence.
 */
template<std::size_t I, typename Sequence>
struct ConcatenateIndexSequence;

template<std::size_t I, std::size_t... Indices>
struct ConcatenateIndexSequence<I,IndexSequence<Indices...> >
    : IndexSequence<Indices...,I> {};

//---------------------------------------------------------------------------//
/*!
 * \brief Create a sequence of size N.
 */
template<std::size_t N>
struct MakeIndexSequence
    : ConcatenateIndexSequence<N-1,typename MakeIndexSequence<N-1>::type>::type
{};

template<>
struct MakeIndexSequence<1> : IndexSequence<0> {};

//---------------------------------------------------------------------------//

} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_INDEXSEQUENCE_HPP
