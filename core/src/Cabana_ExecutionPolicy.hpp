#ifndef CABANA_EXECUTIONPOLICY_HPP
#define CABANA_EXECUTIONPOLICY_HPP

#include <Cabana_Index.hpp>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class IndexRangePolicy
  \brief Execution policy over a range of indices.
*/
template<class ExecutionSpace>
class IndexRangePolicy
{
  public:

    using execution_space = ExecutionSpace;

    IndexRangePolicy( const Index& begin, const Index& end )
        : _begin( begin )
        , _end( end )
    {}

    KOKKOS_INLINE_FUNCTION Index begin() const { return _begin; }
    KOKKOS_INLINE_FUNCTION Index end() const { return _end; }

  private:

    Index _begin;
    Index _end;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_EXECUTIONPOLICY_HPP
