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

    // Range constructor.
    IndexRangePolicy( const Index& begin, const Index& end )
        : _begin( begin )
        , _end( end )
    {}

    // Container constructor. The container must have a begin() and end()
    // function that returns a Cabana::Index. C++ concepts would be really
    // nice here. Valid containers include the AoSoA and MemberSlices.
    template<class Container>
    IndexRangePolicy( Container container )
        : _begin( container.begin() )
        , _end( container.end() )
    {}

    // Range bounds accessors.
    KOKKOS_INLINE_FUNCTION Index begin() const { return _begin; }
    KOKKOS_INLINE_FUNCTION Index end() const { return _end; }

  private:

    Index _begin;
    Index _end;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_EXECUTIONPOLICY_HPP
