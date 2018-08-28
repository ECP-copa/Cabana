#ifndef CABANA_EXECUTIONPOLICY_HPP
#define CABANA_EXECUTIONPOLICY_HPP

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class RangePolicy
  \brief Execution policy over a range of indices.
*/
template<int N, class ExecutionSpace>
class RangePolicy
{
  public:

    static constexpr int array_size = N;

    using execution_space = ExecutionSpace;

    // Range constructor.
    RangePolicy( const int begin, const int end )
        : _begin( begin )
        , _end( end )
    {}

    // Container constructor. The container must have a size() function that
    // returns an int. C++ concepts would be really nice here. Valid
    // containers include the AoSoA.
    template<class Container>
    RangePolicy( Container container )
        : _begin( 0 )
        , _end( container.size() )
    {}

    // Range bounds accessors.
    KOKKOS_INLINE_FUNCTION int begin() const { return _begin; }
    KOKKOS_INLINE_FUNCTION int end() const { return _end; }

  private:

    int _begin;
    int _end;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_EXECUTIONPOLICY_HPP
