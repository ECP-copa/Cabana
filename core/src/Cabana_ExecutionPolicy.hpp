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
    RangePolicy( const std::size_t begin, const std::size_t end )
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
    KOKKOS_INLINE_FUNCTION std::size_t begin() const { return _begin; }
    KOKKOS_INLINE_FUNCTION std::size_t end() const { return _end; }

  private:

    std::size_t _begin;
    std::size_t _end;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_EXECUTIONPOLICY_HPP
