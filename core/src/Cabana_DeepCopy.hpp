#ifndef CABANA_DEEPCOPY_HPP
#define CABANA_DEEPCOPY_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberSlice.hpp>
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ExecPolicy.hpp>

#include <type_traits>
#include <exception>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
// Copy data between two elements of a slice.
template<class DstSliceType, class SrcSliceType>
class DeepCopySliceElement
{
  private:

    DstSliceType _dst;
    SrcSliceType _src;

  public:

    DeepCopySliceElement( DstSliceType dst, SrcSliceType src )
        : _dst( dst )
        , _src( src )
    {}

    template<typename T = typename DstSliceType::data_type>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(0==std::rank<T>::value),void>::type
    operator()( const int particle_index ) const
    {
        _dst( particle_index ) = _src( particle_index );
    }

    template<typename T = typename DstSliceType::data_type>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(1==std::rank<T>::value),void>::type
    operator()( const int particle_index ) const
    {
        for ( int i0 = 0; i0 < _dst.extent(0); ++i0 )
            _dst( particle_index, i0 ) = _src( particle_index, i0 );
    }

    template<typename T = typename DstSliceType::data_type>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(2==std::rank<T>::value),void>::type
    operator()( const int particle_index ) const
    {
        for ( int i0 = 0; i0 < _dst.extent(0); ++i0 )
            for ( int i1 = 0; i1 < _dst.extent(1); ++i1 )
                _dst( particle_index, i0, i1 ) =
                    _src( particle_index, i0, i1 );
    }

    template<typename T = typename DstSliceType::data_type>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(3==std::rank<T>::value),void>::type
    operator()( const int particle_index ) const
    {
        for ( int i0 = 0; i0 < _dst.extent(0); ++i0 )
            for ( int i1 = 0; i1 < _dst.extent(1); ++i1 )
                for ( int i2 = 0; i2 < _dst.extent(2); ++i2 )
                    _dst( particle_index, i0, i1, i2 ) =
                        _src( particle_index, i0, i1, i2 );
    }

    template<typename T = typename DstSliceType::data_type>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(4==std::rank<T>::value),void>::type
    operator()( const int particle_index ) const
    {
        for ( int i0 = 0; i0 < _dst.extent(0); ++i0 )
            for ( int i1 = 0; i1 < _dst.extent(1); ++i1 )
                for ( int i2 = 0; i2 < _dst.extent(2); ++i2 )
                    for ( int i3 = 0; i3 < _dst.extent(3); ++i3 )
                        _dst( particle_index, i0, i1, i2, i3 ) =
                            _src( particle_index, i0, i1, i2, i3 );
    }
};

//---------------------------------------------------------------------------//
/*!
  \brief Deep copy a slice. Both slices must live in the same memory space.

  This function does not fence upon completion. If deep_copy is called in the
  context of an AoSoA the fence will be called after all slice copies have
  been launched. Each slice has its own memory space so this will allow each
  slice deep copy to continue in tandem.
*/
template<class DstAoSoA, class SrcAoSoA, class DstSliceType, class SrcSliceType>
inline void deepCopySlice(
    DstSliceType& dst,
    const SrcSliceType& src,
    bool fence_upon_completion = true,
    typename std::enable_if<(is_member_slice<DstSliceType>::value &&
                             is_member_slice<SrcSliceType>::value)>::type * = 0 )
{
    using dst_memory_space = typename DstAoSoA::traits::memory_space;
    using src_memory_space = typename SrcAoSoA::traits::memory_space;
    using dst_execution_space = typename DstAoSoA::traits::execution_space;
    using src_execution_space = typename SrcAoSoA::traits::execution_space;

    static_assert( std::is_same<typename DstSliceType::value_type,
                   typename SrcSliceType::value_type>::value,
                   "Attempted to copy slices with different value types" );
    static_assert( std::is_same<dst_memory_space,src_memory_space>::value,
                   "Attempted to copy slices in different memory spaces" );
    static_assert( std::is_same<dst_execution_space,src_execution_space>::value,
                   "Attempted to copy slices in different execution spaces" );

    // Check for the same number of values.
    if ( dst.size() != src.size() )
    {
        throw std::runtime_error(
            "Attempted to deep copy slices of different sizes" );
    }

    Kokkos::RangePolicy<dst_execution_space> exec_policy( 0, dst.size() );
    DeepCopySliceElement<DstSliceType,SrcSliceType> copy_func( dst, src );
    Kokkos::parallel_for(
        "deepCopySlice", exec_policy, copy_func );

    if ( fence_upon_completion ) Kokkos::fence();
}

//---------------------------------------------------------------------------//
/*!
  \brief Deep copy an AoSoA member-by-member.
*/
template<std::size_t M, class DstAoSoA, class SrcAoSoA>
struct DeepCopyByMember;

template<class DstAoSoA, class SrcAoSoA>
struct DeepCopyByMember<0,DstAoSoA,SrcAoSoA>
{
    static void copy(
        DstAoSoA& dst,
        const SrcAoSoA& src,
        typename std::enable_if<(is_aosoa<DstAoSoA>::value &&
                                 is_aosoa<SrcAoSoA>::value)>::type *  = 0 )
    {
        auto dst_slice = dst.template view<0>();
        auto src_slice = src.template view<0>();
        deepCopySlice<DstAoSoA,SrcAoSoA>( dst_slice, src_slice, false );
    }
};

template<std::size_t M, class DstAoSoA, class SrcAoSoA>
struct DeepCopyByMember
{
    static void copy(
        DstAoSoA& dst,
        const SrcAoSoA& src,
        typename std::enable_if<(is_aosoa<DstAoSoA>::value &&
                                 is_aosoa<SrcAoSoA>::value)>::type *  = 0 )
    {
        auto dst_slice = dst.template view<M>();
        auto src_slice = src.template view<M>();
        deepCopySlice<DstAoSoA,SrcAoSoA>( dst_slice, src_slice, false );
        DeepCopyByMember<M-1,DstAoSoA,SrcAoSoA>::copy( dst, src );
    }
};

//---------------------------------------------------------------------------//

} // end namepsace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Deep copy data between compatible AoSoA objects.

  \param dst The destination for the copied data.

  \param src The source of the copied data.

  Only AoSoA objects with the same set of member data types and size may be
  copied.
*/
template<class DstAoSoA, class SrcAoSoA>
inline void deep_copy(
    DstAoSoA& dst,
    const SrcAoSoA& src,
    typename std::enable_if<(is_aosoa<DstAoSoA>::value &&
                             is_aosoa<SrcAoSoA>::value)>::type *  = 0 )
{
    using dst_type = DstAoSoA;
    using src_type = SrcAoSoA;
    using dst_memory_space = typename dst_type::traits::memory_space;
    using src_memory_space = typename src_type::traits::memory_space;
    using dst_soa_type = typename dst_type::soa_type;
    using src_soa_type = typename src_type::soa_type;
    using dst_array_layout = typename dst_type::traits::inner_array_layout;
    using src_array_layout = typename src_type::traits::inner_array_layout;

    // Check that the data types are the same.
    static_assert(
        std::is_same<typename dst_type::member_types,
        typename src_type::member_types>::value,
        "Attempted to deep copy AoSoA objects of different member types" );

    // Check for the same number of values.
    if ( dst.size() != src.size() )
    {
        throw std::runtime_error(
            "Attempted to deep copy AoSoA objects of different sizes" );
    }

    // Get the pointers to the beginning of the data blocks.
    void* dst_data = dst.ptr();
    const void* src_data = src.ptr();

    // Return if both pointers are null.
    if ( dst_data == nullptr && src_data == nullptr )
    {
        Kokkos::fence();
        return;
    }

    // Get the number of SoA's in each object.
    auto dst_num_soa = dst.numSoA();
    auto src_num_soa = src.numSoA();

    // Return if the AoSoA memory occupies the same space.
    if ( (dst_data == src_data) &&
         (dst_num_soa * sizeof(dst_soa_type) ==
          src_num_soa * sizeof(src_soa_type)) )
    {
        Kokkos::fence();
        return;
    }

    // If the inner array size is the same and both AoSoAs have the same number
    // of values then we can do a byte-wise copy directly.
    if ( std::is_same<dst_soa_type,src_soa_type>::value &&
         std::is_same<dst_array_layout,src_array_layout>::value )
    {
        Kokkos::fence();
        Kokkos::Impl::DeepCopy<dst_memory_space,src_memory_space>(
            dst_data, src_data, dst_num_soa * sizeof(dst_soa_type) );
        Kokkos::fence();
    }

    // Otherwise copy the data element-by-element because the data layout is
    // different.
    else
    {
        // Define a AoSoA type in the destination space with the same data
        // layout as the source.
        using src_mirror_type = AoSoA<typename src_type::member_types,
                                      src_array_layout,
                                      dst_memory_space>;
        static_assert(
            std::is_same<src_soa_type,typename src_mirror_type::soa_type>::value,
            "Incompatible source mirror type in destination space" );

        // Create an AoSoA in the destination space with the same data layout
        // as the source.
        src_mirror_type src_copy_on_dst( src.size() );

        // Copy the source to the destination space.
        Kokkos::fence();
        Kokkos::Impl::DeepCopy<dst_memory_space,src_memory_space>(
            src_copy_on_dst.ptr(),
            src_data,
            src_num_soa * sizeof(src_soa_type) );
        Kokkos::fence();

        // Iterate through members and copy the source copy to the destination
        // in the destination memory space.
        Impl::DeepCopyByMember<dst_type::number_of_members-1,
                               DstAoSoA,
                               src_mirror_type>::copy( dst, src_copy_on_dst );
        Kokkos::fence();
    }
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_DEEPCOPY_HPP
