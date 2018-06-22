#ifndef CABANA_SORT_HPP
#define CABANA_SORT_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberDataTypes.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <type_traits>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
// Sort members of an AoSoA recursively.
template<class AoSoA_t, class BinSort>
void sortMember( AoSoA_t aosoa,
                 const BinSort& bin_sort,
                 const int begin,
                 const int end,
                 MemberTag<0> )
{
    auto member = slice<0>( aosoa );
    bin_sort.sort( member, begin, end );
}

template<std::size_t M, class AoSoA_t, class BinSort>
void sortMember( AoSoA_t aosoa,
                 const BinSort& bin_sort,
                 const int begin,
                 const int end,
                 MemberTag<M> )
{
    static_assert( 0 <= M && M < AoSoA_t::number_of_members,
                   "Static loop out of bounds!" );
    auto member = slice<M>( aosoa );
    bin_sort.sort( member, begin, end );
    sortMember( aosoa, bin_sort, begin, end,
                MemberTag<M-1>() );
}

//---------------------------------------------------------------------------//
// Sort an AoSoA over a subset of its range using a comparator over the given
// Kokkos View of keys.
template<class AoSoA_t, class KeyViewType, class Comparator>
void kokkosBinSort(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    const bool sort_within_bins,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    Kokkos::BinSort<KeyViewType,Comparator> bin_sort(
        keys, begin, end, comp, sort_within_bins );
    bin_sort.create_permute_vector();
    sortMember(
        aosoa, bin_sort, begin, end,
        MemberTag<AoSoA_t::number_of_members-1>() );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// Given a set of keys, find the minimum and maximum over the given range.
template<class KeyViewType>
Kokkos::MinMaxScalar<typename KeyViewType::non_const_value_type>
keyMinMax( KeyViewType keys, const int begin, const int end )
{
    Kokkos::MinMaxScalar<typename KeyViewType::non_const_value_type> result;
    Kokkos::MinMax<typename KeyViewType::non_const_value_type> reducer(result);
    Kokkos::parallel_reduce(
        "Cabana::keyMinMax",
        Kokkos::RangePolicy<typename KeyViewType::execution_space>(begin,end),
        Kokkos::Impl::min_max_functor<KeyViewType>(keys),
        reducer );
    Kokkos::fence();
    return result;
}

//---------------------------------------------------------------------------//
// Sort an AoSoA over a subset of its range using the given Kokkos View of
// keys.
template<class AoSoA_t, class KeyViewType>
void kokkosBinSort1d(
    AoSoA_t aosoa,
    KeyViewType keys,
    const int nbin,
    const bool sort_within_bins,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    // Find the minimum and maximum key values.
    auto key_bounds = Impl::keyMinMax( keys, begin, end );

    // If the min and max keys are the same then there is no need to sort.
    if( key_bounds.min_val == key_bounds.max_val ) return;

    // Create a sorting comparator.
    Kokkos::BinOp1D<KeyViewType> comp(
        nbin, key_bounds.min_val, key_bounds.max_val );

    // BinSort
    kokkosBinSort( aosoa, keys, comp, sort_within_bins, begin, end );
}

//---------------------------------------------------------------------------//
// Copy the a 1D slice into a Kokkos view.
template<class SliceType>
Kokkos::View<typename SliceType::data_type, typename SliceType::memory_space>
copySliceToKeys( SliceType slice )
{
    using KeyViewType = Kokkos::View<typename SliceType::data_type,
                                     typename SliceType::memory_space>;
    KeyViewType keys( "slice_keys", slice.extent(0) );
    Kokkos::RangePolicy<typename SliceType::execution_space>
        exec_policy( 0, slice.extent(0) );
    auto copy_op = KOKKOS_LAMBDA( const int i ) { keys(i) = slice(i); };
    Kokkos::parallel_for( "Cabana::copySliceToKeys::copy_op",
                          exec_policy,
                          copy_op );
    Kokkos::fence();
    return keys;
}

//---------------------------------------------------------------------------//

} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Sort an AoSoA over a subset of its range using a general comparator
  over the given Kokkos View of keys.

  \tparam AoSoA_t The AoSoA type to sort.

  \tparam KeyViewType The Kokkos::View type for keys.

  \tparam Comparator Kokkos::BinSort compatible comparator type.

  \param aosoa The AoSoA to sort.

  \param keys The key values to use for sorting. A key value is needed for
  every element of the AoSoA.

  \param comp The comparator to use for sorting. Must be compatible with
  Kokkos::BinSort.

  \param begin The begining index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.
*/
template<class AoSoA_t, class KeyViewType, class Comparator>
void sortByKeyWithComparator(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    Impl::kokkosBinSort( aosoa, keys, comp, true, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an entire AoSoA using a general comparator over the given
  Kokkos View of keys.

  \tparam AoSoA_t The AoSoA type to sort.

  \tparam KeyViewType The Kokkos::View type for keys.

  \tparam Comparator Kokkos::BinSort compatible comparator type.

  \param aosoa The AoSoA to sort.

  \param keys The key values to use for sorting. A key value is needed for
  every element of the AoSoA.

  \param comp The comparator to use for sorting. Must be compatible with
  Kokkos::BinSort.
*/
template<class AoSoA_t, class KeyViewType, class Comparator>
void sortByKeyWithComparator(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    Impl::kokkosBinSort( aosoa, keys, comp, true, 0, aosoa.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA over a subset of its range using a general comparator
  over the given Kokkos View of keys.

  \tparam AoSoA_t The AoSoA type to bin.

  \tparam KeyViewType The Kokkos::View type for keys.

  \tparam Comparator Kokkos::BinSort compatible comparator type.

  \param aosoa The AoSoA to bin.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param comp The comparator to use for binning. Must be compatible with
  Kokkos::BinSort.

  \param begin The begining index of the AoSoA range to bin.

  \param end The end index of the AoSoA range to bin.
*/
template<class AoSoA_t, class KeyViewType, class Comparator>
void binByKeyWithComparator(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    Impl::kokkosBinSort( aosoa, keys, comp, false, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA using a general comparator over the given Kokkos
  View of keys.

  \tparam AoSoA_t The AoSoA type to bint.

  \tparam KeyViewType The Kokkos::View type for keys.

  \tparam Comparator Kokkos::BinSort compatible comparator type.

  \param aosoa The AoSoA to bin.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param comp The comparator to use for binning. Must be compatible with
  Kokkos::BinSort.
*/
template<class AoSoA_t, class KeyViewType, class Comparator>
void binByKeyWithComparator(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    Impl::kokkosBinSort( aosoa, keys, comp, false, 0, aosoa.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an AoSoA over a subset of its range based on the associated key
  values.

  \tparam AoSoA_t The AoSoA type to sort.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param aosoa The AoSoA to sort.

  \param keys The key values to use for sorting. A key value is needed for
  every element of the AoSoA.

  \param begin The begining index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.
*/
template<class AoSoA_t, class KeyViewType>
void sortByKey(
    AoSoA_t aosoa,
    KeyViewType keys,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    int nbin = (end - begin) / 2;
    Impl::kokkosBinSort1d( aosoa, keys, nbin, true, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an entire AoSoA based on the associated key values.

  \tparam AoSoA_t The AoSoA type to sort.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param aosoa The AoSoA to sort.

  \param keys The key values to use for sorting. A key value is needed for
  every element of the AoSoA.
*/
template<class AoSoA_t, class KeyViewType>
void sortByKey(
    AoSoA_t aosoa,
    KeyViewType keys,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    sortByKey( aosoa, keys, 0, aosoa.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA over a subset of its range based on the associated key
  values and number of bins. The bins are evenly divided over the range of key
  values.

  \tparam AoSoA_t The AoSoA type to bin.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param aosoa The AoSoA to bin.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.

  \param begin The begining index of the AoSoA range to bin.

  \param end The end index of the AoSoA range to bin.
*/
template<class AoSoA_t, class KeyViewType>
void binByKey(
    AoSoA_t aosoa,
    KeyViewType keys,
    const int nbin,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    Impl::kokkosBinSort1d( aosoa, keys, nbin, false, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA based on the associated key values and number of
  bins. The bins are evenly divided over the range of key values.

  \tparam AoSoA_t The AoSoA type to bin.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param aosoa The AoSoA to bint.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.
*/
template<class AoSoA_t, class KeyViewType>
void binByKey(
    AoSoA_t aosoa,
    KeyViewType keys,
    const int nbin,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    Impl::kokkosBinSort1d( aosoa, keys, nbin, false, 0, aosoa.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an AoSoA over a subset of its range based on the associated
  slice values.

  \tparam Member Member index of the key values.

  \tparam AoSoA_t The AoSoA type to sort.

  \param aosoa The AoSoA to sort.

  \param begin The begining index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.
*/
template<std::size_t Member, class AoSoA_t>
void sortByMember(
    AoSoA_t aosoa,
    MemberTag<Member> member_tag,
    const int begin,
    const int end,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    std::ignore = member_tag;
    auto keys = Impl::copySliceToKeys( slice<Member>(aosoa) );
    sortByKey( aosoa, keys, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an entire AoSoA based on the associated slice values.

  \tparam Member Member index of the key values.

  \tparam AoSoA_t The AoSoA type to sort.

  \param aosoa The AoSoA to sort.
*/
template<std::size_t Member, class AoSoA_t>
void sortByMember(
    AoSoA_t aosoa,
    MemberTag<Member> member_tag,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    sortByMember<Member>( aosoa, member_tag, 0, aosoa.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA over a subset of its range based on the associated
  slice values.

  \tparam Member Member index of the key values.

  \tparam AoSoA_t The AoSoA type to bin.

  \param aosoa The AoSoA to bin.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.

  \param begin The begining index of the AoSoA range to bin.

  \param end The end index of the AoSoA range to bin.
*/
template<std::size_t Member, class AoSoA_t>
void binByMember(
    AoSoA_t aosoa,
    MemberTag<Member> member_tag,
    const int nbin,
    const int begin,
    const int end,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    std::ignore = member_tag;
    auto keys = Impl::copySliceToKeys( slice<Member>(aosoa) );
    binByKey( aosoa, keys, nbin, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA based on the associated slice values.

  \tparam Member Member index of the key values.

  \tparam AoSoA_t The AoSoA type to bin.

  \param aosoa The AoSoA to bin.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.
*/
template<std::size_t Member, class AoSoA_t>
void binByMember(
    AoSoA_t aosoa,
    MemberTag<Member> member_tag,
    const int nbin,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    binByMember<Member>( aosoa, member_tag, nbin, 0, aosoa.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA spatially over a subset of its range within a given
  structured Cartesian grid.

  \tparam PositionMember The member index for particle positions in the
  AoSoA.

  \tparam AoSoA_t The AoSoA type to sort.

  \param aosoa The AoSoA to sort.

  \param position_member Tag for the AoSoA member containing position.

  \param begin The begining index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.

  \param grid_dx Bin size in the x direction.

  \param grid_dy Bin size in the y direction.

  \param grid_dz Bin size in the z direction.

  \param grid_x_min Lower grid bound in the x direction.

  \param grid_y_min Lower grid bound in the y direction.

  \param grid_z_min Lower grid bound in the z direction.

  \param grid_x_max Upper grid bound in the x direction.

  \param grid_y_may Upper grid bound in the y direction.

  \param grid_z_maz Upper grid bound in the z direction.
*/
template<class AoSoA_t, std::size_t PositionMember>
void binByRegularGrid3d(
    AoSoA_t aosoa,
    MemberTag<PositionMember> position_member,
    const int begin,
    const int end,
    const double grid_dx,
    const double grid_dy,
    const double grid_dz,
    const double grid_x_min,
    const double grid_y_min,
    const double grid_z_min,
    const double grid_x_max,
    const double grid_y_max,
    const double grid_z_max,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )

{
    std::ignore = position_member;

    // Get the positions.
    auto position = slice<PositionMember>( aosoa );
    using PositionSlice = decltype(position);

    // Copy the positions into a Kokkos view. For now we need to do this
    // because of internal copy constructors being called within
    // Kokkos::BinSort.
    using KeyViewType = Kokkos::View<typename PositionSlice::data_type,
                                     typename PositionSlice::memory_space>;
    KeyViewType keys( "position_bin_keys", position.extent(0) );
    Kokkos::RangePolicy<typename PositionSlice::execution_space>
        exec_policy( 0, position.extent(0) );
    auto copy_op = KOKKOS_LAMBDA( const int i )
                   { for ( int d = 0; d < 3; ++d ) keys(i,d) = position(i,d); };
    Kokkos::parallel_for( "Cabana::binByRegularGrid3d::copy_op",
                          exec_policy,
                          copy_op );
    Kokkos::fence();

    // Create a binning operator.
    int nbin[3] =
        { static_cast<int>(std::floor((grid_x_max-grid_x_min) / grid_dx)),
          static_cast<int>(std::floor((grid_y_max-grid_y_min) / grid_dy)),
          static_cast<int>(std::floor((grid_z_max-grid_z_min) / grid_dz)) };
    typename PositionSlice::value_type key_min[3] =
        { grid_x_min, grid_y_min, grid_z_min };
    typename PositionSlice::value_type key_max[3] =
        { grid_x_max, grid_y_max, grid_z_max };
    Kokkos::BinOp3D<KeyViewType> comp( nbin, key_min, key_max );

    // Do the binning.
    Impl::kokkosBinSort( aosoa, keys, comp, false, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA spatially within a given structured Cartesian
  grid.

  \tparam PositionMember The member index for particle positions in the
  AoSoA.

  \tparam AoSoA_t The AoSoA type to sort.

  \param aosoa The AoSoA to sort.

  \param position_member Tag for the AoSoA member containing position.

  \param grid_dx Bin size in the x direction.

  \param grid_dy Bin size in the y direction.

  \param grid_dz Bin size in the z direction.

  \param grid_x_min Lower grid bound in the x direction.

  \param grid_y_min Lower grid bound in the y direction.

  \param grid_z_min Lower grid bound in the z direction.

  \param grid_x_max Upper grid bound in the x direction.

  \param grid_y_may Upper grid bound in the y direction.

  \param grid_z_maz Upper grid bound in the z direction.
*/
template<class AoSoA_t, std::size_t PositionMember>
void binByRegularGrid3d(
    AoSoA_t aosoa,
    MemberTag<PositionMember> position_member,
    const double grid_dx,
    const double grid_dy,
    const double grid_dz,
    const double grid_x_min,
    const double grid_y_min,
    const double grid_z_min,
    const double grid_x_max,
    const double grid_y_max,
    const double grid_z_max,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    binByRegularGrid3d( aosoa, position_member,
                        0, aosoa.size(),
                        grid_dx, grid_dy, grid_dz,
                        grid_x_min, grid_y_min, grid_z_min,
                        grid_x_max, grid_y_max, grid_z_max );
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_SORT_HPP
