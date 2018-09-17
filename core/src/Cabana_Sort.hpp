/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_SORT_HPP
#define CABANA_SORT_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>
#include <Cabana_DeepCopy.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class BinningData
  \brief Data describing the bin sizes and offsets resulting from a binning
  operation.
*/
template<class KokkosMemorySpace>
class BinningData
{
  public:

    using memory_space = KokkosMemorySpace;
    using size_type = typename memory_space::size_type;
    using CountView = Kokkos::View<const int*,KokkosMemorySpace>;
    using OffsetView = Kokkos::View<size_type*,KokkosMemorySpace>;

    BinningData()
        : _nbin(0)
    {}

    BinningData( CountView counts,
                 OffsetView offsets,
                 OffsetView permute_vector )
        : _nbin( counts.extent(0) )
        , _counts( counts )
        , _offsets( offsets )
        , _permute_vector( permute_vector )
    {}

    /*!
      \brief Get the number of bins.
      \return The number of bins.
    */
    KOKKOS_INLINE_FUNCTION
    int numBin() const
    { return _nbin; }

    /*!
      \brief Given a bin get the number of particles it contains.
      \param bin_id The bin id.
      \return The number of particles in the bin.
    */
    KOKKOS_INLINE_FUNCTION
    int binSize( const size_type bin_id ) const
    { return _counts( bin_id ); }

    /*!
      \brief Given a bin get the particle index at which it sorts.
      \param bin_id The bin id.
      \return The starting particle index of the bin.
    */
    KOKKOS_INLINE_FUNCTION
    size_type binOffset( const size_type bin_id ) const
    { return _offsets( bin_id ); }

    /*!
      \brief Given a local particle id in the binned layout, get the id of the
      particle in the old (unbinned) layout.
    */
    KOKKOS_INLINE_FUNCTION
    size_type permutation( const size_type particle_id ) const
    { return _permute_vector(particle_id); }

    /*!
      \brief Get the entire bin count view.
    */
    CountView binCountView() const
    { return _counts; }

    /*!
      \brief Get the entire bin offset view.
    */
    OffsetView binOffsetView() const
    { return _offsets; }

    /*!
      \brief Get the entire permutation vector.
    */
    OffsetView permuteVector() const
    { return _permute_vector; }

  private:

    int _nbin;
    CountView _counts;
    OffsetView _offsets;
    OffsetView _permute_vector;
};

//---------------------------------------------------------------------------//
/*!
  \class LinkedCellList
  \brief Data describing the bin sizes and offsets resulting from a binning
  operation on a 3d regular Cartesian grid.
*/
template<class KokkosMemorySpace>
class LinkedCellList
{
  public:

    using memory_space = KokkosMemorySpace;
    using size_type = typename memory_space::size_type;
    using OffsetView = Kokkos::View<size_type*,KokkosMemorySpace>;

    /*!
      \brief Default constructor.
    */
    LinkedCellList()
    {
        _nbin[0] = 0;
        _nbin[1] = 0;
        _nbin[2] = 0;
    }

    /*!
      \brief Constructor
    */
    LinkedCellList( BinningData<KokkosMemorySpace> bin_data_1d,
                    const int nbin[3] )
        : _bin_data( bin_data_1d )
    {
        _nbin[0] = nbin[0];
        _nbin[1] = nbin[1];
        _nbin[2] = nbin[2];
    }

    /*!
      \brief Copy constructor.
    */
    LinkedCellList( const LinkedCellList& data )
    {
        _bin_data = data._bin_data;
        _nbin[0] = data._nbin[0];
        _nbin[1] = data._nbin[1];
        _nbin[2] = data._nbin[2];
    }

    /*!
      \brief Assignment operators.
    */
    LinkedCellList&
    operator=( const LinkedCellList& data )
    {
        _bin_data = data._bin_data;
        _nbin[0] = data._nbin[0];
        _nbin[1] = data._nbin[1];
        _nbin[2] = data._nbin[2];
        return *this;
    }

    /*!
      \brief Get the total number of bins.
      \return the total number of bins.
    */
    KOKKOS_INLINE_FUNCTION
    int totalBins() const
    { return _nbin[0] * _nbin[1] * _nbin[2]; }

    /*!
      \brief Get the number of bins in a given dimension.
      \param dim The dimension to get the number of bins for.
      \return The number of bins.
    */
    KOKKOS_INLINE_FUNCTION
    int numBin( const int dim ) const
    { return _nbin[dim]; }

    /*!
      \brief Given the ijk index of a bin get its cardinal index.
      \param i The i bin index (x).
      \param j The j bin index (y).
      \param k The k bin index (z).
      \return The cardinal bin index.

      Note that the Kokkos sort orders the bins such that the i index moves
      the slowest and the k index mvoes the fastest.
    */
    KOKKOS_INLINE_FUNCTION
    size_type cardinalBinIndex( const int i, const int j, const int k ) const
    { return (i * _nbin[1] + j) * _nbin[2] + k; }

    /*!
      \brief Given the cardinal index of a bin get its ijk indices.
      \param cardinal The cardinal bin index.
      \param i The i bin index (x).
      \param j The j bin index (y).
      \param k The k bin index (z).

      Note that the Kokkos sort orders the bins such that the i index moves
      the slowest and the k index mvoes the fastest.
    */
    KOKKOS_INLINE_FUNCTION
    void ijkBinIndex( const int cardinal, int& i, int& j, int& k ) const
    {
        i = cardinal / (_nbin[1]*_nbin[2]);
        j = ( cardinal / _nbin[2] ) % _nbin[1];
        k = cardinal % _nbin[2];
    }

    /*!
      \brief Given a bin get the number of particles it contains.
      \param i The i bin index (x).
      \param j The j bin index (y).
      \param k The k bin index (z).
      \return The number of particles in the bin.
    */
    KOKKOS_INLINE_FUNCTION
    int binSize( const int i, const int j, const int k ) const
    { return _bin_data.binSize(cardinalBinIndex(i,j,k)); }

    /*!
      \brief Given a bin get the particle index at which it sorts.
      \param i The i bin index (x).
      \param j The j bin index (y).
      \param k The k bin index (z).
      \return The starting particle index of the bin.
    */
    KOKKOS_INLINE_FUNCTION
    size_type binOffset( const int i, const int j, const int k ) const
    { return _bin_data.binOffset(cardinalBinIndex(i,j,k)); }

    /*!
      \brief Given a local particle id in the binned layout, get the id of the
      particle in the old (unbinned) layout.
      \param particle_id The id of the particle in the binned layout.
      \return The particle id in the old (unbinned) layout.
    */
    KOKKOS_INLINE_FUNCTION
    size_type permutation( const int particle_id ) const
    { return _bin_data.permutation(particle_id); }

    /*!
      \brief Get the entire permutation vector.
    */
    OffsetView permuteVector() const
    { return _bin_data.permuteVector(); }

    /*!
      \brief Get the 1d bin data.
      \return The 1d bin data.
    */
    BinningData<KokkosMemorySpace> data1d() const
    { return _bin_data; }

  private:

    BinningData<KokkosMemorySpace> _bin_data;
    int _nbin[3];
};

namespace Impl
{
//---------------------------------------------------------------------------//
// Create a permutation vector over a range subset using a comparator over the
// given Kokkos View of keys.
template<class KeyViewType, class Comparator>
BinningData<typename KeyViewType::memory_space>
kokkosBinSort( KeyViewType keys,
               Comparator comp,
               const bool sort_within_bins,
               const std::size_t begin,
               const std::size_t end )
{
    Kokkos::BinSort<KeyViewType,Comparator> bin_sort(
        keys, begin, end, comp, sort_within_bins );
    bin_sort.create_permute_vector();
    return BinningData<typename KeyViewType::memory_space>(
        bin_sort.get_bin_count(),
        bin_sort.get_bin_offsets(),
        bin_sort.get_permute_vector() );
}

//---------------------------------------------------------------------------//
// Given a set of keys, find the minimum and maximum over the given range.
template<class KeyViewType>
Kokkos::MinMaxScalar<typename KeyViewType::non_const_value_type>
keyMinMax( KeyViewType keys, const std::size_t begin, const std::size_t end )
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
template<class KeyViewType>
BinningData<typename KeyViewType::memory_space>
kokkosBinSort1d( KeyViewType keys,
                 const int nbin,
                 const bool sort_within_bins,
                 const std::size_t begin,
                 const std::size_t end )
{
    // Find the minimum and maximum key values.
    auto key_bounds = Impl::keyMinMax( keys, begin, end );

    // Create a sorting comparator.
    Kokkos::BinOp1D<KeyViewType> comp(
        nbin, key_bounds.min_val, key_bounds.max_val );

    // BinSort
    return kokkosBinSort( keys, comp, sort_within_bins, begin, end );
}

//---------------------------------------------------------------------------//
// Copy the a 1D slice into a Kokkos view.
template<class SliceType>
Kokkos::View<typename SliceType::value_type*,
             typename SliceType::kokkos_memory_space>
copySliceToKeys( SliceType slice )
{
    using KeyViewType = Kokkos::View<typename SliceType::value_type*,
                                     typename SliceType::kokkos_memory_space>;
    KeyViewType keys( "slice_keys", slice.size() );
    Kokkos::RangePolicy<typename SliceType::kokkos_execution_space>
        exec_policy( 0, slice.size() );
    auto copy_op = KOKKOS_LAMBDA( const std::size_t i ) { keys(i) = slice(i); };
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

  \tparam KeyViewType The Kokkos::View type for keys.

  \tparam Comparator Kokkos::BinSort compatible comparator type.

  \param keys The key values to use for sorting. A key value is needed for
  every element of the AoSoA.

  \param comp The comparator to use for sorting. Must be compatible with
  Kokkos::BinSort.

  \param begin The beginning index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.

  \return The permutation vector associated with the sorting.
*/
template<class KeyViewType, class Comparator>
typename BinningData<typename KeyViewType::memory_space>::OffsetView
sortByKeyWithComparator( KeyViewType keys,
                         Comparator comp,
                         const std::size_t begin,
                         const std::size_t end )
{
    auto bin_data = Impl::kokkosBinSort( keys, comp, true, begin, end );
    return bin_data.permuteVector();
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an entire AoSoA using a general comparator over the given
  Kokkos View of keys.

  \tparam KeyViewType The Kokkos::View type for keys.

  \tparam Comparator Kokkos::BinSort compatible comparator type.

  \param keys The key values to use for sorting. A key value is needed for
  every element of the AoSoA.

  \param comp The comparator to use for sorting. Must be compatible with
  Kokkos::BinSort.

  \return The permutation vector associated with the sorting.
*/
template<class KeyViewType, class Comparator>
typename BinningData<typename KeyViewType::memory_space>::OffsetView
sortByKeyWithComparator( KeyViewType keys, Comparator comp )
{
    Impl::kokkosBinSort( keys, comp, true, 0, keys.extent(0) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA over a subset of its range using a general comparator
  over the given Kokkos View of keys.

  \tparam KeyViewType The Kokkos::View type for keys.

  \tparam Comparator Kokkos::BinSort compatible comparator type.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param comp The comparator to use for binning. Must be compatible with
  Kokkos::BinSort.

  \param begin The beginning index of the AoSoA range to bin.

  \param end The end index of the AoSoA range to bin.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class KeyViewType, class Comparator>
BinningData<typename KeyViewType::memory_space>
binByKeyWithComparator(
    KeyViewType keys,
    Comparator comp,
    const std::size_t begin,
    const std::size_t end )
{
    return Impl::kokkosBinSort( keys, comp, false, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA using a general comparator over the given Kokkos
  View of keys.

  \tparam KeyViewType The Kokkos::View type for keys.

  \tparam Comparator Kokkos::BinSort compatible comparator type.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param comp The comparator to use for binning. Must be compatible with
  Kokkos::BinSort.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class KeyViewType, class Comparator>
BinningData<typename KeyViewType::memory_space>
binByKeyWithComparator(
    KeyViewType keys,
    Comparator comp )
{
    return Impl::kokkosBinSort( keys, comp, false, 0, keys.extent(0) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an AoSoA over a subset of its range based on the associated key
  values.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param keys The key values to use for sorting. A key value is needed for
  every element of the AoSoA.

  \param begin The beginning index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.

  \return The permutation vector associated with the sorting.
*/
template<class KeyViewType>
typename BinningData<typename KeyViewType::memory_space>::OffsetView
sortByKey( KeyViewType keys, const std::size_t begin, const std::size_t end )
{
    int nbin = (end - begin) / 2;
    auto bin_data =
        Impl::kokkosBinSort1d( keys, nbin, true, begin, end );
    return bin_data.permuteVector();
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an entire AoSoA based on the associated key values.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param keys The key values to use for sorting. A key value is needed for
  every element of the AoSoA.

  \return The permutation vector associated with the sorting.

*/
template<class KeyViewType>
typename BinningData<typename KeyViewType::memory_space>::OffsetView
sortByKey( KeyViewType keys )
{
    return sortByKey( keys, 0, keys.extent(0) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA over a subset of its range based on the associated key
  values and number of bins. The bins are evenly divided over the range of key
  values.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.

  \param begin The beginning index of the AoSoA range to bin.

  \param end The end index of the AoSoA range to bin.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class KeyViewType>
BinningData<typename KeyViewType::memory_space>
binByKey( KeyViewType keys,
          const int nbin,
          const std::size_t begin,
          const std::size_t end )
{
    return Impl::kokkosBinSort1d( keys, nbin, false, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA based on the associated key values and number of
  bins. The bins are evenly divided over the range of key values.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class KeyViewType>
BinningData<typename KeyViewType::memory_space>
binByKey( KeyViewType keys, const int nbin )
{
    return Impl::kokkosBinSort1d( keys, nbin, false, 0, keys.extent(0) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an AoSoA over a subset of its range based on the associated
  slice values.

  \tparam SliceType Slice type for keys.

  \param begin The beginning index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.

  \return The permutation vector associated with the sorting.
*/
template<class SliceType>
typename BinningData<typename SliceType::kokkos_memory_space>::OffsetView
sortByMember( SliceType slice,
              const std::size_t begin,
              const std::size_t end,
    typename std::enable_if<(is_slice<SliceType>::value),int>::type * = 0 )
{
    auto keys = Impl::copySliceToKeys( slice );
    return sortByKey( keys, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an entire AoSoA based on the associated slice values.

  \tparam SliceType Slice type for keys.

  \param slice Slice of keys.

  \return The permutation vector associated with the sorting.
*/
template<class SliceType>
typename BinningData<typename SliceType::kokkos_memory_space>::OffsetView
sortByMember(
    SliceType slice,
    typename std::enable_if<(is_slice<SliceType>::value),int>::type * = 0 )
{
    return sortByMember( slice, 0, slice.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA over a subset of its range based on the associated
  slice values.

  \tparam SliceType Slice type for keys

  \param slice Slice of keys.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.

  \param begin The beginning index of the AoSoA range to bin.

  \param end The end index of the AoSoA range to bin.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class SliceType>
BinningData<typename SliceType::kokkos_memory_space>
binByMember(
    SliceType slice,
    const int nbin,
    const std::size_t begin,
    const std::size_t end,
    typename std::enable_if<(is_slice<SliceType>::value),int>::type * = 0 )
{
    auto keys = Impl::copySliceToKeys( slice );
    return binByKey( keys, nbin, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA based on the associated slice values.

  \tparam SliceType Slice type for keys.

  \param slice Slice of keys.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class SliceType>
BinningData<typename SliceType::kokkos_memory_space>
binByMember(
    SliceType slice,
    const int nbin,
    typename std::enable_if<(is_slice<SliceType>::value),int>::type * = 0 )
{
    return binByMember( slice, nbin, 0, slice.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA spatially over a subset of its range within given
  structured Cartesian grid.

  \tparam SliceType Slice type for positions.

  \param positions Slice of positions.

  \param begin The beginning index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.

  \param grid_delta Grid sizes in each cardinal direction.

  \param grid_min Grid minimum value in each direction.

  \param grid_max Grid maximum value in each direction.
*/
template<class SliceType>
LinkedCellList<typename SliceType::kokkos_memory_space>
buildLinkedCellList(
    SliceType positions,
    const std::size_t begin,
    const std::size_t end,
    const typename SliceType::value_type grid_delta[3],
    const typename SliceType::value_type grid_min[3],
    const typename SliceType::value_type grid_max[3],
    typename std::enable_if<(is_slice<SliceType>::value),int>::type * = 0 )
{
    // Get the positions.
    using PositionValueType = typename SliceType::value_type;

    // Copy the positions into a Kokkos view. For now we need to do this
    // because of internal copy constructors being called within
    // Kokkos::BinSort.
    using KeyViewType =
        Kokkos::View<typename SliceType::value_type**,
                     typename SliceType::kokkos_memory_space>;
    KeyViewType keys(
        "position_bin_keys", positions.size(), positions.extent(2) );
    Kokkos::RangePolicy<typename SliceType::kokkos_execution_space>
        exec_policy( 0, positions.size() );
    auto copy_op = KOKKOS_LAMBDA( const std::size_t i )
                   { for ( int d = 0; d < 3; ++d ) keys(i,d) = positions(i,d); };
    Kokkos::parallel_for( "Cabana::buildLinkedCellList::copy_op",
                          exec_policy,
                          copy_op );
    Kokkos::fence();

    // Create a binning operator.
    int nbin[3] =
        { static_cast<int>(std::floor((grid_max[0]-grid_min[0]) / grid_delta[0])),
          static_cast<int>(std::floor((grid_max[1]-grid_min[1]) / grid_delta[1])),
          static_cast<int>(std::floor((grid_max[2]-grid_min[2]) / grid_delta[2])) };
    Kokkos::BinOp3D<KeyViewType> comp( nbin, grid_min, grid_max );

    // Do the binning.
    auto bin_data_1d = Impl::kokkosBinSort( keys, comp, false, begin, end );

    // Return the bin data.
    return LinkedCellList<
        typename SliceType::kokkos_memory_space>( bin_data_1d, nbin );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA spatially over a subset of its range within given
  structured Cartesian grid.

  \tparam SliceType Slice type for positions.

  \param positions Slice of positions.

  \param grid_delta Grid sizes in each cardinal direction.

  \param grid_min Grid minimum value in each direction.

  \param grid_max Grid maximum value in each direction.
*/
template<class SliceType>
LinkedCellList<typename SliceType::kokkos_memory_space>
buildLinkedCellList(
    SliceType positions,
    const typename SliceType::value_type grid_delta[3],
    const typename SliceType::value_type grid_min[3],
    const typename SliceType::value_type grid_max[3],
    typename std::enable_if<(is_slice<SliceType>::value),int>::type * = 0 )
{
    return buildLinkedCellList( positions,
                                0, positions.size(),
                                grid_delta, grid_min, grid_max );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a permutation vector permute an AoSoA.

  \tparam ViewType The permutation vector view type.

  \tparm AoSoA_t The AoSoA type.

  \param permute_vector The permutation vector.

  \param begin The first index of the AoSoA to sort.

  \param end The last index of the AoSoA to sort.

  \param aosoa The AoSoA to permute.
 */
template<class ViewType, class AoSoA_t>
void permute( const ViewType& permute_vector,
              const std::size_t begin,
              const std::size_t end,
              AoSoA_t& aosoa )
{
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename ViewType::memory_space>
        scratch_tuples( "scratch_tuples", end - begin );

    auto permute_to_scratch = KOKKOS_LAMBDA( const std::size_t i )
                              {
                                  scratch_tuples( i - begin ) =
                                  aosoa.getTuple( permute_vector(i) );
                              };
    Kokkos::parallel_for(
        "Cabana::kokkosBinSort::permute_to_scratch",
        Kokkos::RangePolicy<typename ViewType::execution_space>(begin,end),
        permute_to_scratch );
    Kokkos::fence();

    auto copy_back = KOKKOS_LAMBDA( const std::size_t i )
                     { aosoa.setTuple( i, scratch_tuples(i-begin) ); };
    Kokkos::parallel_for(
        "Cabana::kokkosBinSort::copy_back",
        Kokkos::RangePolicy<typename ViewType::execution_space>(begin,end),
        copy_back );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a permutation vector permute an AoSoA.

  \tparam ViewType The permutation vector view type.

  \tparm AoSoA_t The AoSoA type.

  \param permute_vector The permutation vector.

  \param aosoa The AoSoA to permute.
 */
template<class ViewType, class AoSoA_t>
void permute( const ViewType& permute_vector,
              AoSoA_t& aosoa )
{
    permute( permute_vector, 0, aosoa.size(), aosoa );
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_SORT_HPP
