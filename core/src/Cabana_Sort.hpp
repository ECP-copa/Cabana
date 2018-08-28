#ifndef CABANA_SORT_HPP
#define CABANA_SORT_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberDataTypes.hpp>

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
    int binSize( const int bin_id ) const
    { return _counts( bin_id ); }

    /*!
      \brief Given a bin get the particle index at which it sorts.
      \param bin_id The bin id.
      \return The starting particle index of the bin.
    */
    KOKKOS_INLINE_FUNCTION
    size_type binOffset( const int bin_id ) const
    { return _offsets( bin_id ); }

    /*!
      \brief Given a local particle id in the binned layout, get the id of the
      particle in the old (unbinned) layout.
    */
    KOKKOS_INLINE_FUNCTION
    int permutation( const int particle_id ) const
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
  \class CartesianGrid3dBinningData
  \brief Data describing the bin sizes and offsets resulting from a binning
  operation on a 3d regular Cartesian grid.
*/
template<class KokkosMemorySpace>
class CartesianGrid3dBinningData
{
  public:

    using memory_space = KokkosMemorySpace;
    using size_type = typename memory_space::size_type;

    /*!
      \brief Default constructor.
    */
    CartesianGrid3dBinningData()
    {
        _nbin[0] = 0;
        _nbin[1] = 0;
        _nbin[2] = 0;
    }

    /*!
      \brief Constructor
    */
    CartesianGrid3dBinningData( BinningData<KokkosMemorySpace> bin_data_1d,
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
    CartesianGrid3dBinningData( const CartesianGrid3dBinningData& data )
    {
        _bin_data = data._bin_data;
        _nbin[0] = data._nbin[0];
        _nbin[1] = data._nbin[1];
        _nbin[2] = data._nbin[2];
    }

    /*!
      \brief Assignment operators.
    */
    CartesianGrid3dBinningData&
    operator=( const CartesianGrid3dBinningData& data )
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
    int cardinalBinIndex( const int i, const int j, const int k ) const
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
    int permutation( const int particle_id ) const
    { return _bin_data.permutation(particle_id); }

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
// Sort an AoSoA over a subset of its range using a comparator over the given
// Kokkos View of keys.
template<class AoSoA_t, class KeyViewType, class Comparator>
BinningData<typename KeyViewType::memory_space>
kokkosBinSort(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    const bool create_data_only,
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
    auto permute_vector = bin_sort.get_permute_vector();

    if ( !create_data_only )
    {
        Kokkos::View<typename AoSoA_t::particle_type*,
                     typename KeyViewType::memory_space>
            scratch_particles( "scratch_particles", end - begin );

        auto permute_to_scratch = KOKKOS_LAMBDA( const int i )
                                  {
                                      scratch_particles( i - begin ) =
                                      aosoa.getParticle( permute_vector(i) );
                                  };
        Kokkos::parallel_for(
            "Cabana::kokkosBinSort::permute_to_scratch",
            Kokkos::RangePolicy<typename KeyViewType::execution_space>(begin,end),
            permute_to_scratch );
        Kokkos::fence();

        auto copy_back = KOKKOS_LAMBDA( const int i )
                         { aosoa.setParticle( i, scratch_particles(i-begin) ); };
        Kokkos::parallel_for(
            "Cabana::kokkosBinSort::copy_back",
            Kokkos::RangePolicy<typename KeyViewType::execution_space>(begin,end),
            copy_back );
        Kokkos::fence();
    }

    return BinningData<typename KeyViewType::memory_space>(
        bin_sort.get_bin_count(),
        bin_sort.get_bin_offsets(),
        bin_sort.get_permute_vector() );
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
BinningData<typename KeyViewType::memory_space>
kokkosBinSort1d(
    AoSoA_t aosoa,
    KeyViewType keys,
    const int nbin,
    const bool create_data_only,
    const bool sort_within_bins,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    // Find the minimum and maximum key values.
    auto key_bounds = Impl::keyMinMax( keys, begin, end );

    // Create a sorting comparator.
    Kokkos::BinOp1D<KeyViewType> comp(
        nbin, key_bounds.min_val, key_bounds.max_val );

    // BinSort
    return kokkosBinSort( aosoa, keys, comp, create_data_only,
                          sort_within_bins, begin, end );
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

  \param create_permute_vector_only True if the permutation vector should be
  created but the data shouldn't actually be sorted.

  \param begin The beginning index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.

  \return The permutation vector associated with the sorting.
*/
template<class AoSoA_t, class KeyViewType, class Comparator>
typename BinningData<typename KeyViewType::memory_space>::OffsetView
sortByKeyWithComparator(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    const bool create_permute_vector_only,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    auto bin_data =
        Impl::kokkosBinSort( aosoa, keys, comp, create_permute_vector_only,
                             true, begin, end );
    return bin_data.permuteVector();
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

  \param create_permute_vector_only True if the permutation vector should be
  created but the data shouldn't actually be sorted.

  \return The permutation vector associated with the sorting.
*/
template<class AoSoA_t, class KeyViewType, class Comparator>
typename BinningData<typename KeyViewType::memory_space>::OffsetView
sortByKeyWithComparator(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    const bool create_permute_vector_only,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    Impl::kokkosBinSort( aosoa, keys, comp, create_permute_vector_only,
                         true, 0, aosoa.size() );
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

  \param create_data_only True if the binning data should be created (i.e. bin
  sizes, offsets, and permutation vector) but the particles should not
  actually be binned.

  \param begin The beginning index of the AoSoA range to bin.

  \param end The end index of the AoSoA range to bin.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class AoSoA_t, class KeyViewType, class Comparator>
BinningData<typename KeyViewType::memory_space>
binByKeyWithComparator(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    const bool create_data_only,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    return Impl::kokkosBinSort(
        aosoa, keys, comp, create_data_only, false, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA using a general comparator over the given Kokkos
  View of keys.

  \tparam AoSoA_t The AoSoA type to bin.

  \tparam KeyViewType The Kokkos::View type for keys.

  \tparam Comparator Kokkos::BinSort compatible comparator type.

  \param aosoa The AoSoA to bin.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param comp The comparator to use for binning. Must be compatible with
  Kokkos::BinSort.

  \param create_data_only True if the binning data should be created (i.e. bin
  sizes, offsets, and permutation vector) but the particles should not
  actually be binned.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class AoSoA_t, class KeyViewType, class Comparator>
BinningData<typename KeyViewType::memory_space>
binByKeyWithComparator(
    AoSoA_t aosoa,
    KeyViewType keys,
    Comparator comp,
    const bool create_data_only,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    return Impl::kokkosBinSort(
        aosoa, keys, comp, create_data_only, false, 0, aosoa.size() );
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

  \param create_permute_vector_only True if the permutation vector should be
  created but the data shouldn't actually be sorted.

  \param begin The beginning index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.

  \return The permutation vector associated with the sorting.
*/
template<class AoSoA_t, class KeyViewType>
typename BinningData<typename KeyViewType::memory_space>::OffsetView
sortByKey(
    AoSoA_t aosoa,
    KeyViewType keys,
    const bool create_permute_vector_only,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    int nbin = (end - begin) / 2;
    auto bin_data =
        Impl::kokkosBinSort1d( aosoa, keys, nbin, create_permute_vector_only,
                               true, begin, end );
    return bin_data.permuteVector();
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an entire AoSoA based on the associated key values.

  \tparam AoSoA_t The AoSoA type to sort.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param aosoa The AoSoA to sort.

  \param keys The key values to use for sorting. A key value is needed for
  every element of the AoSoA.

  \param create_permute_vector_only True if the permutation vector should be
  created but the data shouldn't actually be sorted.

  \return The permutation vector associated with the sorting.

*/
template<class AoSoA_t, class KeyViewType>
typename BinningData<typename KeyViewType::memory_space>::OffsetView
sortByKey(
    AoSoA_t aosoa,
    KeyViewType keys,
    const bool create_permute_vector_only,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    return sortByKey(
        aosoa, keys, create_permute_vector_only, 0, aosoa.size() );
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

  \param create_data_only True if the binning data should be created (i.e. bin
  sizes, offsets, and permutation vector) but the particles should not
  actually be binned.

  \param begin The beginning index of the AoSoA range to bin.

  \param end The end index of the AoSoA range to bin.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class AoSoA_t, class KeyViewType>
BinningData<typename KeyViewType::memory_space>
binByKey(
    AoSoA_t aosoa,
    KeyViewType keys,
    const int nbin,
    const bool create_data_only,
    const int begin,
    const int end,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    return Impl::kokkosBinSort1d(
        aosoa, keys, nbin, create_data_only, false, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA based on the associated key values and number of
  bins. The bins are evenly divided over the range of key values.

  \tparam AoSoA_t The AoSoA type to bin.

  \tparam KeyViewType The Kokkos::View type for keys.

  \param aosoa The AoSoA to bin.

  \param keys The key values to use for binning. A key value is needed for
  every element of the AoSoA.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.

  \param create_data_only True if the binning data should be created (i.e. bin
  sizes, offsets, and permutation vector) but the particles should not
  actually be binned.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<class AoSoA_t, class KeyViewType>
BinningData<typename KeyViewType::memory_space>
binByKey(
    AoSoA_t aosoa,
    KeyViewType keys,
    const int nbin,
    const bool create_data_only,
    typename std::enable_if<(
        is_aosoa<AoSoA_t>::value && Kokkos::is_view<KeyViewType>::value),
    int>::type * = 0 )
{
    return Impl::kokkosBinSort1d(
        aosoa, keys, nbin, create_data_only, false, 0, aosoa.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an AoSoA over a subset of its range based on the associated
  slice values.

  \tparam Member Member index of the key values.

  \tparam AoSoA_t The AoSoA type to sort.

  \param aosoa The AoSoA to sort.

  \param member_tag Tag indicating which member to use as the keys for sorting.

  \param create_permute_vector_only True if the permutation vector should be
  created but the data shouldn't actually be sorted.

  \param begin The beginning index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.

  \return The permutation vector associated with the sorting.
*/
template<std::size_t Member, class AoSoA_t>
typename BinningData<
    typename AoSoA_t::memory_space::kokkos_memory_space>::OffsetView
sortByMember(
    AoSoA_t aosoa,
    MemberTag<Member> member_tag,
    const bool create_permute_vector_only,
    const int begin,
    const int end,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    auto keys = Impl::copySliceToKeys( aosoa.view(member_tag) );
    return sortByKey( aosoa, keys, create_permute_vector_only, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Sort an entire AoSoA based on the associated slice values.

  \tparam Member Member index of the key values.

  \tparam AoSoA_t The AoSoA type to sort.

  \param aosoa The AoSoA to sort.

  \param member_tag Tag indicating which member to use as the keys for sorting.

  \param create_permute_vector_only True if the permutation vector should be
  created but the data shouldn't actually be sorted.

  \return The permutation vector associated with the sorting.
*/
template<std::size_t Member, class AoSoA_t>
typename BinningData<
    typename AoSoA_t::memory_space::kokkos_memory_space>::OffsetView
sortByMember(
    AoSoA_t aosoa,
    MemberTag<Member> member_tag,
    const bool create_permute_vector_only,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    return sortByMember( aosoa, member_tag, create_permute_vector_only,
                         0, aosoa.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA over a subset of its range based on the associated
  slice values.

  \tparam Member Member index of the key values.

  \tparam AoSoA_t The AoSoA type to bin.

  \param aosoa The AoSoA to bin.

  \param member_tag Tag indicating which member to use as the keys for
  binning.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.

  \param create_data_only True if the binning data should be created (i.e. bin
  sizes, offsets, and permutation vector) but the particles should not
  actually be binned.

  \param begin The beginning index of the AoSoA range to bin.

  \param end The end index of the AoSoA range to bin.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<std::size_t Member, class AoSoA_t>
BinningData<typename AoSoA_t::memory_space::kokkos_memory_space>
binByMember(
    AoSoA_t aosoa,
    MemberTag<Member> member_tag,
    const int nbin,
    const bool create_data_only,
    const int begin,
    const int end,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    auto keys = Impl::copySliceToKeys( aosoa.view(member_tag) );
    return binByKey( aosoa, keys, nbin, create_data_only, begin, end );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an entire AoSoA based on the associated slice values.

  \tparam Member Member index of the key values.

  \tparam AoSoA_t The AoSoA type to bin.

  \param aosoa The AoSoA to bin.

  \param member_tag Tag indicating which member to use as the keys for
  binning.

  \param nbin The number of bins to use for binning. The range of key values
  will subdivided equally by the number of bins.

  \param create_data_only True if the binning data should be created (i.e. bin
  sizes, offsets, and permutation vector) but the particles should not
  actually be binned.

  \return The binning data (e.g. bin sizes and offsets).
*/
template<std::size_t Member, class AoSoA_t>
BinningData<typename AoSoA_t::memory_space::kokkos_memory_space>
binByMember(
    AoSoA_t aosoa,
    MemberTag<Member> member_tag,
    const int nbin,
    const bool create_data_only,
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    return binByMember(
        aosoa, member_tag, nbin, create_data_only, 0, aosoa.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA spatially over a subset of its range within given
  structured Cartesian grid.

  \tparam PositionMember The member index for particle positions in the
  AoSoA.

  \tparam AoSoA_t The AoSoA type to sort.

  \param aosoa The AoSoA to sort.

  \param position_member Tag for the AoSoA member containing position.

  \param create_data_only True if the binning data should be created (i.e. bin
  sizes, offsets, and permutation vector) but the particles should not
  actually be binned.

  \param begin The beginning index of the AoSoA range to sort.

  \param end The end index of the AoSoA range to sort.

  \param grid_delta Grid sizes in each cardinal direction.

  \param grid_min Grid minimum value in each direction.

  \param grid_max Grid maximum value in each direction.
*/
template<class AoSoA_t, std::size_t PositionMember>
CartesianGrid3dBinningData<typename AoSoA_t::memory_space::kokkos_memory_space>
binByCartesianGrid3d(
    AoSoA_t aosoa,
    MemberTag<PositionMember> position_member,
    const bool create_data_only,
    const int begin,
    const int end,
    const typename AoSoA_t::template member_value_type<PositionMember>
    grid_delta[3],
    const typename AoSoA_t::template member_value_type<PositionMember>
    grid_min[3],
    const typename AoSoA_t::template member_value_type<PositionMember>
    grid_max[3],
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    // Get the positions.
    auto position = aosoa.view( position_member );
    using PositionSlice = decltype(position);
    using PositionValueType = typename PositionSlice::value_type;

    // Copy the positions into a Kokkos view. For now we need to do this
    // because of internal copy constructors being called within
    // Kokkos::BinSort.
    using KeyViewType =
        Kokkos::View<typename PositionSlice::value_type**,
                     typename PositionSlice::kokkos_memory_space>;
    KeyViewType keys(
        "position_bin_keys", position.size(), position.fieldExtent(0) );
    Kokkos::RangePolicy<typename PositionSlice::kokkos_execution_space>
        exec_policy( 0, position.size() );
    auto copy_op = KOKKOS_LAMBDA( const int i )
                   { for ( int d = 0; d < 3; ++d ) keys(i,d) = position(i,d); };
    Kokkos::parallel_for( "Cabana::binByCartesianGrid3d::copy_op",
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
    auto bin_data_1d = Impl::kokkosBinSort(
        aosoa, keys, comp, create_data_only, false, begin, end );

    // Return the bin data.
    return CartesianGrid3dBinningData<
        typename AoSoA_t::memory_space::kokkos_memory_space>(
            bin_data_1d, nbin );
}

//---------------------------------------------------------------------------//
/*!
  \brief Bin an AoSoA spatially over a subset of its range within given
  structured Cartesian grid.

  \tparam PositionMember The member index for particle positions in the
  AoSoA.

  \tparam AoSoA_t The AoSoA type to sort.

  \param aosoa The AoSoA to sort.

  \param position_member Tag for the AoSoA member containing position.

  \param create_data_only True if the binning data should be created (i.e. bin
  sizes, offsets, and permutation vector) but the particles should not
  actually be binned.

  \param grid_delta Grid sizes in each cardinal direction.

  \param grid_min Grid minimum value in each direction.

  \param grid_max Grid maximum value in each direction.
*/
template<class AoSoA_t, std::size_t PositionMember>
CartesianGrid3dBinningData<typename AoSoA_t::memory_space::kokkos_memory_space>
binByCartesianGrid3d(
    AoSoA_t aosoa,
    MemberTag<PositionMember> position_member,
    const bool create_data_only,
    const typename AoSoA_t::template member_value_type<PositionMember>
    grid_delta[3],
    const typename AoSoA_t::template member_value_type<PositionMember>
    grid_min[3],
    const typename AoSoA_t::template member_value_type<PositionMember>
    grid_max[3],
    typename std::enable_if<(is_aosoa<AoSoA_t>::value),int>::type * = 0 )
{
    return binByCartesianGrid3d( aosoa, position_member, create_data_only,
                                 0, aosoa.size(),
                                 grid_delta, grid_min, grid_max );
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_SORT_HPP
