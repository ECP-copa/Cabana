/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_FASTFOURIERTRANSFORM_HPP
#define CAJITA_FASTFOURIERTRANSFORM_HPP

#include <Cajita_Array.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

#include <heffte_fft3d.h>

#include <array>
#include <memory>
#include <type_traits>

namespace Cajita
{
namespace Experimental
{
//---------------------------------------------------------------------------//

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct HeffteBackendTraits<Kokkos::Cuda>
{
    using backend_type = heffte::backend::cufft;
};
#endif

// TODO: enable heffte::backend::mkl, etc.
template <class BackendType>
struct HeffteBackendTraits
{
    using backend_type = heffte::backend::fftw;
};

class FastFourierTransformParams
{
    bool alltoall = true;
    bool pencils = true;
    bool reorder = true;

  public:
    void set_alltoall( bool value ) { alltoall = value; }
    void set_pencils( bool value ) { pencils = value; }
    void set_reorder( bool value ) { reorder = value; }
    bool get_alltoall() const { return alltoall; }
    bool get_pencils() const { return pencils; }
    bool get_reorder() const { return reorder; }
};

struct FFTScaleFull
{
};
struct FFTScaleNone
{
};
struct FFTScaleSymmetric
{
};

template <class Scalar, class EntityType, class MeshType, class DeviceType>
class FastFourierTransform
{
  public:
    std::array<int, 3> global_high;
    std::array<int, 3> global_low;

    FastFourierTransform( const ArrayLayout<EntityType, MeshType> &layout )
    {
        // Get the local dimensions of the problem.
        auto entity_space =
            layout.localGrid()->indexSpace( Own(), EntityType(), Local() );
        std::array<int, 3> local_num_entity = {
            (int)entity_space.extent( Dim::K ),
            (int)entity_space.extent( Dim::J ),
            (int)entity_space.extent( Dim::I ) };

        // Get the global grid.
        const auto &global_grid = layout.localGrid()->globalGrid();

        // Get the low corner of the global index space on this rank.
        global_low = {(int)global_grid.globalOffset( Dim::K ),
                      (int)global_grid.globalOffset( Dim::J ),
                      (int)global_grid.globalOffset( Dim::I )};

        // Get the high corner of the global index space on this rank.
        global_high = {global_low[Dim::I] + local_num_entity[Dim::I] - 1,
                       global_low[Dim::J] + local_num_entity[Dim::J] - 1,
                       global_low[Dim::K] + local_num_entity[Dim::K] - 1};
    }
};

template <class Scalar, class EntityType, class MeshType, class DeviceType>
class HeffteFastFourierTransform
    : public FastFourierTransform<Scalar, EntityType, MeshType, DeviceType>
{
  public:
    // Types.
    using value_type = Scalar;
    using entity_type = EntityType;
    using mesh_type = MeshType;
    using device_type = DeviceType;
    using exec_space = typename device_type::execution_space;
    using backend_type = typename HeffteBackendTraits<exec_space>::backend_type;

    using FastFourierTransform<Scalar, EntityType, MeshType,
                               DeviceType>::global_high;
    using FastFourierTransform<Scalar, EntityType, MeshType,
                               DeviceType>::global_low;

    /*!
      \brief Constructor
      \param layout The array layout defining the vector space of the transform.
      \param params Parameters for the 3D FFT.
    */
    HeffteFastFourierTransform( const ArrayLayout<EntityType, MeshType> &layout,
                                const FastFourierTransformParams &params )
        : FastFourierTransform<Scalar, EntityType, MeshType, DeviceType>(
              layout )
    {
        if ( 1 != layout.dofsPerEntity() )
            throw std::logic_error(
                "Only 1 complex value per entity allowed in FFT" );

        heffte::box3d inbox = {global_low, global_high};
        heffte::box3d outbox = {global_low, global_high};

        heffte::plan_options heffte_params =
            heffte::default_options<backend_type>();
        heffte_params.use_alltoall = params.get_alltoall();
        heffte_params.use_pencils = params.get_pencils();
        heffte_params.use_reorder = params.get_reorder();

        // Set FFT options from given parameters
        _fft = std::make_shared<heffte::fft3d<backend_type>>(
            inbox, outbox, layout.localGrid()->globalGrid().comm(),
            heffte_params );

        int fftsize = std::max( _fft->size_outbox(), _fft->size_inbox() );

        // Check the size.
        auto entity_space =
            layout.localGrid()->indexSpace( Own(), EntityType(), Local() );
        if ( fftsize < (int)entity_space.size() )
            throw std::logic_error( "Expected FFT allocation size smaller "
                                    "than local grid size" );

        _fft_work = Kokkos::View<std::complex<Scalar> *, DeviceType>(
            Kokkos::ViewAllocateWithoutInitializing( "fft_work" ), fftsize );
    }

    /*!
      \brief Do a forward FFT.
      \param in The array on which to perform the forward transform.
    */
    template <class Array_t>
    void forward( const Array_t &x, const FFTScaleNone )
    {
        compute( x, 1, heffte::scale::none );
    }
    template <class Array_t>
    void forward( const Array_t &x, const FFTScaleFull )
    {
        compute( x, 1, heffte::scale::full );
    }
    template <class Array_t>
    void forward( const Array_t &x, const FFTScaleSymmetric )
    {
        compute( x, 1, heffte::scale::symmetric );
    }

    /*!
     \brief Do a reverse FFT.
     \param out The array on which to perform the reverse transform.
    */
    template <class Array_t>
    void reverse( const Array_t &x, const FFTScaleNone )
    {
        compute( x, -1, heffte::scale::none );
    }
    template <class Array_t>
    void reverse( const Array_t &x, const FFTScaleFull )
    {
        compute( x, -1, heffte::scale::full );
    }
    template <class Array_t>
    void reverse( const Array_t &x, const FFTScaleSymmetric )
    {
        compute( x, -1, heffte::scale::symmetric );
    }

    template <class Array_t>
    void compute( const Array_t &x, const int flag, const FFT_ScaleType scale )
    {
        static_assert( is_array<Array_t>::value, "Must use an array" );
        static_assert(
            std::is_same<typename Array_t::entity_type, entity_type>::value,
            "Array entity type mush match transform entity type" );
        static_assert(
            std::is_same<typename Array_t::mesh_type, mesh_type>::value,
            "Array mesh type mush match transform mesh type" );
        static_assert(
            std::is_same<typename Array_t::device_type, DeviceType>::value,
            "Array device type and transform device type are different." );
        static_assert(
            std::is_same<typename Array_t::value_type,
                         Kokkos::complex<value_type>>::value ||
                std::is_same<typename Array_t::value_type, value_type>::value,
            "Array value type and complex transform value type are "
            "different." );

        if ( 1 != x.layout()->dofsPerEntity() )
            throw std::logic_error(
                "Only 1 complex value per entity allowed in FFT" );

        // Create a subview of the work array to write the local data into.
        auto own_space =
            x.layout()->localGrid()->indexSpace( Own(), EntityType(), Local() );
        // auto work_view_space = appendDimension( own_space, 2 );
        auto work_view =
            createView<std::complex<Scalar>, Kokkos::LayoutRight, DeviceType>(
                own_space, _fft_work.data() );

        // Copy to the work array. The work array only contains owned data.
        auto x_view = x.view();
        Kokkos::parallel_for(
            "fft_copy_x_to_work",
            createExecutionPolicy( own_space,
                                   typename DeviceType::execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                auto iw = i - own_space.min( Dim::I );
                auto jw = j - own_space.min( Dim::J );
                auto kw = k - own_space.min( Dim::K );
                auto realpart = x_view( i, j, k, 0 ).real();
                auto imagpart = x_view( i, j, k, 0 ).imag();
                work_view( iw, jw, kw ).real( realpart );
                work_view( iw, jw, kw ).imag( imagpart );
            } );
////convert int flag of scale to heffte::scale

        if ( flag == 1 )
        {
            _fft->forward( _fft_work.data(), _fft_work.data(), scale );
        }
        else if ( flag == -1 )
        {
            _fft->backward( _fft_work.data(), _fft_work.data(), scale );
        }
        else
        {
            throw std::logic_error(
                "Only 1:forward and -1:backward are allowed as compute flag" );
        }

        // Copy back to output array.
        Kokkos::parallel_for(
            "fft_copy_work_to_x",
            createExecutionPolicy( own_space,
                                   typename DeviceType::execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                auto iw = i - own_space.min( Dim::I );
                auto jw = j - own_space.min( Dim::J );
                auto kw = k - own_space.min( Dim::K );
                x_view( i, j, k, 0 ).real() = work_view( iw, jw, kw ).real();
                x_view( i, j, k, 0 ).imag() = work_view( iw, jw, kw ).imag();
            } );
    }

  private:
    // std::shared_ptr<heffte::fft3d<BackendType>> _fft;
    std::shared_ptr<FFT_type> _fft;
    Kokkos::View<std::complex<Scalar> *, DeviceType> _fft_work;
};

//---------------------------------------------------------------------------//
// FFT creation
//---------------------------------------------------------------------------//
template <class Scalar, class DeviceType, class EntityType, class MeshType>
std::shared_ptr<
    HeffteFastFourierTransform<Scalar, EntityType, MeshType, DeviceType>>
createHeffteFastFourierTransform(
    const ArrayLayout<EntityType, MeshType> &layout,
    const FastFourierTransformParams &params )
{
    return std::make_shared<
        HeffteFastFourierTransform<Scalar, EntityType, MeshType, DeviceType>>(
        layout, params );
}

template <class Scalar, class DeviceType, class EntityType, class MeshType>
std::shared_ptr<
    HeffteFastFourierTransform<Scalar, EntityType, MeshType, DeviceType>>
createHeffteFastFourierTransform(
    const ArrayLayout<EntityType, MeshType> &layout )
{
    using device_type = DeviceType;
    using exec_space = typename device_type::execution_space;
    using backend_type = typename HeffteBackendTraits<exec_space>::backend_type;

    // use default heFFTe params for this backend
    const heffte::plan_options heffte_params =
        heffte::default_options<backend_type>();
    FastFourierTransformParams params;
    params.set_alltoall( heffte_params.use_alltoall );
    params.set_pencils( heffte_params.use_pencils );
    params.set_reorder( heffte_params.use_reorder );

    return std::make_shared<
        HeffteFastFourierTransform<Scalar, EntityType, MeshType, DeviceType>>(
        layout, params );
}

//---------------------------------------------------------------------------//

} // end namespace Experimental
} // end namespace Cajita

#endif // end CAJITA_FASTFOURIERTRANSFORM_HPP
