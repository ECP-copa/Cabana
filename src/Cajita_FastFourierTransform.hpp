/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
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
//---------------------------------------------------------------------------//
template <class MemorySpace>
struct HeffteMemoryTraits;

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct HeffteMemoryTraits<Kokkos::CudaSpace>
{
    static constexpr heffte_memory_type_t value = HEFFTE_MEM_GPU;
};

template <>
struct HeffteMemoryTraits<Kokkos::CudaUVMSpace>
{
    static constexpr heffte_memory_type_t value = HEFFTE_MEM_MANAGED;
};
#endif

template <class MemorySpace>
struct HeffteMemoryTraits
{
    static constexpr heffte_memory_type_t value = HEFFTE_MEM_CPU;
};

//---------------------------------------------------------------------------//
class FastFourierTransformParams
{
  public:
    /*!
      \brief Default constructor to disable aggregate initialization.
    */
    FastFourierTransformParams()
        : collective( 2 )
        , exchange( 0 )
        , packflag( 2 )
        , scaled( 1 )
    {
    }

    /*!
      \brief Set the collective type.
      \param type Collective type.

      0: point-to-point
      1: all-to-all
      2: combination

      If this function is not used to set the type then 2 is used as the
      default.
    */
    FastFourierTransformParams &setCollectiveType( const int type )
    {
        collective = type;
        return *this;
    }

    /*!
      \brief Set the exchange type.
      \param type Exchange type.

      0: reshape direct from pencil to pencil
      1: two reshapes from pencil to brick, then brick to pencil

      If this function is not used to set the type then 0 is used as the
      default.
    */
    FastFourierTransformParams &setExchangeType( const int type )
    {
        exchange = type;
        return *this;
    }

    /*!
      \brief Set the pack type.
      \param type Pack type.

      0: array
      1: pointer
      2: memcpy

      If this function is not used to set the type then 2 is used as the
      default.
    */
    FastFourierTransformParams &setPackType( const int type )
    {
        packflag = type;
        return *this;
    }

    /*!
      \brief Set the scaling type.
      \param type Scaling type.

      0: Not scaling after forward
      1: Scaling after forward

      If this function is not used to set the type then 1 is used as the
      default.
    */
    FastFourierTransformParams &setScalingType( const int type )
    {
        scaled = type;
        return *this;
    }

    // Collective communication type.
    int collective;

    // Parallel decomposition exchange type.
    int exchange;

    // Buffer packing type.
    int packflag;

    // Forward scaling option.
    int scaled;
};

//---------------------------------------------------------------------------//
template <class Scalar, class EntityType, class MeshType, class DeviceType>
class FastFourierTransform
{
  public:
    // Types.
    using value_type = Scalar;
    using entity_type = EntityType;
    using mesh_type = MeshType;
    using device_type = DeviceType;

    /*!
      \brief Constructor
      \param layout The array layout defining the vector space of the
      transform.
      \param params Parameters for the 3D FFT.
    */
    FastFourierTransform( const ArrayLayout<EntityType, MeshType> &layout,
                          const FastFourierTransformParams &params )
        : _fft( layout.localGrid()->globalGrid().comm() )
    {
        if ( 1 != layout.dofsPerEntity() )
            throw std::logic_error(
                "Only 1 complex value per entity allowed in FFT" );

        // Set the memory type. For now we will just do FFTs on the host until
        // we find the HEFFTE GPU memory bug.
        _fft.mem_type = HEFFTE_MEM_CPU;

        // Let the fft allocate its own send/receive buffers.
        _fft.memoryflag = 1;

        // Set parameters.
        _fft.collective = params.collective;
        _fft.exchange = params.exchange;
        _fft.packflag = params.packflag;
        _fft.scaled = params.scaled;

        // Get the global grid.
        const auto &global_grid = layout.localGrid()->globalGrid();

        // Get the global dimensions of the problem. K indices move the
        // fastest because we fix the work array to be layout right.
        std::array<int, 3> global_num_entity = {
            global_grid.globalNumEntity( EntityType(), Dim::K ),
            global_grid.globalNumEntity( EntityType(), Dim::J ),
            global_grid.globalNumEntity( EntityType(), Dim::I )};

        // Get the local dimensions of the problem.
        auto entity_space =
            layout.localGrid()->indexSpace( Own(), EntityType(), Local() );
        std::array<int, 3> local_num_entity = {
            (int)entity_space.extent( Dim::K ),
            (int)entity_space.extent( Dim::J ),
            (int)entity_space.extent( Dim::I )};

        // Get the low corner of the global index space on this rank.
        std::array<int, 3> global_low = {
            (int)global_grid.globalOffset( Dim::K ),
            (int)global_grid.globalOffset( Dim::J ),
            (int)global_grid.globalOffset( Dim::I )};

        // Get the high corner of the global index space on this rank.
        std::array<int, 3> global_high = {
            global_low[Dim::K] + local_num_entity[Dim::K] - 1,
            global_low[Dim::J] + local_num_entity[Dim::J] - 1,
            global_low[Dim::I] + local_num_entity[Dim::I] - 1};

        // Setup the fft.
        int permute = 0;
        int fftsize, sendsize, recvsize;
        Scalar *work = nullptr;
        _fft.setup( work, global_num_entity.data(), global_low.data(),
                    global_high.data(), global_low.data(), global_high.data(),
                    permute, fftsize, sendsize, recvsize );

        // Check the size.
        if ( fftsize < (int)entity_space.size() )
            throw std::logic_error( "HEFFTE expected allocation size smaller "
                                    "than local grid size" );

        // Allocate the work array. The work array only contains owned data.
        auto subview_space = appendDimension( entity_space, 2 );
        _fft_work = createView<Scalar, Kokkos::LayoutRight, DeviceType>(
            "fft_work", subview_space );
    }

    /*!
      \brief Do a forward FFT.
      \param in The array on which to perform the forward transform.
    */
    template <class Array_t>
    void forward( const Array_t &x )
    {
        compute( x, 1 );
    }

    /*!
     \brief Do a reverse FFT.
     \param out The array on which to perform the reverse transform.
    */
    template <class Array_t>
    void reverse( const Array_t &x )
    {
        compute( x, -1 );
    }

  public:
    template <class Array_t>
    void compute( const Array_t &x, const int flag )
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

        // Copy to the work array. The work array only contains owned data.
        auto x_view = x.view();
        auto own_space =
            x.layout()->localGrid()->indexSpace( Own(), EntityType(), Local() );
        Kokkos::parallel_for(
            "fft_copy_x_to_work",
            createExecutionPolicy( own_space,
                                   typename DeviceType::execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                auto iw = i - own_space.min( Dim::I );
                auto jw = j - own_space.min( Dim::J );
                auto kw = k - own_space.min( Dim::K );
                _fft_work( iw, jw, kw, 0 ) = x_view( i, j, k, 0 ).real();
                _fft_work( iw, jw, kw, 1 ) = x_view( i, j, k, 0 ).imag();
            } );

        // Copy to the host. Once we fix the HEFFTE GPU memory bug we wont
        // need this.
        auto fft_work_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _fft_work );

        // Perform FFT.
        _fft.compute( fft_work_mirror.data(), fft_work_mirror.data(), flag );

        // Copy back to the work array. Once we fix the HEFFTE GPU memory bug
        // we wont need this.
        Kokkos::deep_copy( _fft_work, fft_work_mirror );

        // Copy back to output array.
        Kokkos::parallel_for(
            "fft_copy_work_to_x",
            createExecutionPolicy( own_space,
                                   typename DeviceType::execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                auto iw = i - own_space.min( Dim::I );
                auto jw = j - own_space.min( Dim::J );
                auto kw = k - own_space.min( Dim::K );
                x_view( i, j, k, 0 ).real() = _fft_work( iw, jw, kw, 0 );
                x_view( i, j, k, 0 ).imag() = _fft_work( iw, jw, kw, 1 );
            } );
    }

  private:
    HEFFTE_NS::FFT3d<Scalar> _fft;
    Kokkos::View<Scalar ****, Kokkos::LayoutRight, DeviceType> _fft_work;
};

//---------------------------------------------------------------------------//
// FFT creation
//---------------------------------------------------------------------------//
template <class Scalar, class DeviceType, class EntityType, class MeshType>
std::shared_ptr<FastFourierTransform<Scalar, EntityType, MeshType, DeviceType>>
createFastFourierTransform( const ArrayLayout<EntityType, MeshType> &layout,
                            const FastFourierTransformParams &params )
{
    return std::make_shared<
        FastFourierTransform<Scalar, EntityType, MeshType, DeviceType>>(
        layout, params );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_FASTFOURIERTRANSFORM_HPP
