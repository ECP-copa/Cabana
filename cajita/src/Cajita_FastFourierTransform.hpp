/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cajita_FastFourierTransform.hpp
  \brief Fast Fourier transforms
*/
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

//! Tag for full scaling of FFT.
struct FFTScaleFull
{
};
//! Tag for no scaling of FFT.
struct FFTScaleNone
{
};
//! Tag for symmetric scaling of FFT.
struct FFTScaleSymmetric
{
};

//! Tag specifying FFTW backend for FFT (host default).
struct FFTBackendFFTW
{
};
//! Tag specifying MKL backend for FFT.
struct FFTBackendMKL
{
};

namespace Impl
{
//! \cond Impl
struct FFTBackendDefault
{
};
//! \endcond
} // namespace Impl

//! Matching Array static type checker.
template <class ArrayEntity, class ArrayMesh, class ArrayDevice,
          class ArrayScalar, class Entity, class Mesh, class Device,
          class Scalar, typename SFINAE = void>
struct is_matching_array : public std::false_type
{
    static_assert( std::is_same<ArrayEntity, Entity>::value,
                   "Array entity type mush match FFT entity type." );
    static_assert( std::is_same<ArrayMesh, Mesh>::value,
                   "Array mesh type mush match FFT mesh type." );
    static_assert( std::is_same<ArrayDevice, Device>::value,
                   "Array device type must match FFT device type." );
};

//! Matching Array static type checker.
template <class ArrayEntity, class ArrayMesh, class ArrayDevice,
          class ArrayScalar, class Entity, class Mesh, class Device,
          class Scalar>
struct is_matching_array<
    ArrayEntity, ArrayMesh, ArrayDevice, ArrayScalar, Entity, Mesh, Device,
    Scalar,
    typename std::enable_if<std::is_same<ArrayEntity, Entity>::value &&
                            std::is_same<ArrayMesh, Mesh>::value &&
                            std::is_same<ArrayDevice, Device>::value>::type>
    : public std::true_type
{
};

//---------------------------------------------------------------------------//
/*!
  \brief Parameters controlling details for fast Fourier transforms.
*/
class FastFourierTransformParams
{
    bool alltoall = true;
    bool pencils = true;
    bool reorder = true;

  public:
    /*!
      \brief setAllToAll Set MPI communication.
      \param value Use all to all MPI communication.
    */
    void setAllToAll( bool value ) { alltoall = value; }
    /*!
      \brief setPencils Set data exchange type (pencil or slab).
      \param value Use pencil (true) or slab (false) decomposition.
    */
    void setPencils( bool value ) { pencils = value; }
    /*!
      \brief setReorder Set data handling (contiguous or strided memory).
      \param value Use contiguous (true) or strided (false) memory layout.
      Contiguous layout requires tensor transposition; strided layout does not.
    */
    void setReorder( bool value ) { reorder = value; }
    /*!
      \brief Get MPI communication strategy.
      \return Using AllToAll or not.
    */
    bool getAllToAll() const { return alltoall; }
    /*!
      \brief getPencils Get data exchange type (pencil or slab).
      \return Using pencil (true) or slab (false) decomposition.
    */
    bool getPencils() const { return pencils; }
    /*!
      \brief getReorder Get data handling (contiguous or strided memory).
      \return Using contiguous (true) or strided (false) memory layout.
      Contiguous layout requires tensor transposition; strided layout does not.
    */
    bool getReorder() const { return reorder; }
};

//---------------------------------------------------------------------------//
/*!
  \brief 2D/3D distributed fast Fourier transform base implementation.
*/
template <class EntityType, class MeshType, class Scalar, class DeviceType,
          class Derived>
class FastFourierTransform
{
  public:
    //! Array entity type.
    using entity_type = EntityType;
    //! Mesh type.
    using mesh_type = MeshType;
    //! Scalar value type.
    using value_type = Scalar;
    //! Kokkos device type.
    using device_type = DeviceType;
    //! Kokkos execution space.
    using exec_space = typename device_type::execution_space;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    //! Global high box corner.
    std::array<int, num_space_dim> global_high;
    //! Global low box corner.
    std::array<int, num_space_dim> global_low;

    /*!
      \brief Constructor
      \param layout The array layout defining the vector space of the transform.
    */
    FastFourierTransform( const ArrayLayout<EntityType, MeshType>& layout )
    {
        checkArrayDofs( layout.dofsPerEntity() );

        // Get the local dimensions of the problem.
        auto entity_space =
            layout.localGrid()->indexSpace( Own(), EntityType(), Local() );
        // Get the global grid.
        const auto& global_grid = layout.localGrid()->globalGrid();

        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            // Get the low corner of the global index space on this rank.
            global_low[d] =
                (int)global_grid.globalOffset( num_space_dim - d - 1 );

            // Get the high corner of the global index space on this rank.
            int local_num_entity =
                (int)entity_space.extent( num_space_dim - d - 1 );
            global_high[d] = global_low[d] + local_num_entity - 1;
        }
    }

    /*!
      \brief Ensure the FFT compute array has the correct DoFs.
      \param dof Degrees of freedom of array.
    */
    inline void checkArrayDofs( const int dof )
    {
        if ( 2 != dof )
            throw std::logic_error(
                "Only 1 complex value per entity allowed in FFT" );
    }

    /*!
      \brief Do a forward FFT.
      \param x The array on which to perform the forward transform.
      \param scaling Method of scaling data.
    */
    template <class Array_t, class ScaleType>
    void forward(
        const Array_t& x, const ScaleType scaling,
        typename std::enable_if<
            ( is_array<Array_t>::value &&
              is_matching_array<
                  typename Array_t::entity_type, typename Array_t::mesh_type,
                  typename Array_t::device_type, typename Array_t::value_type,
                  entity_type, mesh_type, device_type, value_type>::value ),
            int>::type* = 0 )
    {
        Kokkos::Profiling::pushRegion( "Cabana::FFT::forward" );

        checkArrayDofs( x.layout()->dofsPerEntity() );
        static_cast<Derived*>( this )->forwardImpl( x, scaling );

        Kokkos::Profiling::popRegion();
    }

    /*!
      \brief Do a reverse FFT.
      \param x The array on which to perform the reverse transform.
      \param scaling Method of scaling data.
    */
    template <class Array_t, class ScaleType>
    void reverse(
        const Array_t& x, const ScaleType scaling,
        typename std::enable_if<
            ( is_array<Array_t>::value &&
              is_matching_array<
                  typename Array_t::entity_type, typename Array_t::mesh_type,
                  typename Array_t::device_type, typename Array_t::value_type,
                  entity_type, mesh_type, device_type, value_type>::value ),
            int>::type* = 0 )
    {
        Kokkos::Profiling::pushRegion( "Cabana::FFT::reverse" );

        checkArrayDofs( x.layout()->dofsPerEntity() );
        static_cast<Derived*>( this )->reverseImpl( x, scaling );

        Kokkos::Profiling::popRegion();
    }

    /*!
      \brief Copy owned data for FFT.
    */
    template <class IndexSpaceType, class LViewType, class LGViewType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, void>
    copyToLocal( const IndexSpaceType own_space, LViewType& l_view,
                 const LGViewType lg_view )
    {
        Kokkos::parallel_for(
            "fft_copy_to_work",
            createExecutionPolicy( own_space, exec_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                auto iw = i - own_space.min( Dim::I );
                auto jw = j - own_space.min( Dim::J );
                auto kw = k - own_space.min( Dim::K );
                l_view( iw, jw, kw, 0 ) = lg_view( i, j, k, 0 );
                l_view( iw, jw, kw, 1 ) = lg_view( i, j, k, 1 );
            } );
    }

    /*!
      \brief Copy owned data for FFT.
    */
    template <class IndexSpaceType, class LViewType, class LGViewType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, void>
    copyToLocal( const IndexSpaceType own_space, LViewType& l_view,
                 const LGViewType lg_view )
    {
        Kokkos::parallel_for(
            "fft_copy_to_work",
            createExecutionPolicy( own_space, exec_space() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                auto iw = i - own_space.min( Dim::I );
                auto jw = j - own_space.min( Dim::J );
                l_view( iw, jw, 0 ) = lg_view( i, j, 0 );
                l_view( iw, jw, 1 ) = lg_view( i, j, 1 );
            } );
    }

    /*!
      \brief Copy owned data back after FFT.
    */
    template <class IndexSpaceType, class LViewType, class LGViewType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, void>
    copyFromLocal( const IndexSpaceType own_space, const LViewType l_view,
                   LGViewType& lg_view )
    {
        Kokkos::parallel_for(
            "fft_copy_from_work",
            createExecutionPolicy( own_space, exec_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                auto iw = i - own_space.min( Dim::I );
                auto jw = j - own_space.min( Dim::J );
                auto kw = k - own_space.min( Dim::K );
                lg_view( i, j, k, 0 ) = l_view( iw, jw, kw, 0 );
                lg_view( i, j, k, 1 ) = l_view( iw, jw, kw, 1 );
            } );
    }

    /*!
      \brief Copy owned data back after FFT.
    */
    template <class IndexSpaceType, class LViewType, class LGViewType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, void>
    copyFromLocal( const IndexSpaceType own_space, const LViewType l_view,
                   LGViewType& lg_view )
    {
        Kokkos::parallel_for(
            "fft_copy_from_work",
            createExecutionPolicy( own_space, exec_space() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                auto iw = i - own_space.min( Dim::I );
                auto jw = j - own_space.min( Dim::J );
                lg_view( i, j, 0 ) = l_view( iw, jw, 0 );
                lg_view( i, j, 1 ) = l_view( iw, jw, 1 );
            } );
    }
};

//---------------------------------------------------------------------------//
// heFFTe
//---------------------------------------------------------------------------//
namespace Impl
{
//! \cond Impl
template <class ExecutionSpace, class BackendType>
struct HeffteBackendTraits
{
};
#ifdef Heffte_ENABLE_MKL
template <class ExecutionSpace>
struct HeffteBackendTraits<ExecutionSpace, FFTBackendMKL>
{
    using backend_type = heffte::backend::mkl;
};
#endif
#ifdef Heffte_ENABLE_FFTW
template <class ExecutionSpace>
struct HeffteBackendTraits<ExecutionSpace, FFTBackendFFTW>
{
    using backend_type = heffte::backend::fftw;
};
template <class ExecutionSpace>
struct HeffteBackendTraits<ExecutionSpace, Impl::FFTBackendDefault>
{
    using backend_type = heffte::backend::fftw;
};
#else
#ifdef Heffte_ENABLE_MKL
template <class ExecutionSpace>
struct HeffteBackendTraits<ExecutionSpace, Impl::FFTBackendDefault>
{
    using backend_type = heffte::backend::mkl;
};
#else
throw std::runtime_error( "Must enable at least one host heFFTe backend." );
#endif
#endif
#ifdef Heffte_ENABLE_CUDA
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct HeffteBackendTraits<Kokkos::Cuda, Impl::FFTBackendDefault>
{
    using backend_type = heffte::backend::cufft;
};
#endif
#endif
#ifdef Heffte_ENABLE_ROCM
#ifdef KOKKOS_ENABLE_HIP
template <>
struct HeffteBackendTraits<Kokkos::Experimental::HIP, Impl::FFTBackendDefault>
{
    using backend_type = heffte::backend::rocfft;
};
#endif
#endif
#ifdef Heffte_ENABLE_ONEAPI
#ifdef KOKKOS_ENABLE_SYCL
template <>
struct HeffteBackendTraits<Kokkos::Experimental::SYCL, Impl::FFTBackendDefault>
{
    using backend_type = heffte::backend::onemkl;
};
#endif
#endif

template <class ScaleType>
struct HeffteScalingTraits
{
};
template <>
struct HeffteScalingTraits<FFTScaleNone>
{
    static const auto scaling_type = heffte::scale::none;
};
template <>
struct HeffteScalingTraits<FFTScaleFull>
{
    static const auto scaling_type = heffte::scale::full;
};
template <>
struct HeffteScalingTraits<FFTScaleSymmetric>
{
    static const auto scaling_type = heffte::scale::symmetric;
};

#ifdef KOKKOS_ENABLE_SYCL
// Overload for SYCL.
template <class HeffteBackendType>
auto createHeffte(
    HeffteBackendType, heffte::box3d<> inbox, heffte::box3d<> outbox,
    MPI_Comm comm, heffte::plan_options params,
    typename std::enable_if<
        std::is_same<HeffteBackendType, heffte::backend::onemkl>::value,
        int>::type* = 0 )
{
    // Set FFT options from given parameters
    // heFFTe correctly handles 2D or 3D FFTs within "fft3d"
    // FIXME: get SYCL queue from Kokkos.
    sycl::queue q;
    return std::make_shared<heffte::fft3d<HeffteBackendType>>( q, inbox, outbox,
                                                               comm, params );
}
#endif

template <class HeffteBackendType>
auto createHeffte(
    HeffteBackendType, heffte::box3d<> inbox, heffte::box3d<> outbox,
    MPI_Comm comm, heffte::plan_options params,
    typename std::enable_if<
        !std::is_same<HeffteBackendType, heffte::backend::onemkl>::value,
        int>::type* = 0 )
{
    // Set FFT options from given parameters
    // heFFTe correctly handles 2D or 3D FFTs within "fft3d"
    return std::make_shared<heffte::fft3d<HeffteBackendType>>( inbox, outbox,
                                                               comm, params );
}

//! \endcond
} // namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Interface to heFFTe fast Fourier transform library.
*/
template <class EntityType, class MeshType, class Scalar, class DeviceType,
          class BackendType>
class HeffteFastFourierTransform
    : public FastFourierTransform<
          EntityType, MeshType, Scalar, DeviceType,
          HeffteFastFourierTransform<EntityType, MeshType, Scalar, DeviceType,
                                     BackendType>>
{
  public:
    //! Scalar value type.
    using value_type = Scalar;
    //! Kokkos device type.
    using device_type = DeviceType;
    //! FFT backend type.
    using backend_type = BackendType;
    //! Kokkos execution space.
    using exec_space = typename device_type::execution_space;
    //! Mesh type.
    using mesh_type = MeshType;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    //! heFFTe backend type.
    using heffte_backend_type =
        typename Impl::HeffteBackendTraits<exec_space,
                                           backend_type>::backend_type;

    /*!
      \brief Constructor
      \param layout The array layout defining the vector space of the transform.
      \param params Parameters for the FFT.
    */
    HeffteFastFourierTransform( const ArrayLayout<EntityType, MeshType>& layout,
                                const FastFourierTransformParams& params )
        : FastFourierTransform<
              EntityType, MeshType, Scalar, DeviceType,
              HeffteFastFourierTransform<EntityType, MeshType, Scalar,
                                         DeviceType, BackendType>>( layout )
    {
        // heFFTe correctly handles 2D or 3D domains within "box3d"
        heffte::box3d<> inbox = { this->global_low, this->global_high };
        heffte::box3d<> outbox = { this->global_low, this->global_high };

        heffte::plan_options heffte_params =
            heffte::default_options<heffte_backend_type>();
        // TODO: use all three heffte options for algorithm
        bool alltoall = params.getAllToAll();
        if ( alltoall )
            heffte_params.algorithm = heffte::reshape_algorithm::alltoallv;
        else
            heffte_params.algorithm = heffte::reshape_algorithm::p2p;
        heffte_params.use_pencils = params.getPencils();
        heffte_params.use_reorder = params.getReorder();

        // Create the heFFTe main class (separated to handle SYCL queue
        // correctly).
        _fft = Impl::createHeffte( heffte_backend_type{}, inbox, outbox,
                                   layout.localGrid()->globalGrid().comm(),
                                   heffte_params );
        int fftsize = std::max( _fft->size_outbox(), _fft->size_inbox() );

        // Check the size.
        auto entity_space =
            layout.localGrid()->indexSpace( Own(), EntityType(), Local() );
        if ( fftsize < (int)entity_space.size() )
            throw std::logic_error( "Expected FFT allocation size smaller "
                                    "than local grid size" );

        _fft_work = Kokkos::View<Scalar*, DeviceType>(
            Kokkos::ViewAllocateWithoutInitializing( "fft_work" ),
            2 * fftsize );
        _workspace = Kokkos::View<Scalar* [2], DeviceType>(
            Kokkos::ViewAllocateWithoutInitializing( "workspace" ),
            4 * fftsize );
    }

    /*!
      \brief Do a forward FFT.
      \param x The array on which to perform the forward transform.
      \param ScaleType Method of scaling data.
    */
    template <class Array_t, class ScaleType>
    void forwardImpl( const Array_t& x, const ScaleType )
    {
        compute( x, 1, Impl::HeffteScalingTraits<ScaleType>().scaling_type );
    }

    /*!
     \brief Do a reverse FFT.
     \param x The array on which to perform the reverse transform
     \param ScaleType Method of scaling data.
    */
    template <class Array_t, class ScaleType>
    void reverseImpl( const Array_t& x, const ScaleType )
    {
        compute( x, -1, Impl::HeffteScalingTraits<ScaleType>().scaling_type );
    }

    /*!
     \brief Do the FFT.
     \param x The array on which to perform the transform.
     \param flag Flag for forward or reverse.
     \param scale Method of scaling data.
    */
    template <class Array_t>
    void compute( const Array_t& x, const int flag, const heffte::scale scale )
    {
        // Create a subview of the work array to write the local data into.
        auto own_space =
            x.layout()->localGrid()->indexSpace( Own(), EntityType(), Local() );
        auto local_view_space = appendDimension( own_space, 2 );
        auto local_view = createView<Scalar, Kokkos::LayoutRight, DeviceType>(
            local_view_space, _fft_work.data() );

        // TODO: pull this out to template function
        // Copy to the work array. The work array only contains owned data.
        auto localghost_view = x.view();

        this->copyToLocal( own_space, local_view, localghost_view );

        if ( flag == 1 )
        {
            _fft->forward(
                reinterpret_cast<std::complex<Scalar>*>( _fft_work.data() ),
                reinterpret_cast<std::complex<Scalar>*>( _fft_work.data() ),
                reinterpret_cast<std::complex<Scalar>*>( _workspace.data() ),
                scale );
        }
        else if ( flag == -1 )
        {
            _fft->backward(
                reinterpret_cast<std::complex<Scalar>*>( _fft_work.data() ),
                reinterpret_cast<std::complex<Scalar>*>( _fft_work.data() ),
                reinterpret_cast<std::complex<Scalar>*>( _workspace.data() ),
                scale );
        }
        else
        {
            throw std::logic_error(
                "Only 1:forward and -1:backward are allowed as compute flag" );
        }

        // Copy back to output array.
        this->copyFromLocal( own_space, local_view, localghost_view );
    }

  private:
    // heFFTe correctly handles 2D or 3D FFTs within "fft3d"
    std::shared_ptr<heffte::fft3d<heffte_backend_type>> _fft;
    Kokkos::View<Scalar*, DeviceType> _fft_work;
    Kokkos::View<Scalar* [2], DeviceType> _workspace;
};

//---------------------------------------------------------------------------//
// heFFTe creation
//---------------------------------------------------------------------------//
//! Creation function for heFFTe FFT with explict FFT backend.
//! \param layout FFT entity array
//! \param params FFT parameters
template <class Scalar, class DeviceType, class BackendType, class EntityType,
          class MeshType>
auto createHeffteFastFourierTransform(
    const ArrayLayout<EntityType, MeshType>& layout,
    const FastFourierTransformParams& params )
{
    return std::make_shared<HeffteFastFourierTransform<
        EntityType, MeshType, Scalar, DeviceType, BackendType>>( layout,
                                                                 params );
}

//! Creation function for heFFTe FFT with default FFT backend.
//! \param layout FFT entity array
//! \param params FFT parameters
template <class Scalar, class DeviceType, class EntityType, class MeshType>
auto createHeffteFastFourierTransform(
    const ArrayLayout<EntityType, MeshType>& layout,
    const FastFourierTransformParams& params )
{
    return createHeffteFastFourierTransform<
        Scalar, DeviceType, Impl::FFTBackendDefault, EntityType, MeshType>(
        layout, params );
}

//! Creation function for heFFTe FFT with explict FFT backend and default
//! parameters.
//! \param layout FFT entity array
template <class Scalar, class DeviceType, class BackendType, class EntityType,
          class MeshType>
auto createHeffteFastFourierTransform(
    const ArrayLayout<EntityType, MeshType>& layout )
{
    using device_type = DeviceType;
    using backend_type = BackendType;
    using exec_space = typename device_type::execution_space;
    using heffte_backend_type =
        typename Impl::HeffteBackendTraits<exec_space,
                                           backend_type>::backend_type;

    // use default heFFTe params for this backend
    const heffte::plan_options heffte_params =
        heffte::default_options<heffte_backend_type>();
    FastFourierTransformParams params;
    // TODO: set appropriate default for AllToAll
    params.setAllToAll( true );
    params.setPencils( heffte_params.use_pencils );
    params.setReorder( heffte_params.use_reorder );

    return std::make_shared<HeffteFastFourierTransform<
        EntityType, MeshType, Scalar, DeviceType, BackendType>>( layout,
                                                                 params );
}

//! Creation function for heFFTe FFT with default FFT backend and default
//! parameters.
//! \param layout FFT entity array
template <class Scalar, class DeviceType, class EntityType, class MeshType>
auto createHeffteFastFourierTransform(
    const ArrayLayout<EntityType, MeshType>& layout )
{
    return createHeffteFastFourierTransform<
        Scalar, DeviceType, Impl::FFTBackendDefault, EntityType, MeshType>(
        layout );
}

//---------------------------------------------------------------------------//

} // end namespace Experimental
} // end namespace Cajita

#endif // end CAJITA_FASTFOURIERTRANSFORM_HPP
