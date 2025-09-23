/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/****************************************************************************
 * Copyright (c) 2022 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_HDF5ParticleOutput.hpp
  \brief Write particle output using the HDF5 (XDMF) format.
*/

#ifndef CABANA_HDF5PARTICLEOUTPUT_HPP
#define CABANA_HDF5PARTICLEOUTPUT_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <hdf5.h>
#ifdef H5_HAVE_SUBFILING_VFD
#include "H5FDioc.h"       /* Private header for the IOC VFD */
#include "H5FDsubfiling.h" /* Private header for the subfiling VFD */
#endif

#include <mpi.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Experimental
{
namespace HDF5ParticleOutput
{

namespace Impl
{
// XDMF file creation routines
//! \cond Impl
inline void writeXdmfHeader( const char* xml_file_name, const double time,
                             hsize_t dims0, hsize_t dims1, const char* dtype,
                             uint precision, const char* h5_file_name,
                             const char* coords_name )
{
    std::ofstream xdmf_file( xml_file_name, std::ios::trunc );
    // Set precision to guarantee that conversion to text and back is exact.
    xdmf_file.precision( std::numeric_limits<double>::max_digits10 );
    xdmf_file << "<?xml version=\"1.0\" ?>\n";
    xdmf_file << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    xdmf_file << "<Xdmf Version=\"2.0\">\n";
    xdmf_file << "  <Domain>\n";
    xdmf_file << "    <Grid Name=\"points\" GridType=\"Uniform\">\n";
    xdmf_file << "      <Time Value=\"" << time << "\"/>\n";
    xdmf_file << "      <Topology TopologyType=\"Polyvertex\"";
    xdmf_file << " Dimensions=\"" << dims0 << "\"";
    xdmf_file << " NodesPerElement=\"1\"> </Topology>\n";
    xdmf_file << "      <Geometry Type=\"" << ( dims1 == 3 ? "XYZ" : "XY" )
              << "\">\n";
    xdmf_file << "         <DataItem Dimensions=\"" << dims0 << " " << dims1;
    xdmf_file << "\" NumberType=\"" << dtype;
    xdmf_file << "\" Precision=\"" << precision;
    xdmf_file << "\" Format=\"HDF\"> " << h5_file_name << ":/" << coords_name;
    xdmf_file << " </DataItem>\n";
    xdmf_file << "      </Geometry>\n";
    xdmf_file.close();
}

inline void writeXdmfAttribute( const char* xml_file_name,
                                const char* field_name, hsize_t dims0,
                                hsize_t dims1, hsize_t dims2, const char* dtype,
                                uint precision, const char* h5_file_name,
                                const char* dataitem )
{
    std::string AttributeType = "\"Scalar\"";
    if ( dims2 != 0 )
        AttributeType = "\"Tensor\"";
    else if ( dims1 != 0 )
        AttributeType = "\"Vector\"";

    std::ofstream xdmf_file( xml_file_name, std::ios::app );
    xdmf_file << "      <Attribute AttributeType =" << AttributeType
              << " Center=\"Node\"";
    xdmf_file << " Name=\"" << field_name << "\">\n";
    xdmf_file << "        <DataItem ItemType=\"Uniform\" Dimensions=\""
              << dims0;
    if ( dims1 != 0 )
        xdmf_file << " " << dims1;
    if ( dims2 != 0 )
        xdmf_file << " " << dims2;
    xdmf_file << "\" DataType=\"" << dtype << "\" Precision=\"" << precision
              << "\"";
    xdmf_file << " Format=\"HDF\"> " << h5_file_name << ":/" << dataitem;
    xdmf_file << " </DataItem>\n";
    xdmf_file << "      </Attribute>\n";
    xdmf_file.close();
}

inline void writeXdmfFooter( const char* xml_file_name )
{
    std::ofstream xdmf_file( xml_file_name, std::ios::app );
    xdmf_file << "    </Grid>\n";
    xdmf_file << "  </Domain>\n</Xdmf>\n";
    xdmf_file.close();
}
//! \endcond
} // namespace Impl

/*!
  \brief HDF5 tuning settings.

  Various property list setting to tune HDF5 for a given system. For an
  in-depth description of these settings, see the HDF5 reference manual at
  https://docs.hdfgroup.org/hdf5/develop

  File access property list alignment settings result in any file
  object &ge; threshold bytes aligned on an address which is a multiple of
  alignment.
*/
struct HDF5Config
{
    //! I/O transfer mode to collective or independent (default)
    bool collective = false;

    //! Set alignment on or off
    bool align = false;
    //! Threshold for aligning file objects
    unsigned long threshold = 0;
    //! Alignment value
    unsigned long alignment = 16777216;

    //! Sets metadata I/O mode operations to collective or independent (default)
    bool meta_collective = true;

    //! Cause all metadata for an object to be evicted from the cache
    bool evict_on_close = false;

#ifdef H5_HAVE_SUBFILING_VFD

    //! Use the subfiling file driver
    bool subfiling = false;

    // Optional subfiling file driver configuration parameters

    //! Size (in bytes) of data stripes in subfiles
    int64_t subfiling_stripe_size = H5FD_SUBFILING_DEFAULT_STRIPE_SIZE;

    //! Target number of subfiles to use
    int32_t subfiling_stripe_count = H5FD_SUBFILING_DEFAULT_STRIPE_COUNT;

    //! The method to use for selecting MPI ranks to be I/O concentrators.
    int subfiling_ioc_selection = SELECT_IOC_ONE_PER_NODE;

    //! Number of I/O concentrator worker threads to use
    int32_t subfiling_thread_pool_size = H5FD_IOC_DEFAULT_THREAD_POOL_SIZE;
#endif
};

//! \cond Impl
// Format traits for both HDF5 and XDMF.
template <typename T>
struct HDF5Traits;

template <>
struct HDF5Traits<int>
{
    static hid_t type( std::string* dtype, uint* precision )
    {
        *dtype = "Int";
        *precision = sizeof( int );
        return H5T_NATIVE_INT;
    }
};

template <>
struct HDF5Traits<unsigned int>
{
    static hid_t type( std::string* dtype, uint* precision )
    {
        *dtype = "UInt";
        *precision = sizeof( unsigned int );
        return H5T_NATIVE_UINT;
    }
};

template <>
struct HDF5Traits<long>
{
    static hid_t type( std::string* dtype, uint* precision )
    {
        *dtype = "Int";
        *precision = sizeof( long );
        return H5T_NATIVE_LONG;
    }
};

template <>
struct HDF5Traits<unsigned long>
{
    static hid_t type( std::string* dtype, uint* precision )
    {
        *dtype = "UInt";
        *precision = sizeof( unsigned long );
        return H5T_NATIVE_ULONG;
    }
};

template <>
struct HDF5Traits<float>
{
    static hid_t type( std::string* dtype, uint* precision )
    {
        *dtype = "Float";
        *precision = sizeof( float );
        return H5T_NATIVE_FLOAT;
    }
};

template <>
struct HDF5Traits<double>
{
    static hid_t type( std::string* dtype, uint* precision )
    {
        *dtype = "Float";
        *precision = sizeof( double );
        return H5T_NATIVE_DOUBLE;
    }
};

namespace Impl
{
//---------------------------------------------------------------------------//
// HDF5 (XDMF) Particle Field Output.
//---------------------------------------------------------------------------//

// Rank-0 field
template <class SliceType>
void writeFields(
    HDF5Config h5_config, hid_t file_id, std::size_t n_local,
    std::size_t n_global, hsize_t n_offset, int comm_rank,
    const char* filename_hdf5, const char* filename_xdmf,
    const SliceType& slice,
    typename std::enable_if<
        2 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    hid_t plist_id;
    hid_t dset_id;
    hid_t dcpl_id;
    hid_t filespace_id;
    hid_t memspace_id;

    // HDF5 hyperslab parameters
    hsize_t offset[1];
    hsize_t dimsf[1];
    hsize_t count[1];

    offset[0] = n_offset;
    count[0] = n_local;
    dimsf[0] = n_global;

    // Reorder in a contiguous blocked format.
    Kokkos::View<typename SliceType::value_type*,
                 typename SliceType::memory_space>
        view( Kokkos::ViewAllocateWithoutInitializing( "field" ), n_local );
    copySliceToView( view, slice, 0, n_local );

    // Mirror the field to the host.
    auto host_view =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view );

    std::string dtype;
    uint precision = 0;
    hid_t type_id =
        HDF5Traits<typename SliceType::value_type>::type( &dtype, &precision );

    filespace_id = H5Screate_simple( 1, dimsf, nullptr );

    dcpl_id = H5Pcreate( H5P_DATASET_CREATE );
    H5Pset_fill_time( dcpl_id, H5D_FILL_TIME_NEVER );

    dset_id = H5Dcreate( file_id, slice.label().c_str(), type_id, filespace_id,
                         H5P_DEFAULT, dcpl_id, H5P_DEFAULT );

    H5Sselect_hyperslab( filespace_id, H5S_SELECT_SET, offset, nullptr, count,
                         nullptr );

    memspace_id = H5Screate_simple( 1, count, nullptr );

    plist_id = H5Pcreate( H5P_DATASET_XFER );
    // Default IO in HDF5 is independent
    if ( h5_config.collective )
        H5Pset_dxpl_mpio( plist_id, H5FD_MPIO_COLLECTIVE );

    H5Dwrite( dset_id, type_id, memspace_id, filespace_id, plist_id,
              host_view.data() );

    H5Pclose( plist_id );
    H5Pclose( dcpl_id );
    H5Sclose( memspace_id );
    H5Dclose( dset_id );
    H5Sclose( filespace_id );

    if ( 0 == comm_rank )
    {
        hsize_t zero = 0;
        Impl::writeXdmfAttribute(
            filename_xdmf, slice.label().c_str(), dimsf[0], zero, zero,
            dtype.c_str(), precision, filename_hdf5, slice.label().c_str() );
    }
}

// Rank-1 field
template <class SliceType>
void writeFields(
    HDF5Config h5_config, hid_t file_id, std::size_t n_local,
    std::size_t n_global, hsize_t n_offset, int comm_rank,
    const char* filename_hdf5, const char* filename_xdmf,
    const SliceType& slice,
    typename std::enable_if<
        3 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    hid_t plist_id;
    hid_t dset_id;
    hid_t dcpl_id;
    hid_t filespace_id;
    hid_t memspace_id;

    // Reorder in a contiguous blocked format.
    Kokkos::View<typename SliceType::value_type**, Kokkos::LayoutRight,
                 typename SliceType::memory_space>
        view( Kokkos::ViewAllocateWithoutInitializing( "field" ), n_local,
              slice.extent( 2 ) );
    copySliceToView( view, slice, 0, n_local );

    // Mirror the field to the host.
    auto host_view =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view );

    hsize_t offset[2];
    hsize_t dimsf[2];
    hsize_t dimsm[2];
    hsize_t count[2];

    dimsf[0] = n_global;
    dimsf[1] = host_view.extent( 1 );
    dimsm[0] = n_local;
    dimsm[1] = host_view.extent( 1 );

    offset[0] = n_offset;
    offset[1] = 0;

    count[0] = dimsm[0];
    count[1] = dimsm[1];

    std::string dtype;
    uint precision;
    hid_t type_id =
        HDF5Traits<typename SliceType::value_type>::type( &dtype, &precision );

    filespace_id = H5Screate_simple( 2, dimsf, nullptr );

    dcpl_id = H5Pcreate( H5P_DATASET_CREATE );
    H5Pset_fill_time( dcpl_id, H5D_FILL_TIME_NEVER );

    dset_id = H5Dcreate( file_id, slice.label().c_str(), type_id, filespace_id,
                         H5P_DEFAULT, dcpl_id, H5P_DEFAULT );

    H5Sselect_hyperslab( filespace_id, H5S_SELECT_SET, offset, nullptr, count,
                         nullptr );

    memspace_id = H5Screate_simple( 2, dimsm, nullptr );
    plist_id = H5Pcreate( H5P_DATASET_XFER );
    // Default IO in HDF5 is independent
    if ( h5_config.collective )
        H5Pset_dxpl_mpio( plist_id, H5FD_MPIO_COLLECTIVE );

    H5Dwrite( dset_id, type_id, memspace_id, filespace_id, plist_id,
              host_view.data() );

    H5Pclose( plist_id );
    H5Pclose( dcpl_id );
    H5Sclose( memspace_id );
    H5Dclose( dset_id );
    H5Sclose( filespace_id );

    if ( 0 == comm_rank )
    {
        hsize_t zero = 0;
        Impl::writeXdmfAttribute(
            filename_xdmf, slice.label().c_str(), dimsf[0], dimsf[1], zero,
            dtype.c_str(), precision, filename_hdf5, slice.label().c_str() );
    }
}

// Rank-2 field
template <class SliceType>
void writeFields(
    HDF5Config h5_config, hid_t file_id, std::size_t n_local,
    std::size_t n_global, hsize_t n_offset, int comm_rank,
    const char* filename_hdf5, const char* filename_xdmf,
    const SliceType& slice,
    typename std::enable_if<
        4 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    hid_t plist_id;
    hid_t dset_id;
    hid_t dcpl_id;
    hid_t filespace_id;
    hid_t memspace_id;

    // Reorder in a contiguous blocked format.
    Kokkos::View<typename SliceType::value_type***, Kokkos::LayoutRight,
                 typename SliceType::memory_space>
        view( Kokkos::ViewAllocateWithoutInitializing( "field" ), n_local,
              slice.extent( 2 ), slice.extent( 3 ) );
    copySliceToView( view, slice, 0, n_local );

    // Mirror the field to the host.
    auto host_view =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view );

    hsize_t offset[3];
    hsize_t dimsf[3];
    hsize_t dimsm[3];
    hsize_t count[3];

    dimsf[0] = n_global;
    dimsf[1] = host_view.extent( 1 );
    dimsf[2] = host_view.extent( 2 );
    dimsm[0] = n_local;
    dimsm[1] = host_view.extent( 1 );
    dimsm[2] = host_view.extent( 2 );

    offset[0] = n_offset;
    offset[1] = 0;
    offset[2] = 0;

    count[0] = dimsm[0];
    count[1] = dimsm[1];
    count[2] = dimsm[2];

    std::string dtype;
    uint precision;
    hid_t type_id =
        HDF5Traits<typename SliceType::value_type>::type( &dtype, &precision );

    filespace_id = H5Screate_simple( 3, dimsf, nullptr );

    dcpl_id = H5Pcreate( H5P_DATASET_CREATE );
    H5Pset_fill_time( dcpl_id, H5D_FILL_TIME_NEVER );

    dset_id = H5Dcreate( file_id, slice.label().c_str(), type_id, filespace_id,
                         H5P_DEFAULT, dcpl_id, H5P_DEFAULT );

    H5Sselect_hyperslab( filespace_id, H5S_SELECT_SET, offset, nullptr, count,
                         nullptr );

    memspace_id = H5Screate_simple( 3, dimsm, nullptr );
    plist_id = H5Pcreate( H5P_DATASET_XFER );
    // Default IO in HDF5 is independent
    if ( h5_config.collective )
        H5Pset_dxpl_mpio( plist_id, H5FD_MPIO_COLLECTIVE );

    H5Dwrite( dset_id, type_id, memspace_id, filespace_id, plist_id,
              host_view.data() );

    H5Pclose( plist_id );
    H5Pclose( dcpl_id );
    H5Sclose( memspace_id );
    H5Dclose( dset_id );
    H5Sclose( filespace_id );

    if ( 0 == comm_rank )
    {
        Impl::writeXdmfAttribute(
            filename_xdmf, slice.label().c_str(), dimsf[0], dimsf[1], dimsf[2],
            dtype.c_str(), precision, filename_hdf5, slice.label().c_str() );
    }
}
//! \endcond
} // namespace Impl

//! Write particle data to HDF5 output. Empty overload if only writing coords.
inline void writeFields( HDF5Config, hid_t, std::size_t, std::size_t, hsize_t,
                         int, const char*, const char* )
{
}

//! Write particle data to HDF5 output.
template <class SliceType>
void writeFields( HDF5Config h5_config, hid_t file_id, std::size_t n_local,
                  std::size_t n_global, hsize_t n_offset, int comm_rank,
                  const char* filename_hdf5, const char* filename_xdmf,
                  const SliceType& slice )
{
    Impl::writeFields( h5_config, file_id, n_local, n_global, n_offset,
                       comm_rank, filename_hdf5, filename_xdmf, slice );
}

//! Write particle data to HDF5 output.
template <class SliceType, class... FieldSliceTypes>
void writeFields( HDF5Config h5_config, hid_t file_id, std::size_t n_local,
                  std::size_t n_global, hsize_t n_offset, int comm_rank,
                  const char* filename_hdf5, const char* filename_xdmf,
                  const SliceType& slice, FieldSliceTypes&&... fields )
{
    Impl::writeFields( h5_config, file_id, n_local, n_global, n_offset,
                       comm_rank, filename_hdf5, filename_xdmf, slice );
    writeFields( h5_config, file_id, n_local, n_global, n_offset, comm_rank,
                 filename_hdf5, filename_xdmf, fields... );
}

//---------------------------------------------------------------------------//
/*!
  \brief Write particle output in HDF5 format.
  \param h5_config HDF5 configuration settings.
  \param prefix Filename prefix.
  \param comm MPI communicator.
  \param time_step_index Current simulation step index.
  \param time Current simulation time.
  \param n_local Number of local particles.
  \param coords_slice Particle coordinates.
  \param fields Variadic list of particle property fields.
*/
template <class CoordSliceType, class... FieldSliceTypes>
void writeTimeStep( HDF5Config h5_config, const std::string& prefix,
                    MPI_Comm comm, const int time_step_index, const double time,
                    const std::size_t n_local,
                    const CoordSliceType& coords_slice,
                    FieldSliceTypes&&... fields )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::HDF5ParticleOutput" );

    hid_t plist_id;
    hid_t dset_id;
    hid_t dcpl_id;
    hid_t file_id;
    hid_t filespace_id;
    hid_t memspace_id;

    // HDF5 hyperslab parameters
    hsize_t offset[2];
    hsize_t dimsf[2];
    hsize_t count[2];

    int comm_rank;
    MPI_Comm_rank( comm, &comm_rank );
    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    // Compose a data file name.
    std::stringstream filename_hdf5;
    filename_hdf5 << prefix << "_" << time_step_index << ".h5";

    std::stringstream filename_xdmf;
    filename_xdmf << prefix << "_" << time_step_index << ".xmf";

    plist_id = H5Pcreate( H5P_FILE_ACCESS );
    H5Pset_libver_bounds( plist_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST );

#if H5_VERSION_GE( 1, 10, 1 )
    if ( h5_config.evict_on_close )
    {
        H5Pset_evict_on_close( plist_id, true );
    }
#endif

#if H5_VERSION_GE( 1, 10, 0 )
    if ( h5_config.collective )
    {
        H5Pset_all_coll_metadata_ops( plist_id, true );
        H5Pset_coll_metadata_write( plist_id, true );
    }
#endif

    if ( h5_config.align )
        H5Pset_alignment( plist_id, h5_config.threshold, h5_config.alignment );

#ifdef H5_HAVE_SUBFILING_VFD
    if ( h5_config.subfiling )
    {
        H5FD_subfiling_config_t subfiling_config;
        H5FD_ioc_config_t ioc_config;

        H5FD_subfiling_config_t* subfiling_ptr = nullptr;
        H5FD_ioc_config_t* ioc_ptr = nullptr;

        // Get the default subfiling configuration parameters
        hid_t fapl_id = H5I_INVALID_HID;

        fapl_id = H5Pcreate( H5P_FILE_ACCESS );
        H5Pget_fapl_subfiling( fapl_id, &subfiling_config );

        if ( h5_config.subfiling_stripe_size !=
             subfiling_config.shared_cfg.stripe_size )
        {
            subfiling_config.shared_cfg.stripe_size =
                h5_config.subfiling_stripe_size;
            if ( subfiling_ptr == nullptr )
                subfiling_ptr = &subfiling_config;
        }
        if ( h5_config.subfiling_stripe_count !=
             subfiling_config.shared_cfg.stripe_count )
        {
            subfiling_config.shared_cfg.stripe_count =
                h5_config.subfiling_stripe_count;
            if ( subfiling_ptr == nullptr )
                subfiling_ptr = &subfiling_config;
        }
        if ( h5_config.subfiling_ioc_selection !=
             (int)subfiling_config.shared_cfg.ioc_selection )
        {
            subfiling_config.shared_cfg.ioc_selection =
                (H5FD_subfiling_ioc_select_t)h5_config.subfiling_ioc_selection;
            if ( subfiling_ptr == nullptr )
                subfiling_ptr = &subfiling_config;
        }
        if ( h5_config.subfiling_thread_pool_size !=
             H5FD_IOC_DEFAULT_THREAD_POOL_SIZE )
        {
            H5Pget_fapl_ioc( fapl_id, &ioc_config );
            ioc_config.thread_pool_size = h5_config.subfiling_thread_pool_size;
            if ( ioc_ptr == nullptr )
                ioc_ptr = &ioc_config;
        }
        H5Pclose( fapl_id );

        H5Pset_mpi_params( plist_id, comm, MPI_INFO_NULL );

        if ( ioc_ptr != nullptr )
            H5Pset_fapl_ioc( subfiling_config.ioc_fapl_id, ioc_ptr );

        H5Pset_fapl_subfiling( plist_id, subfiling_ptr );
    }
    else
#endif
    {
        H5Pset_fapl_mpio( plist_id, comm, MPI_INFO_NULL );
    }

    file_id = H5Fcreate( filename_hdf5.str().c_str(), H5F_ACC_TRUNC,
                         H5P_DEFAULT, plist_id );
    H5Pclose( plist_id );

    // Write current simulation time
    hid_t fspace = H5Screate( H5S_SCALAR );
    hid_t attr_id = H5Acreate( file_id, "Time", H5T_NATIVE_DOUBLE, fspace,
                               H5P_DEFAULT, H5P_DEFAULT );
    H5Awrite( attr_id, H5T_NATIVE_DOUBLE, &time );
    H5Aclose( attr_id );
    H5Sclose( fspace );

    // Reorder the coordinates in a blocked format.
    Kokkos::View<typename CoordSliceType::value_type**, Kokkos::LayoutRight,
                 typename CoordSliceType::memory_space>
        coords_view( Kokkos::ViewAllocateWithoutInitializing( "coords" ),
                     coords_slice.size(), coords_slice.extent( 2 ) );
    Kokkos::parallel_for(
        "Cabana::HDF5ParticleOutput::copyCoords",
        Kokkos::RangePolicy<typename CoordSliceType::execution_space>(
            0, coords_slice.size() ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < coords_slice.extent( 2 ); ++d0 )
                coords_view( i, d0 ) = coords_slice( i, d0 );
        } );

    // Mirror the coordinates to the host.
    auto host_coords =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), coords_view );

    std::vector<int> all_offsets( comm_size );
    all_offsets[comm_rank] = n_local;

    MPI_Allreduce( MPI_IN_PLACE, all_offsets.data(), comm_size, MPI_INT,
                   MPI_SUM, comm );

    offset[0] = 0;
    offset[1] = 0;

    size_t n_global = 0;
    for ( int i = 0; i < comm_size; i++ )
    {
        if ( i < comm_rank )
        {
            offset[0] += static_cast<hsize_t>( all_offsets[i] );
        }
        n_global += (size_t)all_offsets[i];
    }
    std::vector<int>().swap( all_offsets );

    dimsf[0] = n_global;
    dimsf[1] = coords_slice.extent( 2 );

    filespace_id = H5Screate_simple( 2, dimsf, nullptr );

    count[0] = n_local;
    count[1] = coords_slice.extent( 2 );

    memspace_id = H5Screate_simple( 2, count, nullptr );

    plist_id = H5Pcreate( H5P_DATASET_XFER );

    // Default IO in HDF5 is independent
    if ( h5_config.collective )
        H5Pset_dxpl_mpio( plist_id, H5FD_MPIO_COLLECTIVE );

    std::string dtype;
    uint precision;
    hid_t type_id = HDF5Traits<typename CoordSliceType::value_type>::type(
        &dtype, &precision );

    dcpl_id = H5Pcreate( H5P_DATASET_CREATE );
    H5Pset_fill_time( dcpl_id, H5D_FILL_TIME_NEVER );

    dset_id = H5Dcreate( file_id, coords_slice.label().c_str(), type_id,
                         filespace_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT );

    H5Sselect_hyperslab( filespace_id, H5S_SELECT_SET, offset, nullptr, count,
                         nullptr );

    H5Dwrite( dset_id, type_id, memspace_id, filespace_id, plist_id,
              host_coords.data() );
    H5Dclose( dset_id );

    H5Pclose( plist_id );
    H5Pclose( dcpl_id );
    H5Sclose( filespace_id );
    H5Sclose( memspace_id );

    if ( 0 == comm_rank )
    {
        Impl::writeXdmfHeader( filename_xdmf.str().c_str(), time, dimsf[0],
                               dimsf[1], dtype.c_str(), precision,
                               filename_hdf5.str().c_str(),
                               coords_slice.label().c_str() );
    }

    // Add variables.
    hsize_t n_offset = offset[0];
    writeFields( h5_config, file_id, n_local, n_global, n_offset, comm_rank,
                 filename_hdf5.str().c_str(), filename_xdmf.str().c_str(),
                 fields... );

    H5Fclose( file_id );

    if ( 0 == comm_rank )
        Impl::writeXdmfFooter( filename_xdmf.str().c_str() );
}

//---------------------------------------------------------------------------//
// HDF5 (XDMF) Particle Field Input.
//---------------------------------------------------------------------------//
//! Read particle data from HDF5 output. Rank-0
template <class SliceType>
void readField(
    hid_t dset_id, hid_t dtype_id, hid_t memspace_id, hid_t filespace_id,
    hid_t plist_id, std::size_t n_local, const SliceType& slice,
    typename std::enable_if<
        2 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    // Read the field into a View.
    Kokkos::View<typename SliceType::value_type*, Kokkos::HostSpace> host_view(
        Kokkos::ViewAllocateWithoutInitializing( "field" ), n_local );
    H5Dread( dset_id, dtype_id, memspace_id, filespace_id, plist_id,
             host_view.data() );

    // Mirror the field and copy.
    auto view = Kokkos::create_mirror_view_and_copy(
        typename SliceType::memory_space(), host_view );
    copyViewToSlice( slice, view, 0, n_local );
}

//! Read particle data from HDF5 output. Rank-1
template <class SliceType>
void readField(
    hid_t dset_id, hid_t dtype_id, hid_t memspace_id, hid_t filespace_id,
    hid_t plist_id, std::size_t n_local, const SliceType& slice,
    typename std::enable_if<
        3 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    // Read the field into a View.
    Kokkos::View<typename SliceType::value_type**, Kokkos::LayoutRight,
                 Kokkos::HostSpace>
        host_view( Kokkos::ViewAllocateWithoutInitializing( "field" ), n_local,
                   slice.extent( 2 ) );
    H5Dread( dset_id, dtype_id, memspace_id, filespace_id, plist_id,
             host_view.data() );

    // Mirror the field and copy.
    auto view = Kokkos::create_mirror_view_and_copy(
        typename SliceType::memory_space(), host_view );
    copyViewToSlice( slice, view, 0, n_local );
}

//! Read particle data from HDF5 output. Rank-2
template <class SliceType>
void readField(
    hid_t dset_id, hid_t dtype_id, hid_t memspace_id, hid_t filespace_id,
    hid_t plist_id, std::size_t n_local, const SliceType& slice,
    typename std::enable_if<
        4 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    // Read the field into a View.
    Kokkos::View<typename SliceType::value_type***, Kokkos::LayoutRight,
                 Kokkos::HostSpace>
        host_view( Kokkos::ViewAllocateWithoutInitializing( "field" ), n_local,
                   slice.extent( 2 ), slice.extent( 3 ) );
    H5Dread( dset_id, dtype_id, memspace_id, filespace_id, plist_id,
             host_view.data() );

    // Mirror the field and copy.
    auto view = Kokkos::create_mirror_view_and_copy(
        typename SliceType::memory_space(), host_view );
    copyViewToSlice( slice, view, 0, n_local );
}

//---------------------------------------------------------------------------//
/*!
  \brief Read particle output from an HDF5 file.
  \param h5_config HDF5 configuration settings.
  \param prefix Filename prefix.
  \param comm MPI communicator.
  \param time_step_index Current simulation step index.
  \param n_local Number of local particles.
  \param dataset_name Dataset name to read data from.
  \param time Current simulation time.
  \param field Particle property field slice.
*/
template <class FieldSliceType>
void readTimeStep( HDF5Config h5_config, const std::string& prefix,
                   MPI_Comm comm, const int time_step_index,
                   const std::size_t n_local, const std::string& dataset_name,
                   double& time, FieldSliceType& field )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::HDF5ParticleInput" );

    hid_t plist_id;
    hid_t dset_id;
    hid_t file_id;
    hid_t dtype_id;
    hid_t filespace_id;
    hid_t memspace_id;
    int ndims;

    // HDF5 hyperslab parameters
    hsize_t offset[3] = { 0, 0, 0 };
    hsize_t dimsf[3] = { 0, 0, 0 };
    hsize_t count[3] = { 0, 0, 0 };

    int comm_rank;
    MPI_Comm_rank( comm, &comm_rank );
    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    // Retrieve data file name.
    std::stringstream filename_hdf5;
    filename_hdf5 << prefix << "_" << time_step_index << ".h5";

    plist_id = H5Pcreate( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio( plist_id, comm, MPI_INFO_NULL );

#if H5_VERSION_GE( 1, 10, 0 )
    if ( h5_config.collective )
    {
        H5Pset_all_coll_metadata_ops( plist_id, true );
    }
#endif

    // Open the HDF5 file.
    file_id = H5Fopen( filename_hdf5.str().c_str(), H5F_ACC_RDONLY, plist_id );
    H5Pclose( plist_id );

    // Get current simulation time associated with time_step_index
    hid_t attr_id = H5Aopen( file_id, "Time", H5P_DEFAULT );
    H5Aread( attr_id, H5T_NATIVE_DOUBLE, &time );
    H5Aclose( attr_id );

    // Open the dataset.
    dset_id = H5Dopen( file_id, dataset_name.c_str(), H5P_DEFAULT );

    // Get the datatype of the dataset.
    dtype_id = H5Dget_type( dset_id );

    // Get the dataspace of the dataset.
    filespace_id = H5Dget_space( dset_id );

    // Get the rank of the dataspace.
    ndims = H5Sget_simple_extent_ndims( filespace_id );

    // Get the extents of the file dataspace.
    H5Sget_simple_extent_dims( filespace_id, dimsf, nullptr );

    std::vector<int> all_offsets( comm_size );
    all_offsets[comm_rank] = n_local;

    MPI_Allreduce( MPI_IN_PLACE, all_offsets.data(), comm_size, MPI_INT,
                   MPI_SUM, comm );

    for ( int i = 0; i < comm_size; i++ )
    {
        if ( i < comm_rank )
        {
            offset[0] += static_cast<hsize_t>( all_offsets[i] );
        }
    }
    std::vector<int>().swap( all_offsets );

    count[0] = n_local;
    count[1] = dimsf[1];
    count[2] = dimsf[2];

    memspace_id = H5Screate_simple( ndims, count, nullptr );

    plist_id = H5Pcreate( H5P_DATASET_XFER );

    // Default IO in HDF5 is independent
    if ( h5_config.collective )
        H5Pset_dxpl_mpio( plist_id, H5FD_MPIO_COLLECTIVE );

    H5Sselect_hyperslab( filespace_id, H5S_SELECT_SET, offset, nullptr, count,
                         nullptr );

    readField( dset_id, dtype_id, memspace_id, filespace_id, plist_id, n_local,
               field );

    H5Pclose( plist_id );
    H5Sclose( memspace_id );
    H5Sclose( filespace_id );
    H5Dclose( dset_id );
    H5Fclose( file_id );
}

//---------------------------------------------------------------------------//

} // namespace HDF5ParticleOutput
} // namespace Experimental
} // end namespace Cabana

#endif // CABANA_HDF5PARTICLEOUTPUT_HPP
