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

/*!
  \file Cabana_Grid_BovWriter.hpp
  \brief Brick of values (BOV) grid output
*/
#ifndef CABANA_GRID_BOVWRITER_HPP
#define CABANA_GRID_BOVWRITER_HPP

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_HaloBase.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_MpiTraits.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

namespace Cabana
{
namespace Grid
{
namespace Experimental
{
namespace BovWriter
{
//---------------------------------------------------------------------------//
// VisIt Brick-of-Values (BOV) grid field writer.
//---------------------------------------------------------------------------//

//! BOV Format traits.
//! \tparam Scalar type.
template <typename T>
struct BovFormat;

//! BOV Format traits.
template <>
struct BovFormat<short>
{
    //! Get BOV value type.
    static std::string value() { return "SHORT"; }
};

//! BOV Format traits.
template <>
struct BovFormat<int>
{
    //! Get BOV value type.
    static std::string value() { return "INT"; }
};

//! BOV Format traits.
template <>
struct BovFormat<float>
{
    //! Get BOV value type.
    static std::string value() { return "FLOAT"; }
};

//! BOV Format traits.
template <>
struct BovFormat<double>
{
    //! Get BOV value type.
    static std::string value() { return "DOUBLE"; }
};

//! BOV Centering
//! \tparam Entity type.
template <typename T>
struct BovCentering;

//! BOV Centering
template <>
struct BovCentering<Cell>
{
    //! Get BOV value type.
    static std::string value() { return "zonal"; }
};

//! BOV Centering
template <>
struct BovCentering<Node>
{
    //! Get BOV value type.
    static std::string value() { return "nodal"; }
};

//---------------------------------------------------------------------------//
//! Create the MPI subarray for the given array.
template <class Array_t, std::size_t N>
MPI_Datatype createSubarray( const Array_t& array,
                             const std::array<long, N>& owned_extents,
                             const std::array<long, N>& global_extents )
{
    using value_type = typename Array_t::value_type;
    const auto& global_grid = array.layout()->localGrid()->globalGrid();

    std::array<int, N> local_start;
    std::array<int, N> local_size;
    std::array<int, N> global_size;
    for ( std::size_t i = 0; i < N - 1; ++i )
    {
        local_start[i] =
            static_cast<int>( global_grid.globalOffset( N - i - 2 ) );
        local_size[i] = static_cast<int>( owned_extents[N - i - 2] );
        global_size[i] = static_cast<int>( global_extents[N - i - 2] );
    }
    local_start.back() = 0;
    local_size.back() = owned_extents.back();
    global_size.back() = global_extents.back();

    MPI_Datatype subarray;
    MPI_Type_create_subarray( N, global_size.data(), local_size.data(),
                              local_start.data(), MPI_ORDER_C,
                              MpiTraits<value_type>::type(), &subarray );

    return subarray;
}

//---------------------------------------------------------------------------//
//! Reorder a view to the required ordering for I/O
template <class TargetView, class SourceView, class Indices, class ExecSpace>
std::enable_if_t<4 == TargetView::rank, void>
reorderView( TargetView& target, const SourceView& source,
             const Indices& index_space, const ExecSpace& exec_space )
{
    Kokkos::parallel_for(
        "Cabana::Grid::BovWriter::Reorder",
        createExecutionPolicy( index_space, exec_space ),
        KOKKOS_LAMBDA( const int k, const int j, const int i, const int l ) {
            target( k, j, i, l ) = source( i, j, k, l );
        } );
    exec_space.fence();
}

//! Reorder a view to the required ordering for I/O
template <class TargetView, class SourceView, class Indices, class ExecSpace>
std::enable_if_t<3 == TargetView::rank, void>
reorderView( TargetView& target, const SourceView& source,
             const Indices& index_space, const ExecSpace& exec_space )
{
    Kokkos::parallel_for(
        "Cabana::Grid::BovWriter::Reorder",
        createExecutionPolicy( index_space, exec_space ),
        KOKKOS_LAMBDA( const int j, const int i, const int l ) {
            target( j, i, l ) = source( i, j, l );
        } );
    exec_space.fence();
}

//---------------------------------------------------------------------------//
/*!
  \brief Write a grid array to a VisIt BOV.

  This version writes a single output and does not use bricklets. We will do
  this in the future to improve parallel visualization.

  \param prefix The filename prefix
  \param time_step_index The index of the time step we are writing.
  \param time The current time
  \param array The array to write
  \param gather_array Gather the array before writing to make parallel
  consistent.
*/
template <class ExecutionSpace, class Array_t>
void writeTimeStep( ExecutionSpace, const std::string& prefix,
                    const int time_step_index, const double time,
                    const Array_t& array, const bool gather_array = true )
{
    static_assert( isUniformMesh<typename Array_t::mesh_type>::value,
                   "ViSIT BOV writer can only be used with uniform mesh" );

    // Types
    using entity_type = typename Array_t::entity_type;
    using value_type = typename Array_t::value_type;
    using memory_space = typename Array_t::memory_space;
    const std::size_t num_space_dim = Array_t::num_space_dim;

    // Get the global grid.
    const auto& global_grid = array.layout()->localGrid()->globalGrid();

    // Get the global mesh.
    const auto& global_mesh = global_grid.globalMesh();

    // If this is a node field, determine periodicity so we can add the last
    // node back to the visualization if needed.
    std::array<long, num_space_dim + 1> global_extents;
    for ( std::size_t i = 0; i < num_space_dim + 1; ++i )
    {
        global_extents[i] = -1;
    }
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        if ( std::is_same<entity_type, Cell>::value )
            global_extents[d] = global_grid.globalNumEntity( Cell(), d );
        else if ( std::is_same<entity_type, Node>::value )
            global_extents[d] = global_grid.globalNumEntity( Cell(), d ) + 1;
    }
    global_extents[num_space_dim] = array.layout()->dofsPerEntity();

    auto owned_index_space = array.layout()->indexSpace( Own(), Local() );
    std::array<long, num_space_dim + 1> owned_extents;
    for ( std::size_t i = 0; i < num_space_dim + 1; ++i )
    {
        owned_extents[i] = -1;
    }
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        if ( std::is_same<entity_type, Cell>::value )
        {
            owned_extents[d] = owned_index_space.extent( d );
        }
        else if ( std::is_same<entity_type, Node>::value )
        {
            if ( !global_grid.isPeriodic( d ) ||
                 global_grid.dimBlockId( d ) <
                     global_grid.dimNumBlock( d ) - 1 )
                owned_extents[d] = owned_index_space.extent( d );
            else
                owned_extents[d] = owned_index_space.extent( d ) + 1;
        }
    }
    owned_extents[num_space_dim] = array.layout()->dofsPerEntity();

    // Gather halo data if any dimensions are periodic.
    if ( gather_array )
    {
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            if ( global_grid.isPeriodic( d ) )
            {
                auto halo =
                    createHalo( NodeHaloPattern<num_space_dim>(), 0, array );
                halo->gather( ExecutionSpace(), array );
                break;
            }
        }
    }

    // Create a contiguous array of the owned array values. Note that we
    // reorder to KJI grid ordering to conform to the BOV format.
    std::array<long, num_space_dim + 1> local_space_min;
    std::array<long, num_space_dim + 1> local_space_max;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        local_space_min[d] = owned_index_space.min( d );
        local_space_max[d] = owned_index_space.min( d ) + owned_extents[d];
    }
    local_space_min.back() = 0;
    local_space_max.back() = owned_extents.back();
    IndexSpace<num_space_dim + 1> local_space( local_space_min,
                                               local_space_max );
    auto owned_subview = createSubview( array.view(), local_space );

    std::array<long, num_space_dim + 1> reorder_space_size;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        reorder_space_size[d] = owned_extents[num_space_dim - d - 1];
    }
    reorder_space_size.back() = owned_extents.back();
    IndexSpace<num_space_dim + 1> reorder_space( reorder_space_size );
    auto owned_view = createView<value_type, Kokkos::LayoutRight, memory_space>(
        array.label(), reorder_space );
    reorderView( owned_view, owned_subview, reorder_space, ExecutionSpace() );

    // Compose a data file name prefix.
    std::stringstream file_name;
    file_name << prefix << "_" << std::setfill( '0' ) << std::setw( 6 )
              << time_step_index;

    // Open a binary data file.
    std::string data_file_name = file_name.str() + ".dat";
    MPI_File data_file;
    MPI_File_open( global_grid.comm(), data_file_name.c_str(),
                   MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                   &data_file );

    // Create the global subarray in which we are writing the local data.
    auto subarray = createSubarray( array, owned_extents, global_extents );
    MPI_Type_commit( &subarray );

    // Set the data in the file this process is going to write to.
    MPI_File_set_view( data_file, 0, MpiTraits<value_type>::type(), subarray,
                       "native", MPI_INFO_NULL );

    // Write the view to binary.
    MPI_Status status;
    MPI_File_write_all( data_file, owned_view.data(), owned_view.size(),
                        MpiTraits<value_type>::type(), &status );

    // Clean up.
    MPI_File_close( &data_file );
    MPI_Type_free( &subarray );

    // Create a VisIt BOV header with global data. Only create the header
    // on rank 0.
    int rank;
    MPI_Comm_rank( global_grid.comm(), &rank );
    if ( 0 == rank )
    {
        // Open a file for writing.
        std::string header_file_name = file_name.str() + ".bov";
        std::fstream header;
        header.open( header_file_name, std::fstream::out );

        // Write the current time.
        header << "TIME: " << time << std::endl;

        // Data file name.
        header << "DATA_FILE: " << data_file_name << std::endl;

        // Global data size.
        header << "DATA_SIZE: ";
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            header << global_extents[d] << " ";
        }
        for ( std::size_t d = num_space_dim; d < 3; ++d )
        {
            header << 1;
        }
        header << std::endl;

        // Data format.
        header << "DATA_FORMAT: " << BovFormat<value_type>::value()
               << std::endl;

        // Variable name.
        header << "VARIABLE: " << array.label() << std::endl;

        // Endian order
        header << "DATA_ENDIAN: LITTLE" << std::endl;

        // Data location.
        header << "CENTERING: " << BovCentering<entity_type>::value()
               << std::endl;

        // Mesh low corner.
        header << "BRICK_ORIGIN: ";
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            header << global_mesh.lowCorner( d ) << " ";
        }
        for ( std::size_t d = num_space_dim; d < 3; ++d )
        {
            header << 0.0;
        }
        header << std::endl;

        // Mesh global width
        header << "BRICK_SIZE: ";
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            header << global_grid.globalNumEntity( Cell(), d ) *
                          global_mesh.cellSize( d )
                   << " ";
        }
        for ( std::size_t d = num_space_dim; d < 3; ++d )
        {
            header << 0.0;
        }
        header << std::endl;

        // Number of data components. Scalar and vector types are
        // supported.
        header << "DATA_COMPONENTS: " << global_extents[num_space_dim]
               << std::endl;

        // Close the header.
        header.close();
    }
}

/*!
  \brief Write a grid array to a VisIt BOV.

  This version writes a single output and does not use bricklets. We will do
  this in the future to improve parallel visualization.

  \param prefix The filename prefix
  \param time_step_index The index of the time step we are writing.
  \param time The current time
  \param array The array to write
  \param gather_array Gather the array before writing to make parallel
  consistent.
*/
template <class Array_t>
void writeTimeStep( const std::string& prefix, const int time_step_index,
                    const double time, const Array_t& array,
                    const bool gather_array = true )
{
    using exec_space = typename Array_t::execution_space;
    writeTimeStep( exec_space{}, prefix, time_step_index, time, array,
                   gather_array );
}

/*!
  \brief Write a grid array to a VisIt BOV.

  This version writes a single output and does not use bricklets. We will do
  this in the future to improve parallel visualization.

  \param time_step_index The index of the time step we are writing.
  \param time The current time
  \param array The array to write
  \param gather_array Gather the array before writing to make parallel
  consistent.
*/
template <class ExecutionSpace, class Array_t,
          typename std::enable_if<
              Kokkos::is_execution_space<ExecutionSpace>::value, int>::type = 0>
void writeTimeStep( ExecutionSpace, const int time_step_index,
                    const double time, const Array_t& array,
                    const bool gather_array = true )
{
    writeTimeStep( ExecutionSpace{}, "grid_" + array.label(), time_step_index,
                   time, array, gather_array );
}

/*!
  \brief Write a grid array to a VisIt BOV.

  This version writes a single output and does not use bricklets. We will do
  this in the future to improve parallel visualization.

  \param time_step_index The index of the time step we are writing.
  \param time The current time
  \param array The array to write
  \param gather_array Gather the array before writing to make parallel
  consistent.
*/
template <class Array_t>
void writeTimeStep( const int time_step_index, const double time,
                    const Array_t& array, const bool gather_array = true )
{
    using exec_space = typename Array_t::execution_space;
    writeTimeStep( exec_space{}, "grid_" + array.label(), time_step_index, time,
                   array, gather_array );
}

//---------------------------------------------------------------------------//

} // end namespace BovWriter
} // end namespace Experimental
} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_BOVWRITER_HPP
