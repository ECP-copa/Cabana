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

#ifndef CAJITA_BOVWRITER_HPP
#define CAJITA_BOVWRITER_HPP

#include <Cajita_Array.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_MpiTraits.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

namespace Cajita
{
namespace BovWriter
{
namespace Experimental
{
//---------------------------------------------------------------------------//
// VisIt Brick-of-Values (BOV) grid field writer.
//---------------------------------------------------------------------------//
// BOV Format traits.
template <typename T>
struct BovFormat;

template <>
struct BovFormat<short>
{
    static std::string value() { return "SHORT"; }
};

template <>
struct BovFormat<int>
{
    static std::string value() { return "INT"; }
};

template <>
struct BovFormat<float>
{
    static std::string value() { return "FLOAT"; }
};

template <>
struct BovFormat<double>
{
    static std::string value() { return "DOUBLE"; }
};

// BOV Centering
template <typename T>
struct BovCentering;

template <>
struct BovCentering<Cell>
{
    static std::string value() { return "zonal"; }
};

template <>
struct BovCentering<Node>
{
    static std::string value() { return "nodal"; }
};

//---------------------------------------------------------------------------//
// Create the MPI subarray for the given array.
template <class Array_t>
MPI_Datatype createSubarray( const Array_t &array,
                             const std::array<long, 4> &owned_extents,
                             const std::array<long, 4> &global_extents )
{
    using value_type = typename Array_t::value_type;
    const auto &global_grid = array.layout()->localGrid()->globalGrid();

    int local_start[4] = {
        static_cast<int>( global_grid.globalOffset( Dim::K ) ),
        static_cast<int>( global_grid.globalOffset( Dim::J ) ),
        static_cast<int>( global_grid.globalOffset( Dim::I ) ), 0 };
    int local_size[4] = { static_cast<int>( owned_extents[Dim::K] ),
                          static_cast<int>( owned_extents[Dim::J] ),
                          static_cast<int>( owned_extents[Dim::I] ),
                          static_cast<int>( owned_extents[3] ) };
    int global_size[4] = { static_cast<int>( global_extents[Dim::K] ),
                           static_cast<int>( global_extents[Dim::J] ),
                           static_cast<int>( global_extents[Dim::I] ),
                           static_cast<int>( global_extents[3] ) };

    MPI_Datatype subarray;
    MPI_Type_create_subarray( 4, global_size, local_size, local_start,
                              MPI_ORDER_C, MpiTraits<value_type>::type(),
                              &subarray );

    return subarray;
}

//---------------------------------------------------------------------------//
/*!
  \brief Write a grid array to a VisIt BOV.

  This version writes a single output and does not use bricklets. We will do
  this in the future to improve parallel visualization.

  \param time_step_index The index of the time step we are writing.
  \param time The current time
  \param array The array to write
*/
template <class Array_t>
void writeTimeStep( const int time_step_index, const double time,
                    const Array_t &array )
{
    static_assert( isUniformMesh<typename Array_t::mesh_type>::value,
                   "ViSIT BOV writer can only be used with uniform mesh" );

    // Types
    using entity_type = typename Array_t::entity_type;
    using value_type = typename Array_t::value_type;
    using device_type = typename Array_t::device_type;
    using execution_space = typename device_type::execution_space;

    // Get the global grid.
    const auto &global_grid = array.layout()->localGrid()->globalGrid();

    // Get the global mesh.
    const auto &global_mesh = global_grid.globalMesh();

    // If this is a node field, determine periodicity so we can add the last
    // node back to the visualization if needed.
    std::array<long, 4> global_extents = { -1, -1, -1, -1 };
    for ( int d = 0; d < 3; ++d )
    {
        if ( std::is_same<entity_type, Cell>::value )
            global_extents[d] = global_grid.globalNumEntity( Cell(), d );
        else if ( std::is_same<entity_type, Node>::value )
            global_extents[d] = global_grid.globalNumEntity( Cell(), d ) + 1;
    }
    global_extents[3] = array.layout()->dofsPerEntity();
    auto owned_index_space = array.layout()->indexSpace( Own(), Local() );
    std::array<long, 4> owned_extents = { -1, -1, -1, -1 };
    for ( int d = 0; d < 3; ++d )
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
    owned_extents[3] = array.layout()->dofsPerEntity();

    // Create a contiguous array of the owned array values. Note that we
    // reorder to KJI grid ordering to conform to the BOV format.
    IndexSpace<4> local_space(
        { owned_index_space.min( Dim::I ), owned_index_space.min( Dim::J ),
          owned_index_space.min( Dim::K ), 0 },
        { owned_index_space.min( Dim::I ) + owned_extents[Dim::I],
          owned_index_space.min( Dim::J ) + owned_extents[Dim::J],
          owned_index_space.min( Dim::K ) + owned_extents[Dim::K],
          owned_extents[3] } );
    auto owned_subview = createSubview( array.view(), local_space );
    IndexSpace<4> reorder_space( { owned_extents[Dim::K], owned_extents[Dim::J],
                                   owned_extents[Dim::I], owned_extents[3] } );
    auto owned_view = createView<value_type, Kokkos::LayoutRight, device_type>(
        array.label(), reorder_space );
    Kokkos::parallel_for(
        "bov_reorder",
        createExecutionPolicy( reorder_space, execution_space() ),
        KOKKOS_LAMBDA( const int k, const int j, const int i, const int l ) {
            owned_view( k, j, i, l ) = owned_subview( i, j, k, l );
        } );
    Kokkos::fence();

    // Compose a data file name prefix.
    std::stringstream file_name;
    file_name << "grid_" << array.label() << "_" << std::setfill( '0' )
              << std::setw( 6 ) << time_step_index;

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
        header << "DATA_SIZE: " << global_extents[Dim::I] << " "
               << global_extents[Dim::J] << " " << global_extents[Dim::K]
               << std::endl;

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
        header << "BRICK_ORIGIN: " << global_mesh.lowCorner( Dim::I ) << " "
               << global_mesh.lowCorner( Dim::J ) << " "
               << global_mesh.lowCorner( Dim::K ) << std::endl;

        // Mesh global width
        header << "BRICK_SIZE: "
               << global_grid.globalNumEntity( Cell(), Dim::I ) *
                      global_mesh.cellSize( Dim::I )
               << " "
               << global_grid.globalNumEntity( Cell(), Dim::J ) *
                      global_mesh.cellSize( Dim::J )
               << " "
               << global_grid.globalNumEntity( Cell(), Dim::K ) *
                      global_mesh.cellSize( Dim::K )
               << std::endl;

        // Number of data components. Scalar and vector types are
        // supported.
        header << "DATA_COMPONENTS: " << global_extents[3] << std::endl;

        // Close the header.
        header.close();
    }
}

//---------------------------------------------------------------------------//

} // end namespace Experimental
} // end namespace BovWriter
} // end namespace Cajita

#endif // end CAJITA_BOVWRITER_HPP
