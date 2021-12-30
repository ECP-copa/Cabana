/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_SILOPARTICLEOUTPUT_HPP
#define CABANA_SILOPARTICLEOUTPUT_HPP

#include <Kokkos_Core.hpp>

#include <silo.h>

#include <mpi.h>

#include <pmpio.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace SiloParticleOutput
{
//---------------------------------------------------------------------------//
// Silo Particle Field Output.
//---------------------------------------------------------------------------//
// Format traits.
template <typename T>
struct SiloTraits;

template <>
struct SiloTraits<short>
{
    static int type() { return DB_SHORT; }
};

template <>
struct SiloTraits<int>
{
    static int type() { return DB_INT; }
};

template <>
struct SiloTraits<float>
{
    static int type() { return DB_FLOAT; }
};

template <>
struct SiloTraits<double>
{
    static int type() { return DB_DOUBLE; }
};

namespace Impl
{
//---------------------------------------------------------------------------//
// Rank-0 field
template <class SliceType>
void writeFieldsImpl(
    DBfile* silo_file, const std::string& mesh_name, const SliceType& slice,
    typename std::enable_if<
        2 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    // Reorder in a contiguous blocked format.
    Kokkos::View<typename SliceType::value_type*,
                 typename SliceType::device_type>
        view( "field", slice.size() );
    Kokkos::parallel_for(
        "Cabana::SiloParticleOutput::writeFieldRank0",
        Kokkos::RangePolicy<typename SliceType::execution_space>(
            0, slice.size() ),
        KOKKOS_LAMBDA( const int i ) { view( i ) = slice( i ); } );

    // Mirror the field to the host.
    auto host_view =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view );

    // Write the field.
    DBPutPointvar1( silo_file, slice.label().c_str(), mesh_name.c_str(),
                    host_view.data(), host_view.extent( 0 ),
                    SiloTraits<typename SliceType::value_type>::type(),
                    nullptr );
}

// Rank-1 field
template <class SliceType>
void writeFieldsImpl(
    DBfile* silo_file, const std::string& mesh_name, const SliceType& slice,
    typename std::enable_if<
        3 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    // Reorder in a contiguous blocked format.
    Kokkos::View<typename SliceType::value_type**, Kokkos::LayoutLeft,
                 typename SliceType::device_type>
        view( "field", slice.size(), slice.extent( 2 ) );
    Kokkos::parallel_for(
        "Cabana::SiloParticleOutput::writeFieldRank1",
        Kokkos::RangePolicy<typename SliceType::execution_space>(
            0, slice.size() ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                view( i, d0 ) = slice( i, d0 );
        } );

    // Mirror the field to the host.
    auto host_view =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view );

    // Get the data pointers.
    std::vector<typename SliceType::value_type*> ptrs( host_view.extent( 1 ) );
    for ( std::size_t d0 = 0; d0 < host_view.extent( 1 ); ++d0 )
        ptrs[d0] = &host_view( 0, d0 );

    // Write the field.
    DBPutPointvar( silo_file, slice.label().c_str(), mesh_name.c_str(),
                   host_view.extent( 1 ), ptrs.data(), host_view.extent( 0 ),
                   SiloTraits<typename SliceType::value_type>::type(),
                   nullptr );
}

// Rank-2 field
template <class SliceType>
void writeFieldsImpl(
    DBfile* silo_file, const std::string& mesh_name, const SliceType& slice,
    typename std::enable_if<
        4 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    // Reorder in a contiguous blocked format.
    Kokkos::View<typename SliceType::value_type***, Kokkos::LayoutLeft,
                 typename SliceType::device_type>
        view( "field", slice.size(), slice.extent( 2 ), slice.extent( 3 ) );
    Kokkos::parallel_for(
        "Cabana::SiloParticleOutput::writeFieldRank2",
        Kokkos::RangePolicy<typename SliceType::execution_space>(
            0, slice.size() ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                for ( std::size_t d1 = 0; d1 < slice.extent( 3 ); ++d1 )
                    view( i, d0, d1 ) = slice( i, d0, d1 );
        } );

    // Mirror the field to the host.
    auto host_view =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view );

    // Get the data pointers.
    std::vector<typename SliceType::value_type*> ptrs;
    ptrs.reserve( host_view.extent( 1 ) * host_view.extent( 2 ) );
    for ( unsigned d0 = 0; d0 < host_view.extent( 1 ); ++d0 )
        for ( unsigned d1 = 0; d1 < host_view.extent( 2 ); ++d1 )
            ptrs.push_back( &host_view( 0, d0, d1 ) );

    // Write the field.
    DBPutPointvar( silo_file, slice.label().c_str(), mesh_name.c_str(),
                   host_view.extent( 1 ) * host_view.extent( 2 ), ptrs.data(),
                   host_view.extent( 0 ),
                   SiloTraits<typename SliceType::value_type>::type(),
                   nullptr );
}

} // namespace Impl

template <class SliceType>
void writeFields( DBfile* silo_file, const std::string& mesh_name,
                  const SliceType& slice )
{
    writeFieldsImpl( silo_file, mesh_name, slice );
}

template <class SliceType, class... FieldSliceTypes>
void writeFields( DBfile* silo_file, const std::string& mesh_name,
                  const SliceType& slice, FieldSliceTypes&&... fields )
{
    writeFieldsImpl( silo_file, mesh_name, slice );
    writeFields( silo_file, mesh_name, fields... );
}

//---------------------------------------------------------------------------//
// parallel i/o callbacks
void* createFile( const char* file_name, const char* dir_name, void* user_data )
{
    std::ignore = user_data;
    DBfile* silo_file =
        DBCreate( file_name, DB_CLOBBER, DB_LOCAL, nullptr, DB_PDB );
    if ( silo_file )
    {
        DBMkDir( silo_file, dir_name );
        DBSetDir( silo_file, dir_name );
    }

    return (void*)silo_file;
}

void* openFile( const char* file_name, const char* dir_name,
                PMPIO_iomode_t io_mode, void* user_data )
{
    std::ignore = io_mode;
    std::ignore = user_data;
    DBfile* silo_file = DBOpen( file_name, DB_PDB, DB_APPEND );
    if ( silo_file )
    {
        DBMkDir( silo_file, dir_name );
        DBSetDir( silo_file, dir_name );
    }
    return (void*)silo_file;
}

void closeFile( void* file, void* user_data )
{
    std::ignore = user_data;
    DBfile* silo_file = (DBfile*)file;
    if ( silo_file )
        DBClose( silo_file );
}

namespace Impl
{
//---------------------------------------------------------------------------//
// Get field names.
template <class SliceType>
void getFieldNamesImpl( std::vector<std::string>& names,
                        const SliceType& slice )
{
    names.push_back( slice.label() );
}

// Get field names.
template <class SliceType, class... FieldSliceTypes>
void getFieldNamesImpl( std::vector<std::string>& names, const SliceType& slice,
                        FieldSliceTypes&&... fields )
{
    getFieldNamesImpl( names, slice );
    getFieldNamesImpl( names, fields... );
}

template <class... FieldSliceTypes>
std::vector<std::string> getFieldNames( FieldSliceTypes&&... fields )
{
    std::vector<std::string> names;
    getFieldNamesImpl( names, fields... );
    return names;
}

} // namespace Impl

//---------------------------------------------------------------------------//
// Write a multimesh hierarchy.
template <class... FieldSliceTypes>
void writeMultiMesh( PMPIO_baton_t* baton, DBfile* silo_file,
                     const int comm_size, const std::string& mesh_name,
                     const int time_step_index, const double time,
                     FieldSliceTypes&&... fields )
{
    // Go to the root directory of the file.
    DBSetDir( silo_file, "/" );

    // Create the mesh block names.
    std::vector<std::string> mb_names;
    for ( int r = 0; r < comm_size; ++r )
    {
        int group_rank = PMPIO_GroupRank( baton, r );
        if ( 0 == group_rank )
        {
            std::stringstream bname;
            bname << "rank_" << r << "/" << mesh_name;
            mb_names.push_back( bname.str() );
        }
        else
        {
            std::stringstream bname;
            bname << "particles_" << time_step_index << "_group_" << group_rank
                  << ".silo:/rank_" << r << "/" << mesh_name;
            mb_names.push_back( bname.str() );
        }
    }
    char** mesh_block_names = (char**)malloc( comm_size * sizeof( char* ) );
    for ( int r = 0; r < comm_size; ++r )
        mesh_block_names[r] = const_cast<char*>( mb_names[r].c_str() );

    std::vector<int> mesh_block_types( comm_size, DB_POINTMESH );

    // Get the names of the fields.
    std::vector<std::string> field_names = getFieldNames( fields... );

    // Create the field block names.
    int num_field = field_names.size();
    std::vector<std::vector<std::string>> fb_names( num_field );
    for ( int f = 0; f < num_field; ++f )
    {
        for ( int r = 0; r < comm_size; ++r )
        {
            int group_rank = PMPIO_GroupRank( baton, r );
            if ( 0 == group_rank )
            {
                std::stringstream bname;
                bname << "rank_" << r << "/" << field_names[f];
                fb_names[f].push_back( bname.str() );
            }
            else
            {

                std::stringstream bname;
                bname << "particles_" << time_step_index << "_group_"
                      << group_rank << ".silo:/rank_" << r << "/"
                      << field_names[f];
                fb_names[f].push_back( bname.str() );
            }
        }
    }

    std::vector<char**> field_block_names( num_field );
    for ( int f = 0; f < num_field; ++f )
    {
        field_block_names[f] = (char**)malloc( comm_size * sizeof( char* ) );
        for ( int r = 0; r < comm_size; ++r )
            field_block_names[f][r] =
                const_cast<char*>( fb_names[f][r].c_str() );
    }

    std::vector<int> field_block_types( comm_size, DB_POINTVAR );

    // Create options.
    DBoptlist* options = DBMakeOptlist( 1 );
    DBAddOption( options, DBOPT_DTIME, (void*)&time );
    DBAddOption( options, DBOPT_CYCLE, (void*)&time_step_index );

    // Add the multiblock mesh.
    std::stringstream mbname;
    mbname << "multi_" << mesh_name;
    DBPutMultimesh( silo_file, mbname.str().c_str(), comm_size,
                    mesh_block_names, mesh_block_types.data(), options );

    // Add the multiblock fields.
    for ( int f = 0; f < num_field; ++f )
    {
        std::stringstream mfname;
        mfname << "multi_" << field_names[f];
        DBPutMultivar( silo_file, mfname.str().c_str(), comm_size,
                       field_block_names[f], field_block_types.data(),
                       options );
    }

    // Cleanup.
    free( mesh_block_names );
    for ( auto& f_name : field_block_names )
        free( f_name );
    DBFreeOptlist( options );
}

//---------------------------------------------------------------------------//
// Write a time step.
template <class GlobalGridType, class CoordSliceType, class... FieldSliceTypes>
void writeTimeStep( const GlobalGridType& global_grid,
                    const int time_step_index, const double time,
                    const CoordSliceType& coords, FieldSliceTypes&&... fields )
{
    // Pick a number of groups. We want to write approximately the N^3 blocks
    // to N^2 groups. Pick the block dimension with the largest number of
    // ranks as the number of groups. We may want to tweak this as an optional
    // input later with this behavior as the default.
    int num_group = 0;
    for ( int d = 0; d < 3; ++d )
        if ( global_grid.dimNumBlock( d ) > num_group )
            num_group = global_grid.dimNumBlock( d );

    // Create the parallel baton.
    int mpi_tag = 1948;
    PMPIO_baton_t* baton =
        PMPIO_Init( num_group, PMPIO_WRITE, global_grid.comm(), mpi_tag,
                    createFile, openFile, closeFile, nullptr );

    // Compose a data file name.
    int comm_rank;
    MPI_Comm_rank( global_grid.comm(), &comm_rank );
    int group_rank = PMPIO_GroupRank( baton, comm_rank );
    std::stringstream file_name;

    // Group 0 writes a master file for the time step.
    if ( 0 == group_rank )
        file_name << "particles_" << time_step_index << ".silo";

    // The other groups write auxiliary files.
    else
        file_name << "particles_" << time_step_index << "_group_" << group_rank
                  << ".silo";

    // Compose a directory name.
    std::stringstream dir_name;
    dir_name << "rank_" << comm_rank;

    // Wait for our turn to write to the file.
    DBfile* silo_file = (DBfile*)PMPIO_WaitForBaton(
        baton, file_name.str().c_str(), dir_name.str().c_str() );

    // Reorder the coordinates in a blocked format.
    Kokkos::View<typename CoordSliceType::value_type**, Kokkos::LayoutLeft,
                 typename CoordSliceType::device_type>
        view( "coords", coords.size(), coords.extent( 2 ) );
    Kokkos::parallel_for(
        "Cabana::SiloParticleOutput::writeCoords",
        Kokkos::RangePolicy<typename CoordSliceType::execution_space>(
            0, coords.size() ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < coords.extent( 2 ); ++d0 )
                view( i, d0 ) = coords( i, d0 );
        } );

    // Mirror the coordinates to the host.
    auto host_coords =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view );

    // Add the point mesh.
    std::string mesh_name = "particles";
    double* ptrs[3] = { &host_coords( 0, 0 ), &host_coords( 0, 1 ),
                        &host_coords( 0, 2 ) };
    DBPutPointmesh( silo_file, mesh_name.c_str(), host_coords.extent( 1 ), ptrs,
                    host_coords.extent( 0 ),
                    SiloTraits<typename CoordSliceType::value_type>::type(),
                    nullptr );

    // Add variables.
    writeFields( silo_file, mesh_name, fields... );

    // Root rank writes the global multimesh hierarchy for parallel
    // simulations.
    int comm_size;
    MPI_Comm_size( global_grid.comm(), &comm_size );
    if ( 0 == comm_rank && comm_size > 1 )
        writeMultiMesh( baton, silo_file, comm_size, mesh_name, time_step_index,
                        time, fields... );

    // Hand off the baton.
    PMPIO_HandOffBaton( baton, silo_file );

    // Finish.
    PMPIO_Finish( baton );
}

//---------------------------------------------------------------------------//

} // namespace SiloParticleOutput
} // end namespace Cabana

#endif // CABANA_SILOPARTICLEOUTPUT_HPP
