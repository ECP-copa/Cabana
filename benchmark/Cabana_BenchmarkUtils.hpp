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

#include <Cabana_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef Cabana_ENABLE_MPI
#include <mpi.h>
#endif

namespace Cabana
{
namespace Benchmark
{
//---------------------------------------------------------------------------//
// Local timer. Carries multiple data points (the independent variable in
// the parameter sweep) for each timer to allow for parametric sweeps. Each
// timer can do multiple runs over each data point in the parameter sweep. The
// name of the data point and its values can then be injected into the output
// table.
class Timer
{
  public:
    // Create the timer.
    Timer( const std::string& name, const int num_data )
        : _name( name )
        , _starts( num_data )
        , _data( num_data )
        , _is_stopped( num_data, true )
    {
    }

    // Start the timer for the given data point.
    void start( const int data_point )
    {
        if ( !_is_stopped[data_point] )
            throw std::logic_error( "attempted to start a running timer" );
        _starts[data_point] = std::chrono::high_resolution_clock::now();
        _is_stopped[data_point] = false;
    }

    // Stop the timer at the given data point.
    void stop( const int data_point )
    {
        if ( _is_stopped[data_point] )
            throw std::logic_error( "attempted to stop a stopped timer" );
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> fp_micro =
            now - _starts[data_point];
        _data[data_point].push_back( fp_micro.count() );
        _is_stopped[data_point] = true;
    }

  public:
    std::string _name;
    std::vector<std::chrono::high_resolution_clock::time_point> _starts;
    std::vector<std::vector<double>> _data;
    std::vector<bool> _is_stopped;
};

//---------------------------------------------------------------------------//
// Local output.
// Write timer results. Provide the values of the data points so
// they can be injected into the table.
template <class Scalar>
void outputResults( std::ostream& stream, const std::string& data_point_name,
                    const std::vector<Scalar>& data_point_vals,
                    const Timer& timer )
{
    // Write the data header.
    stream << "\n";
    stream << timer._name << "\n";
    stream << data_point_name << " min max ave"
           << "\n";

    // Write out each data point
    for ( std::size_t n = 0; n < timer._data.size(); ++n )
    {
        if ( !timer._is_stopped[n] )
            throw std::logic_error(
                "attempted to output from a running timer" );

        // Compute the minimum.
        double local_min =
            *std::min_element( timer._data[n].begin(), timer._data[n].end() );

        // Compute the maximum.
        double local_max =
            *std::max_element( timer._data[n].begin(), timer._data[n].end() );

        // Compute the average.
        double local_sum = std::accumulate( timer._data[n].begin(),
                                            timer._data[n].end(), 0.0 );
        double average = local_sum / timer._data[n].size();

        // Output.
        stream << data_point_vals[n] << " " << local_min << " " << local_max
               << " " << average << "\n";
    }
}

//---------------------------------------------------------------------------//
// Parallel output.
// Write timer results on rank 0. Provide the values of the data points so
// they can be injected into the table. This function does collective
// communication.
#ifdef Cabana_ENABLE_MPI
template <class Scalar>
void outputResults( std::ostream& stream, const std::string& data_point_name,
                    const std::vector<Scalar>& data_point_vals,
                    const Timer& timer, MPI_Comm comm )
{
    // Get comm rank;
    int comm_rank;
    MPI_Comm_rank( comm, &comm_rank );

    // Get comm size;
    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    // Write the data header.
    if ( 0 == comm_rank )
    {
        stream << "\n";
        stream << timer._name << "\n";
        stream << "num_rank " << data_point_name << " min max ave"
               << "\n";
    }

    // Write out each data point
    for ( std::size_t n = 0; n < timer._data.size(); ++n )
    {
        if ( !timer._is_stopped[n] )
            throw std::logic_error(
                "attempted to output from a running timer" );

        // Compute the minimum.
        double local_min =
            *std::min_element( timer._data[n].begin(), timer._data[n].end() );
        double global_min = 0.0;
        MPI_Reduce( &local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm );

        // Compute the maximum.
        double local_max =
            *std::max_element( timer._data[n].begin(), timer._data[n].end() );
        double global_max = 0.0;
        MPI_Reduce( &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm );

        // Compute the average.
        double local_sum = std::accumulate( timer._data[n].begin(),
                                            timer._data[n].end(), 0.0 );
        double average = 0.0;
        MPI_Reduce( &local_sum, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm );
        average /= timer._data[n].size() * comm_size;

        // Output on rank 0.
        if ( 0 == comm_rank )
        {
            stream << comm_size << " " << data_point_vals[n] << " "
                   << global_min << " " << global_max << " " << average << "\n";
        }
    }
}
#endif

//---------------------------------------------------------------------------//

} // end namespace Benchmark
} // end namespace Cabana

//---------------------------------------------------------------------------//
