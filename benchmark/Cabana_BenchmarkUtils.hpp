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

#include <Cabana_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
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

//---------------------------------------------------------------------------//
// Generate random particles.
template <class Slice>
void createParticles( Slice& x, const double x_min, const double x_max,
                      const double min_dist )
{
    auto num_particles = x.size();
    double min_dist_sqr = min_dist * min_dist;
    std::default_random_engine rand_gen;
    std::uniform_real_distribution<double> rand_dist( x_min, x_max );

    // Create some coarse bins to accelerate construction.
    int nbinx = 15;
    int nbin = nbinx * nbinx * nbinx;
    double bin_dx = ( x_max - x_min ) / nbinx;
    std::vector<std::vector<int>> bins( nbin );
    auto bin_id = [=]( const int i, const int j, const int k ) {
        return k * nbinx * nbinx + j * nbinx + i;
    };

    // Seed the distribution with a particle at the origin.
    x( 0, 0 ) = ( x_max - x_min ) / 2.0;
    x( 0, 1 ) = ( x_max - x_min ) / 2.0;
    x( 0, 2 ) = ( x_max - x_min ) / 2.0;
    int p0_i = std::floor( ( x( 0, 0 ) - x_min ) / bin_dx );
    int p0_j = std::floor( ( x( 0, 1 ) - x_min ) / bin_dx );
    int p0_k = std::floor( ( x( 0, 2 ) - x_min ) / bin_dx );
    bins[bin_id( p0_i, p0_j, p0_k )].push_back( 0 );

    // Create particles. Only add particles that are outside a minimum
    // distance from other particles.
    for ( std::size_t p = 1; p < num_particles; ++p )
    {
        if ( 0 == ( p - 1 ) % ( num_particles / 4 ) )
            std::cout << "Inserting " << p << " / " << num_particles
                      << std::endl;

        bool found_neighbor = true;

        // Keep trying new random coordinates until we insert one that is not
        // within the minimum distance of any other particle.
        while ( found_neighbor )
        {
            found_neighbor = false;

            // Create particle coordinates.
            x( p, 0 ) = rand_dist( rand_gen );
            x( p, 1 ) = rand_dist( rand_gen );
            x( p, 2 ) = rand_dist( rand_gen );

            // Figure out which bin we are in.
            int bin_i = std::floor( ( x( p, 0 ) - x_min ) / bin_dx );
            int bin_j = std::floor( ( x( p, 1 ) - x_min ) / bin_dx );
            int bin_k = std::floor( ( x( p, 2 ) - x_min ) / bin_dx );
            int i_min = ( 0 < bin_i ) ? bin_i - 1 : 0;
            int j_min = ( 0 < bin_j ) ? bin_j - 1 : 0;
            int k_min = ( 0 < bin_k ) ? bin_k - 1 : 0;
            int i_max = ( nbinx > bin_i ) ? bin_i + 1 : nbinx;
            int j_max = ( nbinx > bin_j ) ? bin_j + 1 : nbinx;
            int k_max = ( nbinx > bin_k ) ? bin_k + 1 : nbinx;

            // Search the adjacent bins for neighbors.
            for ( int i = i_min; i < i_max; ++i )
            {
                for ( int j = j_min; j < j_max; ++j )
                {
                    for ( int k = k_min; k < k_max; ++k )
                    {
                        for ( auto& n : bins[bin_id( i, j, k )] )
                        {
                            double dx = x( n, 0 ) - x( p, 0 );
                            double dy = x( n, 1 ) - x( p, 1 );
                            double dz = x( n, 2 ) - x( p, 2 );
                            double dist = dx * dx + dy * dy + dz * dz;

                            if ( dist < min_dist_sqr )
                            {
                                found_neighbor = true;
                                break;
                            }
                        }
                    }
                    if ( found_neighbor )
                        break;
                }
                if ( found_neighbor )
                    break;
            }

            // Add the particle to its bin if we didn't find any neighbors.
            if ( !found_neighbor )
                bins[bin_id( bin_i, bin_j, bin_k )].push_back( p );
        }
    }
    std::cout << std::endl;
}

} // end namespace Benchmark
} // end namespace Cabana

//---------------------------------------------------------------------------//
