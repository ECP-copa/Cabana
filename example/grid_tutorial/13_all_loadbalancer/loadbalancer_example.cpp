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

#include <Cabana_Grid.hpp>

#include <cmath>
#include <iostream>

//---------------------------------------------------------------------------//
// Helper function.
//---------------------------------------------------------------------------//
// Creates an artificial work for a rectangular domain according to a Gaussian
// work density.
double workGauss( const std::array<double, 4>& verts,
                  std::array<double, 2>& global_low_corner,
                  std::array<double, 2>& global_high_corner )
{
    // The work density function is a radial gauss curve centred at (x0,y0)
    // work density function: r = sqrt((x-x0)^2+(y-y0)^2)
    //                        wd = 1/(sigma sqrt(2Pi)) Exp(-.5*( r/sigma )^2 )
    // -.5*sqrt(Pi/2) sigma (
    //                        Erf( (x1-x0)/(sqrt(2) sigma) ) -
    //                        Erf( (x2-x0)/(sqrt(2) sigma) )
    //                      ) (
    //                        Erf( (y1-y0)/(sqrt(2) sigma) ) -
    //                        Erf( (y2-y0)/(sqrt(2) sigma) )
    //                      )
    const double x1 = verts[0];
    const double y1 = verts[1];
    const double x2 = verts[2];
    const double y2 = verts[3];
    const double x0 = ( global_low_corner[0] + global_high_corner[0] ) / 2.;
    const double y0 = ( global_low_corner[1] + global_high_corner[1] ) / 2.;
    const double sigma = 1.;
    const double s2sigma = 1. / ( sqrt( 2. ) * sigma );
    const double Erfx1 = erf( ( x1 - x0 ) * s2sigma );
    const double Erfx2 = erf( ( x2 - x0 ) * s2sigma );
    const double Erfy1 = erf( ( y0 - y1 ) * s2sigma );
    const double Erfy2 = erf( ( y0 - y2 ) * s2sigma );
    return -0.5 * sqrt( 0.5 * M_PI ) * sigma * ( Erfx1 - Erfx2 ) *
           ( Erfy1 - Erfy2 );
}
//---------------------------------------------------------------------------//
// Loadbalancer example.
//---------------------------------------------------------------------------//
void loadBalancerExample()
{
    /*
     * The LoadBalancer class generates a new GlobalGrid in order to minimize
     * differences in the local work across MPI ranks. This capability relies on
     * the optional ALL library dependency. Most is the same as in any
     * application using Cabana::Grid without load balancing; the comments will
     * be focused on the additions/changes due to including the load balancer.
     */

    /*
     * The example system is 2D and its size based on the number of ranks. Every
     * rank will have the same initial 47 x 38 domain. The cell size is chosen
     * at arbitrarily as is the offset of the system (global_low_corner). The
     * system is non periodic. These choices are arbitrary with regards to the
     * load balancer and were chosen for ease of use/consistency for different
     * number of ranks.
     *
     * The partitioner, global mesh, and (initial) global grid are set up as
     * usual.
     */
    const Cabana::Grid::DimBlockPartitioner<2> partitioner;

    std::array<int, 2> empty_array = { 0 };
    std::array<int, 2> ranks =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, empty_array );
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 47 * ranks[0], 38 * ranks[0] };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    std::array<bool, 2> is_dim_periodic = { false, false };
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Get the current rank for printing output.
    int comm_rank = global_grid->blockId();
    if ( comm_rank == 0 )
    {
        std::cout << "Cabana::Grid Load Balancer Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    /*
     * The load balancer is initialized using the global grid and MPI
     * communicator. The additional minimum domain size is optional and, if
     * given, can be a scalar value or array for each dimension. The domain
     * decomposition from the global grid is chosen as the starting point, which
     * is equisized.
     */
    const std::array<double, 2> min_domain_size = { 3 * cell_size,
                                                    5 * cell_size };
    auto lb = Cabana::Grid::Experimental::createLoadBalancer(
        MPI_COMM_WORLD, global_grid, min_domain_size );

    /*
     * The vertices are retrieved in this example to calculate the work of a
     * domain. The work is the quantity that is balanced across processes.
     * Usually the work is a measure or approximation of the workload of a
     * domain: in particle simulations the number of particles, in particle-grid
     * simulations some combination of both. It can also be the measured time
     * taken for the calculations of the domain, although this should not
     * include time waiting for other processes.
     */
    std::array<double, 4> vertices = lb->getVertices();
    for ( std::size_t step = 0; step < 20; ++step )
    {
        /*
         * During the simulation, every time the domains need to be recreated,
         * calculate the current work of the domain, then let the load balancer
         * create a new global grid that is more optimally balanced.
         */
        double work =
            workGauss( vertices, global_low_corner, global_high_corner );
        global_grid =
            lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
        vertices = lb->getVertices();
        /*
         * The imbalance is a measure showing the difference between the
         * minimum and maximum work of a process.
         */
        double imbalance = lb->getImbalance();
        if ( comm_rank == 0 )
            printf( "LB Imbalance: %g\n", imbalance );
        /*
         * After the new global grid is created, all classes that depend on it
         * or its dependents have to be recreated, such as the local grid,
         * local mesh, grid array layout, and the grid arrays themselves.
         */
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        loadBalancerExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
