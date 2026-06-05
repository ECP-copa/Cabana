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

// References
// [1] Krüger, Timm, et al. The lattice Boltzmann method. Vol. 10. No. 978-3. Cham: Springer International Publishing, 2017.
// [2] Zou, Qisu, and Xiaoyi He. "On pressure and velocity boundary conditions for the lattice Boltzmann BGK model." Physics of fluids 9.6 (1997): 1591-1598.
// [3] Bhatnagar, P. L., Gross, E. P., and Krook, M. "A model for collision processes in gases." Physical Review 94.3 (1954): 511.
//     (BGK relaxation operator underlying the collision step)
//
// Algorithm outline — D2Q9 BGK-LBM with body force (EDM) and Zou-He walls
// -------------------------------------------------------------------------
// Each timestep:
//   1. Copy f -> f_old  (preserve pre-streaming distributions)
//
//   2. Streaming (pull scheme)
//        f_q(x, t+1) = f_old_q(x - c_q, t)
//      x-direction is periodic; y wrap is overwritten by BCs below.
//
//   3. Zou-He no-slip BC on bottom wall (y=0) and top wall (y=H)  [2]
//      Enforces zero wall velocity by deriving the three unknown
//      post-stream distributions from mass and momentum constraints.
//
//   4. BGK collision  [1, 3]
//        f_q(x, t+1) -= (1/tau) * (f_q - f_eq_q)
//      where f_eq_q = w_q * rho * [1 + (c_q·u)/cs² + (c_q·u)²/(2cs⁴) - u²/(2cs²)]
//
//   5. Exact-Difference Method (EDM) body force in x-direction  [1, §8.4]
//        f_q += w_q * rho * c_qx * g / cs²
//      Corrected macroscopic velocity: ux_phys = ux + g/2

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <iostream>




int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        // =====================================================
        // D2Q9 lattice constants — device-accessible Views
        // =====================================================

        Kokkos::View<int[9]>    d_cx("d2q9_cx");
        Kokkos::View<int[9]>    d_cy("d2q9_cy");
        Kokkos::View<double[9]> d_w ("d2q9_w");

        {
            auto h_cx = Kokkos::create_mirror_view(d_cx);
            auto h_cy = Kokkos::create_mirror_view(d_cy);
            auto h_w  = Kokkos::create_mirror_view(d_w);

            h_cx(0)= 0; h_cx(1)= 1; h_cx(2)= 0; h_cx(3)=-1; h_cx(4)= 0;
            h_cx(5)= 1; h_cx(6)=-1; h_cx(7)=-1; h_cx(8)= 1;

            h_cy(0)= 0; h_cy(1)= 0; h_cy(2)= 1; h_cy(3)= 0; h_cy(4)=-1;
            h_cy(5)= 1; h_cy(6)= 1; h_cy(7)=-1; h_cy(8)=-1;

            h_w(0) = 4.0/9.0;
            h_w(1) = 1.0/9.0;  h_w(2) = 1.0/9.0;  h_w(3) = 1.0/9.0;  h_w(4) = 1.0/9.0;
            h_w(5) = 1.0/36.0; h_w(6) = 1.0/36.0; h_w(7) = 1.0/36.0; h_w(8) = 1.0/36.0;

            Kokkos::deep_copy(d_cx, h_cx);
            Kokkos::deep_copy(d_cy, h_cy);
            Kokkos::deep_copy(d_w,  h_w);
        }

        // =====================================================
        // LBM parameters
        // =====================================================

        const int    Nx      = 100;
        const int    Ny      = 50;
        const double tau     = 0.6;
        const double g       = 1e-6;
        const double nu      = (tau - 0.5) / 3.0;
        const int    n_steps      = 90000;
        const int    write_every  = 1000;

        // =====================================================
        // Create grid
        // =====================================================

        std::array<double,2> low_corner  = { 0.0, 0.0 };
        std::array<double,2> high_corner = { (double)Nx, (double)Ny };
        std::array<int,2>    num_cell    = { Nx, Ny };

        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            low_corner, high_corner, num_cell );

        std::array<bool,2> periodic = { true, false };

        Cabana::Grid::DimBlockPartitioner<2> partitioner;

        auto global_grid = Cabana::Grid::createGlobalGrid(
            MPI_COMM_WORLD, global_mesh, periodic, partitioner );

        const int halo_width = 1;

        auto local_grid = Cabana::Grid::createLocalGrid(
            global_grid, halo_width );

        // =====================================================
        // Cell layouts
        // =====================================================

        auto scalar_layout = Cabana::Grid::createArrayLayout(
            global_grid, halo_width, 1, Cabana::Grid::Cell() );

        auto d2q9_layout = Cabana::Grid::createArrayLayout(
            global_grid, halo_width, 9, Cabana::Grid::Cell() );

        // =====================================================
        // Create fields
        // =====================================================

        auto rho   = Cabana::Grid::createArray<double>( "rho",   scalar_layout );
        auto ux    = Cabana::Grid::createArray<double>( "ux",    scalar_layout );
        auto uy    = Cabana::Grid::createArray<double>( "uy",    scalar_layout );
        auto f     = Cabana::Grid::createArray<double>( "f",     d2q9_layout );
        auto f_old = Cabana::Grid::createArray<double>( "f_old", d2q9_layout );

        auto Rho   = rho->view();
        auto Ux    = ux->view();
        auto Uy    = uy->view();
        auto F     = f->view();
        auto F_old = f_old->view();

        // =====================================================
        // Initialize: f = w[q] * rho, rho = 1, u = 0
        // =====================================================
        Cabana::Grid::grid_parallel_for(
            "initialize",
            Kokkos::DefaultExecutionSpace(),
            *local_grid,
            Cabana::Grid::Own(),
            Cabana::Grid::Cell(),
            KOKKOS_LAMBDA( const int i, const int j )
        {
            Rho(i,j,0) = 1.0;
            Ux(i,j,0)  = 0.0;
            Uy(i,j,0)  = 0.0;

            for ( int q = 0; q < 9; ++q )
                F(i,j,q) = d_w(q);
        });

        Kokkos::fence();

        // =====================================================
        // Time loop
        // =====================================================
        for ( int step = 0; step < n_steps; ++step )
        {
            // ── 1. Copy F → F_old ─────────────────────────────────
            Kokkos::deep_copy( F_old, F );

            // ── 2. Streaming: pull scheme with periodic wrap ───────
            Cabana::Grid::grid_parallel_for(
                "stream",
                Kokkos::DefaultExecutionSpace(),
                *local_grid,
                Cabana::Grid::Own(),
                Cabana::Grid::Cell(),
                KOKKOS_LAMBDA( const int i, const int j )
            {
                for ( int q = 0; q < 9; ++q )
                {
                    const int i0 = i - halo_width;
                    const int j0 = j - halo_width;
                    const int si = ( i0 - d_cx(q) + Nx ) % Nx + halo_width;
                    const int sj = ( j0 - d_cy(q) + Ny ) % Ny + halo_width;
                    F(i,j,q) = F_old(si, sj, q);
                }
            });

            Kokkos::fence();

            // ── 3. Zou-He no-slip BC — bottom wall ────────────────
            //    Physical j = halo_width (logical j=0).
            //    f2, f5, f6 are unknown after streaming; effective wall
            //    velocity = -g/2 so physical ux = 0.
            Kokkos::parallel_for(
                "zou_he_bottom",
                Kokkos::RangePolicy<>( halo_width, halo_width + Nx ),
                KOKKOS_LAMBDA( const int i )
            {
                const int    j  = halo_width;
                const double f0 = F(i,j,0), f1 = F(i,j,1), f3 = F(i,j,3);
                const double f4 = F(i,j,4), f7 = F(i,j,7), f8 = F(i,j,8);
                const double r  = f0 + f1 + f3 + 2.0*(f4 + f7 + f8);
                F(i,j,2) = f4;
                F(i,j,5) = f7 - 0.5*(f1 - f3) - 0.25*g*r;
                F(i,j,6) = f8 + 0.5*(f1 - f3) + 0.25*g*r;
            });

            // ── 4. Zou-He no-slip BC — top wall ───────────────────
            //    Physical j = halo_width + Ny - 1 (logical j=Ny-1).
            //    f4, f7, f8 are unknown after streaming.
            Kokkos::parallel_for(
                "zou_he_top",
                Kokkos::RangePolicy<>( halo_width, halo_width + Nx ),
                KOKKOS_LAMBDA( const int i )
            {
                const int    j  = halo_width + Ny - 1;
                const double f0 = F(i,j,0), f1 = F(i,j,1), f3 = F(i,j,3);
                const double f2 = F(i,j,2), f5 = F(i,j,5), f6 = F(i,j,6);
                const double r  = f0 + f1 + f3 + 2.0*(f2 + f5 + f6);
                F(i,j,4) = f2;
                F(i,j,7) = f5 + 0.5*(f1 - f3) + 0.25*g*r;
                F(i,j,8) = f6 - 0.5*(f1 - f3) - 0.25*g*r;
            });

            Kokkos::fence();

            // ── 5. BGK collision + EDM body force ─────────────────
            //    EDM: S_k = w_k * rho * (c_k · g) / cs²
            //    Added after BGK with no 1/tau factor.
            //    Ux corrected to midpoint velocity (u + g/2) for output.
            Cabana::Grid::grid_parallel_for(
                "collide",
                Kokkos::DefaultExecutionSpace(),
                *local_grid,
                Cabana::Grid::Own(),
                Cabana::Grid::Cell(),
                KOKKOS_LAMBDA( const int i, const int j )
            {
                const double cs2 = 1.0 / 3.0;

                // Macroscopic fields
                double r = 0.0, vx = 0.0, vy = 0.0;
                for ( int q = 0; q < 9; ++q )
                {
                    const double fq = F(i,j,q);
                    r  += fq;
                    vx += d_cx(q) * fq;
                    vy += d_cy(q) * fq;
                }
                Rho(i,j,0) = r;
                const double ux_loc = vx / r;
                const double uy_loc = vy / r;

                // Equilibrium + BGK relaxation
                const double u_sq = ux_loc*ux_loc + uy_loc*uy_loc;
                for ( int q = 0; q < 9; ++q )
                {
                    const double cu    = d_cx(q)*ux_loc + d_cy(q)*uy_loc;
                    const double feq_q = d_w(q) * r * (
                        1.0 + cu/cs2 + cu*cu/(2.0*cs2*cs2) - u_sq/(2.0*cs2)
                    );
                    F(i,j,q) -= (1.0/tau) * (F(i,j,q) - feq_q);
                }

                // EDM body force (g acts in x-direction only)
                for ( int q = 0; q < 9; ++q )
                    F(i,j,q) += d_w(q) * r * d_cx(q) * g / cs2;

                // Corrected velocity for post-processing
                Ux(i,j,0) = ux_loc + 0.5*g;
                Uy(i,j,0) = uy_loc;
            });

            Kokkos::fence();

            if ( ( step + 1 ) % write_every == 0 )
            {
                Cabana::Grid::Experimental::BovWriter::writeTimeStep(
                    "poiseuille_rho", step + 1, (double)( step + 1 ), *rho );
                Cabana::Grid::Experimental::BovWriter::writeTimeStep(
                    "poiseuille_ux",  step + 1, (double)( step + 1 ), *ux  );
                Cabana::Grid::Experimental::BovWriter::writeTimeStep(
                    "poiseuille_uy",  step + 1, (double)( step + 1 ), *uy  );
            }
        }

        // =====================================================
        // Write output fields via BOV writer
        // =====================================================

        Cabana::Grid::Experimental::BovWriter::writeTimeStep(
            "poiseuille_rho", n_steps, (double)n_steps, *rho );
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(
            "poiseuille_ux",  n_steps, (double)n_steps, *ux  );
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(
            "poiseuille_uy",  n_steps, (double)n_steps, *uy  );

        // =====================================================
        // Validation: compare to analytical Poiseuille profile
        //   u(y) = g/(2*nu) * y * (H - y),  H = Ny-1
        // =====================================================

        auto ux_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), Ux );

        const int    H       = Ny - 1;
        double       ux_max  = 0.0;
        double       err_max = 0.0;

        // Physical index j runs [halo_width, halo_width+Ny); logical j = j_phys - halo_width
        std::cout << "\n  j    ux_sim         ux_analytical  err\n";
        for ( int j = halo_width; j < halo_width + Ny; ++j )
        {
            const int j_log = j - halo_width;

            double ux_avg = 0.0;
            for ( int i = halo_width; i < halo_width + Nx; ++i )
                ux_avg += ux_host(i, j, 0);
            ux_avg /= Nx;

            const double y_d   = (double)j_log;
            const double ux_an = g / (2.0 * nu) * y_d * (H - y_d);
            const double err   = std::abs( ux_avg - ux_an );

            std::cout << "  " << j_log
                      << "  " << ux_avg
                      << "  " << ux_an
                      << "  " << err
                      << "\n";

            if ( ux_avg  > ux_max  ) ux_max  = ux_avg;
            if ( err     > err_max ) err_max  = err;
        }

        std::cout << "\nux_max  = " << ux_max
                  << "   Ma = " << ux_max / std::sqrt(1.0/3.0) << "\n";
        std::cout << "err_max = " << err_max << "\n";
    }

    MPI_Finalize();

    return 0;
}
