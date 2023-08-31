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

#include <iostream>

//---------------------------------------------------------------------------//
// Global Mesh example.
//---------------------------------------------------------------------------//
void globalMeshExample()
{
    /*
      All meshes in Cabana::Grid are logically recitlinear: each mesh cell is
      either cubic or rectangular (either uniform or non-uniform) and can
      directly index it's 26 nearest neighbors (in 3D or 8 in 2D), even though
      they may not be directly adjacent (e.g. if it is sparse).
    */
    std::cout << "Cabana::Grid Global Mesh Example\n" << std::endl;

    /*
      The simplest Cabana::Grid mesh is uniform, defined by a constant cell
      size; however, this cell size can be different per dimension.
    */
    double cell_size = 0.23;
    std::array<double, 3> cell_size_array = { 0.23, 0.19, 0.05 };

    /*
      The number of cells defines the computational size of the system and also
      does not need to be uniform for a uniform mesh (only a constant cell size
      per dimension).
    */
    std::array<int, 3> global_num_cell = { 22, 19, 21 };

    /*
      The low and high corners define the extent of the box. Here the cell size
      and total number of cells per dimension are calculated to make the box
      extents consistent.
    */
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };

    /*
      Now create the global mesh with the low and high corners and total cells.
    */
    auto global_mesh_num_cell = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    /*
      Instead create the global mesh with the cell size.
    */
    auto global_mesh_cell_size = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    /*
      Finally, the mesh can be created with a non-uniform cell size per
      dimension instead.
    */
    std::array<double, 3> global_high_corner_2 = {
        global_low_corner[0] + cell_size_array[0] * global_num_cell[0],
        global_low_corner[1] + cell_size_array[1] * global_num_cell[1],
        global_low_corner[2] + cell_size_array[2] * global_num_cell[2] };
    auto global_mesh_cell_size_array = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner_2, cell_size_array );

    /*
      The mesh details can now be extracted for calculations: the corner
      information used to create the mesh, the extent in each dimension, and the
      total number of mesh cells per dimension.

      Note that the mesh is returned as a shared pointer.
    */
    double low_x =
        global_mesh_cell_size_array->lowCorner( Cabana::Grid::Dim::I );
    double high_z =
        global_mesh_cell_size_array->highCorner( Cabana::Grid::Dim::K );
    double extent_y =
        global_mesh_cell_size_array->extent( Cabana::Grid::Dim::J );
    double cells_y =
        global_mesh_cell_size_array->globalNumCell( Cabana::Grid::Dim::J );

    std::cout << "Mesh created with low X corner " << low_x
              << " and high Z corner " << high_z << std::endl;
    std::cout << "Extent in Y is " << extent_y << " with " << cells_y
              << " total cells." << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    globalMeshExample();

    return 0;
}

//---------------------------------------------------------------------------//
