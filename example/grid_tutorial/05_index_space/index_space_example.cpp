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
// Index Space example.
//---------------------------------------------------------------------------//
void indexSpaceExample()
{
    /*
      Each index space represents a contiguous set of structured
      multidimensional indices which is then used to describe how to iterate
      over Cabana::Grid grids and arrays in parallel. It is generally used to
      thread over a local grid (discussed in the following example) and used
      tightly coupled to the parallel operations (also discussed in a later
      example).
    */
    std::cout << "Cabana::Grid Index Space Example\n" << std::endl;

    /*
      Index spaces can be one dimensional and up to 4D, mapping to physical
      dimensions on the grid or grid data. They can be constructed with
      initializer_lists or arrays.

      In this very simple 1D case we use the size constructor to build the index
      space {0,1,2,3,4}:
    */
    Cabana::Grid::IndexSpace<1> is1( { 5 } );
    std::cout << "1D index space:\nMin: ";
    std::cout << is1.min( 0 ) << " Max: " << is1.max( 0 ) << "\n" << std::endl;

    /*
      Next, we set both the start and end values for the index space resulting
      in {5,6,7,8,9}.
    */
    std::cout << "1D index space:\nMin: ";
    Cabana::Grid::IndexSpace<1> is1_2( { 5 }, { 10 } );
    std::cout << is1_2.min( 0 ) << " Max: " << is1_2.max( 0 ) << "\n"
              << std::endl;

    /*
      Next we create a 2D space.
    */
    Cabana::Grid::IndexSpace<2> is2( { 4, 3 }, { 8, 9 } );
    std::cout << "2D index space:\nMin: ";
    for ( int d = 0; d < 2; ++d )
        std::cout << is2.min( d ) << " ";
    std::cout << std::endl << "Max: ";
    for ( int d = 0; d < 2; ++d )
        std::cout << is2.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      For this 3D case, each dimension should again go from zero to one less
      than the value given (the value given is an exclusive upper bound).
    */
    Cabana::Grid::IndexSpace<3> is3( { 5, 4, 8 } );
    std::cout << "3D index space:\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << is3.min( d ) << " ";
    std::cout << std::endl << "Max: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << is3.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      There are additional interfaces to extract information
      about a given index space: rank (dimension), extent and
      range (pair of start and end) in each dimension, total
      size of the index space (product).
    */
    std::cout << "Rank: " << is3.rank() << std::endl;
    std::cout << "Extent: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << is3.extent( d ) << " ";
    std::cout << std::endl << "Range: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << "(" << is3.range( d ).first << ", "
                  << is3.range( d ).second << ") ";
    std::cout << std::endl;
    std::cout << "Size: " << is3.size() << std::endl;

    /*
      Finally, it is possible to check whether a given set of indices in within
      the range of an index space (in this case false and then true based on the
      second indices).
    */
    long is_check1[3] = { 3, 9, 3 };
    std::cout << "Is {3, 9, 3} in {5, 4, 8}: " << is3.inRange( is_check1 )
              << std::endl;
    long is_check2[3] = { 3, 1, 3 };
    std::cout << "Is {3, 1, 3} in {5, 4, 8}: " << is3.inRange( is_check2 )
              << std::endl;

    /*
      Index spaces can also have dimensions appended. Using the 3D index space
      created above, we create a 4D index space (the minimum input can be
      omitted to start from zero).
    */
    auto is4 = Cabana::Grid::appendDimension( is3, 3, 10 );
    std::cout << "\nappended 4D index space:\nExtent: ";
    for ( int d = 0; d < 4; ++d )
        std::cout << is4.extent( d ) << " ";
    std::cout << std::endl;

    /*
      Using index spaces in practice often means create a Kokkos execution
      policy directly. Internally this creates a multidimensional range policy
      (except for the 1D case, which is a standard linear policy).
    */
    using exec_space = Kokkos::DefaultHostExecutionSpace;
    auto exec_policy1 = createExecutionPolicy( is1, exec_space() );
    std::cout << "\nexecution policy from 1D index space:\nExtent: ";
    std::cout << exec_policy1.begin() << " " << exec_policy1.end() << std::endl;

    /*
      Next we pass a multidimensional range policy created from an index space
      to a Kokkos parallel loop and sum all indices. Note that the functor
      signature uses i,j,k indexing to match the index space and corresponding
      execution policy (unused in this very simple kernel). This total size
      should match the one reported by the index space.
    */
    auto exec_policy3 = createExecutionPolicy( is3, exec_space() );
    int total = 0;
    Kokkos::parallel_reduce(
        "md_range_loop", exec_policy3,
        KOKKOS_LAMBDA( const int, const int, const int, int& sum ) { sum++; },
        total );
    std::cout << "\nexecution policy from 3D index space:\nSize: ";
    std::cout << total << std::endl;

    /*
      Similarly, index spaces can be used to create Kokkos Views and subviews of
      existing Views with sizes matching the extents of the index space.
    */
    auto v1 = Cabana::Grid::createView<double>( "v1", is1 );
    std::cout << "\n1D View created from 1D index space\nExtent: "
              << v1.extent( 0 ) << std::endl;
    ;

    Kokkos::View<double**, exec_space> v2( "v2", 12, 7 );
    auto sv1 = Cabana::Grid::createSubview( v2, is2 );
    std::cout << "\n2D subview from 2D index space\nExtent: " << sv1.extent( 0 )
              << " " << sv1.extent( 1 ) << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    indexSpaceExample();

    return 0;
}

//---------------------------------------------------------------------------//
