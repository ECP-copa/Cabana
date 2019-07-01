#include <Cajita_Halo.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_Array.hpp>
#include <Cajita_ManualPartitioner.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void gatherScatterTest( const ManualPartitioner& partitioner,
                        const std::vector<bool>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 32, 23, 41 };
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         partitioner,
                                         is_dim_periodic,
                                         global_low_corner,
                                         global_high_corner,
                                         cell_size );

    // Create an array on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, halo_width, dofs_per_cell, Cell() );
    auto array = createArray<double,TEST_DEVICE>( "array", cell_layout );

    // Assign the owned cells a value of 1.
    auto owned_space = cell_layout->indexSpace( Own(), Local() );
    auto owned_subview = createSubview( array->view(), owned_space );
    Kokkos::deep_copy( owned_subview, 1.0 );

    // Create a halo.
    auto halo = createHalo( *array, FullHaloPattern() );

    // Gather into the ghosts.
    halo->gather( *array, 124 );

    // Check the gather. We should get 1 everywhere in the array now.
    auto ghosted_space = cell_layout->indexSpace( Ghost(), Local() );
    auto host_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), array->view() );
    for ( unsigned i = 0; i < ghosted_space.extent(0); ++i )
        for ( unsigned j = 0; j < ghosted_space.extent(1); ++j )
            for ( unsigned k = 0; k < ghosted_space.extent(2); ++k )
                for ( unsigned l = 0; l < ghosted_space.extent(3); ++l )
                    EXPECT_EQ( host_view(i,j,k,l), 1.0 );

    // Scatter from the ghosts back to owned.
    halo->scatter( *array, 125 );

    // Check the scatter. The value of the cell should be a function of how
    // many neighbors it has. Corner neighbors get 8, edge neighbors get 4,
    // face neighbors get 2, and no neighbors remain at 1.

    // This function checks if an index is in the halo of a low neighbor in
    // the given dimension
    auto in_dim_min_halo =
        [=]( const int i, const int dim ){
            if ( is_dim_periodic[dim] || global_grid->dimBlockId(dim) > 0 )
                return i < (owned_space.min(dim) + halo_width);
            else
                return false;
        };

    // This function checks if an index is in the halo of a high neighbor in
    // the given dimension
    auto in_dim_max_halo =
        [=]( const int i, const int dim ){
            if ( is_dim_periodic[dim] ||
                 global_grid->dimBlockId(dim) <
                 global_grid->dimNumBlock(dim) - 1 )
                return i >= (owned_space.max(dim) - halo_width);
            else
                return false;
        };

    // Check results. Use the halo functions to figure out how many neighbor a
    // given cell was ghosted to.
    Kokkos::deep_copy( host_view, array->view() );
    for ( unsigned i = owned_space.min(0); i < owned_space.max(0); ++i )
        for ( unsigned j = owned_space.min(1); j < owned_space.max(1); ++j )
            for ( unsigned k = owned_space.min(2); k < owned_space.max(2); ++k )
            {
                int num_n = 0;
                if ( in_dim_min_halo(i,Dim::I) || in_dim_max_halo(i,Dim::I) )
                    ++num_n;
                if ( in_dim_min_halo(j,Dim::J) || in_dim_max_halo(j,Dim::J) )
                    ++num_n;
                if ( in_dim_min_halo(k,Dim::K) || in_dim_max_halo(k,Dim::K) )
                    ++num_n;
                double scatter_val = std::pow( 2.0, num_n );
                for ( unsigned l = 0; l < owned_space.extent(3); ++l )
                    EXPECT_EQ( host_view(i,j,k,l), scatter_val );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, not_periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    ManualPartitioner partitioner( ranks_per_dim );

    // Boundaries are not periodic.
    std::vector<bool> is_dim_periodic = {false,false,false};

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    ManualPartitioner partitioner( ranks_per_dim );

    // Every boundary is periodic
    std::vector<bool> is_dim_periodic = {true,true,true};

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
}

//---------------------------------------------------------------------------//

} // end namespace Test
