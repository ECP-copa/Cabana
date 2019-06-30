#include <Cajita_UniformDimPartitioner.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
std::vector<int> UniformDimPartitioner::ranksPerDimension(
    MPI_Comm comm,
    const std::vector<int>& global_cells_per_dim ) const
{
    std::ignore = global_cells_per_dim;

    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    return ranks_per_dim;
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
