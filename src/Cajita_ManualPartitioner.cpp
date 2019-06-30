#include <Cajita_ManualPartitioner.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
ManualPartitioner::ManualPartitioner( const std::vector<int>& ranks_per_dim )
    : _ranks_per_dim( ranks_per_dim )
{}

//---------------------------------------------------------------------------//
std::vector<int> ManualPartitioner::ranksPerDimension(
    MPI_Comm,
    const std::vector<int>& ) const
{
    return _ranks_per_dim;
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
