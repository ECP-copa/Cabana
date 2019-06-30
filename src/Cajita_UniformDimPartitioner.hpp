#ifndef CAJITA_UNIFORMDIMPARTITIONER_HPP
#define CAJITA_UNIFORMDIMPARTITIONER_HPP

#include <Cajita_Partitioner.hpp>

#include <vector>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
class UniformDimPartitioner : public Partitioner
{
  public:

    std::vector<int> ranksPerDimension(
        MPI_Comm comm,
        const std::vector<int>& global_cells_per_dim ) const override;
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_UNIFORMDIMPARTITIONER_HPP
