#ifndef CAJITA_PARTITIONER_HPP
#define CAJITA_PARTITIONER_HPP

#include <vector>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
class Partitioner
{
  public:

    ~Partitioner() = default;

    /*!
      \brief Get the number of MPI ranks in each dimension of the grid.
      \param comm The communicator to use for the partitioning.
      \param global_cells_per_dim The number of global cells in each dimension.
      \return The number of MPI ranks in each dimension of the grid.
    */
    virtual std::vector<int> ranksPerDimension(
        MPI_Comm comm,
        const std::vector<int>& global_cells_per_dim ) const = 0;
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_PARTITIONER_HPP
