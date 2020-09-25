/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_REFERENCESTRUCTUREDSOLVER_HPP
#define CAJITA_REFERENCESTRUCTUREDSOLVER_HPP

#include <Cajita_Array.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_Halo.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_MpiTraits.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Reference preconditioned structured solver interface.
template <class Scalar, class EntityType, class MeshType, class DeviceType>
class ReferenceStructuredSolver
{
  public:
    // Types.
    using entity_type = EntityType;
    using device_type = DeviceType;
    using memory_space = typename device_type::memory_space;
    using execution_space = typename device_type::execution_space;
    using value_type = Scalar;
    using Array_t = Array<Scalar, EntityType, MeshType, DeviceType>;

    // Destructor.
    virtual ~ReferenceStructuredSolver() {}

    /*!
      \brief Set the matrix stencil.
      \param stencil The (i,j,k) offsets describing the structured matrix
      entries at each grid point. Offsets are defined relative to an index.
      \param is_symmetric If true the matrix is designated as symmetric. The
      stencil entries should only contain one entry from each symmetric
      component if this is true.
    */
    virtual void
    setMatrixStencil( const std::vector<std::array<int, 3>> &stencil,
                      const bool is_symmetric ) = 0;

    /*!
      \brief Get the matrix values.
      \return The matrix entry values which the user can fill with their
      entries. For each entity over which the vector space is defined an entry
      for each stencil element is required. The order of the stencil elements
      is that same as that in the stencil definition. Note that values
      corresponding to stencil entries outside of the domain should be set to
      zero.
    */
    virtual const Array_t &getMatrixValues() = 0;

    /*!
      \brief Set the preconditioner stencil.
      \param stencil The (i,j,k) offsets describing the structured
      preconditioner entries at each grid point. Offsets are defined relative to
      an index. \param is_symmetric If true the preconditioner is designated as
      symmetric. The stencil entries should only contain one entry from each
      symmetric component if this is true.
    */
    virtual void
    setPreconditionerStencil( const std::vector<std::array<int, 3>> &stencil,
                              const bool is_symmetric ) = 0;

    /*!
      \brief Get the preconditioner values.
      \return The preconditioner entry values which the user can fill with their
      entries. For each entity over which the vector space is defined an entry
      for each stencil element is required. The order of the stencil elements
      is that same as that in the stencil definition. Note that values
      corresponding to stencil entries outside of the domain should be set to
      zero.
    */
    virtual const Array_t &getPreconditionerValues() = 0;

    // Set convergence tolerance implementation.
    virtual void setTolerance( const double tol ) = 0;

    // Set maximum iteration implementation.
    virtual void setMaxIter( const int max_iter ) = 0;

    // Set the output level.
    virtual void setPrintLevel( const int print_level ) = 0;

    // Setup the problem.
    virtual void setup() = 0;

    /*!
      \brief Solve the problem Ax = b for x.
      \param b The forcing term.
      \param x The solution.
    */
    virtual void solve( const Array_t &b, Array_t &x ) = 0;

    // Get the number of iterations taken on the last solve.
    virtual int getNumIter() = 0;

    // Get the relative residual norm achieved on the last solve.
    virtual double getFinalRelativeResidualNorm() = 0;
};

//---------------------------------------------------------------------------//
// Reference structured preconditioned block conjugate gradient implementation.
template <class Scalar, class EntityType, class MeshType, class DeviceType>
class ReferenceConjugateGradient
    : public ReferenceStructuredSolver<Scalar, EntityType, MeshType, DeviceType>
{
  public:
    // Types.
    using entity_type = EntityType;
    using device_type = DeviceType;
    using value_type = Scalar;
    using execution_space = typename device_type::execution_space;
    using memory_space = typename device_type::memory_space;
    using Array_t = Array<Scalar, EntityType, MeshType, DeviceType>;

    // Array-like container to hold layout and data information.
    template <class ScalarT, class MemorySpaceT, class ArrayLayoutT>
    struct LayoutContainer
    {
        using value_type = ScalarT;
        using memory_space = MemorySpaceT;
        const ArrayLayoutT &array_layout;
        const ArrayLayoutT *layout() const { return &array_layout; }
    };

    /*!
      \brief Constructor.
    */
    ReferenceConjugateGradient(
        const ArrayLayout<EntityType, MeshType> &layout )
        : _tol( 1.0e-6 )
        , _max_iter( 1000 )
        , _print_level( 0 )
        , _num_iter( 0 )
        , _residual_norm( 0.0 )
    {
        // Array layout for vectors (p_old,z,r_old,q,p_new,r_new).
        auto vector_layout =
            createArrayLayout( layout.localGrid(), 6, EntityType() );
        _vectors =
            createArray<Scalar, DeviceType>( "cg_vectors", vector_layout );
    }

    /*!
      \brief Set the matrix stencil.
      \param stencil The (i,j,k) offsets describing the structured matrix
      entries at each grid point. Offsets are defined relative to an index.
      \param is_symmetric If true the matrix is designated as symmetric. The
      stencil entries should only contain one entry from each symmetric
      component if this is true.
    */
    void setMatrixStencil( const std::vector<std::array<int, 3>> &stencil,
                           const bool is_symmetric = false ) override
    {
        setStencil( stencil, is_symmetric, _A_stencil, _A_halo, _A );
    }

    /*!
      \brief Get the matrix values.
      \return The matrix entry values. For each entity over which the
      vector space is defined an entry for each stencil element is
      required. The order of the stencil elements is that same as that in the
      stencil definition. Note that values corresponding to stencil entries
      outside of the domain should be set to zero.
    */
    const Array_t &getMatrixValues() override { return *_A; }

    /*!
      \brief Set the preconditioner stencil.
      \param stencil The (i,j,k) offsets describing the structured
      preconditioner entries at each grid point. Offsets are defined relative to
      an index. \param is_symmetric If true the preconditioner is designated as
      symmetric. The stencil entries should only contain one entry from each
      symmetric component if this is true.
    */
    void
    setPreconditionerStencil( const std::vector<std::array<int, 3>> &stencil,
                              const bool is_symmetric = false ) override
    {
        setStencil( stencil, is_symmetric, _M_stencil, _M_halo, _M );
    }

    /*!
      \brief Get the preconditioner values.
      \return The preconditioner entry values. For each entity over which the
      vector space is defined an entry for each stencil element is
      required. The order of the stencil elements is that same as that in the
      stencil definition. Note that values corresponding to stencil entries
      outside of the domain should be set to zero.
    */
    const Array_t &getPreconditionerValues() override { return *_M; }

    // Set convergence tolerance implementation.
    void setTolerance( const double tol ) override { _tol = tol; }

    // Set maximum iteration implementation.
    void setMaxIter( const int max_iter ) override { _max_iter = max_iter; }

    // Set the output level.
    void setPrintLevel( const int print_level ) override
    {
        _print_level = print_level;
    }

    // Setup the problem.
    void setup() override {}

    /*!
      \brief Solve the problem Ax = b for x.
      \param b The forcing term.
      \param x The solution.
    */
    void solve( const Array_t &b, Array_t &x ) override
    {
        // Get the local grid.
        auto local_grid = _vectors->layout()->localGrid();

        // Print banner
        if ( 1 <= _print_level && 0 == local_grid->globalGrid().blockId() )
            std::cout << std::endl
                      << "Preconditioned conjugate gradient" << std::endl;

        // Index space.
        auto entity_space =
            local_grid->indexSpace( Own(), EntityType(), Local() );

        // Subarrays.
        auto p_old = createSubarray( *_vectors, 0, 1 );
        auto z = createSubarray( *_vectors, 1, 2 );
        auto r_old = createSubarray( *_vectors, 2, 3 );
        auto q = createSubarray( *_vectors, 3, 4 );
        auto p_new = createSubarray( *_vectors, 4, 5 );
        auto r_new = createSubarray( *_vectors, 5, 6 );
        auto A_halo_vectors = createSubarray( *_vectors, 0, 2 );
        auto M_halo_vectors = createSubarray( *_vectors, 2, 4 );

        // Views.
        auto x_view = x.view();
        auto b_view = b.view();
        auto M_view = _M->view();
        auto A_view = _A->view();
        auto p_old_view = p_old->view();
        auto z_view = z->view();
        auto r_old_view = r_old->view();
        auto q_view = q->view();
        auto p_new_view = p_new->view();
        auto r_new_view = r_new->view();

        // Reset iteration count.
        _num_iter = 0;

        // Compute the norm of the RHS.
        std::vector<Scalar> b_norm( 1 );
        ArrayOp::norm2( b, b_norm );

        // Copy the LHS into p so we can gather it.
        Kokkos::deep_copy( p_old_view, x_view );

        // Gather the LHS through gatheing p and z.
        _A_halo->gather( execution_space(), *A_halo_vectors );

        // Compute the initial residual and norm.
        _residual_norm = 0.0;
        Kokkos::parallel_reduce(
            "compute_r0",
            createExecutionPolicy( entity_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k,
                           Scalar &result ) {
                // Compute the local contribution from matrix-vector
                // multiplication. Note that we copied x into p for this
                // operation to easily perform the gather. Only apply the
                // stencil entry if it is greater than 0.
                Scalar Ax = 0.0;
                for ( unsigned c = 0; c < _A_stencil.extent( 0 ); ++c )
                    if ( fabs( A_view( i, j, k, c ) ) > 0.0 )
                        Ax += A_view( i, j, k, c ) *
                              p_old_view( i + _A_stencil( c, Dim::I ),
                                          j + _A_stencil( c, Dim::J ),
                                          k + _A_stencil( c, Dim::K ), 0 );

                // Compute the residual.
                auto r_new = b_view( i, j, k, 0 ) - Ax;

                // Assign the residual.
                r_old_view( i, j, k, 0 ) = r_new;

                // Contribute to the reduction.
                result += r_new * r_new;
            },
            _residual_norm );

        // Finish the global norm reduction.
        MPI_Allreduce( MPI_IN_PLACE, &_residual_norm, 1,
                       MpiTraits<Scalar>::type(), MPI_SUM,
                       local_grid->globalGrid().comm() );

        // If we already have met our criteria then return.
        _residual_norm = std::sqrt( _residual_norm ) / b_norm[0];
        if ( 2 == _print_level && 0 == local_grid->globalGrid().blockId() )
            std::cout << "Iteration " << _num_iter
                      << ": |r|_2 / |b|_2 = " << _residual_norm << std::endl;
        if ( _residual_norm <= _tol )
            return;

        // r and q.
        _M_halo->gather( execution_space(), *M_halo_vectors );

        // Compute the initial preconditioned residual.
        Scalar zTr_old = 0.0;
        Kokkos::parallel_reduce(
            "compute_z0",
            createExecutionPolicy( entity_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k,
                           Scalar &result ) {
                // Compute the local contribution from matrix-vector
                // multiplication. Only apply the stencil entry if it is
                // greater than 0.
                Scalar Mr = 0.0;
                for ( unsigned c = 0; c < _M_stencil.extent( 0 ); ++c )
                    if ( fabs( M_view( i, j, k, c ) ) > 0.0 )
                        Mr += M_view( i, j, k, c ) *
                              r_old_view( i + _M_stencil( c, Dim::I ),
                                          j + _M_stencil( c, Dim::J ),
                                          k + _M_stencil( c, Dim::K ), 0 );
                // Write values.
                z_view( i, j, k, 0 ) = Mr;
                p_old_view( i, j, k, 0 ) = Mr;
                p_new_view( i, j, k, 0 ) = Mr;

                // Compute zTr
                result += Mr * r_old_view( i, j, k, 0 );
            },
            zTr_old );

        // Finish computation of zTr
        MPI_Allreduce( MPI_IN_PLACE, &zTr_old, 1, MpiTraits<Scalar>::type(),
                       MPI_SUM, local_grid->globalGrid().comm() );

        // Gather the LHS through gatheing p and z.
        _A_halo->gather( execution_space(), *A_halo_vectors );

        // Compute A*p and pT*A*p.
        Scalar pTAp = 0.0;
        Kokkos::parallel_reduce(
            "compute_q0",
            createExecutionPolicy( entity_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k,
                           Scalar &result ) {
                // Compute the local contribution from matrix-vector
                // multiplication. This computes the updated p vector
                // in-line to avoid another kernel launch. Only apply the
                // stencil entry if it is greater than 0.
                Scalar Ap = 0.0;
                for ( unsigned c = 0; c < _A_stencil.extent( 0 ); ++c )
                    if ( fabs( A_view( i, j, k, c ) ) > 0.0 )
                        Ap += A_view( i, j, k, c ) *
                              ( p_old_view( i + _A_stencil( c, Dim::I ),
                                            j + _A_stencil( c, Dim::J ),
                                            k + _A_stencil( c, Dim::K ), 0 ) );

                // Write values.
                q_view( i, j, k, 0 ) = Ap;

                // Compute contribution to the dot product.
                result += p_old_view( i, j, k, 0 ) * Ap;
            },
            pTAp );

        // Finish the global reduction on pTAp.
        MPI_Allreduce( MPI_IN_PLACE, &pTAp, 1, MpiTraits<Scalar>::type(),
                       MPI_SUM, local_grid->globalGrid().comm() );

        // Iterate.
        bool converged = false;
        Scalar zTr_new = 0.0;
        Scalar alpha;
        Scalar beta;
        while ( _residual_norm > _tol && _num_iter < _max_iter )
        {
            // Gather r and q.
            _M_halo->gather( execution_space(), *M_halo_vectors );

            // Kernel 1: Compute x, r, residual norm, and zTr
            alpha = zTr_old / pTAp;
            zTr_new = 0.0;
            Kokkos::parallel_reduce(
                "cg_kernel_1",
                createExecutionPolicy( entity_space, execution_space() ),
                KOKKOS_LAMBDA( const int i, const int j, const int k,
                               Scalar &result ) {
                    // Compute the local contribution from matrix-vector
                    // multiplication. This computes the updated q vector
                    // in-line to avoid another kernel launch. Only apply the
                    // stencil entry if it is greater than 0.
                    Scalar Mr = 0.0;
                    for ( unsigned c = 0; c < _M_stencil.extent( 0 ); ++c )
                        if ( fabs( M_view( i, j, k, c ) ) > 0.0 )
                            Mr +=
                                M_view( i, j, k, c ) *
                                ( r_old_view( i + _M_stencil( c, Dim::I ),
                                              j + _M_stencil( c, Dim::J ),
                                              k + _M_stencil( c, Dim::K ), 0 ) -
                                  alpha * q_view( i + _M_stencil( c, Dim::I ),
                                                  j + _M_stencil( c, Dim::J ),
                                                  k + _M_stencil( c, Dim::K ),
                                                  0 ) );

                    // Compute the updated x.
                    Scalar x_new =
                        x_view( i, j, k, 0 ) + alpha * p_new_view( i, j, k, 0 );

                    // Compute the updated residual.
                    Scalar r_new =
                        r_old_view( i, j, k, 0 ) - alpha * q_view( i, j, k, 0 );

                    // Write to old p vector.
                    p_old_view( i, j, k, 0 ) = p_new_view( i, j, k, 0 );

                    // Write values.
                    x_view( i, j, k, 0 ) = x_new;
                    r_new_view( i, j, k, 0 ) = r_new;
                    z_view( i, j, k, 0 ) = Mr;

                    // Compute contribution to the zTr.
                    result += Mr * r_new;
                },
                zTr_new );

            // Finish the global reduction on zTr and r_norm.
            MPI_Allreduce( MPI_IN_PLACE, &zTr_new, 1, MpiTraits<Scalar>::type(),
                           MPI_SUM, local_grid->globalGrid().comm() );

            // Update residual norm
            _residual_norm = std::sqrt( fabs( zTr_new ) ) / b_norm[0];

            // Increment iteration count.
            _num_iter++;

            // Output result
            if ( 2 == _print_level && 0 == local_grid->globalGrid().blockId() )
                std::cout << "Iteration " << _num_iter
                          << ": |r|_2 / |b|_2 = " << _residual_norm
                          << std::endl;

            // Check for convergence.
            if ( _residual_norm <= _tol )
            {
                converged = true;
                break;
            }

            // Gather p and z.
            _A_halo->gather( execution_space(), *A_halo_vectors );

            // Kernel 2: Compute p, A*p, and p^T*A*p
            beta = zTr_new / zTr_old;
            pTAp = 0.0;
            Kokkos::parallel_reduce(
                "cg_kernel_2",
                createExecutionPolicy( entity_space, execution_space() ),
                KOKKOS_LAMBDA( const int i, const int j, const int k,
                               Scalar &result ) {
                    // Compute the local contribution from matrix-vector
                    // multiplication. This computes the updated p vector
                    // in-line to avoid another kernel launch. Only apply the
                    // stencil entry if it is greater than 0.
                    Scalar Ap = 0.0;
                    for ( unsigned c = 0; c < _A_stencil.extent( 0 ); ++c )
                        if ( fabs( A_view( i, j, k, c ) ) > 0.0 )
                            Ap += A_view( i, j, k, c ) *
                                  ( z_view( i + _A_stencil( c, Dim::I ),
                                            j + _A_stencil( c, Dim::J ),
                                            k + _A_stencil( c, Dim::K ), 0 ) +
                                    beta *
                                        p_old_view( i + _A_stencil( c, Dim::I ),
                                                    j + _A_stencil( c, Dim::J ),
                                                    k + _A_stencil( c, Dim::K ),
                                                    0 ) );

                    // Compute the updated p.
                    Scalar p_new =
                        z_view( i, j, k, 0 ) + beta * p_old_view( i, j, k, 0 );

                    // Write to old residual.
                    r_old_view( i, j, k, 0 ) = r_new_view( i, j, k, 0 );

                    // Write values.
                    q_view( i, j, k, 0 ) = Ap;
                    p_new_view( i, j, k, 0 ) = p_new;

                    // Compute contribution to the dot product.
                    result += p_new * Ap;
                },
                pTAp );

            // Finish the global reduction on pTAp.
            MPI_Allreduce( MPI_IN_PLACE, &pTAp, 1, MpiTraits<Scalar>::type(),
                           MPI_SUM, local_grid->globalGrid().comm() );

            // Update zTr
            zTr_old = zTr_new;
        }

        // Output end state.
        if ( 1 <= _print_level && 0 == local_grid->globalGrid().blockId() )
            std::cout << "Finished in " << _num_iter
                      << " iterations, converged to " << _residual_norm
                      << std::endl
                      << std::endl;

        // If we didn't converge throw.
        if ( !converged )
            throw std::runtime_error( "CG solver did not converge" );
    }

    // Get the number of iterations taken on the last solve.
    int getNumIter() override { return _num_iter; }

    // Get the relative residual norm achieved on the last solve.
    double getFinalRelativeResidualNorm() override { return _residual_norm; }

  private:
    // Set the stencil of a matrix.
    void setStencil( const std::vector<std::array<int, 3>> &stencil,
                     const bool is_symmetric,
                     Kokkos::View<int *[3], DeviceType> &device_stencil,
                     std::shared_ptr<Halo<memory_space>> &halo,
                     std::shared_ptr<Array_t> &matrix )
    {
        // For now we don't support symmetry.
        if ( is_symmetric )
            throw std::logic_error(
                "Reference CG currently does not support symmetry" );

        // Get the local grid.
        auto local_grid = _vectors->layout()->localGrid();

        // Copy stencil to the device.
        device_stencil = Kokkos::View<int *[3], DeviceType>(
            Kokkos::ViewAllocateWithoutInitializing( "stencil" ),
            stencil.size() );
        auto stencil_mirror =
            Kokkos::create_mirror_view( Kokkos::HostSpace(), device_stencil );
        for ( unsigned s = 0; s < stencil.size(); ++s )
            for ( int d = 0; d < 3; ++d )
                stencil_mirror( s, d ) = stencil[s][d];
        Kokkos::deep_copy( device_stencil, stencil_mirror );

        // Compose the halo pattern and compute how wide the halo needs to be
        // to gather all elements accessed by the stencil.
        std::set<std::array<int, 3>> neighbor_set;
        std::array<int, 3> neighbor;
        int width = 0;
        for ( auto s : stencil )
        {
            // Compse a set of the neighbor ranks based on the stencil.
            for ( int d = 0; d < 3; ++d )
                neighbor[d] = ( s[d] == 0 ) ? 0 : s[d] / std::abs( s[d] );
            neighbor_set.emplace( neighbor );

            // Compute the width of the halo needed to apply the stencil.
            for ( int d = 0; d < 3; ++d )
                width = std::max( width, std::abs( s[d] ) );
        }
        std::vector<std::array<int, 3>> halo_neighbors( neighbor_set.size() );
        std::copy( neighbor_set.begin(), neighbor_set.end(),
                   halo_neighbors.begin() );

        // Build the halo. We put 2 entries as each operator application will
        // gather 2 vectors with fused kernels.
        auto halo_layout = createArrayLayout( local_grid, 2, EntityType() );
        HaloPattern pattern;
        pattern.setNeighbors( halo_neighbors );
        halo = createHalo<Scalar, DeviceType>( *halo_layout, pattern, width );

        // Create a new layout.
        auto matrix_layout =
            createArrayLayout( local_grid, stencil.size(), EntityType() );

        // Allocate the matrix.
        matrix = createArray<Scalar, DeviceType>( "matrix", matrix_layout );
    }

  private:
    Scalar _tol;
    int _max_iter;
    int _print_level;
    int _num_iter;
    Scalar _residual_norm;
    int _diag_entry;
    Kokkos::View<int *[3], DeviceType> _A_stencil;
    Kokkos::View<int *[3], DeviceType> _M_stencil;
    std::shared_ptr<Halo<memory_space>> _A_halo;
    std::shared_ptr<Halo<memory_space>> _M_halo;
    std::shared_ptr<Array_t> _A;
    std::shared_ptr<Array_t> _M;
    std::shared_ptr<Array_t> _vectors;
};

//---------------------------------------------------------------------------//
// Builders.
//---------------------------------------------------------------------------//
template <class Scalar, class DeviceType, class EntityType, class MeshType>
std::shared_ptr<
    ReferenceConjugateGradient<Scalar, EntityType, MeshType, DeviceType>>
createReferenceConjugateGradient(
    const ArrayLayout<EntityType, MeshType> &layout )
{
    return std::make_shared<
        ReferenceConjugateGradient<Scalar, EntityType, MeshType, DeviceType>>(
        layout );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_REFERENCESTRUCTUREDSOLVER_HPP
