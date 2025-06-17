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

/*!
  \file Cabana_Grid_HypreStructuredSolver.hpp
  \brief HYPRE structured solver interface
*/
#ifndef CABANA_GRID_HYPRESTRUCTUREDSOLVER_HPP
#define CABANA_GRID_HYPRESTRUCTUREDSOLVER_HPP

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_Hypre.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_Types.hpp>

#include <HYPRE_config.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <array>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
//! Hypre structured solver interface for scalar fields.
template <class Scalar, class EntityType, class MemorySpace>
class HypreStructuredSolver
{
  public:
    //! Entity type.
    using entity_type = EntityType;
    //! Kokkos memory space..
    using memory_space = MemorySpace;
    //! Scalar value type.
    using value_type = Scalar;
    //! Hypre memory space compatibility check.
    static_assert( HypreIsCompatibleWithMemorySpace<memory_space>::value,
                   "HYPRE not compatible with solver memory space" );

    /*!
      \brief Constructor.
      \param layout The array layout defining the vector space of the solver.
      \param is_preconditioner Flag indicating if this solver will be used as
      a preconditioner.
    */
    template <class ArrayLayout_t>
    HypreStructuredSolver( const ArrayLayout_t& layout,
                           const bool is_preconditioner = false )
        : _comm( layout.localGrid()->globalGrid().comm() )
        , _is_preconditioner( is_preconditioner )
    {
        static_assert( is_array_layout<ArrayLayout_t>::value,
                       "Must use an array layout" );
        static_assert(
            std::is_same<typename ArrayLayout_t::entity_type,
                         entity_type>::value,
            "Array layout entity type mush match solver entity type" );

        // Spatial dimension.
        const std::size_t num_space_dim = ArrayLayout_t::num_space_dim;

        // Only create data structures if this is not a preconditioner.
        if ( !_is_preconditioner )
        {
            // Create the grid.
            auto error = HYPRE_StructGridCreate( _comm, num_space_dim, &_grid );
            checkHypreError( error );

            // Get the global index space spanned by the local grid on this
            // rank. Note that the upper bound is not a bound but rather the
            // last index as this is what Hypre wants. Note that we reordered
            // this to KJI from IJK to be consistent with HYPRE ordering. By
            // setting up the grid like this, HYPRE will then want layout-right
            // data indexed as (i,j,k) or (i,j,k,l) which will allow us to
            // directly use Kokkos::deep_copy to move data between arrays and
            // HYPRE data structures.
            auto global_space = layout.indexSpace( Own(), Global() );
            _lower.resize( num_space_dim );
            _upper.resize( num_space_dim );
            for ( std::size_t d = 0; d < num_space_dim; ++d )
            {
                _lower[d] = static_cast<HYPRE_Int>(
                    global_space.min( num_space_dim - d - 1 ) );
                _upper[d] = static_cast<HYPRE_Int>(
                    global_space.max( num_space_dim - d - 1 ) - 1 );
            }
            error = HYPRE_StructGridSetExtents( _grid, _lower.data(),
                                                _upper.data() );
            checkHypreError( error );

            // Get periodicity. Note we invert the order of this to KJI as well.
            const auto& global_grid = layout.localGrid()->globalGrid();
            HYPRE_Int periodic[num_space_dim];
            for ( std::size_t d = 0; d < num_space_dim; ++d )
                periodic[num_space_dim - 1 - d] =
                    global_grid.isPeriodic( d )
                        ? layout.localGrid()->globalGrid().globalNumEntity(
                              EntityType(), d )
                        : 0;
            error = HYPRE_StructGridSetPeriodic( _grid, periodic );
            checkHypreError( error );

            // Assemble the grid.
            error = HYPRE_StructGridAssemble( _grid );
            checkHypreError( error );

            // Allocate LHS and RHS vectors and initialize to zero. Note that we
            // are fixing the views under these vectors to layout-right.
            std::array<long, num_space_dim> reorder_size;
            for ( std::size_t d = 0; d < num_space_dim; ++d )
            {
                reorder_size[d] = global_space.extent( d );
            }
            IndexSpace<num_space_dim> reorder_space( reorder_size );
            auto vector_values =
                createView<HYPRE_Complex, Kokkos::LayoutRight, memory_space>(
                    "vector_values", reorder_space );
            Kokkos::deep_copy( vector_values, 0.0 );

            error = HYPRE_StructVectorCreate( _comm, _grid, &_b );
            checkHypreError( error );
            error = HYPRE_StructVectorInitialize( _b );
            checkHypreError( error );
            error = HYPRE_StructVectorSetBoxValues(
                _b, _lower.data(), _upper.data(), vector_values.data() );
            checkHypreError( error );
            error = HYPRE_StructVectorAssemble( _b );
            checkHypreError( error );

            error = HYPRE_StructVectorCreate( _comm, _grid, &_x );
            checkHypreError( error );
            error = HYPRE_StructVectorInitialize( _x );
            checkHypreError( error );
            error = HYPRE_StructVectorSetBoxValues(
                _x, _lower.data(), _upper.data(), vector_values.data() );
            checkHypreError( error );
            error = HYPRE_StructVectorAssemble( _x );
            checkHypreError( error );
        }
    }

    // Destructor.
    virtual ~HypreStructuredSolver()
    {
        // We only make data if this is not a preconditioner.
        if ( !_is_preconditioner )
        {
            HYPRE_StructVectorDestroy( _x );
            HYPRE_StructVectorDestroy( _b );
            HYPRE_StructMatrixDestroy( _A );
            HYPRE_StructStencilDestroy( _stencil );
            HYPRE_StructGridDestroy( _grid );
        }
    }

    //! Return if this solver is a preconditioner.
    bool isPreconditioner() const { return _is_preconditioner; }

    /*!
      \brief Set the operator stencil.
      \param stencil The (i,j,k) offsets describing the structured matrix
      entries at each grid point. Offsets are defined relative to an index.
      \param is_symmetric If true the matrix is designated as symmetric. The
      stencil entries should only contain one entry from each symmetric
      component if this is true.
    */
    template <std::size_t NumSpaceDim>
    void
    setMatrixStencil( const std::vector<std::array<int, NumSpaceDim>>& stencil,
                      const bool is_symmetric = false )
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call setMatrixStencil() on preconditioners" );

        // Create the stencil.
        _stencil_size = stencil.size();
        auto error =
            HYPRE_StructStencilCreate( NumSpaceDim, _stencil_size, &_stencil );
        checkHypreError( error );
        std::array<HYPRE_Int, NumSpaceDim> offset;
        for ( unsigned n = 0; n < stencil.size(); ++n )
        {
            for ( std::size_t d = 0; d < NumSpaceDim; ++d )
                offset[d] = stencil[n][d];
            error = HYPRE_StructStencilSetElement( _stencil, n, offset.data() );
            checkHypreError( error );
        }

        // Create the matrix object. Must be done after the stencil is setup
        error = HYPRE_StructMatrixCreate( _comm, _grid, _stencil, &_A );
        checkHypreError( error );
        error = HYPRE_StructMatrixSetSymmetric( _A, is_symmetric );
        checkHypreError( error );
        error = HYPRE_StructMatrixInitialize( _A );
        checkHypreError( error );
    }

    /*!
      \brief Set the matrix values.
      \param values The matrix entry values. For each entity over which the
      vector space is defined an entry for each stencil element is
      required. The order of the stencil elements is that same as that in the
      stencil definition. Note that values corresponding to stencil entries
      outside of the domain should be set to zero.
    */
    template <class Array_t>
    void setMatrixValues( const Array_t& values )
    {
        static_assert( is_array<Array_t>::value, "Must use an array" );
        static_assert(
            std::is_same<typename Array_t::entity_type, entity_type>::value,
            "Cabana::Grid::HypreStructuredSolver:setMatrixValues: Array entity "
            "type mush match solver entity type" );
        static_assert(
            std::is_same<typename Array_t::memory_space, MemorySpace>::value,
            "Cabana::Grid::HypreStructuredSolver:setMatrixValues: Array memory "
            "space and solver memory space are different." );

        static_assert(
            std::is_same<typename Array_t::value_type, value_type>::value,
            "Cabana::Grid::HypreStructuredSolver:setMatrixValues: Array value "
            "type and solver value type are different." );

        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cabana::Grid::HypreStructuredSolver:setMatrixValues: Cannot "
                "call setMatrixValues() on preconditioners" );

        if ( values.layout()->dofsPerEntity() !=
             static_cast<int>( _stencil_size ) )
            throw std::runtime_error(
                "Cabana::Grid::HypreStructuredSolver:setMatrixValues: Number "
                "of matrix values does not match stencil size" );

        // Spatial dimension.
        const std::size_t num_space_dim = Array_t::num_space_dim;

        // Copy the matrix entries into HYPRE. The HYPRE layout is fixed as
        // layout-right.
        auto owned_space = values.layout()->indexSpace( Own(), Local() );
        std::array<long, num_space_dim + 1> reorder_size;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            reorder_size[d] = owned_space.extent( d );
        }
        reorder_size.back() = _stencil_size;
        IndexSpace<num_space_dim + 1> reorder_space( reorder_size );
        auto a_values =
            createView<HYPRE_Complex, Kokkos::LayoutRight, memory_space>(
                "a_values", reorder_space );
        auto values_subv = createSubview( values.view(), owned_space );
        Kokkos::deep_copy( a_values, values_subv );

        // Insert values into the HYPRE matrix.
        std::vector<HYPRE_Int> indices( _stencil_size );
        std::iota( indices.begin(), indices.end(), 0 );
        auto error = HYPRE_StructMatrixSetBoxValues(
            _A, _lower.data(), _upper.data(), indices.size(), indices.data(),
            a_values.data() );
        checkHypreError( error );
        error = HYPRE_StructMatrixAssemble( _A );
        checkHypreError( error );
    }

    /*!
      \brief Print the hypre matrix to output file
      \param prefix File prefix for where hypre output is written
    */
    void printMatrix( const char* prefix )
    {
        HYPRE_StructMatrixPrint( prefix, _A, 0 );
    }

    /*!
      \brief Print the hypre LHS to output file
      \param prefix File prefix for where hypre output is written
    */
    void printLHS( const char* prefix )
    {
        HYPRE_StructVectorPrint( prefix, _x, 0 );
    }

    /*!
      \brief Print the hypre RHS to output file
      \param prefix File prefix for where hypre output is written
    */
    void printRHS( const char* prefix )
    {
        HYPRE_StructVectorPrint( prefix, _b, 0 );
    }

    //! Set convergence tolerance implementation.
    void setTolerance( const double tol ) { this->setToleranceImpl( tol ); }

    //! Set maximum iteration implementation.
    void setMaxIter( const int max_iter ) { this->setMaxIterImpl( max_iter ); }

    //! Set the output level.
    void setPrintLevel( const int print_level )
    {
        this->setPrintLevelImpl( print_level );
    }

    //! Set a preconditioner.
    void
    setPreconditioner( const std::shared_ptr<HypreStructuredSolver<
                           Scalar, EntityType, MemorySpace>>& preconditioner )
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cabana::Grid::HypreStructuredSolver:setPreconditioner: Cannot "
                "call setPreconditioner() on a preconditioner" );

        // Only a preconditioner can be used as a preconditioner.
        if ( !preconditioner->isPreconditioner() )
            throw std::logic_error( "Cabana::Grid::HypreStructuredSolver:"
                                    "setPreconditioner: Not a preconditioner" );

        _preconditioner = preconditioner;
        this->setPreconditionerImpl( *_preconditioner );
    }

    //! Setup the problem.
    void setup()
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error( "Cabana::Grid::HypreStructuredSolver:setup:"
                                    " Cannot call setup() on preconditioners" );

        // FIXME: appears to be a memory issue in the call to this function
        this->setupImpl();
    }

    /*!
      \brief Solve the problem Ax = b for x.
      \param b The forcing term.
      \param x The solution.
    */
    template <class Array_t>
    void solve( const Array_t& b, Array_t& x )
    {
        Kokkos::Profiling::ScopedRegion region(
            "Cabana::Grid::HypreStructuredSolver::solve" );

        static_assert(
            is_array<Array_t>::value,
            "Cabana::Grid::HypreStructuredSolver::solve: Must use an array" );
        static_assert(
            std::is_same<typename Array_t::entity_type, entity_type>::value,
            "Cabana::Grid::HypreStructuredSolver::solve: Array entity type "
            "mush match solver entity type" );
        static_assert(
            std::is_same<typename Array_t::memory_space, MemorySpace>::value,
            "Cabana::Grid::HypreStructuredSolver::solve: Array memory space "
            "and solver memory space are different." );

        static_assert(
            std::is_same<typename Array_t::value_type, value_type>::value,
            "Cabana::Grid::HypreStructuredSolver::solve: Array value type and "
            "solver value type are different." );

        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cabana::Grid::HypreStructuredSolver::solve: Cannot call "
                "solve() on preconditioners" );

        if ( b.layout()->dofsPerEntity() != 1 ||
             x.layout()->dofsPerEntity() != 1 )
            throw std::runtime_error(
                "Cabana::Grid::HypreStructuredSolver::solve: Structured solver "
                "only for scalar fields" );

        // Spatial dimension.
        const std::size_t num_space_dim = Array_t::num_space_dim;

        // Copy the RHS into HYPRE. The HYPRE layout is fixed as layout-right.
        auto owned_space = b.layout()->indexSpace( Own(), Local() );
        std::array<long, num_space_dim + 1> reorder_size;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            reorder_size[d] = owned_space.extent( d );
        }
        reorder_size.back() = 1;
        IndexSpace<num_space_dim + 1> reorder_space( reorder_size );
        auto vector_values =
            createView<HYPRE_Complex, Kokkos::LayoutRight, memory_space>(
                "vector_values", reorder_space );
        auto b_subv = createSubview( b.view(), owned_space );
        Kokkos::deep_copy( vector_values, b_subv );

        // Insert b values into the HYPRE vector.
        auto error = HYPRE_StructVectorSetBoxValues(
            _b, _lower.data(), _upper.data(), vector_values.data() );
        checkHypreError( error );
        error = HYPRE_StructVectorAssemble( _b );
        checkHypreError( error );

        // Solve the problem
        this->solveImpl();

        // Extract the solution from the LHS
        error = HYPRE_StructVectorGetBoxValues(
            _x, _lower.data(), _upper.data(), vector_values.data() );
        checkHypreError( error );

        // Copy the HYPRE solution to the LHS.
        auto x_subv = createSubview( x.view(), owned_space );
        Kokkos::deep_copy( x_subv, vector_values );
    }

    //! Get the number of iterations taken on the last solve.
    int getNumIter() { return this->getNumIterImpl(); }

    //! Get the relative residual norm achieved on the last solve.
    double getFinalRelativeResidualNorm()
    {
        return this->getFinalRelativeResidualNormImpl();
    }

    //! Get the preconditioner.
    virtual HYPRE_StructSolver getHypreSolver() const = 0;
    //! Get the preconditioner setup function.
    virtual HYPRE_PtrToStructSolverFcn getHypreSetupFunction() const = 0;
    //! Get the preconditioner solve function.
    virtual HYPRE_PtrToStructSolverFcn getHypreSolveFunction() const = 0;

  protected:
    //! Set convergence tolerance implementation.
    virtual void setToleranceImpl( const double tol ) = 0;

    //! Set maximum iteration implementation.
    virtual void setMaxIterImpl( const int max_iter ) = 0;

    //! Set the output level.
    virtual void setPrintLevelImpl( const int print_level ) = 0;

    //! Setup implementation.
    virtual void setupImpl() = 0;

    //! Solver implementation.
    virtual void solveImpl() = 0;

    //! Get the number of iterations taken on the last solve.
    virtual int getNumIterImpl() = 0;

    //! Get the relative residual norm achieved on the last solve.
    virtual double getFinalRelativeResidualNormImpl() = 0;

    //! Set a preconditioner.
    virtual void setPreconditionerImpl(
        const HypreStructuredSolver<Scalar, EntityType, MemorySpace>&
            preconditioner ) = 0;

    //! Check a hypre error.
    void checkHypreError( const int error ) const
    {
        if ( error > 0 )
        {
            char error_msg[256];
            HYPRE_DescribeError( error, error_msg );
            std::stringstream out;
            out << "HYPRE structured solver error: ";
            out << error << " " << error_msg;
            HYPRE_ClearError( error );
            throw std::runtime_error( out.str() );
        }
    }

  protected:
    //! Matrix for the problem Ax = b.
    HYPRE_StructMatrix _A;
    //! Forcing term for the problem Ax = b.
    HYPRE_StructVector _b;
    //! Solution to the problem Ax = b.
    HYPRE_StructVector _x;

  private:
    MPI_Comm _comm;
    bool _is_preconditioner;
    HYPRE_StructGrid _grid;
    std::vector<HYPRE_Int> _lower;
    std::vector<HYPRE_Int> _upper;
    HYPRE_StructStencil _stencil;
    unsigned _stencil_size;
    std::shared_ptr<HypreStructuredSolver<Scalar, EntityType, MemorySpace>>
        _preconditioner;
};

//---------------------------------------------------------------------------//
//! PCG solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreStructPCG
    : public HypreStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using base_type = HypreStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreStructPCG( const ArrayLayout_t& layout,
                    const bool is_preconditioner = false )
        : base_type( layout, is_preconditioner )
    {
        if ( is_preconditioner )
            throw std::logic_error(
                "HYPRE PCG cannot be used as a preconditioner" );

        auto error = HYPRE_StructPCGCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );

        HYPRE_StructPCGSetTwoNorm( _solver, 1 );
    }

    ~HypreStructPCG() { HYPRE_StructPCGDestroy( _solver ); }

    // PCG SETTINGS

    //! Set the absolute tolerance
    void setAbsoluteTol( const double tol )
    {
        auto error = HYPRE_StructPCGSetAbsoluteTol( _solver, tol );
        this->checkHypreError( error );
    }

    //! Additionally require that the relative difference in successive
    //! iterates be small.
    void setRelChange( const int rel_change )
    {
        auto error = HYPRE_StructPCGSetRelChange( _solver, rel_change );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_StructPCGSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_StructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_StructPCGSetup;
    }
    HYPRE_PtrToStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_StructPCGSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_StructPCGSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_StructPCGSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_StructPCGSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl() override
    {
        auto error = HYPRE_StructPCGSetup( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    void solveImpl() override
    {
        auto error = HYPRE_StructPCGSolve( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_StructPCGGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_StructPCGGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreStructuredSolver<Scalar, EntityType, MemorySpace>&
            preconditioner ) override
    {
        auto error = HYPRE_StructPCGSetPrecond(
            _solver, preconditioner.getHypreSolveFunction(),
            preconditioner.getHypreSetupFunction(),
            preconditioner.getHypreSolver() );
        this->checkHypreError( error );
    }

  private:
    HYPRE_StructSolver _solver;
    using base_type::_A;
    using base_type::_b;
    using base_type::_x;
};

//---------------------------------------------------------------------------//
//! GMRES solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreStructGMRES
    : public HypreStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using base_type = HypreStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreStructGMRES( const ArrayLayout_t& layout,
                      const bool is_preconditioner = false )
        : base_type( layout, is_preconditioner )
    {
        if ( is_preconditioner )
            throw std::logic_error(
                "HYPRE GMRES cannot be used as a preconditioner" );

        auto error = HYPRE_StructGMRESCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreStructGMRES() { HYPRE_StructGMRESDestroy( _solver ); }

    // GMRES SETTINGS

    //! Set the absolute tolerance
    void setAbsoluteTol( const double tol )
    {
        auto error = HYPRE_StructGMRESSetAbsoluteTol( _solver, tol );
        this->checkHypreError( error );
    }

    //! Set the max size of the Krylov space.
    void setKDim( const int k_dim )
    {
        auto error = HYPRE_StructGMRESSetKDim( _solver, k_dim );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_StructGMRESSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_StructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_StructGMRESSetup;
    }
    HYPRE_PtrToStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_StructGMRESSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_StructGMRESSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_StructGMRESSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_StructGMRESSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl() override
    {
        auto error = HYPRE_StructGMRESSetup( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    void solveImpl() override
    {
        auto error = HYPRE_StructGMRESSolve( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_StructGMRESGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_StructGMRESGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreStructuredSolver<Scalar, EntityType, MemorySpace>&
            preconditioner ) override
    {
        auto error = HYPRE_StructGMRESSetPrecond(
            _solver, preconditioner.getHypreSolveFunction(),
            preconditioner.getHypreSetupFunction(),
            preconditioner.getHypreSolver() );
        this->checkHypreError( error );
    }

  private:
    HYPRE_StructSolver _solver;
    using base_type::_A;
    using base_type::_b;
    using base_type::_x;
};

//---------------------------------------------------------------------------//
//! BiCGSTAB solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreStructBiCGSTAB
    : public HypreStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using base_type = HypreStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreStructBiCGSTAB( const ArrayLayout_t& layout,
                         const bool is_preconditioner = false )
        : base_type( layout, is_preconditioner )
    {
        if ( is_preconditioner )
            throw std::logic_error(
                "HYPRE BiCGSTAB cannot be used as a preconditioner" );

        auto error = HYPRE_StructBiCGSTABCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreStructBiCGSTAB() { HYPRE_StructBiCGSTABDestroy( _solver ); }

    // BiCGSTAB SETTINGS

    //! Set the absolute tolerance
    void setAbsoluteTol( const double tol )
    {
        auto error = HYPRE_StructBiCGSTABSetAbsoluteTol( _solver, tol );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_StructBiCGSTABSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_StructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_StructBiCGSTABSetup;
    }
    HYPRE_PtrToStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_StructBiCGSTABSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_StructBiCGSTABSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_StructBiCGSTABSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_StructBiCGSTABSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl() override
    {
        auto error = HYPRE_StructBiCGSTABSetup( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    void solveImpl() override
    {
        auto error = HYPRE_StructBiCGSTABSolve( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_StructBiCGSTABGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreStructuredSolver<Scalar, EntityType, MemorySpace>&
            preconditioner ) override
    {
        auto error = HYPRE_StructBiCGSTABSetPrecond(
            _solver, preconditioner.getHypreSolveFunction(),
            preconditioner.getHypreSetupFunction(),
            preconditioner.getHypreSolver() );
        this->checkHypreError( error );
    }

  private:
    HYPRE_StructSolver _solver;
    using base_type::_A;
    using base_type::_b;
    using base_type::_x;
};

//---------------------------------------------------------------------------//
//! PFMG solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreStructPFMG
    : public HypreStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using base_type = HypreStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreStructPFMG( const ArrayLayout_t& layout,
                     const bool is_preconditioner = false )
        : base_type( layout, is_preconditioner )
    {
        auto error = HYPRE_StructPFMGCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );

        if ( is_preconditioner )
        {
            error = HYPRE_StructPFMGSetZeroGuess( _solver );
            this->checkHypreError( error );
        }
    }

    ~HypreStructPFMG() { HYPRE_StructPFMGDestroy( _solver ); }

    // PFMG SETTINGS

    //! Set the maximum number of multigrid levels.
    void setMaxLevels( const int max_levels )
    {
        auto error = HYPRE_StructPFMGSetMaxLevels( _solver, max_levels );
        this->checkHypreError( error );
    }

    //! Additionally require that the relative difference in successive
    //! iterates be small.
    void setRelChange( const int rel_change )
    {
        auto error = HYPRE_StructPFMGSetRelChange( _solver, rel_change );
        this->checkHypreError( error );
    }

    /*!
      \brief Set relaxation type.

      0 - Jacobi
      1 - Weighted Jacobi (default)
      2 - Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR
      post-relaxation)
      3 - Red/Black Gauss-Seidel (nonsymmetric: RB pre- and post-relaxation)
    */
    void setRelaxType( const int relax_type )
    {
        auto error = HYPRE_StructPFMGSetRelaxType( _solver, relax_type );
        this->checkHypreError( error );
    }

    //! Set the Jacobi weight
    void setJacobiWeight( const double weight )
    {
        auto error = HYPRE_StructPFMGSetJacobiWeight( _solver, weight );
        this->checkHypreError( error );
    }

    /*!
      \brief Set type of coarse-grid operator to use.

      0 - Galerkin (default)
      1 - non-Galerkin 5-pt or 7-pt stencils

      Both operators are constructed algebraically.  The non-Galerkin option
      maintains a 5-pt stencil in 2D and a 7-pt stencil in 3D on all grid
      levels. The stencil coefficients are computed by averaging techniques.
    */
    void setRAPType( const int rap_type )
    {
        auto error = HYPRE_StructPFMGSetRAPType( _solver, rap_type );
        this->checkHypreError( error );
    }

    //! Set number of relaxation sweeps before coarse-grid correction.
    void setNumPreRelax( const int num_pre_relax )
    {
        auto error = HYPRE_StructPFMGSetNumPreRelax( _solver, num_pre_relax );
        this->checkHypreError( error );
    }

    //! Set number of relaxation sweeps before coarse-grid correction.
    void setNumPostRelax( const int num_post_relax )
    {
        auto error = HYPRE_StructPFMGSetNumPostRelax( _solver, num_post_relax );
        this->checkHypreError( error );
    }

    //! Skip relaxation on certain grids for isotropic problems.  This can
    //! greatly improve efficiency by eliminating unnecessary relaxations when
    //! the underlying problem is isotropic.
    void setSkipRelax( const int skip_relax )
    {
        auto error = HYPRE_StructPFMGSetSkipRelax( _solver, skip_relax );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_StructPFMGSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_StructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_StructPFMGSetup;
    }
    HYPRE_PtrToStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_StructPFMGSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_StructPFMGSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_StructPFMGSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_StructPFMGSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl() override
    {
        auto error = HYPRE_StructPFMGSetup( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    void solveImpl() override
    {
        auto error = HYPRE_StructPFMGSolve( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_StructPFMGGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_StructPFMGGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreStructuredSolver<Scalar, EntityType, MemorySpace>& ) override
    {
        throw std::logic_error(
            "HYPRE PFMG solver does not support preconditioning." );
    }

  private:
    HYPRE_StructSolver _solver;
    using base_type::_A;
    using base_type::_b;
    using base_type::_x;
};

//---------------------------------------------------------------------------//
//! SMG solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreStructSMG
    : public HypreStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using base_type = HypreStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreStructSMG( const ArrayLayout_t& layout,
                    const bool is_preconditioner = false )
        : base_type( layout, is_preconditioner )
    {
        auto error = HYPRE_StructSMGCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );

        if ( is_preconditioner )
        {
            error = HYPRE_StructSMGSetZeroGuess( _solver );
            this->checkHypreError( error );
        }
    }

    ~HypreStructSMG() { HYPRE_StructSMGDestroy( _solver ); }

    // SMG Settings

    //! Additionally require that the relative difference in successive
    //! iterates be small.
    void setRelChange( const int rel_change )
    {
        auto error = HYPRE_StructSMGSetRelChange( _solver, rel_change );
        this->checkHypreError( error );
    }

    //! Set number of relaxation sweeps before coarse-grid correction.
    void setNumPreRelax( const int num_pre_relax )
    {
        auto error = HYPRE_StructSMGSetNumPreRelax( _solver, num_pre_relax );
        this->checkHypreError( error );
    }

    //! Set number of relaxation sweeps before coarse-grid correction.
    void setNumPostRelax( const int num_post_relax )
    {
        auto error = HYPRE_StructSMGSetNumPostRelax( _solver, num_post_relax );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_StructSMGSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_StructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_StructSMGSetup;
    }
    HYPRE_PtrToStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_StructSMGSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_StructSMGSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_StructSMGSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_StructSMGSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl() override
    {
        auto error = HYPRE_StructSMGSetup( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    void solveImpl() override
    {
        auto error = HYPRE_StructSMGSolve( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_StructSMGGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_StructSMGGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreStructuredSolver<Scalar, EntityType, MemorySpace>& ) override
    {
        throw std::logic_error(
            "HYPRE SMG solver does not support preconditioning." );
    }

  private:
    HYPRE_StructSolver _solver;
    using base_type::_A;
    using base_type::_b;
    using base_type::_x;
};

//---------------------------------------------------------------------------//
//! Jacobi solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreStructJacobi
    : public HypreStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using base_type = HypreStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreStructJacobi( const ArrayLayout_t& layout,
                       const bool is_preconditioner = false )
        : base_type( layout, is_preconditioner )
    {
        auto error = HYPRE_StructJacobiCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );

        if ( is_preconditioner )
        {
            error = HYPRE_StructJacobiSetZeroGuess( _solver );
            this->checkHypreError( error );
        }
    }

    ~HypreStructJacobi() { HYPRE_StructJacobiDestroy( _solver ); }

    HYPRE_StructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_StructJacobiSetup;
    }
    HYPRE_PtrToStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_StructJacobiSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_StructJacobiSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_StructJacobiSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int ) override
    {
        // The Jacobi solver does not support a print level.
    }

    void setupImpl() override
    {
        auto error = HYPRE_StructJacobiSetup( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    void solveImpl() override
    {
        auto error = HYPRE_StructJacobiSolve( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_StructJacobiGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_StructJacobiGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreStructuredSolver<Scalar, EntityType, MemorySpace>& ) override
    {
        throw std::logic_error(
            "HYPRE Jacobi solver does not support preconditioning." );
    }

  private:
    HYPRE_StructSolver _solver;
    using base_type::_A;
    using base_type::_b;
    using base_type::_x;
};

//---------------------------------------------------------------------------//
//! Diagonal preconditioner.
template <class Scalar, class EntityType, class MemorySpace>
class HypreStructDiagonal
    : public HypreStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using base_type = HypreStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreStructDiagonal( const ArrayLayout_t& layout,
                         const bool is_preconditioner = false )
        : base_type( layout, is_preconditioner )
    {
        if ( !is_preconditioner )
            throw std::logic_error(
                "Diagonal preconditioner cannot be used as a solver" );
    }

    HYPRE_StructSolver getHypreSolver() const override { return nullptr; }
    HYPRE_PtrToStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_StructDiagScaleSetup;
    }
    HYPRE_PtrToStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_StructDiagScale;
    }

  protected:
    void setToleranceImpl( const double ) override
    {
        throw std::logic_error(
            "Diagonal preconditioner cannot be used as a solver" );
    }

    void setMaxIterImpl( const int ) override
    {
        throw std::logic_error(
            "Diagonal preconditioner cannot be used as a solver" );
    }

    void setPrintLevelImpl( const int ) override
    {
        throw std::logic_error(
            "Diagonal preconditioner cannot be used as a solver" );
    }

    void setupImpl() override
    {
        throw std::logic_error(
            "Diagonal preconditioner cannot be used as a solver" );
    }

    void solveImpl() override
    {
        throw std::logic_error(
            "Diagonal preconditioner cannot be used as a solver" );
    }

    int getNumIterImpl() override
    {
        throw std::logic_error(
            "Diagonal preconditioner cannot be used as a solver" );
    }

    double getFinalRelativeResidualNormImpl() override
    {
        throw std::logic_error(
            "Diagonal preconditioner cannot be used as a solver" );
    }

    void setPreconditionerImpl(
        const HypreStructuredSolver<Scalar, EntityType, MemorySpace>& ) override
    {
        throw std::logic_error(
            "Diagonal preconditioner does not support preconditioning." );
    }
};

//---------------------------------------------------------------------------//
// Builders
//---------------------------------------------------------------------------//
//! Create a HYPRE PCG structured solver.
//! \return Shared pointer to HypreStructPCG.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreStructPCG<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreStructPCG( const ArrayLayout_t& layout,
                      const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructPCG<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE GMRES structured solver.
//! \return Shared pointer to HypreStructGMRES.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreStructGMRES<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreStructGMRES( const ArrayLayout_t& layout,
                        const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructGMRES<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE BiCGSTAB structured solver.
//! \return Shared pointer to HypreStructBiCGSTAB.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreStructBiCGSTAB<Scalar, typename ArrayLayout_t::entity_type,
                                    MemorySpace>>
createHypreStructBiCGSTAB( const ArrayLayout_t& layout,
                           const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructBiCGSTAB<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE PFMG structured solver.
//! \return Shared pointer to HypreStructPFMG.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreStructPFMG<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreStructPFMG( const ArrayLayout_t& layout,
                       const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructPFMG<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE SMG structured solver.
//! \return Shared pointer to HypreStructSMG.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreStructSMG<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreStructSMG( const ArrayLayout_t& layout,
                      const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructSMG<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE Jacobi structured solver.
//! \return Shared pointer to HypreStructJacobi.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreStructJacobi<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreStructJacobi( const ArrayLayout_t& layout,
                         const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructJacobi<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE Diagonal structured solver.
//! \return Shared pointer to HypreStructDiagonal.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreStructDiagonal<Scalar, typename ArrayLayout_t::entity_type,
                                    MemorySpace>>
createHypreStructDiagonal( const ArrayLayout_t& layout,
                           const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructDiagonal<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//---------------------------------------------------------------------------//
// Factory
//---------------------------------------------------------------------------//
/*!
  \brief Create a HYPRE structured solver.

  \param solver_type Solver name.
  \param layout The ArrayLayout defining the vector space of the solver.
  \param is_preconditioner Use as a preconditioner.
  \return Shared pointer to a HypreStructuredSolver.
*/
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreStructuredSolver<
    Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreStructuredSolver( const std::string& solver_type,
                             const ArrayLayout_t& layout,
                             const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );

    if ( "PCG" == solver_type )
        return createHypreStructPCG<Scalar, MemorySpace>( layout,
                                                          is_preconditioner );
    else if ( "GMRES" == solver_type )
        return createHypreStructGMRES<Scalar, MemorySpace>( layout,
                                                            is_preconditioner );
    else if ( "BiCGSTAB" == solver_type )
        return createHypreStructBiCGSTAB<Scalar, MemorySpace>(
            layout, is_preconditioner );
    else if ( "PFMG" == solver_type )
        return createHypreStructPFMG<Scalar, MemorySpace>( layout,
                                                           is_preconditioner );
    else if ( "SMG" == solver_type )
        return createHypreStructSMG<Scalar, MemorySpace>( layout,
                                                          is_preconditioner );
    else if ( "Jacobi" == solver_type )
        return createHypreStructJacobi<Scalar, MemorySpace>(
            layout, is_preconditioner );
    else if ( "Diagonal" == solver_type )
        return createHypreStructDiagonal<Scalar, MemorySpace>(
            layout, is_preconditioner );
    else
        throw std::runtime_error(
            "Cabana::Grid::createHypreStructuredSolver: Invalid solver type" );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_HYPRESTRUCTUREDSOLVER_HPP
