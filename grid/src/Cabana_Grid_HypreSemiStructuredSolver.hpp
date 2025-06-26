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
  \file Cabana_Grid_HypreSemiStructuredSolver.hpp
  \brief HYPRE semi-structured solver interface
*/
#ifndef CABANA_GRID_HYPRESEMISTRUCTUREDSOLVER_HPP
#define CABANA_GRID_HYPRESEMISTRUCTUREDSOLVER_HPP

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_Hypre.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_Types.hpp>

#include <HYPRE_config.h>
#include <HYPRE_sstruct_ls.h>
#include <HYPRE_sstruct_mv.h>
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
//! Hypre semi-structured solver interface for scalar fields.
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructuredSolver
{
  public:
    //! Entity type.
    using entity_type = EntityType;
    //! Kokkos memory space..
    using memory_space = MemorySpace;
    //! Scalar value type.
    using value_type = Scalar;
    //! Object Type for SStruct
    const int object_type = HYPRE_SSTRUCT;
    //! Hypre memory space compatibility check.
    static_assert( HypreIsCompatibleWithMemorySpace<memory_space>::value,
                   "HYPRE not compatible with solver memory space" );

    /*!
      \brief Constructor.
      \param layout The array layout defining the vector space of the solver.
      \param n_vars Number of variable in the system domain
      \param is_preconditioner Flag indicating if this solver will be used as
      a preconditioner.
    */
    template <class ArrayLayout_t>
    HypreSemiStructuredSolver( const ArrayLayout_t& layout, int n_vars,
                               const bool is_preconditioner = false )
        : _comm( layout.localGrid()->globalGrid().comm() )
        , _is_preconditioner( is_preconditioner )
    {
        HYPRE_Init();

        static_assert( is_array_layout<ArrayLayout_t>::value,
                       "Must use an array layout" );
        static_assert(
            std::is_same<typename ArrayLayout_t::entity_type,
                         entity_type>::value,
            "Array layout entity type must match solver entity type" );

        // Spatial dimension.
        const std::size_t num_space_dim = ArrayLayout_t::num_space_dim;

        // Only a single part grid is supported initially
        int n_parts = 1;
        int part = 0;

        // Only create data structures if this is not a preconditioner.
        if ( !_is_preconditioner )
        {
            // Create the grid.
            auto error = HYPRE_SStructGridCreate( _comm, num_space_dim, n_parts,
                                                  &_grid );
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
            error = HYPRE_SStructGridSetExtents( _grid, part, _lower.data(),
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
            error = HYPRE_SStructGridSetPeriodic( _grid, part, periodic );
            checkHypreError( error );

            // Set the variables on the HYPRE grid
            std::vector<HYPRE_SStructVariable> vartypes;
            vartypes.resize( n_vars );
            for ( int i = 0; i < n_vars; ++i )
            {
                vartypes[i] = HYPRE_SSTRUCT_VARIABLE_CELL;
            }
            error = HYPRE_SStructGridSetVariables( _grid, part, n_vars,
                                                   vartypes.data() );

            // Assemble the grid.
            error = HYPRE_SStructGridAssemble( _grid );
            checkHypreError( error );

            // Allocate LHS and RHS vectors and initialize to zero. Note that we
            // are fixing the views under these vectors to layout-right.

            std::array<long, num_space_dim + 1> reorder_size;
            for ( std::size_t d = 0; d < num_space_dim; ++d )
            {
                reorder_size[d] = global_space.extent( d );
            }
            reorder_size.back() = n_vars;
            IndexSpace<num_space_dim + 1> reorder_space( reorder_size );
            auto vector_values =
                createView<HYPRE_Complex, Kokkos::LayoutRight, memory_space>(
                    "vector_values0", reorder_space );
            Kokkos::deep_copy( vector_values, 0.0 );

            _stencils.resize( n_vars );
            _stencil_size.resize( n_vars );
            _stencil_index.resize( n_vars,
                                   std::vector<unsigned>( n_vars + 1 ) );

            error = HYPRE_SStructVectorCreate( _comm, _grid, &_b );
            checkHypreError( error );
            error = HYPRE_SStructVectorSetObjectType( _b, object_type );
            checkHypreError( error );
            error = HYPRE_SStructVectorInitialize( _b );
            checkHypreError( error );
            for ( int i = 0; i < n_vars; ++i )
            {
                error = HYPRE_SStructVectorSetBoxValues(
                    _b, part, _lower.data(), _upper.data(), i,
                    vector_values.data() );
                checkHypreError( error );
            }
            error = HYPRE_SStructVectorAssemble( _b );
            checkHypreError( error );

            error = HYPRE_SStructVectorCreate( _comm, _grid, &_x );
            checkHypreError( error );
            error = HYPRE_SStructVectorSetObjectType( _x, object_type );
            checkHypreError( error );
            error = HYPRE_SStructVectorInitialize( _x );
            checkHypreError( error );
            for ( int i = 0; i < n_vars; ++i )
            {
                error = HYPRE_SStructVectorSetBoxValues(
                    _x, part, _lower.data(), _upper.data(), i,
                    vector_values.data() );
                checkHypreError( error );
            }
            checkHypreError( error );
            error = HYPRE_SStructVectorAssemble( _x );
            checkHypreError( error );
        }
    }

    // Destructor.
    virtual ~HypreSemiStructuredSolver()
    {
        // We only make data if this is not a preconditioner.
        if ( !_is_preconditioner )
        {
            HYPRE_SStructVectorDestroy( _x );
            HYPRE_SStructVectorDestroy( _b );
            HYPRE_SStructMatrixDestroy( _A );
            for ( std::size_t i = 0; i < _stencils.size(); ++i )
            {
                HYPRE_SStructStencilDestroy( _stencils[i] );
            }
            HYPRE_SStructGridDestroy( _grid );
            HYPRE_SStructGraphDestroy( _graph );

            HYPRE_Finalize();
        }
    }

    //! Return if this solver is a preconditioner.
    bool isPreconditioner() const { return _is_preconditioner; }

    /*!
      \brief Create the operator stencil to be filled by setMatrixStencil
      \param NumSpaceDim The number of spatial dimensions in the linear system
      being solved.
      \param var The variable number that the stencil corresponds to, in essence
      which equation number in the linear system
      \param n_vars number of variables in the linear system
      \param stencil_length A vector containing the length of the stencil for
      variable `var` for each variable in the system to be created for HYPRE
    */
    void createMatrixStencil( int NumSpaceDim, int var = 0, int n_vars = 3,
                              std::vector<unsigned> stencil_length = { 7, 7,
                                                                       7 } )
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call createMatrixStencil() on preconditioners" );

        // Generate the stencil indexing
        unsigned index = 0;
        for ( int i = 0; i < n_vars; ++i )
        {
            _stencil_index[var][i] = index;
            index += stencil_length[i];
        }
        _stencil_index[var][n_vars] = index;

        // Create the stencil.
        _stencil_size[var] = index;
        auto error = HYPRE_SStructStencilCreate(
            NumSpaceDim, _stencil_size[var], &_stencils[var] );
        checkHypreError( error );
    }

    /*!
      \brief Set the operator stencil.
      \param stencil The (i,j,k) offsets describing the structured matrix
      entries at each grid point. Offsets are defined relative to an index.
      \param var The variable number that the stencil corresponds to, in essence
       which equation number in the linear system
      \param dep The integer for the independent variable in the linear system
      that is currently being set
    */
    template <std::size_t NumSpaceDim>
    void
    setMatrixStencil( const std::vector<std::array<int, NumSpaceDim>>& stencil,
                      int var = 0, int dep = 0 )
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call setMatrixStencil() on preconditioners" );

        std::array<HYPRE_Int, NumSpaceDim> offset;

        auto index = _stencil_index[var][dep];
        for ( unsigned n = index; n < index + stencil.size(); ++n )
        {
            for ( std::size_t d = 0; d < NumSpaceDim; ++d )
                offset[d] = stencil[n - index][d];
            auto error = HYPRE_SStructStencilSetEntry( _stencils[var], n,
                                                       offset.data(), dep );
            checkHypreError( error );
        }
    }

    /*!
      \brief Set the solver graph
      \param n_vars The number of variables (and equations) in the
      specified problem domain
    */
    void setSolverGraph( int n_vars )
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call setSolverGraph() on preconditioners" );

        int part = 0;

        // Setup the Graph for the non-zero structure of the matrix
        // Create the graph with hypre
        auto error = HYPRE_SStructGraphCreate( _comm, _grid, &_graph );
        checkHypreError( error );

        // Set up the object type
        error = HYPRE_SStructGraphSetObjectType( _graph, object_type );
        checkHypreError( error );

        // Set the stencil to the graph
        for ( int i = 0; i < n_vars; ++i )
        {
            error =
                HYPRE_SStructGraphSetStencil( _graph, part, i, _stencils[i] );
            checkHypreError( error );
        }

        // Assemble the graph
        error = HYPRE_SStructGraphAssemble( _graph );
        checkHypreError( error );

        // Create the matrix. must be done after graph is assembled
        error = HYPRE_SStructMatrixCreate( _comm, _graph, &_A );
        checkHypreError( error );

        // Set the SStruct matrix object type
        error = HYPRE_SStructMatrixSetObjectType( _A, object_type );
        checkHypreError( error );

        // Prepare the matrix for setting values
        error = HYPRE_SStructMatrixInitialize( _A );
        checkHypreError( error );
    }

    /*!
      \brief Set the matrix values.
      \param values The matrix entry values. For each entity over which the
      vector space is defined an entry for each stencil element is
      required. The order of the stencil elements is that same as that in the
      stencil definition. Note that values corresponding to stencil entries
      outside of the domain should be set to zero.
      \param v_x The variable index for the independent variable (column)
      that is being set by the current call to setMatrixValues
      \param v_h The variable index for the equation (row) that is being
      set by the current call to setMatrixValues
    */
    template <class Array_t>
    void setMatrixValues( const Array_t& values, int v_x, int v_h )
    {
        static_assert( is_array<Array_t>::value, "Must use an array" );
        static_assert(
            std::is_same<typename Array_t::entity_type, entity_type>::value,
            "Array entity type must match solver entity type" );
        static_assert(
            std::is_same<typename Array_t::memory_space, MemorySpace>::value,
            "Array device type and solver device type are different." );

        static_assert(
            std::is_same<typename Array_t::value_type, value_type>::value,
            "Array value type and solver value type are different." );

        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call setMatrixValues() on preconditioners" );

        int index_size =
            _stencil_index[v_h][v_x + 1] - _stencil_index[v_h][v_x];

        // Ensure the values array matches up in dimension with the stencil size
        if ( values.layout()->dofsPerEntity() !=
             static_cast<int>( index_size ) )
            throw std::runtime_error(
                "Cabana::Grid::HypreSemiStructuredSolver::setMatrixValues: "
                "Number of matrix values does not match stencil size" );

        // Spatial dimension.
        const std::size_t num_space_dim = Array_t::num_space_dim;

        int part = 0;

        // Copy the matrix entries into HYPRE. The HYPRE layout is fixed as
        // layout-right.
        auto owned_space = values.layout()->indexSpace( Own(), Local() );
        std::array<long, num_space_dim + 1> reorder_size;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            reorder_size[d] = owned_space.extent( d );
        }

        reorder_size.back() = index_size;
        IndexSpace<num_space_dim + 1> reorder_space( reorder_size );
        auto a_values =
            createView<HYPRE_Complex, Kokkos::LayoutRight, memory_space>(
                "a_values", reorder_space );

        auto values_subv = createSubview( values.view(), owned_space );
        Kokkos::deep_copy( a_values, values_subv );

        // Insert values into the HYPRE matrix.
        std::vector<HYPRE_Int> indices( index_size );
        int start = _stencil_index[v_h][v_x];
        std::iota( indices.begin(), indices.end(), start );
        auto error = HYPRE_SStructMatrixSetBoxValues(
            _A, part, _lower.data(), _upper.data(), v_h, indices.size(),
            indices.data(), a_values.data() );
        checkHypreError( error );
    }

    /*!
      \brief Print the hypre matrix to output file
      \param prefix File prefix for where hypre output is written
    */
    void printMatrix( const char* prefix )
    {
        HYPRE_SStructMatrixPrint( prefix, _A, 0 );
    }

    /*!
      \brief Print the hypre LHS to output file
      \param prefix File prefix for where hypre output is written
    */
    void printLHS( const char* prefix )
    {
        HYPRE_SStructVectorPrint( prefix, _x, 0 );
    }

    /*!
      \brief Print the hypre RHS to output file
      \param prefix File prefix for where hypre output is written
    */
    void printRHS( const char* prefix )
    {
        HYPRE_SStructVectorPrint( prefix, _b, 0 );
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
    setPreconditioner( const std::shared_ptr<HypreSemiStructuredSolver<
                           Scalar, EntityType, MemorySpace>>& preconditioner )
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call setPreconditioner() on a preconditioner" );

        // Only a preconditioner can be used as a preconditioner.
        if ( !preconditioner->isPreconditioner() )
            throw std::logic_error( "Not a preconditioner" );

        _preconditioner = preconditioner;
        this->setPreconditionerImpl( *_preconditioner );
    }

    //! Setup the problem.
    void setup()
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error( "Cannot call setup() on preconditioners" );

        auto error = HYPRE_SStructMatrixAssemble( _A );
        checkHypreError( error );

        this->setupImpl();
    }

    /*!
      \brief Solve the problem Ax = b for x.
      \param b The forcing term.
      \param x The solution.
      \param n_vars Number of variables in the solution domain
    */
    template <class Array_t>
    void solve( const Array_t& b, Array_t& x, int n_vars = 3 )
    {
        Kokkos::Profiling::ScopedRegion region(
            "Cabana::Grid::HypreSemiStructuredSolver::solve" );

        static_assert( is_array<Array_t>::value, "Must use an array" );
        static_assert(
            std::is_same<typename Array_t::entity_type, entity_type>::value,
            "Array entity type must match solver entity type" );
        static_assert(
            std::is_same<typename Array_t::memory_space, MemorySpace>::value,
            "Array device type and solver device type are different." );

        static_assert(
            std::is_same<typename Array_t::value_type, value_type>::value,
            "Array value type and solver value type are different." );

        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error( "Cannot call solve() on preconditioners" );

        // Spatial dimension.
        const std::size_t num_space_dim = Array_t::num_space_dim;

        int part = 0;

        // Copy the RHS into HYPRE. The HYPRE layout is fixed as layout-right.
        auto owned_space = b.layout()->indexSpace( Own(), Local() );
        std::array<long, num_space_dim + 1> reorder_min;
        std::array<long, num_space_dim + 1> reorder_max;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            reorder_min[d] = owned_space.min( d );
            reorder_max[d] = owned_space.max( d );
        }

        // Insert b values into the HYPRE vector.
        // The process of creating the view and then deep copying each
        // variable is functional, but we should avoid this process
        // for performance if possible
        int error;
        for ( int var = 0; var < n_vars; ++var )
        {
            reorder_min.back() = var;
            reorder_max.back() = var + 1;

            IndexSpace<num_space_dim + 1> reorder_space( reorder_min,
                                                         reorder_max );
            auto b_values =
                createView<HYPRE_Complex, Kokkos::LayoutRight, memory_space>(
                    "vector_values", reorder_space );
            // Extract one variable at at time.
            auto b_subv = createSubview( b.view(), reorder_space );

            Kokkos::deep_copy( b_values, b_subv );
            error = HYPRE_SStructVectorSetBoxValues(
                _b, part, _lower.data(), _upper.data(), var, b_values.data() );
            checkHypreError( error );
        }

        error = HYPRE_SStructVectorAssemble( _b );
        checkHypreError( error );

        // Solve the problem
        this->solveImpl();

        // Extract the solution from the LHS
        for ( int var = 0; var < n_vars; ++var )
        {
            reorder_min.back() = var;
            reorder_max.back() = var + 1;

            IndexSpace<num_space_dim + 1> reorder_space( reorder_min,
                                                         reorder_max );
            auto x_values =
                createView<HYPRE_Complex, Kokkos::LayoutRight, memory_space>(
                    "vector_values", reorder_space );

            // Extract one variable at at time.
            // Use a pair here to retain the view rank.

            error = HYPRE_SStructVectorGetBoxValues(
                _x, part, _lower.data(), _upper.data(), var, x_values.data() );
            checkHypreError( error );

            // Copy the HYPRE solution to the LHS.
            auto x_subv = createSubview( x.view(), reorder_space );
            Kokkos::deep_copy( x_subv, x_values );
        }
    }

    //! Get the number of iterations taken on the last solve.
    int getNumIter() { return this->getNumIterImpl(); }

    //! Get the relative residual norm achieved on the last solve.
    double getFinalRelativeResidualNorm()
    {
        return this->getFinalRelativeResidualNormImpl();
    }

    //! Get the preconditioner.
    virtual HYPRE_SStructSolver getHypreSolver() const = 0;
    //! Get the preconditioner setup function.
    virtual HYPRE_PtrToSStructSolverFcn getHypreSetupFunction() const = 0;
    //! Get the preconditioner solve function.
    virtual HYPRE_PtrToSStructSolverFcn getHypreSolveFunction() const = 0;

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
        const HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>&
            preconditioner ) = 0;

    //! Check a hypre error.
    void checkHypreError( const int error ) const
    {
        if ( error > 0 )
        {
            char error_msg[256];
            HYPRE_DescribeError( error, error_msg );
            std::stringstream out;
            out << "HYPRE semi-structured solver error: ";
            out << error << " " << error_msg;
            HYPRE_ClearError( error );
            throw std::runtime_error( out.str() );
        }
    }

    //! Matrix for the problem Ax = b.
    HYPRE_SStructMatrix _A;
    //! Forcing term for the problem Ax = b.
    HYPRE_SStructVector _b;
    //! Solution to the problem Ax = b.
    HYPRE_SStructVector _x;

  private:
    MPI_Comm _comm;
    bool _is_preconditioner;
    HYPRE_SStructGrid _grid;
    std::vector<HYPRE_Int> _lower;
    std::vector<HYPRE_Int> _upper;
    std::vector<HYPRE_SStructStencil> _stencils;
    HYPRE_SStructGraph _graph;
    std::vector<unsigned> _stencil_size;
    std::vector<std::vector<unsigned>> _stencil_index;
    std::shared_ptr<HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>>
        _preconditioner;
};

//---------------------------------------------------------------------------//
//! PCG solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructPCG
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE semi-structured solver type.
    using base_type =
        HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructPCG( const ArrayLayout_t& layout, int n_vars,
                        const bool is_preconditioner = false )
        : base_type( layout, n_vars, is_preconditioner )
    {
        if ( is_preconditioner )
            throw std::logic_error(
                "HYPRE PCG cannot be used as a preconditioner" );

        auto error = HYPRE_SStructPCGCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );

        HYPRE_SStructPCGSetTwoNorm( _solver, 1 );
    }

    ~HypreSemiStructPCG() { HYPRE_SStructPCGDestroy( _solver ); }

    // PCG SETTINGS

    //! Set the absolute tolerance
    void setAbsoluteTol( const double tol )
    {
        auto error = HYPRE_SStructPCGSetAbsoluteTol( _solver, tol );
        this->checkHypreError( error );
    }

    //! Additionally require that the relative difference in successive
    //! iterates be small.
    void setRelChange( const int rel_change )
    {
        auto error = HYPRE_SStructPCGSetRelChange( _solver, rel_change );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_SStructPCGSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_SStructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToSStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_SStructPCGSetup;
    }
    HYPRE_PtrToSStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_SStructPCGSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_SStructPCGSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_SStructPCGSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_SStructPCGSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl() override
    {
        auto error = HYPRE_SStructPCGSetup( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    void solveImpl() override
    {
        auto error = HYPRE_SStructPCGSolve( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_SStructPCGGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_SStructPCGGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>&
            preconditioner ) override
    {
        auto error = HYPRE_SStructPCGSetPrecond(
            _solver, preconditioner.getHypreSolveFunction(),
            preconditioner.getHypreSetupFunction(),
            preconditioner.getHypreSolver() );
        this->checkHypreError( error );
    }

  private:
    HYPRE_SStructSolver _solver;
    using base_type::_A;
    using base_type::_b;
    using base_type::_x;
};

//---------------------------------------------------------------------------//
//! GMRES solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructGMRES
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE semi-structured solver type.
    using base_type =
        HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructGMRES( const ArrayLayout_t& layout, int n_vars,
                          const bool is_preconditioner = false )
        : base_type( layout, n_vars, is_preconditioner )
    {
        if ( is_preconditioner )
            throw std::logic_error(
                "HYPRE GMRES cannot be used as a preconditioner" );

        auto error = HYPRE_SStructGMRESCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreSemiStructGMRES() { HYPRE_SStructGMRESDestroy( _solver ); }

    // GMRES SETTINGS

    //! Set the absolute tolerance
    void setAbsoluteTol( const double tol )
    {
        auto error = HYPRE_SStructGMRESSetAbsoluteTol( _solver, tol );
        this->checkHypreError( error );
    }

    //! Set the max size of the Krylov space.
    void setKDim( const int k_dim )
    {
        auto error = HYPRE_SStructGMRESSetKDim( _solver, k_dim );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_SStructGMRESSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_SStructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToSStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_SStructGMRESSetup;
    }
    HYPRE_PtrToSStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_SStructGMRESSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_SStructGMRESSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_SStructGMRESSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_SStructGMRESSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl() override
    {
        auto error = HYPRE_SStructGMRESSetup( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    void solveImpl() override
    {
        auto error = HYPRE_SStructGMRESSolve( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_SStructGMRESGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_SStructGMRESGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>&
            preconditioner ) override
    {
        auto error = HYPRE_SStructGMRESSetPrecond(
            _solver, preconditioner.getHypreSolveFunction(),
            preconditioner.getHypreSetupFunction(),
            preconditioner.getHypreSolver() );
        this->checkHypreError( error );
    }

  private:
    HYPRE_SStructSolver _solver;
    using base_type::_A;
    using base_type::_b;
    using base_type::_x;
};

//---------------------------------------------------------------------------//
//! BiCGSTAB solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructBiCGSTAB
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE semi-structured solver type.
    using base_type =
        HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructBiCGSTAB( const ArrayLayout_t& layout,
                             const bool is_preconditioner = false,
                             int n_vars = 3 )
        : base_type( layout, n_vars, is_preconditioner )
    {
        if ( is_preconditioner )
            throw std::logic_error(
                "HYPRE BiCGSTAB cannot be used as a preconditioner" );

        auto error = HYPRE_SStructBiCGSTABCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreSemiStructBiCGSTAB() { HYPRE_SStructBiCGSTABDestroy( _solver ); }

    // BiCGSTAB SETTINGS

    //! Set the absolute tolerance
    void setAbsoluteTol( const double tol )
    {
        auto error = HYPRE_SStructBiCGSTABSetAbsoluteTol( _solver, tol );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_SStructBiCGSTABSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_SStructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToSStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_SStructBiCGSTABSetup;
    }
    HYPRE_PtrToSStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_SStructBiCGSTABSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_SStructBiCGSTABSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_SStructBiCGSTABSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_SStructBiCGSTABSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl() override
    {
        auto error = HYPRE_SStructBiCGSTABSetup( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    void solveImpl() override
    {
        auto error = HYPRE_SStructBiCGSTABSolve( _solver, _A, _b, _x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error =
            HYPRE_SStructBiCGSTABGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>&
            preconditioner ) override
    {
        auto error = HYPRE_SStructBiCGSTABSetPrecond(
            _solver, preconditioner.getHypreSolveFunction(),
            preconditioner.getHypreSetupFunction(),
            preconditioner.getHypreSolver() );
        this->checkHypreError( error );
    }

  private:
    HYPRE_SStructSolver _solver;
    using base_type::_A;
    using base_type::_b;
    using base_type::_x;
};

//---------------------------------------------------------------------------//
//! Diagonal preconditioner.
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructDiagonal
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE semi-structured solver type.
    using base_type =
        HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructDiagonal( const ArrayLayout_t& layout,
                             const bool is_preconditioner = false,
                             int n_vars = 3 )
        : base_type( layout, n_vars, is_preconditioner )
    {
        if ( !is_preconditioner )
            throw std::logic_error(
                "Diagonal preconditioner cannot be used as a solver" );
    }

    HYPRE_SStructSolver getHypreSolver() const override { return nullptr; }
    HYPRE_PtrToSStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_SStructDiagScaleSetup;
    }
    HYPRE_PtrToSStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_SStructDiagScale;
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
        const HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>& )
        override
    {
        throw std::logic_error(
            "Diagonal preconditioner does not support preconditioning." );
    }
};

//---------------------------------------------------------------------------//
// Builders
//---------------------------------------------------------------------------//
//! Create a HYPRE PCG semi-structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreSemiStructPCG<Scalar, typename ArrayLayout_t::entity_type,
                                   MemorySpace>>
createHypreSemiStructPCG( const ArrayLayout_t& layout,
                          const bool is_preconditioner = false, int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructPCG<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, n_vars, is_preconditioner );
}

//! Create a HYPRE GMRES semi-structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreSemiStructGMRES<
    Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructGMRES( const ArrayLayout_t& layout,
                            const bool is_preconditioner = false,
                            int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructGMRES<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, n_vars, is_preconditioner );
}

//! Create a HYPRE BiCGSTAB semi-structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreSemiStructBiCGSTAB<
    Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructBiCGSTAB( const ArrayLayout_t& layout,
                               const bool is_preconditioner = false,
                               int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructBiCGSTAB<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner, n_vars );
}

//! Create a HYPRE Diagonal semi-structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreSemiStructDiagonal<
    Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructDiagonal( const ArrayLayout_t& layout,
                               const bool is_preconditioner = false,
                               int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructDiagonal<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner, n_vars );
}

//---------------------------------------------------------------------------//
// Factory
//---------------------------------------------------------------------------//
/*!
  \brief Create a HYPRE semi-structured solver.

  \param solver_type Solver name.
  \param layout The ArrayLayout defining the vector space of the solver.
  \param is_preconditioner Use as a preconditioner.
  \param n_vars Number of variables in the solver
*/
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreSemiStructuredSolver<
    Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructuredSolver( const std::string& solver_type,
                                 const ArrayLayout_t& layout,
                                 const bool is_preconditioner = false,
                                 int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );

    if ( "PCG" == solver_type )
        return createHypreSemiStructPCG<Scalar, MemorySpace>(
            layout, is_preconditioner, n_vars );
    else if ( "GMRES" == solver_type )
        return createHypreSemiStructGMRES<Scalar, MemorySpace>(
            layout, is_preconditioner, n_vars );
    else if ( "BiCGSTAB" == solver_type )
        return createHypreSemiStructBiCGSTAB<Scalar, MemorySpace>(
            layout, is_preconditioner, n_vars );
    else if ( "Diagonal" == solver_type )
        return createHypreSemiStructDiagonal<Scalar, MemorySpace>(
            layout, is_preconditioner, n_vars );
    else
        throw std::runtime_error(
            "Cabana::Grid::createHypreSemiStructuredSolver: Invalid solver "
            "type" );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_HypreSemiStRUCTUREDSOLVER_HPP
