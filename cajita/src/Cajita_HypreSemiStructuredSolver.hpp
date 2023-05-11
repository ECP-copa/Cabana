/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cajita_HypreSemiStructuredSolver.hpp
  \brief HYPRE semi-structured solver interface
*/
#ifndef CAJITA_HYPRESEMISTRUCTUREDSOLVER_HPP
#define CAJITA_HYPRESEMISTRUCTUREDSOLVER_HPP

#include <Cajita_Array.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_Types.hpp>

#include <HYPRE_config.h>
#include <HYPRE_sstruct_ls.h>
#include <HYPRE_sstruct_mv.h>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Hypre memory space selection. Don't compile if HYPRE wasn't configured to
// use the device.
// ---------------------------------------------------------------------------//

/*

//! Hypre device compatibility check.
template <class MemorySpace>
struct HypreIsCompatibleWithMemorySpace : std::false_type
{
};

// FIXME: This is currently written in this structure because HYPRE only has
// compile-time switches for backends and hence only one can be used at a
// time. Once they have a run-time switch we can use that instead.
#ifdef HYPRE_USING_CUDA
#ifdef KOKKOS_ENABLE_CUDA
#ifdef HYPRE_USING_DEVICE_MEMORY
//! Hypre device compatibility check - CUDA memory.
template <>
struct HypreIsCompatibleWithMemorySpace<Kokkos::CudaSpace> : std::true_type
{
};
#endif // end HYPRE_USING_DEVICE_MEMORY

//! Hypre device compatibility check - CUDA UVM memory.
#ifdef HYPRE_USING_UNIFIED_MEMORY
template <>
struct HypreIsCompatibleWithMemorySpace<Kokkos::CudaUVMSpace> : std::true_type
{
};
#endif // end HYPRE_USING_UNIFIED_MEMORY
#endif // end KOKKOS_ENABLE_CUDA
#endif // end HYPRE_USING_CUDA

#ifdef HYPRE_USING_HIP
#ifdef KOKKOS_ENABLE_HIP
//! Hypre device compatibility check - HIP memory. FIXME - make this true when
//! the HYPRE CMake includes HIP
template <>
struct HypreIsCompatibleWithMemorySpace<Kokkos::ExperimentalHIPSpace>
    : std::false_type
{
};
#endif // end KOKKOS_ENABLE_HIP
#endif // end HYPRE_USING_HIP

#ifndef HYPRE_USING_GPU
//! Hypre device compatibility check - host memory.
template <>
struct HypreIsCompatibleWithMemorySpace<Kokkos::HostSpace> : std::true_type
{
};
#endif // end HYPRE_USING_GPU

*/

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
    int object_type = HYPRE_SSTRUCT;
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
    HypreSemiStructuredSolver( const ArrayLayout_t& layout,
                           const bool is_preconditioner = false, int n_vars = 3 )
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

        // Only a single part grid is supported initially
        int n_parts = 1;
        int part = 0;

        // Only create data structures if this is not a preconditioner.
        if ( !_is_preconditioner )
        {
            // Create the grid.
            auto error = HYPRE_SStructGridCreate( _comm, num_space_dim, n_parts, &_grid );
            checkHypreError( error );

            // Get the global index space spanned by the local grid on this
            // rank. Note that the upper bound is not a bound but rather the
            // last index as this is what Hypre wants. Note that we reordered
            // this to KJI from IJK to be consistent with HYPRE ordering. By
            // setting up the grid like this, HYPRE will then want layout-right
            // data indexed as (i,j,k) or (i,j,k,l) which will allow us to
            // directly use Kokkos::deep_copy to move data between Cajita arrays
            // and HYPRE data structures.
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

            // *FIX* set up for n-variate solution 
            // Set the variables on the HYPRE grid
            std::vector<HYPRE_SStructVariable> vartypes;
            vartypes.resize( n_vars );
//            HYPRE_SStructVariable vartypes[3]= { HYPRE_SSTRUCT_VARIABLE_NODE,
//                                                 HYPRE_SSTRUCT_VARIABLE_NODE,
//                                                 HYPRE_SSTRUCT_VARIABLE_NODE};
            for (int i = 0; i < n_vars; ++i) 
            {
                vartypes[ i ] = HYPRE_SSTRUCT_VARIABLE_NODE;
            }
            error = HYPRE_SStructGridSetVariables( _grid, part, n_vars, vartypes.data() );

            // Assemble the grid.
            error = HYPRE_SStructGridAssemble( _grid );
            checkHypreError( error );

            // Allocate LHS and RHS vectors and initialize to zero. Note that we
            // are fixing the views under these vectors to layout-right.

            std::array<long, num_space_dim> reorder_size;
            for ( std::size_t d = 0; d < num_space_dim; ++d )
            {
                reorder_size[d] = global_space.extent( d );
            }
            // Is the size of the vector_values array correct?
            IndexSpace<num_space_dim> reorder_space( reorder_size );
            auto vector_values =
                createView<HYPRE_Complex, Kokkos::LayoutRight, memory_space>(
                    "vector_values0", reorder_space );
            Kokkos::deep_copy( vector_values, 0.0 );

            _stencils.resize( n_vars );
            _stencil_size.resize( n_vars );

            error = HYPRE_SStructVectorCreate( _comm, _grid, &_b );
            checkHypreError( error );
            error = HYPRE_SStructVectorSetObjectType(_b, object_type);
            checkHypreError( error );
            error = HYPRE_SStructVectorInitialize( _b );
            checkHypreError( error );
            for ( int i = 0; i < n_vars; ++i )
            {
                error = HYPRE_SStructVectorSetBoxValues(
                    _b, part, _lower.data(), _upper.data(), i, vector_values.data() );
                checkHypreError( error );
            }
            error = HYPRE_SStructVectorAssemble( _b );
            checkHypreError( error );

            error = HYPRE_SStructVectorCreate( _comm, _grid, &_x );
            checkHypreError( error );
            error = HYPRE_SStructVectorSetObjectType(_x, object_type);
            checkHypreError( error );
            error = HYPRE_SStructVectorInitialize( _x );
            checkHypreError( error );
            for ( int i = 0; i < n_vars; ++i )
            {
                error = HYPRE_SStructVectorSetBoxValues(
                    _x, part, _lower.data(), _upper.data(), i, vector_values.data() );
                checkHypreError( error );
            }
            checkHypreError( error );
            error = HYPRE_SStructVectorAssemble( _x );
            checkHypreError( error );
        }
    }

    // *FIX* destroy the vector of HYPRE stencils `_stencils`
    // Destructor.
    virtual ~HypreSemiStructuredSolver()
    {
        // We only make data if this is not a preconditioner.
        if ( !_is_preconditioner )
        {
            HYPRE_SStructVectorDestroy( _x );
            HYPRE_SStructVectorDestroy( _b );
            HYPRE_SStructMatrixDestroy( _A );
            for (std::size_t i = 0; i < _stencils.size(); ++i ){
            HYPRE_SStructStencilDestroy( _stencils[i] );
            }
//            HYPRE_SStructStencilDestroy( _stencil1 );
//            HYPRE_SStructStencilDestroy( _stencil2 );
            HYPRE_SStructGridDestroy( _grid );
            HYPRE_SStructGraphDestroy( _graph );
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
                      const bool is_symmetric = false, int var = 0, int n_vars = 3 )
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call setMatrixStencil() on preconditioners" );

        // Create the stencil.
        _stencil_size[var] = stencil.size();
        auto error =
            HYPRE_SStructStencilCreate( NumSpaceDim, _stencil_size[var], &_stencils[var] );
        checkHypreError( error );

        std::array<HYPRE_Int, NumSpaceDim> offset;
        for ( int dep = 0; dep < n_vars; ++dep )
        {
            for ( unsigned n = 0; n < stencil.size(); ++n )
            {
                for ( std::size_t d = 0; d < NumSpaceDim; ++d )
                    offset[d] = stencil[n][d];
                error = HYPRE_SStructStencilSetEntry( _stencils[var], n, offset.data(), dep );
               checkHypreError( error );
            }
        }
//        _stencil_size[var] = _stencils[var]->size();

    }


    /*
      \brief Set the solver graph
      \param n_vars The number of variables (and equations) in the
      specified problem domain
    */
    void setSolverGraph( int n_vars )
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call setMatrixStencil() on preconditioners" );

        int part = 1;

        // Setup the Graph for the non-zero structure of the matrix
        // Create the graph with hypre
        auto error = HYPRE_SStructGraphCreate( _comm, _grid, &_graph );
        checkHypreError( error );
        
        // Set up the object type
        error = HYPRE_SStructGraphSetObjectType( _graph, object_type );
        checkHypreError( error );

        // Set the stencil to the graph
        for ( int i = 1; i < n_vars; ++i)
        {
            error = HYPRE_SStructGraphSetStencil( _graph, part, i, _stencils[i] );
            checkHypreError( error );
        }

        // Assemble the graph
        error = HYPRE_SStructGraphAssemble( _graph );
        checkHypreError( error );

        // Create the matrix.
        error = HYPRE_SStructMatrixCreate( _comm, _graph, &_A );
        checkHypreError( error );
//        error = HYPRE_SStructMatrixSetSymmetric( _A, part,  is_symmetric );
//        checkHypreError( error );
        
    }

    /*!
      \brief Set the matrix values.
      \param values The matrix entry values. For each entity over which the
      vector space is defined an entry for each stencil element is
      required. The order of the stencil elements is that same as that in the
      stencil definition. Note that values corresponding to stencil entries
      outside of the domain should be set to zero.
      \param var The variable number that the matrix entry values apply to.
      This will correspond to the stencil and graph setup.
    */
    template <class Array_t>
    void setMatrixValues( const Array_t& values, int v_x, int v_h )
    {
        static_assert( is_array<Array_t>::value, "Must use an array" );
        static_assert(
            std::is_same<typename Array_t::entity_type, entity_type>::value,
            "Array entity type mush match solver entity type" );
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

        // Ensure the values array matches up in dimension with the stencil size
        if ( values.layout()->dofsPerEntity() !=
             static_cast<int>( _stencil_size[v_x] ) )
            throw std::runtime_error(
                "Number of matrix values does not match stencil size" );

        // Spatial dimension.
        const std::size_t num_space_dim = Array_t::num_space_dim;

        int part = 0;

        // Intialize the matrix for setting values.
        auto error = HYPRE_SStructMatrixInitialize( _A );
        checkHypreError( error );

        // Copy the matrix entries into HYPRE. The HYPRE layout is fixed as
        // layout-right.
        auto owned_space = values.layout()->indexSpace( Own(), Local() );
        std::array<long, num_space_dim + 1> reorder_size;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            reorder_size[d] = owned_space.extent( d );
        }
        reorder_size.back() = _stencil_size[v_x];
        IndexSpace<num_space_dim + 1> reorder_space( reorder_size );
        auto a_values =
            createView<HYPRE_Complex, Kokkos::LayoutRight, memory_space>(
                "a_values", reorder_space );
        auto values_subv = createSubview( values.view(), owned_space );
        Kokkos::deep_copy( a_values, values_subv );

        // Insert values into the HYPRE matrix.
        std::vector<HYPRE_Int> indices( _stencil_size[v_x] );
        int start = _stencil_size[v_x] * v_x;
        std::iota( indices.begin(), indices.end(), start );
        error = HYPRE_SStructMatrixSetBoxValues(
            _A, part, _lower.data(), _upper.data(), v_h, indices.size(), indices.data(),
            a_values.data() );
        checkHypreError( error );
        error = HYPRE_SStructMatrixAssemble( _A );
        checkHypreError( error );
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

        this->setupImpl( _A, _b, _x );
    }

    /*!
      \brief Solve the problem Ax = b for x.
      \param b The forcing term.
      \param x The solution.
    */
    template <class Array_t>
    void solve( const Array_t& b, Array_t& x, int n_vars = 3 )
    {
        Kokkos::Profiling::pushRegion( "Cajita::HypreSemiStructuredSolver::solve" );

        static_assert( is_array<Array_t>::value, "Must use an array" );
        static_assert(
            std::is_same<typename Array_t::entity_type, entity_type>::value,
            "Array entity type mush match solver entity type" );
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

        // Initialize the RHS.
        auto error = HYPRE_SStructVectorInitialize( _b );
        checkHypreError( error );

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
        for ( int var = 0; var < n_vars; ++var )
        {
            error = HYPRE_SStructVectorSetBoxValues(
                _b, part, _lower.data(), _upper.data(), var, vector_values.data() );
        }

        checkHypreError( error );
        error = HYPRE_SStructVectorAssemble( _b );
        checkHypreError( error );

        // Solve the problem
        this->solveImpl( _A, _b, _x );

        // Extract the solution from the LHS
        for ( int var = 0; var < n_vars; ++var )
        {
            error = HYPRE_SStructVectorSetBoxValues(
                _x, part, _lower.data(), _upper.data(), var, vector_values.data() );
        }

        // Copy the HYPRE solution to the LHS.
        auto x_subv = createSubview( x.view(), owned_space );
        Kokkos::deep_copy( x_subv, vector_values );

        Kokkos::Profiling::popRegion();
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
    virtual void setupImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                            HYPRE_SStructVector x ) = 0;

    //! Solver implementation.
    virtual void solveImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                            HYPRE_SStructVector x ) = 0;

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
            out << "HYPRE structured solver error: ";
            out << error << " " << error_msg;
            HYPRE_ClearError( error );
            throw std::runtime_error( out.str() );
        }
    }

  private:
    MPI_Comm _comm;
    bool _is_preconditioner;
    HYPRE_SStructGrid _grid;
    std::vector<HYPRE_Int> _lower;
    std::vector<HYPRE_Int> _upper;
    std::vector<HYPRE_SStructStencil> _stencils;
//    HYPRE_SStructStencil _stencil0;
//    HYPRE_SStructStencil _stencil1;
//    HYPRE_SStructStencil _stencil2;
    HYPRE_SStructGraph _graph;
    std::vector<unsigned> _stencil_size;
    HYPRE_SStructMatrix _A;
    HYPRE_SStructVector _b;
    HYPRE_SStructVector _x;
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
    //! Base HYPRE structured solver type.
    using Base = HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructPCG( const ArrayLayout_t& layout,
                    const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
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

    void setupImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructPCGSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructPCGSolve( _solver, A, b, x );
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
};

//---------------------------------------------------------------------------//
//! GMRES solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructGMRES
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using Base = HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructGMRES( const ArrayLayout_t& layout,
                      const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
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

    void setupImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructGMRESSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructGMRESSolve( _solver, A, b, x );
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
};

//---------------------------------------------------------------------------//
//! BiCGSTAB solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructBiCGSTAB
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using Base = HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructBiCGSTAB( const ArrayLayout_t& layout,
                         const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
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

    void setupImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructBiCGSTABSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructBiCGSTABSolve( _solver, A, b, x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_SStructBiCGSTABGetNumIterations( _solver, &num_iter );
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
};

//---------------------------------------------------------------------------//
//! PFMG solver.
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructPFMG
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using Base = HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructPFMG( const ArrayLayout_t& layout,
                     const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
    {
        auto error = HYPRE_SStructSysPFMGCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );

        if ( is_preconditioner )
        {
            error = HYPRE_SStructSysPFMGSetZeroGuess( _solver );
            this->checkHypreError( error );
        }
    }

    ~HypreSemiStructPFMG() { HYPRE_SStructSysPFMGDestroy( _solver ); }

    // PFMG SETTINGS

/*
    //! Set the maximum number of multigrid levels.
    void setMaxLevels( const int max_levels )
    {
        auto error = HYPRE_SStructSysPFMGSetMaxLevels( _solver, max_levels );
        this->checkHypreError( error );
    }
*/

    //! Additionally require that the relative difference in successive
    //! iterates be small.
    void setRelChange( const int rel_change )
    {
        auto error = HYPRE_SStructSysPFMGSetRelChange( _solver, rel_change );
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
        auto error = HYPRE_SStructSysPFMGSetRelaxType( _solver, relax_type );
        this->checkHypreError( error );
    }

    //! Set the Jacobi weight
    void setJacobiWeight( const double weight )
    {
        auto error = HYPRE_SStructSysPFMGSetJacobiWeight( _solver, weight );
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

/*
    void setRAPType( const int rap_type )
    {
//        auto error = HYPRE_SStructSysPFMGSetRAPType( _solver, rap_type );
//        this->checkHypreError( error );
    }
*/

    //! Set number of relaxation sweeps before coarse-grid correction.
    void setNumPreRelax( const int num_pre_relax )
    {
        auto error = HYPRE_SStructSysPFMGSetNumPreRelax( _solver, num_pre_relax );
        this->checkHypreError( error );
    }

    //! Set number of relaxation sweeps before coarse-grid correction.
    void setNumPostRelax( const int num_post_relax )
    {
        auto error = HYPRE_SStructSysPFMGSetNumPostRelax( _solver, num_post_relax );
        this->checkHypreError( error );
    }

    //! Skip relaxation on certain grids for isotropic problems.  This can
    //! greatly improve efficiency by eliminating unnecessary relaxations when
    //! the underlying problem is isotropic.
    void setSkipRelax( const int skip_relax )
    {
        auto error = HYPRE_SStructSysPFMGSetSkipRelax( _solver, skip_relax );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_SStructSysPFMGSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_SStructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToSStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_SStructSysPFMGSetup;
    }
    HYPRE_PtrToSStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_SStructSysPFMGSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_SStructSysPFMGSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_SStructSysPFMGSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_SStructSysPFMGSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructSysPFMGSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructSysPFMGSolve( _solver, A, b, x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_SStructSysPFMGGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>& ) override
    {
        throw std::logic_error(
            "HYPRE PFMG solver does not support preconditioning." );
    }

  private:
    HYPRE_SStructSolver _solver;
};

//---------------------------------------------------------------------------//
//! SMG solver.
// Appears Unsupported by HYPRE currently
/*
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructSMG
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using Base = HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructSMG( const ArrayLayout_t& layout,
                    const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
    {
        auto error = HYPRE_SStructSMGCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );

        if ( is_preconditioner )
        {
            error = HYPRE_SStructSMGSetZeroGuess( _solver );
            this->checkHypreError( error );
        }
    }

    ~HypreSemiStructSMG() { HYPRE_SStructSMGDestroy( _solver ); }

    // SMG Settings

    //! Additionally require that the relative difference in successive
    //! iterates be small.
    void setRelChange( const int rel_change )
    {
        auto error = HYPRE_SStructSMGSetRelChange( _solver, rel_change );
        this->checkHypreError( error );
    }

    //! Set number of relaxation sweeps before coarse-grid correction.
    void setNumPreRelax( const int num_pre_relax )
    {
        auto error = HYPRE_SStructSMGSetNumPreRelax( _solver, num_pre_relax );
        this->checkHypreError( error );
    }

    //! Set number of relaxation sweeps before coarse-grid correction.
    void setNumPostRelax( const int num_post_relax )
    {
        auto error = HYPRE_SStructSMGSetNumPostRelax( _solver, num_post_relax );
        this->checkHypreError( error );
    }

    //! Set the amount of logging to do.
    void setLogging( const int logging )
    {
        auto error = HYPRE_SStructSMGSetLogging( _solver, logging );
        this->checkHypreError( error );
    }

    HYPRE_SStructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToSStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_SStructSMGSetup;
    }
    HYPRE_SPtrToStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_SStructSMGSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_SStructSMGSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_SStructSMGSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int print_level ) override
    {
        auto error = HYPRE_SStructSMGSetPrintLevel( _solver, print_level );
        this->checkHypreError( error );
    }

    void setupImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructSMGSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructSMGSolve( _solver, A, b, x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_SStructSMGGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_SStructSMGGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>& ) override
    {
        throw std::logic_error(
            "HYPRE SMG solver does not support preconditioning." );
    }

  private:
    HYPRE_SStructSolver _solver;
};
*/

//---------------------------------------------------------------------------//
//! Jacobi solver.
// Appears unimplemented by HYPRE currently
/*
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructJacobi
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using Base = HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructJacobi( const ArrayLayout_t& layout,
                       const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
    {
        auto error = HYPRE_SStructJacobiCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );

        if ( is_preconditioner )
        {
            error = HYPRE_SStructJacobiSetZeroGuess( _solver );
            this->checkHypreError( error );
        }
    }

    ~HypreSemiStructJacobi() { HYPRE_SStructJacobiDestroy( _solver ); }

    HYPRE_SStructSolver getHypreSolver() const override { return _solver; }
    HYPRE_PtrToSStructSolverFcn getHypreSetupFunction() const override
    {
        return HYPRE_SStructJacobiSetup;
    }
    HYPRE_PtrToSStructSolverFcn getHypreSolveFunction() const override
    {
        return HYPRE_SStructJacobiSolve;
    }

  protected:
    void setToleranceImpl( const double tol ) override
    {
        auto error = HYPRE_SStructJacobiSetTol( _solver, tol );
        this->checkHypreError( error );
    }

    void setMaxIterImpl( const int max_iter ) override
    {
        auto error = HYPRE_SStructJacobiSetMaxIter( _solver, max_iter );
        this->checkHypreError( error );
    }

    void setPrintLevelImpl( const int ) override
    {
        // The Jacobi solver does not support a print level.
    }

    void setupImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructJacobiSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_SStructMatrix A, HYPRE_SStructVector b,
                    HYPRE_SStructVector x ) override
    {
        auto error = HYPRE_SStructJacobiSolve( _solver, A, b, x );
        this->checkHypreError( error );
    }

    int getNumIterImpl() override
    {
        HYPRE_Int num_iter;
        auto error = HYPRE_SStructJacobiGetNumIterations( _solver, &num_iter );
        this->checkHypreError( error );
        return num_iter;
    }

    double getFinalRelativeResidualNormImpl() override
    {
        HYPRE_Real norm;
        auto error =
            HYPRE_SStructJacobiGetFinalRelativeResidualNorm( _solver, &norm );
        this->checkHypreError( error );
        return norm;
    }

    void setPreconditionerImpl(
        const HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>& ) override
    {
        throw std::logic_error(
            "HYPRE Jacobi solver does not support preconditioning." );
    }

  private:
    HYPRE_SStructSolver _solver;
};

*/

//---------------------------------------------------------------------------//
//! Diagonal preconditioner.
template <class Scalar, class EntityType, class MemorySpace>
class HypreSemiStructDiagonal
    : public HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>
{
  public:
    //! Base HYPRE structured solver type.
    using Base = HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>;
    //! Constructor
    template <class ArrayLayout_t>
    HypreSemiStructDiagonal( const ArrayLayout_t& layout,
                         const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
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

    void setupImpl( HYPRE_SStructMatrix, HYPRE_SStructVector,
                    HYPRE_SStructVector ) override
    {
        throw std::logic_error(
            "Diagonal preconditioner cannot be used as a solver" );
    }

    void solveImpl( HYPRE_SStructMatrix, HYPRE_SStructVector,
                    HYPRE_SStructVector ) override
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
        const HypreSemiStructuredSolver<Scalar, EntityType, MemorySpace>& ) override
    {
        throw std::logic_error(
            "Diagonal preconditioner does not support preconditioning." );
    }
};

//---------------------------------------------------------------------------//
// Builders
//---------------------------------------------------------------------------//
//! Create a HYPRE PCG structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreSemiStructPCG<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructPCG( const ArrayLayout_t& layout,
                      const bool is_preconditioner = false, int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructPCG<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE GMRES structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreSemiStructGMRES<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructGMRES( const ArrayLayout_t& layout,
                        const bool is_preconditioner = false, int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructGMRES<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE BiCGSTAB structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreSemiStructBiCGSTAB<Scalar, typename ArrayLayout_t::entity_type,
                                    MemorySpace>>
createHypreSemiStructBiCGSTAB( const ArrayLayout_t& layout,
                           const bool is_preconditioner = false, int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructBiCGSTAB<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE PFMG structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreSemiStructPFMG<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructPFMG( const ArrayLayout_t& layout,
                       const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructPFMG<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}
/*
//! Create a HYPRE SMG structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreSemiStructSMG<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructSMG( const ArrayLayout_t& layout,
                      const bool is_preconditioner = false, int n_vars )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructSMG<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}

//! Create a HYPRE Jacobi structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<
    HypreSemiStructJacobi<Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructJacobi( const ArrayLayout_t& layout,
                         const bool is_preconditioner = false, int n_vars )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructJacobi<
        Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>(
        layout, is_preconditioner );
}
*/
//! Create a HYPRE Diagonal structured solver.
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreSemiStructDiagonal<Scalar, typename ArrayLayout_t::entity_type,
                                    MemorySpace>>
createHypreSemiStructDiagonal( const ArrayLayout_t& layout,
                           const bool is_preconditioner = false, int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreSemiStructDiagonal<
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
*/
template <class Scalar, class MemorySpace, class ArrayLayout_t>
std::shared_ptr<HypreSemiStructuredSolver<
    Scalar, typename ArrayLayout_t::entity_type, MemorySpace>>
createHypreSemiStructuredSolver( const std::string& solver_type,
                             const ArrayLayout_t& layout, 
                             const bool is_preconditioner = false, int n_vars = 3 )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );

    if ( "PCG" == solver_type )
        return createHypreSemiStructPCG<Scalar, MemorySpace>( layout,
                                                          is_preconditioner, n_vars );
    else if ( "GMRES" == solver_type )
        return createHypreSemiStructGMRES<Scalar, MemorySpace>( layout,
                                                            is_preconditioner, n_vars );
    else if ( "BiCGSTAB" == solver_type )
        return createHypreSemiStructBiCGSTAB<Scalar, MemorySpace>(
            layout, is_preconditioner, n_vars );
    else if ( "PFMG" == solver_type )
        return createHypreSemiStructPFMG<Scalar, MemorySpace>( layout,
                                                           is_preconditioner );
/*
    else if ( "SMG" == solver_type )
        return createHypreSemiStructSMG<Scalar, MemorySpace>( layout,
                                                          is_preconditioner );
    else if ( "Jacobi" == solver_type )
        return createHypreSemiStructJacobi<Scalar, MemorySpace>(
            layout, is_preconditioner );
*/
    else if ( "Diagonal" == solver_type )
        return createHypreSemiStructDiagonal<Scalar, MemorySpace>(
            layout, is_preconditioner, n_vars );
    else
        throw std::runtime_error( "Invalid solver type" );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_HypreSemiStRUCTUREDSOLVER_HPP
