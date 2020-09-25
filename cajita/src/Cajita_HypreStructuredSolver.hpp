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

#ifndef CAJITA_HYPRESTRUCTUREDSOLVER_HPP
#define CAJITA_HYPRESTRUCTUREDSOLVER_HPP

#include <Cajita_Array.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_Types.hpp>

#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Hypre structured solver interface for scalar fields.
template <class Scalar, class EntityType, class DeviceType>
class HypreStructuredSolver
{
  public:
    // Types.
    using entity_type = EntityType;
    using device_type = DeviceType;
    using value_type = Scalar;

    /*!
      \brief Constructor.
      \param layout The array layout defining the vector space of the solver.
      \param is_preconditioner Flag indicating if this solver will be used as
      a preconditioner.
    */
    template <class ArrayLayout_t>
    HypreStructuredSolver( const ArrayLayout_t &layout,
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

        // Only create data structures if this is not a preconditioner.
        if ( !_is_preconditioner )
        {
            // Create the grid.
            auto error = HYPRE_StructGridCreate( _comm, 3, &_grid );
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
            _lower = { static_cast<HYPRE_Int>( global_space.min( Dim::K ) ),
                       static_cast<HYPRE_Int>( global_space.min( Dim::J ) ),
                       static_cast<HYPRE_Int>( global_space.min( Dim::I ) ) };
            _upper = {
                static_cast<HYPRE_Int>( global_space.max( Dim::K ) - 1 ),
                static_cast<HYPRE_Int>( global_space.max( Dim::J ) - 1 ),
                static_cast<HYPRE_Int>( global_space.max( Dim::I ) - 1 ) };
            error = HYPRE_StructGridSetExtents( _grid, _lower.data(),
                                                _upper.data() );
            checkHypreError( error );

            // Get periodicity. Note we invert the order of this to KJI as well.
            const auto &global_grid = layout.localGrid()->globalGrid();
            HYPRE_Int periodic[3];
            for ( int d = 0; d < 3; ++d )
                periodic[2 - d] =
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
            IndexSpace<3> reorder_space( { global_space.extent( Dim::I ),
                                           global_space.extent( Dim::J ),
                                           global_space.extent( Dim::K ) } );
            auto vector_values =
                createView<HYPRE_Complex, Kokkos::LayoutRight,
                           Kokkos::HostSpace>( "vector_values", reorder_space );
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
    void setMatrixStencil( const std::vector<std::array<int, 3>> &stencil,
                           const bool is_symmetric = false )
    {
        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call setMatrixStencil() on preconditioners" );

        // Create the stencil.
        _stencil_size = stencil.size();
        auto error = HYPRE_StructStencilCreate( 3, _stencil_size, &_stencil );
        checkHypreError( error );
        for ( unsigned n = 0; n < stencil.size(); ++n )
        {
            HYPRE_Int offset[3] = { stencil[n][Dim::I], stencil[n][Dim::J],
                                    stencil[n][Dim::K] };
            error = HYPRE_StructStencilSetElement( _stencil, n, offset );
            checkHypreError( error );
        }

        // Create the matrix.
        error = HYPRE_StructMatrixCreate( _comm, _grid, _stencil, &_A );
        checkHypreError( error );
        error = HYPRE_StructMatrixSetSymmetric( _A, is_symmetric );
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
    void setMatrixValues( const Array_t &values )
    {
        static_assert( is_array<Array_t>::value, "Must use an array" );
        static_assert(
            std::is_same<typename Array_t::entity_type, entity_type>::value,
            "Array entity type mush match solver entity type" );
        static_assert(
            std::is_same<typename Array_t::device_type, DeviceType>::value,
            "Array device type and solver device type are different." );

        static_assert(
            std::is_same<typename Array_t::value_type, value_type>::value,
            "Array value type and solver value type are different." );

        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error(
                "Cannot call setMatrixValues() on preconditioners" );

        if ( values.layout()->dofsPerEntity() !=
             static_cast<int>( _stencil_size ) )
            throw std::runtime_error(
                "Number of matrix values does not match stencil size" );

        // Intialize the matrix for setting values.
        auto error = HYPRE_StructMatrixInitialize( _A );
        checkHypreError( error );

        // Get a view of the matrix values on the host.
        auto values_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), values.view() );

        // Copy the matrix entries into HYPRE. The HYPRE layout is fixed as
        // layout-right.
        auto owned_space = values.layout()->indexSpace( Own(), Local() );
        IndexSpace<4> reorder_space(
            { owned_space.extent( Dim::I ), owned_space.extent( Dim::J ),
              owned_space.extent( Dim::K ), _stencil_size } );
        auto a_values =
            createView<HYPRE_Complex, Kokkos::LayoutRight, Kokkos::HostSpace>(
                "a_values", reorder_space );
        Kokkos::parallel_for(
            "a_copy",
            createExecutionPolicy( owned_space,
                                   Kokkos::HostSpace::execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k,
                           const int n ) {
                a_values( i - owned_space.min( Dim::I ),
                          j - owned_space.min( Dim::J ),
                          k - owned_space.min( Dim::K ), n ) =
                    values_mirror( i, j, k, n );
            } );

        // Insert values into the HYPRE matrix.
        std::vector<HYPRE_Int> indices( _stencil_size );
        std::iota( indices.begin(), indices.end(), 0 );
        error = HYPRE_StructMatrixSetBoxValues(
            _A, _lower.data(), _upper.data(), indices.size(), indices.data(),
            a_values.data() );
        checkHypreError( error );
        error = HYPRE_StructMatrixAssemble( _A );
        checkHypreError( error );
    }

    // Set convergence tolerance implementation.
    void setTolerance( const double tol ) { this->setToleranceImpl( tol ); }

    // Set maximum iteration implementation.
    void setMaxIter( const int max_iter ) { this->setMaxIterImpl( max_iter ); }

    // Set the output level.
    void setPrintLevel( const int print_level )
    {
        this->setPrintLevelImpl( print_level );
    }

    // Set a preconditioner.
    void
    setPreconditioner( const std::shared_ptr<HypreStructuredSolver<
                           Scalar, EntityType, DeviceType>> &preconditioner )
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

    // Setup the problem.
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
    void solve( const Array_t &b, Array_t &x )
    {
        static_assert( is_array<Array_t>::value, "Must use an array" );
        static_assert(
            std::is_same<typename Array_t::entity_type, entity_type>::value,
            "Array entity type mush match solver entity type" );
        static_assert(
            std::is_same<typename Array_t::device_type, DeviceType>::value,
            "Array device type and solver device type are different." );

        static_assert(
            std::is_same<typename Array_t::value_type, value_type>::value,
            "Array value type and solver value type are different." );

        // This function is only valid for non-preconditioners.
        if ( _is_preconditioner )
            throw std::logic_error( "Cannot call solve() on preconditioners" );

        if ( b.layout()->dofsPerEntity() != 1 ||
             x.layout()->dofsPerEntity() != 1 )
            throw std::runtime_error(
                "Structured solver only for scalar fields" );

        // Initialize the RHS.
        auto error = HYPRE_StructVectorInitialize( _b );
        checkHypreError( error );

        // Get a local view of RHS on the host.
        auto b_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), b.view() );

        // Copy the RHS into HYPRE. The HYPRE layout is fixed as layout-right.
        auto owned_space = b.layout()->indexSpace( Own(), Local() );
        IndexSpace<4> reorder_space( { owned_space.extent( Dim::I ),
                                       owned_space.extent( Dim::J ),
                                       owned_space.extent( Dim::K ), 1 } );
        auto vector_values =
            createView<HYPRE_Complex, Kokkos::LayoutRight, Kokkos::HostSpace>(
                "vector_values", reorder_space );
        Kokkos::parallel_for(
            "b_copy",
            createExecutionPolicy( owned_space,
                                   Kokkos::HostSpace::execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k,
                           const int n ) {
                vector_values( i - owned_space.min( Dim::I ),
                               j - owned_space.min( Dim::J ),
                               k - owned_space.min( Dim::K ), n ) =
                    b_mirror( i, j, k, n );
            } );

        // Insert b values into the HYPRE vector.
        error = HYPRE_StructVectorSetBoxValues(
            _b, _lower.data(), _upper.data(), vector_values.data() );
        checkHypreError( error );
        error = HYPRE_StructVectorAssemble( _b );
        checkHypreError( error );

        // Solve the problem
        this->solveImpl( _A, _b, _x );

        // Extract the solution from the LHS
        error = HYPRE_StructVectorGetBoxValues(
            _x, _lower.data(), _upper.data(), vector_values.data() );
        checkHypreError( error );

        // Get a local view of x on the host.
        auto x_mirror =
            Kokkos::create_mirror_view( Kokkos::HostSpace(), x.view() );

        // Copy the HYPRE solution to the LHS.
        Kokkos::parallel_for(
            "x_copy",
            createExecutionPolicy( owned_space,
                                   Kokkos::HostSpace::execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k,
                           const int n ) {
                x_mirror( i, j, k, n ) =
                    vector_values( i - owned_space.min( Dim::I ),
                                   j - owned_space.min( Dim::J ),
                                   k - owned_space.min( Dim::K ), n );
            } );

        // Copy back to the device.
        Kokkos::deep_copy( x.view(), x_mirror );
    }

    // Get the number of iterations taken on the last solve.
    int getNumIter() { return this->getNumIterImpl(); }

    // Get the relative residual norm achieved on the last solve.
    double getFinalRelativeResidualNorm()
    {
        return this->getFinalRelativeResidualNormImpl();
    }

    //! Preconditioner implementation details.
    virtual HYPRE_StructSolver getHypreSolver() const = 0;
    virtual HYPRE_PtrToStructSolverFcn getHypreSetupFunction() const = 0;
    virtual HYPRE_PtrToStructSolverFcn getHypreSolveFunction() const = 0;

  protected:
    // Set convergence tolerance implementation.
    virtual void setToleranceImpl( const double tol ) = 0;

    // Set maximum iteration implementation.
    virtual void setMaxIterImpl( const int max_iter ) = 0;

    // Set the output level.
    virtual void setPrintLevelImpl( const int print_level ) = 0;

    // Setup implementation.
    virtual void setupImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                            HYPRE_StructVector x ) = 0;

    // Solver implementation.
    virtual void solveImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                            HYPRE_StructVector x ) = 0;

    // Get the number of iterations taken on the last solve.
    virtual int getNumIterImpl() = 0;

    // Get the relative residual norm achieved on the last solve.
    virtual double getFinalRelativeResidualNormImpl() = 0;

    // Set a preconditioner.
    virtual void setPreconditionerImpl(
        const HypreStructuredSolver<Scalar, EntityType, DeviceType>
            &preconditioner ) = 0;

    // Check a hypre error.
    void checkHypreError( const int error ) const
    {
        if ( error > 0 )
        {
            std::stringstream out;
            out << "HYPRE structured solver error: ";
            out << error;
            throw std::runtime_error( out.str() );
        }
    }

  private:
    MPI_Comm _comm;
    bool _is_preconditioner;
    HYPRE_StructGrid _grid;
    std::array<HYPRE_Int, 3> _lower;
    std::array<HYPRE_Int, 3> _upper;
    HYPRE_StructStencil _stencil;
    unsigned _stencil_size;
    HYPRE_StructMatrix _A;
    HYPRE_StructVector _b;
    HYPRE_StructVector _x;
    std::shared_ptr<HypreStructuredSolver<Scalar, EntityType, DeviceType>>
        _preconditioner;
};

//---------------------------------------------------------------------------//
// PCG solver.
template <class Scalar, class EntityType, class DeviceType>
class HypreStructPCG
    : public HypreStructuredSolver<Scalar, EntityType, DeviceType>
{
  public:
    using Base = HypreStructuredSolver<Scalar, EntityType, DeviceType>;

    template <class ArrayLayout_t>
    HypreStructPCG( const ArrayLayout_t &layout,
                    const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
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

    //! PCG SETTINGS

    // Set the absolute tolerance
    void setAbsoluteTol( const double tol )
    {
        auto error = HYPRE_StructPCGSetAbsoluteTol( _solver, tol );
        this->checkHypreError( error );
    }

    // Additionally require that the relative difference in successive
    // iterates be small.
    void setRelChange( const int rel_change )
    {
        auto error = HYPRE_StructPCGSetRelChange( _solver, rel_change );
        this->checkHypreError( error );
    }

    // Set the amount of logging to do.
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

    void setupImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructPCGSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructPCGSolve( _solver, A, b, x );
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
        const HypreStructuredSolver<Scalar, EntityType, DeviceType>
            &preconditioner ) override
    {
        auto error = HYPRE_StructPCGSetPrecond(
            _solver, preconditioner.getHypreSolveFunction(),
            preconditioner.getHypreSetupFunction(),
            preconditioner.getHypreSolver() );
        this->checkHypreError( error );
    }

  private:
    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// GMRES solver.
template <class Scalar, class EntityType, class DeviceType>
class HypreStructGMRES
    : public HypreStructuredSolver<Scalar, EntityType, DeviceType>
{
  public:
    using Base = HypreStructuredSolver<Scalar, EntityType, DeviceType>;

    template <class ArrayLayout_t>
    HypreStructGMRES( const ArrayLayout_t &layout,
                      const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
    {
        if ( is_preconditioner )
            throw std::logic_error(
                "HYPRE GMRES cannot be used as a preconditioner" );

        auto error = HYPRE_StructGMRESCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreStructGMRES() { HYPRE_StructGMRESDestroy( _solver ); }

    //! GMRES SETTINGS

    // Set the absolute tolerance
    void setAbsoluteTol( const double tol )
    {
        auto error = HYPRE_StructGMRESSetAbsoluteTol( _solver, tol );
        this->checkHypreError( error );
    }

    // Set the max size of the Krylov space.
    void setKDim( const int k_dim )
    {
        auto error = HYPRE_StructGMRESSetKDim( _solver, k_dim );
        this->checkHypreError( error );
    }

    // Set the amount of logging to do.
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

    void setupImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructGMRESSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructGMRESSolve( _solver, A, b, x );
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
        const HypreStructuredSolver<Scalar, EntityType, DeviceType>
            &preconditioner ) override
    {
        auto error = HYPRE_StructGMRESSetPrecond(
            _solver, preconditioner.getHypreSolveFunction(),
            preconditioner.getHypreSetupFunction(),
            preconditioner.getHypreSolver() );
        this->checkHypreError( error );
    }

  private:
    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// BiCGSTAB solver.
template <class Scalar, class EntityType, class DeviceType>
class HypreStructBiCGSTAB
    : public HypreStructuredSolver<Scalar, EntityType, DeviceType>
{
  public:
    using Base = HypreStructuredSolver<Scalar, EntityType, DeviceType>;

    template <class ArrayLayout_t>
    HypreStructBiCGSTAB( const ArrayLayout_t &layout,
                         const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
    {
        if ( is_preconditioner )
            throw std::logic_error(
                "HYPRE BiCGSTAB cannot be used as a preconditioner" );

        auto error = HYPRE_StructBiCGSTABCreate(
            layout.localGrid()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreStructBiCGSTAB() { HYPRE_StructBiCGSTABDestroy( _solver ); }

    //! BiCGSTAB SETTINGS

    // Set the absolute tolerance
    void setAbsoluteTol( const double tol )
    {
        auto error = HYPRE_StructBiCGSTABSetAbsoluteTol( _solver, tol );
        this->checkHypreError( error );
    }

    // Set the amount of logging to do.
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

    void setupImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructBiCGSTABSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructBiCGSTABSolve( _solver, A, b, x );
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
        const HypreStructuredSolver<Scalar, EntityType, DeviceType>
            &preconditioner ) override
    {
        auto error = HYPRE_StructBiCGSTABSetPrecond(
            _solver, preconditioner.getHypreSolveFunction(),
            preconditioner.getHypreSetupFunction(),
            preconditioner.getHypreSolver() );
        this->checkHypreError( error );
    }

  private:
    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// PFMG solver.
template <class Scalar, class EntityType, class DeviceType>
class HypreStructPFMG
    : public HypreStructuredSolver<Scalar, EntityType, DeviceType>
{
  public:
    using Base = HypreStructuredSolver<Scalar, EntityType, DeviceType>;

    template <class ArrayLayout_t>
    HypreStructPFMG( const ArrayLayout_t &layout,
                     const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
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

    //! PFMG SETTINGS

    // Set the maximum number of multigrid levels.
    void setMaxLevels( const int max_levels )
    {
        auto error = HYPRE_StructPFMGSetMaxLevels( _solver, max_levels );
        this->checkHypreError( error );
    }

    // Additionally require that the relative difference in successive
    // iterates be small.
    void setRelChange( const int rel_change )
    {
        auto error = HYPRE_StructPFMGSetRelChange( _solver, rel_change );
        this->checkHypreError( error );
    }

    // Set relaxation type.
    //
    // 0 - Jacobi
    // 1 - Weighted Jacobi (default)
    // 2 - Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR
    // post-relaxation)
    // 3 - Red/Black Gauss-Seidel (nonsymmetric: RB pre- and post-relaxation)
    void setRelaxType( const int relax_type )
    {
        auto error = HYPRE_StructPFMGSetRelaxType( _solver, relax_type );
        this->checkHypreError( error );
    }

    // Set the Jacobi weight
    void setJacobiWeight( const double weight )
    {
        auto error = HYPRE_StructPFMGSetJacobiWeight( _solver, weight );
        this->checkHypreError( error );
    }

    // Set type of coarse-grid operator to use.
    //
    // 0 - Galerkin (default)
    // 1 - non-Galerkin 5-pt or 7-pt stencils
    //
    // Both operators are constructed algebraically.  The non-Galerkin option
    // maintains a 5-pt stencil in 2D and a 7-pt stencil in 3D on all grid
    // levels. The stencil coefficients are computed by averaging techniques.
    void setRAPType( const int rap_type )
    {
        auto error = HYPRE_StructPFMGSetRAPType( _solver, rap_type );
        this->checkHypreError( error );
    }

    // Set number of relaxation sweeps before coarse-grid correction.
    void setNumPreRelax( const int num_pre_relax )
    {
        auto error = HYPRE_StructPFMGSetNumPreRelax( _solver, num_pre_relax );
        this->checkHypreError( error );
    }

    // Set number of relaxation sweeps before coarse-grid correction.
    void setNumPostRelax( const int num_post_relax )
    {
        auto error = HYPRE_StructPFMGSetNumPostRelax( _solver, num_post_relax );
        this->checkHypreError( error );
    }

    // Skip relaxation on certain grids for isotropic problems.  This can
    // greatly improve efficiency by eliminating unnecessary relaxations when
    // the underlying problem is isotropic.
    void setSkipRelax( const int skip_relax )
    {
        auto error = HYPRE_StructPFMGSetSkipRelax( _solver, skip_relax );
        this->checkHypreError( error );
    }

    // Set the amount of logging to do.
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

    void setupImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructPFMGSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructPFMGSolve( _solver, A, b, x );
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
        const HypreStructuredSolver<Scalar, EntityType, DeviceType> & ) override
    {
        throw std::logic_error(
            "HYPRE PFMG solver does not support preconditioning." );
    }

  private:
    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// SMG solver.
template <class Scalar, class EntityType, class DeviceType>
class HypreStructSMG
    : public HypreStructuredSolver<Scalar, EntityType, DeviceType>
{
  public:
    using Base = HypreStructuredSolver<Scalar, EntityType, DeviceType>;

    template <class ArrayLayout_t>
    HypreStructSMG( const ArrayLayout_t &layout,
                    const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
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

    //! SMG Settings

    // Additionally require that the relative difference in successive
    // iterates be small.
    void setRelChange( const int rel_change )
    {
        auto error = HYPRE_StructSMGSetRelChange( _solver, rel_change );
        this->checkHypreError( error );
    }

    // Set number of relaxation sweeps before coarse-grid correction.
    void setNumPreRelax( const int num_pre_relax )
    {
        auto error = HYPRE_StructSMGSetNumPreRelax( _solver, num_pre_relax );
        this->checkHypreError( error );
    }

    // Set number of relaxation sweeps before coarse-grid correction.
    void setNumPostRelax( const int num_post_relax )
    {
        auto error = HYPRE_StructSMGSetNumPostRelax( _solver, num_post_relax );
        this->checkHypreError( error );
    }

    // Set the amount of logging to do.
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

    void setupImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructSMGSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructSMGSolve( _solver, A, b, x );
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
        const HypreStructuredSolver<Scalar, EntityType, DeviceType> & ) override
    {
        throw std::logic_error(
            "HYPRE SMG solver does not support preconditioning." );
    }

  private:
    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// Jacobi solver.
template <class Scalar, class EntityType, class DeviceType>
class HypreStructJacobi
    : public HypreStructuredSolver<Scalar, EntityType, DeviceType>
{
  public:
    using Base = HypreStructuredSolver<Scalar, EntityType, DeviceType>;

    template <class ArrayLayout_t>
    HypreStructJacobi( const ArrayLayout_t &layout,
                       const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
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

    void setupImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructJacobiSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A, HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructJacobiSolve( _solver, A, b, x );
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
        const HypreStructuredSolver<Scalar, EntityType, DeviceType> & ) override
    {
        throw std::logic_error(
            "HYPRE Jacobi solver does not support preconditioning." );
    }

  private:
    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// Diagonal preconditioner.
template <class Scalar, class EntityType, class DeviceType>
class HypreStructDiagonal
    : public HypreStructuredSolver<Scalar, EntityType, DeviceType>
{
  public:
    using Base = HypreStructuredSolver<Scalar, EntityType, DeviceType>;

    template <class ArrayLayout_t>
    HypreStructDiagonal( const ArrayLayout_t &layout,
                         const bool is_preconditioner = false )
        : Base( layout, is_preconditioner )
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

    void setupImpl( HYPRE_StructMatrix, HYPRE_StructVector,
                    HYPRE_StructVector ) override
    {
        throw std::logic_error(
            "Diagonal preconditioner cannot be used as a solver" );
    }

    void solveImpl( HYPRE_StructMatrix, HYPRE_StructVector,
                    HYPRE_StructVector ) override
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
        const HypreStructuredSolver<Scalar, EntityType, DeviceType> & ) override
    {
        throw std::logic_error(
            "Diagonal preconditioner does not support preconditioning." );
    }
};

//---------------------------------------------------------------------------//
// Builders
//---------------------------------------------------------------------------//
template <class Scalar, class DeviceType, class ArrayLayout_t>
std::shared_ptr<
    HypreStructPCG<Scalar, typename ArrayLayout_t::entity_type, DeviceType>>
createHypreStructPCG( const ArrayLayout_t &layout,
                      const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructPCG<
        Scalar, typename ArrayLayout_t::entity_type, DeviceType>>(
        layout, is_preconditioner );
}

template <class Scalar, class DeviceType, class ArrayLayout_t>
std::shared_ptr<
    HypreStructGMRES<Scalar, typename ArrayLayout_t::entity_type, DeviceType>>
createHypreStructGMRES( const ArrayLayout_t &layout,
                        const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructGMRES<
        Scalar, typename ArrayLayout_t::entity_type, DeviceType>>(
        layout, is_preconditioner );
}

template <class Scalar, class DeviceType, class ArrayLayout_t>
std::shared_ptr<HypreStructBiCGSTAB<Scalar, typename ArrayLayout_t::entity_type,
                                    DeviceType>>
createHypreStructBiCGSTAB( const ArrayLayout_t &layout,
                           const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructBiCGSTAB<
        Scalar, typename ArrayLayout_t::entity_type, DeviceType>>(
        layout, is_preconditioner );
}

template <class Scalar, class DeviceType, class ArrayLayout_t>
std::shared_ptr<
    HypreStructPFMG<Scalar, typename ArrayLayout_t::entity_type, DeviceType>>
createHypreStructPFMG( const ArrayLayout_t &layout,
                       const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructPFMG<
        Scalar, typename ArrayLayout_t::entity_type, DeviceType>>(
        layout, is_preconditioner );
}

template <class Scalar, class DeviceType, class ArrayLayout_t>
std::shared_ptr<
    HypreStructSMG<Scalar, typename ArrayLayout_t::entity_type, DeviceType>>
createHypreStructSMG( const ArrayLayout_t &layout,
                      const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructSMG<
        Scalar, typename ArrayLayout_t::entity_type, DeviceType>>(
        layout, is_preconditioner );
}

template <class Scalar, class DeviceType, class ArrayLayout_t>
std::shared_ptr<
    HypreStructJacobi<Scalar, typename ArrayLayout_t::entity_type, DeviceType>>
createHypreStructJacobi( const ArrayLayout_t &layout,
                         const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructJacobi<
        Scalar, typename ArrayLayout_t::entity_type, DeviceType>>(
        layout, is_preconditioner );
}

template <class Scalar, class DeviceType, class ArrayLayout_t>
std::shared_ptr<HypreStructDiagonal<Scalar, typename ArrayLayout_t::entity_type,
                                    DeviceType>>
createHypreStructDiagonal( const ArrayLayout_t &layout,
                           const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );
    return std::make_shared<HypreStructDiagonal<
        Scalar, typename ArrayLayout_t::entity_type, DeviceType>>(
        layout, is_preconditioner );
}

//---------------------------------------------------------------------------//
// Factory
//---------------------------------------------------------------------------//
template <class Scalar, class DeviceType, class ArrayLayout_t>
std::shared_ptr<HypreStructuredSolver<
    Scalar, typename ArrayLayout_t::entity_type, DeviceType>>
createHypreStructuredSolver( const std::string &solver_type,
                             const ArrayLayout_t &layout,
                             const bool is_preconditioner = false )
{
    static_assert( is_array_layout<ArrayLayout_t>::value,
                   "Must use an array layout" );

    if ( "PCG" == solver_type )
        return createHypreStructPCG<Scalar, DeviceType>( layout,
                                                         is_preconditioner );
    else if ( "GMRES" == solver_type )
        return createHypreStructGMRES<Scalar, DeviceType>( layout,
                                                           is_preconditioner );
    else if ( "BiCGSTAB" == solver_type )
        return createHypreStructBiCGSTAB<Scalar, DeviceType>(
            layout, is_preconditioner );
    else if ( "PFMG" == solver_type )
        return createHypreStructPFMG<Scalar, DeviceType>( layout,
                                                          is_preconditioner );
    else if ( "SMG" == solver_type )
        return createHypreStructSMG<Scalar, DeviceType>( layout,
                                                         is_preconditioner );
    else if ( "Jacobi" == solver_type )
        return createHypreStructJacobi<Scalar, DeviceType>( layout,
                                                            is_preconditioner );
    else if ( "Diagonal" == solver_type )
        return createHypreStructDiagonal<Scalar, DeviceType>(
            layout, is_preconditioner );
    else
        throw std::runtime_error( "Invalid solver type" );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_HYPRESTRUCTUREDSOLVER_HPP
