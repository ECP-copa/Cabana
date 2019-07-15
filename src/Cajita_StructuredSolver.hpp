#ifndef CAJITA_STRUCTUREDSOLVER_HPP
#define CAJITA_STRUCTUREDSOLVER_HPP

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_Block.hpp>
#include <Cajita_Array.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <HYPRE_struct_mv.h>
#include <HYPRE_struct_ls.h>

#include <Kokkos_Core.hpp>

#include <memory>
#include <vector>
#include <array>
#include <numeric>
#include <string>
#include <sstream>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Hypre structured solver interface for scalar fields.
template<class EntityType, class DeviceType>
class StructuredSolver
{
  public:

    // Types.
    using entity_type = EntityType;
    using device_type = DeviceType;

    /*!
      \brief Constructor.
      \param layout The array layout defining the vector space of the solver.
    */
    StructuredSolver( const ArrayLayout<EntityType>& layout )
        : _comm( layout.block()->globalGrid().comm() )
    {
        // Create the grid.
        auto error = HYPRE_StructGridCreate( _comm, 3, &_grid );
        checkHypreError( error );

        // Get the global index space spanned by the block on this rank. Note
        // that the upper bound is not a bound but rather the last index as
        // this is what Hypre wants.
        auto global_space = layout.indexSpace( Own(), Global() );
        _lower = { static_cast<HYPRE_Int>(global_space.min(Dim::I)),
                   static_cast<HYPRE_Int>(global_space.min(Dim::J)),
                   static_cast<HYPRE_Int>(global_space.min(Dim::K)) };
        _upper = { static_cast<HYPRE_Int>(global_space.max(Dim::I)) - 1,
                   static_cast<HYPRE_Int>(global_space.max(Dim::J)) - 1,
                   static_cast<HYPRE_Int>(global_space.max(Dim::K)) - 1 };
        error =
            HYPRE_StructGridSetExtents( _grid, _lower.data(), _upper.data() );
        checkHypreError( error );

        // Get periodicity
        const auto& domain = layout.block()->globalGrid().domain();
        HYPRE_Int periodic[3];
        for ( int d = 0; d < 3; ++d )
            periodic[d] =
                domain.isPeriodic(d)
                ? layout.block()->globalGrid().globalNumEntity(EntityType(),d)
                : 0;
        error = HYPRE_StructGridSetPeriodic( _grid, periodic );
        checkHypreError( error );

        // Assemble the grid.
        error = HYPRE_StructGridAssemble( _grid );
        checkHypreError( error );

        // Allocate LHS and RHS vectors and initialize to zero.
        IndexSpace<3> reorder_space( { global_space.extent(Dim::K),
                                       global_space.extent(Dim::J),
                                       global_space.extent(Dim::I) } );
        auto vector_values =
            createView<HYPRE_Complex,Kokkos::LayoutRight,Kokkos::HostSpace>(
                "vector_values", reorder_space );
        Kokkos::deep_copy( vector_values, 0.0 );

        error = HYPRE_StructVectorCreate( _comm, _grid, &_b );
        checkHypreError( error );
        error = HYPRE_StructVectorInitialize(_b);
        checkHypreError( error );
        error = HYPRE_StructVectorSetBoxValues( _b,
                                                _lower.data(),
                                                _upper.data(),
                                                vector_values.data() );
        checkHypreError( error );
        error = HYPRE_StructVectorAssemble( _b );
        checkHypreError( error );

        error = HYPRE_StructVectorCreate( _comm, _grid, &_x );
        checkHypreError( error );
        error = HYPRE_StructVectorInitialize(_x);
        checkHypreError( error );
        error = HYPRE_StructVectorSetBoxValues( _x,
                                                _lower.data(),
                                                _upper.data(),
                                                vector_values.data() );
        checkHypreError( error );
        error = HYPRE_StructVectorAssemble( _x );
        checkHypreError( error );
    }

    // Destructor.
    virtual ~StructuredSolver()
    {
        HYPRE_StructVectorDestroy( _x );
        HYPRE_StructVectorDestroy( _b );
        HYPRE_StructMatrixDestroy( _A );
        HYPRE_StructStencilDestroy( _stencil );
        HYPRE_StructGridDestroy( _grid );
    }

    /*!
      \brief Set the operator stencil.
      \param stencil The (i,j,k) offsets describing the structured matrix
      entries at each grid point. Offsets are defined relative to an index.
      \param is_symmetric If true the matrix is designated as symmetric. The
      stencil entries should only contain one entry from each symmetric
      component if this is true.
    */
    void setMatrixStencil( const std::vector<std::array<int,3> >& stencil,
                           const bool is_symmetric = false )
    {
        // Create the stencil.
        _stencil_size = stencil.size();
        auto error = HYPRE_StructStencilCreate( 3, _stencil_size, &_stencil );
        checkHypreError( error );
        for ( unsigned n = 0; n < stencil.size(); ++n )
        {
            HYPRE_Int offset[3] = { stencil[n][Dim::I],
                                    stencil[n][Dim::J],
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
    template<class Scalar>
    void setMatrixValues( const Array<Scalar,EntityType,DeviceType>& values )
    {
        if ( values.layout()->dofsPerEntity() !=
             static_cast<int>(_stencil_size) )
            throw std::runtime_error(
                "Number of matrix values does not match stencil size" );

        // Intialize the matrix for setting values.
        auto error = HYPRE_StructMatrixInitialize( _A );
        checkHypreError( error );

        // Get a view of the matrix values on the host.
        auto owned_space = values.layout()->indexSpace( Own(), Local() );
        auto owned_values = createSubview( values.view(), owned_space );
        auto host_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), owned_values );

        // Reorder the matrix values the way HYPRE wants them ordered.
        IndexSpace<4> reorder_space( { owned_space.extent(Dim::K),
                                       owned_space.extent(Dim::J),
                                       owned_space.extent(Dim::I),
                                       _stencil_size } );
        auto a_values =
            createView<HYPRE_Complex,Kokkos::LayoutRight,Kokkos::HostSpace>(
                "a_values", reorder_space );

        using host_exec_space = decltype(a_values)::execution_space;
        Kokkos::parallel_for(
            "StructuredSolver::setMatrixValues",
            createExecutionPolicy( reorder_space, host_exec_space() ),
            KOKKOS_LAMBDA( const int k, const int j, const int i, const int l )
            {
                a_values(k,j,i,l) = host_values(i,j,k,l);
            });

        // Insert values into the HYPRE matrix.
        std::vector<HYPRE_Int> indices( _stencil_size );
        std::iota( indices.begin(), indices.end(), 0 );
        error = HYPRE_StructMatrixSetBoxValues( _A,
                                                _lower.data(),
                                                _upper.data(),
                                                indices.size(),
                                                indices.data(),
                                                a_values.data() );
        checkHypreError( error );
        error = HYPRE_StructMatrixAssemble( _A );
        checkHypreError( error );
    }

    // Set convergence tolerance implementation.
    void setTolerance( const double tol )
    {
        this->setToleranceImpl( tol );
    }

    // Set maximum iteration implementation.
    void setMaxIter( const int max_iter )
    {
        this->setMaxIterImpl( max_iter );
    }

    // Set the output level.
    void setPrintLevel( const int print_level )
    {
        this->setPrintLevelImpl( print_level );
    }

    // Setup the problem.
    void setup()
    {
        this->setupImpl( _A, _b, _x );
    }

    /*!
      \brief Solve the problem Ax = b for x.
      \param b The forcing term.
      \param x The solution.
    */
    template<class Scalar>
    void solve( const Array<Scalar,EntityType,DeviceType>& b,
                Array<Scalar,EntityType,DeviceType>& x )
    {
        if ( b.layout()->dofsPerEntity() != 1 ||
             x.layout()->dofsPerEntity() != 1 )
            throw std::runtime_error( "Structured solver only for scalar fields" );

        // Initialize the RHS.
        auto error = HYPRE_StructVectorInitialize( _b );
        checkHypreError( error );

        // Get a local view of b on the host.
        auto owned_space = b.layout()->indexSpace( Own(), Local() );
        auto owned_b = createSubview( b.view(), owned_space );
        auto b_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), owned_b );

        // Reorder b the way HYPRE wants it to be ordered
        IndexSpace<3> reorder_space( { owned_space.extent(Dim::K),
                                       owned_space.extent(Dim::J),
                                       owned_space.extent(Dim::I) } );
        auto vector_values =
            createView<HYPRE_Complex,Kokkos::LayoutRight,Kokkos::HostSpace>(
                "vector_values", reorder_space );

        using host_exec_space = decltype(vector_values)::execution_space;
        Kokkos::parallel_for(
            "StructuredSolver::setVectorValues",
            createExecutionPolicy( reorder_space, host_exec_space() ),
            KOKKOS_LAMBDA( const int k, const int j, const int i )
            {
                vector_values(k,j,i) = b_values(i,j,k,0);
            });

        // Insert b values into the HYPRE vector.
        error = HYPRE_StructVectorSetBoxValues( _b,
                                                _lower.data(),
                                                _upper.data(),
                                                vector_values.data() );
        checkHypreError( error );
        error = HYPRE_StructVectorAssemble( _b );
        checkHypreError( error );

        // Solve the problem
        this->solveImpl( _A, _b, _x );

        // Extract the solution from the LHS
        error = HYPRE_StructVectorGetBoxValues( _x,
                                                _lower.data(),
                                                _upper.data(),
                                                vector_values.data() );
        checkHypreError( error );

        // Get a local view of x on the host.
        auto owned_x = createSubview( x.view(), owned_space );
        auto x_values = Kokkos::create_mirror_view(
            Kokkos::HostSpace(), owned_x );

        // Reorder x from the HYPRE order.
        Kokkos::parallel_for(
            "StructuredSolver::setVectorValues",
            createExecutionPolicy( reorder_space, host_exec_space() ),
            KOKKOS_LAMBDA( const int k, const int j, const int i )
            {
                x_values(i,j,k,0) = vector_values(k,j,i);
            });

        // Copy the solution to the LHS.
        Kokkos::deep_copy( owned_x, x_values );
    }

    // Get the number of iterations taken on the last solve.
    int getNumIter()
    {
        return this->getNumIterImpl();
    }

    // Get the relative residual norm achieved on the last solve.
    double getFinalRelativeResidualNorm()
    {
        return this->getFinalRelativeResidualNormImpl();
    }

  protected:

    // Set convergence tolerance implementation.
    virtual void setToleranceImpl( const double tol ) = 0;

    // Set maximum iteration implementation.
    virtual void setMaxIterImpl( const int max_iter ) = 0;

    // Set the output level.
    virtual void setPrintLevelImpl( const int print_level ) = 0;

    // Setup implementation.
    virtual void setupImpl( HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x ) = 0;

    // Solver implementation.
    virtual void solveImpl( HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x ) = 0;

    // Get the number of iterations taken on the last solve.
    virtual int getNumIterImpl() = 0;

    // Get the relative residual norm achieved on the last solve.
    virtual double getFinalRelativeResidualNormImpl() = 0;

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
    HYPRE_StructGrid _grid;
    std::array<HYPRE_Int,3> _lower;
    std::array<HYPRE_Int,3> _upper;
    HYPRE_StructStencil _stencil;
    unsigned _stencil_size;
    HYPRE_StructMatrix _A;
    HYPRE_StructVector _b;
    HYPRE_StructVector _x;
};

//---------------------------------------------------------------------------//
// PCG solver.
template<class EntityType, class DeviceType>
class HypreStructPCG : public StructuredSolver<EntityType,DeviceType>
{
  public:

    using Base = StructuredSolver<EntityType,DeviceType>;

    HypreStructPCG( const ArrayLayout<EntityType>& layout )
        : Base( layout )
    {
        auto error = HYPRE_StructPCGCreate(
            layout.block()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreStructPCG()
    {
        HYPRE_StructPCGDestroy( _solver );
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

    void setupImpl( HYPRE_StructMatrix A,
                    HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructPCGSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A,
                    HYPRE_StructVector b,
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

  private:

    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// GMRES solver.
template<class EntityType, class DeviceType>
class HypreStructGMRES : public StructuredSolver<EntityType,DeviceType>
{
  public:

    using Base = StructuredSolver<EntityType,DeviceType>;

    HypreStructGMRES( const ArrayLayout<EntityType>& layout )
        : Base( layout )
    {
        auto error = HYPRE_StructGMRESCreate(
            layout.block()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreStructGMRES()
    {
        HYPRE_StructGMRESDestroy( _solver );
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

    void setupImpl( HYPRE_StructMatrix A,
                    HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructGMRESSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A,
                    HYPRE_StructVector b,
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

  private:

    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// PFMG solver.
template<class EntityType, class DeviceType>
class HypreStructPFMG : public StructuredSolver<EntityType,DeviceType>
{
  public:

    using Base = StructuredSolver<EntityType,DeviceType>;

    HypreStructPFMG( const ArrayLayout<EntityType>& layout )
        : Base( layout )
    {
        auto error = HYPRE_StructPFMGCreate(
            layout.block()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreStructPFMG()
    {
        HYPRE_StructPFMGDestroy( _solver );
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

    void setupImpl( HYPRE_StructMatrix A,
                    HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructPFMGSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A,
                    HYPRE_StructVector b,
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

  private:

    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// SMG solver.
template<class EntityType, class DeviceType>
class HypreStructSMG : public StructuredSolver<EntityType,DeviceType>
{
  public:

    using Base = StructuredSolver<EntityType,DeviceType>;

    HypreStructSMG( const ArrayLayout<EntityType>& layout )
        : Base( layout )
    {
        auto error = HYPRE_StructSMGCreate(
            layout.block()->globalGrid().comm(), &_solver );
        this->checkHypreError( error );
    }

    ~HypreStructSMG()
    {
        HYPRE_StructSMGDestroy( _solver );
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

    void setupImpl( HYPRE_StructMatrix A,
                    HYPRE_StructVector b,
                    HYPRE_StructVector x ) override
    {
        auto error = HYPRE_StructSMGSetup( _solver, A, b, x );
        this->checkHypreError( error );
    }

    void solveImpl( HYPRE_StructMatrix A,
                    HYPRE_StructVector b,
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

  private:

    HYPRE_StructSolver _solver;
};

//---------------------------------------------------------------------------//
// Factory
//---------------------------------------------------------------------------//
template<class EntityType, class DeviceType>
std::shared_ptr<StructuredSolver<EntityType,DeviceType> >
createStructuredSolver( const std::string& solver_type,
                        const ArrayLayout<EntityType>& layout,
                        DeviceType )
{
    if ( "PCG" == solver_type )
        return std::make_shared<HypreStructPCG<EntityType,DeviceType>>( layout );
    else if ( "GMRES" == solver_type )
        return std::make_shared<HypreStructGMRES<EntityType,DeviceType>>( layout );
    else if ( "PFMG" == solver_type )
        return std::make_shared<HypreStructPFMG<EntityType,DeviceType>>( layout );
    else if ( "SMG" == solver_type )
        return std::make_shared<HypreStructSMG<EntityType,DeviceType>>( layout );
    else
        throw std::runtime_error( "Invalid solver type");
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_STRUCTUREDSOLVER_HPP
