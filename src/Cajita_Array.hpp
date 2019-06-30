#ifndef CAJITA_ARRAY_HPP
#define CAJITA_ARRAY_HPP

#include <Cajita_Block.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_MpiTraits.hpp>

#include <Kokkos_Core.hpp>

#include <memory>
#include <vector>
#include <cmath>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Array layout.
//---------------------------------------------------------------------------//
template<class EntityType>
class ArrayLayout
{
  public:

    // Entity type.
    using entity_type = EntityType;

    /*!
      \brief Constructor.
      \param block The grid block over which the layout will be constructed.
      \param dofs_per_entity The number of degrees-of-freedom per grid entity.
    */
    ArrayLayout( const std::shared_ptr<Block>& block,
                 const int dofs_per_entity )
        : _block( block )
        , _dofs_per_entity( dofs_per_entity )
    {}

    // Get the grid block over which this layout is defined.
    const Block& block() const
    { return *_block; }

    // Get the number of degrees-of-freedom on each grid entity.
    int dofsPerEntity() const
    { return _dofs_per_entity; }

    // Get the index space of the array elements in the given
    // decomposition.
    template<class DecompositionTag, class IndexType>
    IndexSpace<4> indexSpace( DecompositionTag decomposition_tag,
                              IndexType index_type ) const
    {
        return appendDimension(
            _block->indexSpace(decomposition_tag,EntityType(),index_type),
            _dofs_per_entity );
    }

    // Get the local index space of the array elements we shared with the
    // given neighbor in the given decomposition.
    template<class DecompositionTag>
    IndexSpace<4> sharedIndexSpace( DecompositionTag decomposition_tag,
                                    const int off_i,
                                    const int off_j,
                                    const int off_k ) const
    {
        return appendDimension(
            _block->sharedIndexSpace(decomposition_tag,EntityType(),
                                     off_i,off_j,off_k),
            _dofs_per_entity );
    }

  private:

    std::shared_ptr<Block> _block;
    int _dofs_per_entity;
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array layout over the entities of a block.
  \param block The grid block over which to create the layout.
  \param dofs_per_entity The number of degrees-of-freedom per grid entity.
*/
template<class EntityType>
std::shared_ptr<ArrayLayout<EntityType>>
createArrayLayout( const std::shared_ptr<Block>& block,
                   const int dofs_per_entity,
                   EntityType )
{
    return std::make_shared<ArrayLayout<EntityType>>( block, dofs_per_entity );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create an array layout over the entities of a grid given block
  parameters. An intermediate block will be created and assigned to the
  layout.
  \param block The grid block over which to create the layout.
  \param dofs_per_entity The number of degrees-of-freedom per grid entity.
*/
template<class EntityType>
std::shared_ptr<ArrayLayout<EntityType>>
createArrayLayout( const std::shared_ptr<GlobalGrid>& global_grid,
                   const int halo_cell_width,
                   const int dofs_per_entity,
                   EntityType )
{
    return std::make_shared<ArrayLayout<EntityType>>(
        createBlock(global_grid,halo_cell_width), dofs_per_entity );
}

//---------------------------------------------------------------------------//
// Array
//---------------------------------------------------------------------------//
template<class Scalar, class EntityType, class ... Params>
class Array
{
  public:

    // Value type.
    using value_type = Scalar;

    // Entity type.
    using entity_type = EntityType;

    // Array layout type.
    using array_layout = ArrayLayout<entity_type>;

    // View type.
    using view_type = Kokkos::View<value_type****,Params...>;

    // Device type.
    using device_type = typename view_type::device_type;

    // Memory space.
    using memory_space = typename view_type::memory_space;

    // Execution space.
    using execution_space = typename view_type::execution_space;

    /*!
      \brief Create an array with the given layout. Arrays are constructed
      over the ghosted index space of the layout.
      \param label A label for the array.
      \param layout The array layout over which to construct the view.
    */
    Array( const std::string& label,
           const std::shared_ptr<array_layout>& layout )
        : _layout( layout )
        , _data(
            createView<value_type,Params...>(
                label,layout->indexSpace(Ghost(),Local())) )
    {}

    //! Get the layout of the array.
    const array_layout& layout() const
    { return *_layout; }

    //! Get a view of the array data.
    view_type view() const
    { return _data; }

    //! Get the array label.
    std::string label() const
    { return _data.label(); }

  private:

    std::shared_ptr<array_layout> _layout;
    view_type _data;
};

//---------------------------------------------------------------------------//
// Scatic type checker.
//---------------------------------------------------------------------------//
// Static type checker.
template<class >
struct is_array : public std::false_type {};

template<class Scalar, class EntityType, class ... Params>
struct is_array<Array<Scalar,EntityType,Params...> >
    : public std::true_type {};

template<class Scalar, class EntityType, class ... Params>
struct is_array<const Array<Scalar,EntityType,Params...> >
    : public std::true_type {};

//---------------------------------------------------------------------------//
// Array creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array with the given array layout. Views are constructed
  over the ghosted index space of the layout.
  \param label A label for the view.
  \param layout The array layout over which to construct the view.
 */
template<class Scalar, class ... Params, class EntityType>
std::shared_ptr<Array<Scalar,EntityType,Params...>>
createArray( const std::string& label,
             const std::shared_ptr<ArrayLayout<EntityType>>& layout )
{
    return std::make_shared<Array<Scalar,EntityType,Params...>>(
        label, layout );
}

//---------------------------------------------------------------------------//
// Array operations.
//---------------------------------------------------------------------------//
namespace ArrayOp
{
//---------------------------------------------------------------------------//
/*!
  \brief Assign a scalar value to every element of an array.
  \param array The array to assign the value to.
  \param alpha The value to assign to the array.
  \param tag The tag for the decomposition over which to perform the
  operation.
*/
template<class Array_t, class DecompositionTag>
void assign( Array_t& array,
             const typename Array_t::value_type alpha,
             DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto subview =
        createSubview( array.view(), array.layout().indexSpace(tag,Local()) );
    Kokkos::deep_copy( subview, alpha );
}

//---------------------------------------------------------------------------//
/*!
  \brief Scale every element of an array by a scalar value.
  \param array The array to scale.
  \param alpha The value to scale the array by.
  \param tag The tag for the decomposition over which to perform the
  operation.
*/
template<class Array_t, class DecompositionTag>
void scale( Array_t& array,
            const typename Array_t::value_type alpha,
            DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto view = array.view();
    Kokkos::parallel_for(
        "ArrayOp::scale",
        createExecutionPolicy( array.layout().indexSpace(tag,Local()),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ){
            view(i,j,k,l) *= alpha;
        });
}

//---------------------------------------------------------------------------//
/*!
  \brief Scale every element of an array by a scalar
  \param array The array to scale.
  \param alpha The values to scale the array by. A value must be provided for
  each entity degree-of-freedom in the array.
  \param tag The tag for the decomposition over which to perform the
  operation.
*/
template<class Array_t, class DecompositionTag>
void scale( Array_t& array,
            const std::vector<typename Array_t::value_type>& alpha,
            DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( alpha.size() != static_cast<unsigned>(array.layout().dofsPerEntity()) )
        throw std::runtime_error( "Incorrect vector size" );

    Kokkos::View<const typename Array_t::value_type*,
                 Kokkos::HostSpace,
                 Kokkos::MemoryUnmanaged>
        alpha_view_host( alpha.data(), alpha.size() );
    auto alpha_view = Kokkos::create_mirror_view_and_copy(
        typename Array_t::device_type(), alpha_view_host );

    auto array_view = array.view();
    Kokkos::parallel_for(
        "ArrayOp::scale",
        createExecutionPolicy( array.layout().indexSpace(tag,Local()),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ){
            array_view(i,j,k,l) *= alpha_view(l);
        });
}

//---------------------------------------------------------------------------//
/*!
  \brief Copy one array into another over the designated decomposition. A <- B
  \param a The array to which the data will be copied.
  \param b The array from which the data will be copied.
  \param tag The tag for the decomposition over which to perform the
  operation.
*/
template<class Array_t, class DecompositionTag>
void copy( Array_t& a,
           const Array_t& b,
           DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto a_space = a.layout().indexSpace( tag, Local() );
    auto b_space = b.layout().indexSpace( tag, Local() );
    if ( a_space != b_space )
        throw std::logic_error( "Incompatible index spaces" );
    auto subview_a = createSubview( a.view(), a_space );
    auto subview_b = createSubview( b.view(), b_space );
    Kokkos::deep_copy( subview_a, subview_b );
}

//---------------------------------------------------------------------------//
/*!
  \brief Update two vectors auch that a = alpha * a + beta * b.
  \param a The array that will be updated.
  \param alpha The value to scale a by.
  \param b The array to add to a.
  \param beta The value to scale b by.
  \param tag The tag for the decomposition over which to perform the
  operation.
 */
template<class Array_t, class DecompositionTag>
void update( Array_t& a,
             const typename Array_t::value_type alpha,
             const Array_t& b,
             const typename Array_t::value_type beta,
             DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto a_view = a.view();
    auto b_view = b.view();
    Kokkos::parallel_for(
        "ArrayOp::update",
        createExecutionPolicy( a.layout().indexSpace(tag,Local()),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ){
            a_view(i,j,k,l) = alpha * a_view(i,j,k,l) + beta * b_view(i,j,k,l);
        });
}

//---------------------------------------------------------------------------//
// Dot product
template<class ViewType>
struct DotFunctor
{
    typedef typename ViewType::value_type value_type[];
    typedef typename ViewType::size_type size_type;
    size_type value_count;
    ViewType _a;
    ViewType _b;

    DotFunctor( const ViewType& a, const ViewType& b )
        : value_count( a.extent(3) )
        , _a( a )
        , _b( b )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator() ( const size_type i,
                      const size_type j,
                      const size_type k,
                      const size_type l,
                      value_type sum ) const
    {
        sum[l] += _a(i,j,k,l) * _b(i,j,k,l);
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile value_type dst,
               const volatile value_type src ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            dst[j] += src[j];
    }

    KOKKOS_INLINE_FUNCTION void init ( value_type sum ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            sum[j] = 0.0;
    }
};

/*!
  \brief Compute the dot product of owned space of two arrays.
  \param a The first array in the dot product.
  \param b The second array in the dot product.
  \param products The dot product of each entity degree-of-freedom in the
  array. This vector should be pre-sized to the number of degrees-of-freedom
  per entity.
*/
template<class Array_t>
void dot( const Array_t& a,
          const Array_t& b,
          std::vector<typename Array_t::value_type>& products )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( products.size() != static_cast<unsigned>(a.layout().dofsPerEntity()) )
        throw std::runtime_error( "Incorrect vector size" );

    for ( auto& p : products ) p = 0.0;

    DotFunctor<typename Array_t::view_type> functor( a.view(), b.view() );
    Kokkos::parallel_reduce(
        "ArrayOp::dot",
        createExecutionPolicy( a.layout().indexSpace(Own(),Local()),
                               typename Array_t::execution_space() ),
        functor,
        products.data() );

    MPI_Allreduce( MPI_IN_PLACE,
                   products.data(),
                   products.size(),
                   MpiTraits<typename Array_t::value_type>::type(),
                   MPI_SUM,
                   a.layout().block().globalGrid().comm() );
}

//---------------------------------------------------------------------------//
// Infinity norm
template<class ViewType>
struct NormInfFunctor
{
    typedef typename ViewType::value_type value_type[];
    typedef typename ViewType::size_type size_type;
    size_type value_count;
    ViewType _view;

    NormInfFunctor( const ViewType& view )
        : value_count( view.extent(3) )
        , _view( view )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator() ( const size_type i,
                      const size_type j,
                      const size_type k,
                      const size_type l,
                      value_type norm ) const
    {
        auto v_abs = fabs(_view(i,j,k,l));
        if ( v_abs > norm[l] )
            norm[l] = v_abs;
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile value_type dst,
               const volatile value_type src ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            if ( src[j] > dst[j] )
                dst[j] = src[j];
    }

    KOKKOS_INLINE_FUNCTION void init ( value_type norm ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            norm[j] = 0.0;
    }
};

/*!
  \brief Calculate the infinity-norm of the owned elements of the array.
  \param array The array to compute the norm for.
  \param norms The norms for each degree-of-freedom in the array. This vector
  should be pre-sized to the number of degrees-of-freedom per entity.
*/
template<class Array_t>
void normInf( const Array_t& array,
              std::vector<typename Array_t::value_type>& norms )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( norms.size() != static_cast<unsigned>(array.layout().dofsPerEntity()) )
        throw std::runtime_error( "Incorrect vector size" );

    for ( auto& n : norms ) n = 0.0;

    NormInfFunctor<typename Array_t::view_type> functor( array.view() );
    Kokkos::parallel_reduce(
        "ArrayOp::normInf",
        createExecutionPolicy( array.layout().indexSpace(Own(),Local()),
                               typename Array_t::execution_space() ),
        functor,
        norms.data() );

    MPI_Allreduce( MPI_IN_PLACE,
                   norms.data(),
                   norms.size(),
                   MpiTraits<typename Array_t::value_type>::type(),
                   MPI_MAX,
                   array.layout().block().globalGrid().comm() );
}

//---------------------------------------------------------------------------//
// One norm
template<class ViewType>
struct Norm1Functor
{
    typedef typename ViewType::value_type value_type[];
    typedef typename ViewType::size_type size_type;
    size_type value_count;
    ViewType _view;

    Norm1Functor( const ViewType& view )
        : value_count( view.extent(3) )
        , _view( view )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator() ( const size_type i,
                      const size_type j,
                      const size_type k,
                      const size_type l,
                      value_type norm ) const
    {
        norm[l] += fabs(_view(i,j,k,l));
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile value_type dst,
               const volatile value_type src ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            dst[j] += src[j];
    }

    KOKKOS_INLINE_FUNCTION void init ( value_type norm ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            norm[j] = 0.0;
    }
};

/*!
  \brief Calculate the one-norm of the owned elements of the array.
  \param array The array to compute the norm for.
  \param norms The norms for each degree-of-freedom in the array. This vector
  should be pre-sized to the number of degrees-of-freedom per entity.
*/
template<class Array_t>
void norm1( const Array_t& array,
            std::vector<typename Array_t::value_type>& norms )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( norms.size() != static_cast<unsigned>(array.layout().dofsPerEntity()) )
        throw std::runtime_error( "Incorrect vector size" );

    for ( auto& n : norms ) n = 0.0;

    Norm1Functor<typename Array_t::view_type> functor( array.view() );
    Kokkos::parallel_reduce(
        "ArrayOp::norm1",
        createExecutionPolicy( array.layout().indexSpace(Own(),Local()),
                               typename Array_t::execution_space() ),
        functor,
        norms.data() );

    MPI_Allreduce( MPI_IN_PLACE,
                   norms.data(),
                   norms.size(),
                   MpiTraits<typename Array_t::value_type>::type(),
                   MPI_SUM,
                   array.layout().block().globalGrid().comm() );
}

//---------------------------------------------------------------------------//
// Two norm
template<class ViewType>
struct Norm2Functor
{
    typedef typename ViewType::value_type value_type[];
    typedef typename ViewType::size_type size_type;
    size_type value_count;
    ViewType _view;

    Norm2Functor( const ViewType& view )
        : value_count( view.extent(3) )
        , _view( view )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator() ( const size_type i,
                      const size_type j,
                      const size_type k,
                      const size_type l,
                      value_type norm ) const
    {
        norm[l] += _view(i,j,k,l) * _view(i,j,k,l);
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile value_type dst,
               const volatile value_type src ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            dst[j] += src[j];
    }

    KOKKOS_INLINE_FUNCTION void init ( value_type norm ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            norm[j] = 0.0;
    }
};

/*!
  \brief Calculate the two-norm of the owned elements of the array.
  \param array The array to compute the norm for.
  \param norms The norms for each entity degree-of-freedom in the array. This
  vector should be pre-sized to the number of degrees-of-freedom per entity.
*/
template<class Array_t>
void norm2( const Array_t& array,
            std::vector<typename Array_t::value_type>& norms )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( norms.size() != static_cast<unsigned>(array.layout().dofsPerEntity()) )
        throw std::runtime_error( "Incorrect vector size" );

    for ( auto& n : norms ) n = 0.0;

    Norm2Functor<typename Array_t::view_type> functor( array.view() );
    Kokkos::parallel_reduce(
        "ArrayOp::norm2",
        createExecutionPolicy( array.layout().indexSpace(Own(),Local()),
                               typename Array_t::execution_space() ),
        functor,
        norms.data() );

    MPI_Allreduce( MPI_IN_PLACE,
                   norms.data(),
                   norms.size(),
                   MpiTraits<typename Array_t::value_type>::type(),
                   MPI_SUM,
                   array.layout().block().globalGrid().comm() );

    for ( auto& n : norms ) n = std::sqrt( n );
}

//---------------------------------------------------------------------------//

} // end namespace ArrayOp

//---------------------------------------------------------------------------//

} // end namespace Cajita.

#endif // end CAJITA_ARRAY_HPP
