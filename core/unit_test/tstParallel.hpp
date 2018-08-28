#include <Cabana_Parallel.hpp>
#include <Cabana_ExecutionPolicy.hpp>
#include <Cabana_AoSoA.hpp>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template<class aosoa_type>
void checkDataMembers(
    aosoa_type aosoa,
    const float fval, const double dval, const int ival,
    const int dim_1, const int dim_2,
    const int dim_3, const int dim_4 )
{
    auto slice_0 = aosoa.slice( Cabana::MemberTag<0>() );
    auto slice_1 = aosoa.slice( Cabana::MemberTag<1>() );
    auto slice_2 = aosoa.slice( Cabana::MemberTag<2>() );
    auto slice_3 = aosoa.slice( Cabana::MemberTag<3>() );
    auto slice_4 = aosoa.slice( Cabana::MemberTag<4>() );

    for ( auto idx = 0; idx != aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    EXPECT_EQ( slice_0( idx, i, j, k ),
                               fval * (i+j+k) );

        // Member 1.
        EXPECT_EQ( slice_1( idx ), ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        EXPECT_EQ( slice_2( idx, i, j, k, l ),
                                   fval * (i+j+k+l) );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( slice_3( idx, i ), dval * i );

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( slice_4( idx, i, j ), dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// Assignment operator.
template<class AoSoA_t,
         class SliceType0,
         class SliceType1,
         class SliceType2,
         class SliceType3,
         class SliceType4>
class AssignmentOp
{
  public:
    AssignmentOp( AoSoA_t aosoa,
                  float fval,
                  double dval,
                  int ival )
        : _aosoa( aosoa )
        , _slice_0( aosoa.slice(Cabana::MemberTag<0>()) )
        , _slice_1( aosoa.slice(Cabana::MemberTag<1>()) )
        , _slice_2( aosoa.slice(Cabana::MemberTag<2>()) )
        , _slice_3( aosoa.slice(Cabana::MemberTag<3>()) )
        , _slice_4( aosoa.slice(Cabana::MemberTag<4>()) )
        , _fval( fval )
        , _dval( dval )
        , _ival( ival )
        , _dim_1( _slice_2.extent(2) )
        , _dim_2( _slice_2.extent(3) )
        , _dim_3( _slice_2.extent(4) )
        , _dim_4( _slice_2.extent(5) )
    {}

    KOKKOS_INLINE_FUNCTION void operator()( const int idx ) const
    {
        // Member 0.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                for ( int k = 0; k < _dim_3; ++k )
                    _slice_0( idx, i, j, k ) = _fval * (i+j+k);

        // Member 1.
        _slice_1( idx ) = _ival;

        // Member 2.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                for ( int k = 0; k < _dim_3; ++k )
                    for ( int l = 0; l < _dim_4; ++l )
                        _slice_2( idx, i, j, k, l ) = _fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < _dim_1; ++i )
            _slice_3( idx, i ) = _dval * i;

        // Member 4.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                _slice_4( idx, i, j ) = _dval * (i+j);
    }

  private:

    AoSoA_t _aosoa;
    SliceType0 _slice_0;
    SliceType1 _slice_1;
    SliceType2 _slice_2;
    SliceType3 _slice_3;
    SliceType4 _slice_4;
    float _fval;
    double _dval;
    int _ival;
    int _dim_1 = 3;
    int _dim_2 = 2;
    int _dim_3 = 4;
    int _dim_4 = 3;
};

//---------------------------------------------------------------------------//
// Parallel for test.
void runTest()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;
    const int dim_4 = 3;

    // Declare data types.
    using DataTypes =
        Cabana::MemberDataTypes<float[dim_1][dim_2][dim_3],
                                int,
                                float[dim_1][dim_2][dim_3][dim_4],
                                double[dim_1],
                                double[dim_1][dim_2]
                                >;

    // Declare the AoSoA type. Let the library pick an inner array size based
    // on the execution space.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 155;
    AoSoA_t aosoa( num_data );

    // Create an execution policy using the begin and end of the AoSoA.
    Cabana::RangePolicy<AoSoA_t::vector_length,TEST_EXECSPACE>
        range_policy( 0, aosoa.size() );

    // Create a functor to operate on.
    using OpType = AssignmentOp<AoSoA_t,
                                decltype(aosoa.slice(Cabana::MemberTag<0>())),
                                decltype(aosoa.slice(Cabana::MemberTag<1>())),
                                decltype(aosoa.slice(Cabana::MemberTag<2>())),
                                decltype(aosoa.slice(Cabana::MemberTag<3>())),
                                decltype(aosoa.slice(Cabana::MemberTag<4>()))>;
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    OpType func_1( aosoa, fval, dval, ival );

    // Loop in parallel using 1D struct parallelism.
    Cabana::parallel_for( range_policy, func_1, Cabana::StructParallelTag() );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Change values and write a second functor.
    fval = 93.4;
    dval = 12.1;
    ival = 4;
    OpType func_2( aosoa, fval, dval, ival );

    // Loop in parallel using 1D array parallelism.
    Cabana::parallel_for( range_policy, func_2, Cabana::ArrayParallelTag() );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Change values and write a third functor.
    fval = 7.7;
    dval = 3.2;
    ival = 9;
    OpType func_3( aosoa, fval, dval, ival );

    // Loop in parallel using 2D struct and array parallelism.
    Cabana::parallel_for(
        range_policy, func_3, Cabana::StructAndArrayParallelTag() );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Do one more loop but this time auto-dispatch. Reuse the first functor
    // but this time create an execution policy that automatically grabs begin
    // and end from the aosoa.
    Cabana::RangePolicy<AoSoA_t::vector_length,TEST_EXECSPACE> aosoa_policy( aosoa );
    Cabana::parallel_for( aosoa_policy, func_1 );

    // Check data members for proper initialization.
    fval = 3.4;
    dval = 1.23;
    ival = 1;
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, parallel_for_test )
{
    runTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
