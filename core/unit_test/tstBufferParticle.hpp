#include <Cabana_BufferParticle.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template<class aosoa_type>
void checkDataMembers(
    aosoa_type aosoa,
    const float fval, const double dval, const int ival,
    const std::size_t dim_1, const std::size_t dim_2,
    const std::size_t dim_3, const std::size_t dim_4 )
{
    for ( auto idx = aosoa.begin(); idx != aosoa.end(); ++idx )
    {
        // Member 0.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    BOOST_CHECK( aosoa.template get<0>( idx, i, j, k ) ==
                                fval * (i+j+k) );

        // Member 1.
        BOOST_CHECK( aosoa.template get<1>( idx ) == ival );

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    for ( std::size_t l = 0; l < dim_4; ++l )
                        BOOST_CHECK( aosoa.template get<2>( idx, i, j, k, l ) ==
                                    fval * (i+j+k+l) );

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            BOOST_CHECK( aosoa.template get<3>( idx, i ) == dval * i );

        // Member 4.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                BOOST_CHECK( aosoa.template get<4>( idx, i, j ) == dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( pack_unpack_test )
{
    // Data dimensions.
    const std::size_t dim_1 = 3;
    const std::size_t dim_2 = 2;
    const std::size_t dim_3 = 4;
    const std::size_t dim_4 = 3;

    // Declare member types.
    using T0 = float[dim_1][dim_2][dim_3];
    using T1 = int;
    using T2 = float[dim_1][dim_2][dim_3][dim_4];
    using T3 = double[dim_1];
    using T4 = double[dim_1][dim_2];

    // Declare data types.
    using DataTypes = Cabana::MemberDataTypes<T0,T1,T2,T3,T4>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    std::size_t num_data = 453;
    AoSoA_t aosoa( num_data );

    // Initialize data with the rank accessors.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( auto idx = aosoa.begin(); idx != aosoa.end(); ++idx )
    {
        // Member 0.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    aosoa.get<0>( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        aosoa.get<1>( idx ) = ival;

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    for ( std::size_t l = 0; l < dim_4; ++l )
                        aosoa.get<2>( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            aosoa.get<3>( idx, i ) = dval * i;

        // Member 4.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                aosoa.get<4>( idx, i, j ) = dval * (i+j);
    }

    // Check the buffer sizes and offsets.
    using BufferParticle = Cabana::Impl::BufferParticle<DataTypes>;

    BOOST_CHECK( BufferParticle::total_bytes ==
                 sizeof(T0) + sizeof(T1) + sizeof(T2) + sizeof(T3) + sizeof(T4) );

    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<0,DataTypes>::size ==
                  sizeof(T0)) );
    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<1,DataTypes>::size ==
                  sizeof(T1)) );
    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<2,DataTypes>::size ==
                  sizeof(T2)) );
    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<3,DataTypes>::size ==
                  sizeof(T3)) );
    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<4,DataTypes>::size ==
                  sizeof(T4)) );

    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<0,DataTypes>::offset ==
                  0) );
    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<1,DataTypes>::offset ==
                  sizeof(T0)) );
    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<2,DataTypes>::offset ==
                  sizeof(T0) + sizeof(T1)) );
    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<3,DataTypes>::offset ==
                  sizeof(T0) + sizeof(T1) + sizeof(T2)) );
    BOOST_CHECK( (Cabana::Impl::BufferParticleMemberTraits<4,DataTypes>::offset ==
                  sizeof(T0) + sizeof(T1) + sizeof(T2) + sizeof(T3)) );

    // Create an execution policy for packing/unpacking.
    Cabana::IndexRangePolicy<TEST_EXECSPACE> policy( aosoa );

    // Pack the data into a buffer.
    Kokkos::View<BufferParticle*,TEST_MEMSPACE> buffer( "buffer", num_data );
    auto pack_op = KOKKOS_LAMBDA( const Cabana::Index idx )
    {
        Cabana::Impl::pack( idx, aosoa, buffer(idx.oneD()) );
    };
    Cabana::parallel_for( policy, pack_op );

    // Make a new AoSoA.
    AoSoA_t aosoa_2( num_data );

    // Unpack the buffer into the new AoSoA.
    auto unpack_op = KOKKOS_LAMBDA( const Cabana::Index idx )
    {
        Cabana::Impl::unpack( idx, buffer(idx.oneD()), aosoa_2 );
    };
    Cabana::parallel_for( policy, unpack_op );

    // Check data members of the new AoSoAfor proper initialization.
    checkDataMembers( aosoa_2, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}
