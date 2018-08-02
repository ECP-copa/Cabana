#include <Cabana_Particle.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template<class view_type>
void checkDataMembers(
    view_type view,
    const float fval, const double dval, const int ival,
    const std::size_t dim_1, const std::size_t dim_2,
    const std::size_t dim_3, const std::size_t dim_4 )
{
    for ( std::size_t idx = 0; idx < view.extent(0); ++idx )
    {
        // Member 0.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    EXPECT_EQ( view(idx).template get<0>( i, j, k ),
                               fval * (i+j+k) );

        // Member 1.
        EXPECT_EQ( view(idx).template get<1>(), ival );

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    for ( std::size_t l = 0; l < dim_4; ++l )
                        EXPECT_EQ( view(idx).template get<2>( i, j, k, l ),
                                   fval * (i+j+k+l) );

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            EXPECT_EQ( view(idx).template get<3>( i ), dval * i );

        // Member 4.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                EXPECT_EQ( view(idx).template get<4>( i, j ), dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// Particle test
void runTest()
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

    // Declare the particle type.
    using Particle_t = Cabana::Particle<DataTypes>;

    // Create a view of particles.
    std::size_t num_data = 453;
    Kokkos::View<Particle_t*,TEST_MEMSPACE> particles( "particles", num_data );

    // Initialize data.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    auto init_func = KOKKOS_LAMBDA( const std::size_t idx )
    {
        // Member 0.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    particles( idx ).get<0>( i, j, k ) = fval * (i+j+k);

        // Member 1.
        particles( idx ).get<1>() = ival;

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    for ( std::size_t l = 0; l < dim_4; ++l )
                        particles( idx ).get<2>( i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            particles( idx ).get<3>( i ) = dval * i;

        // Member 4.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                particles( idx ).get<4>( i, j ) = dval * (i+j);
    };
    Kokkos::fence();

    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_data );

    Kokkos::parallel_for( policy, init_func );
    Kokkos::fence();

    // Check data members of the for proper initialization.
    checkDataMembers( particles, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, particle_test )
{
    runTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
