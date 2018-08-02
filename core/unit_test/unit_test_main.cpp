#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

int main( int argc, char* argv[] )
{
    Kokkos::initialize( argc, argv );
    ::testing::InitGoogleTest( &argc, argv );
    int return_val = RUN_ALL_TESTS();
    Kokkos::finalize();
    return return_val;
}
