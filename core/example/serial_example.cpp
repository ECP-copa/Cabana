#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberDataTypes.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>

//---------------------------------------------------------------------------//
// Define particle data.
//---------------------------------------------------------------------------//
// Spatial dimension.
const int space_dim = 3;

// User field enumeration. These will be used to index into the data set. Must
// start at 0 and increment contiguously.
//
// NOTE: Users don't have to make this enum (or some other set of integral
// constants) but it is a nice way to provide meaning to the different data
// types and values assigned to the particles.
//
// NOTE: These enums are also ordered in the same way as the data in the
// template parameters below.
enum UserParticleFields
{
    PositionX = 0,
    PositionY,
    PositionZ,
    Velocity,
    Stress,
    Status
};

// Designate the types that the particles will hold.
using ParticleDataTypes =
    Cabana::MemberDataTypes<float,                        // (0) x-position type
                            float,                        // (1) y-position type
                            float,                        // (2) z-position type
                            double[space_dim],            // (3) velocity type
                            double[space_dim][space_dim], // (4) stress type
                            int                           // (5) status type
                            >;

// Declare the memory space.
using MemorySpace = Kokkos::HostSpace;

// Declare the inner array layout.
using ArrayLayout = Cabana::InnerArrayLayout<32,Kokkos::LayoutRight>;

// Set the type for the particle AoSoA.
using ParticleList = Cabana::AoSoA<ParticleDataTypes,ArrayLayout,MemorySpace>;

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//
// Function to intitialize the particles.
void initializeParticles( ParticleList particles )
{
    for ( auto idx = 0; idx != particles.size(); ++idx )
    {
        // Initialize position.
        particles.get<PositionX>( idx ) = 1.1;
        particles.get<PositionY>( idx ) = 2.2;
        particles.get<PositionZ>( idx ) = 3.3;

        // Initialize velocity.
        for ( int d = 0; d < space_dim; ++d )
            particles.get<Velocity>( idx, d ) = 1.1 * d;

        // Initialize stress to the identity matrix.
        for ( int j = 0; j < space_dim; ++j )
            for ( int i = 0; i < space_dim; ++i )
                particles.get<Stress>( idx, i, j ) =
                    ( i == j ) ? 1.0 : 0.0;

        // Initialize all particles to a status of 1.
        particles.get<Status>( idx ) = 1;
    }
}

// Function to print out the data for every particle.
void printParticles( const ParticleList particles )
{
    for ( auto idx = 0; idx != particles.size(); ++idx )
    {
        auto aosoa_idx = Cabana::Impl::Index<32>::aosoa( idx );

        std::cout << std::endl;

        std::cout << "Struct id: " << aosoa_idx.first << std::endl;
        std::cout << "Struct offset: " << aosoa_idx.second << std::endl;
        std::cout << "Position: "
                  << particles.get<PositionX>( idx ) << " "
                  << particles.get<PositionY>( idx ) << " "
                  << particles.get<PositionZ>( idx ) << std::endl;

        std::cout << "Velocity ";
        for ( int d = 0; d < space_dim; ++d )
            std::cout << particles.get<Velocity>( idx, d ) << " ";
        std::cout << std::endl;

        std::cout << "Stress ";
        for ( int j = 0; j < space_dim; ++j )
        {
            std::cout << "{ ";
            for ( int i = 0; i < space_dim; ++i )
                std::cout << particles.get<Stress>( idx, i, j ) << " " ;
            std::cout << "}";
        }
        std::cout << std::endl;

        std::cout << "Status " << particles.get<Status>(idx) << std::endl;
    }
}

// example main
void exampleMain()
{
    // Declare a number of particles.
    int num_particle = 45;

    // Create the particle list.
    ParticleList particles( num_particle );

    // Initialize particles.
    initializeParticles( particles );

    // Print particles.
    printParticles( particles );
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // Initialize the kokkos runtime.
    Kokkos::initialize( argc, argv );

    // Run the test.
    exampleMain();

    // Finalize.
    Kokkos::finalize();
    return 0;
}

//---------------------------------------------------------------------------//
