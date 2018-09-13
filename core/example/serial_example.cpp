#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberTypes.hpp>

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
    Cabana::MemberTypes<float,                        // (0) x-position type
                        float,                        // (1) y-position type
                        float,                        // (2) z-position type
                        double[space_dim],            // (3) velocity type
                        double[space_dim][space_dim], // (4) stress type
                        int                           // (5) status type
                        >;

// Declare the memory space.
using MemorySpace = Cabana::HostSpace;

// Declare the inner array layout.
const int vector_length = 32;

// Set the type for the particle AoSoA.
using ParticleList = Cabana::AoSoA<ParticleDataTypes,MemorySpace,vector_length>;

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//
// Function to intitialize the particles.
void initializeParticles( ParticleList particles )
{
    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();
    auto velocity = particles.slice<Velocity>();
    auto stress = particles.slice<Stress>();
    auto status = particles.slice<Status>();

    for ( auto idx = 0; idx != particles.size(); ++idx )
    {
        // Initialize position.
        position_x( idx ) = 1.1;
        position_y( idx ) = 2.2;
        position_z( idx ) = 3.3;

        // Initialize velocity.
        for ( int d = 0; d < space_dim; ++d )
            velocity( idx, d ) = 1.1 * d;

        // Initialize stress to the identity matrix.
        for ( int j = 0; j < space_dim; ++j )
            for ( int i = 0; i < space_dim; ++i )
                stress( idx, i, j ) =
                    ( i == j ) ? 1.0 : 0.0;

        // Initialize all particles to a status of 1.
        status( idx ) = 1;
    }
}

// Function to print out the data for every particle.
void printParticles( const ParticleList particles )
{
    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();
    auto velocity = particles.slice<Velocity>();
    auto stress = particles.slice<Stress>();
    auto status = particles.slice<Status>();

    for ( auto idx = 0; idx != particles.size(); ++idx )
    {
        auto aosoa_idx_s = Cabana::Impl::Index<32>::s( idx );
        auto aosoa_idx_i = Cabana::Impl::Index<32>::i( idx );

        std::cout << std::endl;

        std::cout << "Struct id: " << aosoa_idx_s << std::endl;
        std::cout << "Struct offset: " << aosoa_idx_i << std::endl;
        std::cout << "Position: "
                  << position_x( idx ) << " "
                  << position_y( idx ) << " "
                  << position_z( idx ) << std::endl;

        std::cout << "Velocity ";
        for ( int d = 0; d < space_dim; ++d )
            std::cout << velocity( idx, d ) << " ";
        std::cout << std::endl;

        std::cout << "Stress ";
        for ( int j = 0; j < space_dim; ++j )
        {
            std::cout << "{ ";
            for ( int i = 0; i < space_dim; ++i )
                std::cout << stress( idx, i, j ) << " " ;
            std::cout << "}";
        }
        std::cout << std::endl;

        std::cout << "Status " << status(idx) << std::endl;
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
