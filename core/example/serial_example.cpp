#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberDataTypes.hpp>
#include <Cabana_Serial.hpp>

#include <cstdlib>
#include <iostream>

//---------------------------------------------------------------------------//
// Define particle data.
//---------------------------------------------------------------------------//
// Inner array size (the size of the arrays in the structs-of-arrays).
//
// NOTE: It would be nice to have a dynamic option for this if possible. This
// would allow for a true SoA-only configuration where only enough memory for
// all the particles is allocated. Although AoSoA with very large inner array
// sizes may not perform any worse than SoA anyways so maybe this is not
// needed.
const std::size_t array_size = 10;

// Spatial dimension.
const std::size_t space_dim = 3;

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

// Set the type for the particle AoSoA.
using ParticleList =
    Cabana::AoSoA<ParticleDataTypes,Cabana::Serial,array_size>;

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//
// Function to intitialize the particles.
void initializeParticles( ParticleList particles )
{
    for ( auto idx = particles.begin();
          idx != particles.end();
          ++idx )
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
    for ( auto idx = particles.begin();
          idx != particles.end();
          ++idx )
    {
        std::cout << std::endl;

        std::cout << "Struct id: " << idx.s() << std::endl;
        std::cout << "Struct offset: " << idx.i() << std::endl;
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

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main()
{
    // Declare a number of particles.
    int num_particle = 45;

    // Create the particle list.
    ParticleList particles( num_particle );

    // Initialize particles.
    initializeParticles( particles );

    // Print particles.
    printParticles( particles );

    return 0;
}

//---------------------------------------------------------------------------//
