#include "Cabana_AoSoA.hpp"
#include "Cabana_Serial.hpp"

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
    std::size_t num_s = particles.numSoA();
    for ( std::size_t s = 0; s < num_s; ++s )
    {
        auto pos_x = particles.array<PositionX>( s );
        auto pos_y = particles.array<PositionY>( s );
        auto pos_z = particles.array<PositionZ>( s );
        auto velocity = particles.array<Velocity>( s );
        auto stress = particles.array<Stress>( s );
        auto status = particles.array<Status>( s );

        std::size_t num_p = particles.arraySize( s );
        for ( int p = 0; p < num_p; ++p )
        {
            pos_x[p] = 1.1;
            pos_y[p] = 2.2;
            pos_z[p] = 3.3;

            for ( int d = 0; d < space_dim; ++d )
                velocity[p][d] = 1.1 * d;

            for ( int j = 0; j < space_dim; ++j )
                for ( int i = 0; i < space_dim; ++i )
                    stress[p][i][j] = ( i == j ) ? 1.0 : 0.0;

            status[p] = 1;
        }
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
