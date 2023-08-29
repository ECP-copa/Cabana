/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Kokkos_Core.hpp>

#include <Cabana_DeepCopy.hpp>
#include <Cabana_Fields.hpp>
#include <Cabana_ParticleList.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// Fields.
struct CommRank : Cabana::Field::Scalar<int>
{
    static std::string label() { return "comm_rank"; }
};

struct Foo : Cabana::Field::Scalar<double>
{
    static std::string label() { return "foo"; }
};

struct Bar : Cabana::Field::Matrix<double, 3, 3>
{
    static std::string label() { return "bar"; }
};

template <class ParticleList>
void setParticleListTestData( ParticleList& plist )
{
    // Resize the aosoa.
    auto& aosoa = plist.aosoa();
    std::size_t num_p = 10;
    aosoa.resize( num_p );

    // Populate fields.
    auto px = plist.slice( Cabana::Field::Position<3>() );
    auto pm = plist.slice( Foo() );
    auto pc = plist.slice( CommRank() );
    auto pf = plist.slice( Bar() );

    Cabana::deep_copy( px, 1.23 );
    Cabana::deep_copy( pm, 3.3 );
    Cabana::deep_copy( pc, 5 );
    Cabana::deep_copy( pf, -1.2 );
}

template <class ParticleList>
void checkParticleListLabels( const ParticleList plist )
{
    EXPECT_EQ( plist.size(), 10 );

    // Check the slices.
    auto px = plist.slice( Cabana::Field::Position<3>() );
    auto pm = plist.slice( Foo() );
    auto pc = plist.slice( CommRank() );
    auto pf = plist.slice( Bar() );
    EXPECT_EQ( px.label(), "position" );
    EXPECT_EQ( pm.label(), "foo" );
    EXPECT_EQ( pc.label(), "comm_rank" );
    EXPECT_EQ( pf.label(), "bar" );
}

template <class ParticleList>
void checkParticleListInitial( const ParticleList plist )
{
    // Check initial state after deep copy.
    auto& aosoa = plist.aosoa();
    auto aosoa_host =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto px_h = Cabana::slice<0>( aosoa_host );
    auto pm_h = Cabana::slice<1>( aosoa_host );
    auto pc_h = Cabana::slice<2>( aosoa_host );
    auto pf_h = Cabana::slice<3>( aosoa_host );
    for ( std::size_t p = 0; p < aosoa_host.size(); ++p )
    {
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( px_h( p, d ), 1.23 );

        EXPECT_DOUBLE_EQ( pm_h( p ), 3.3 );

        EXPECT_EQ( pc_h( p ), 5 );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( pf_h( p, i, j ), -1.2 );
    }
}

template <class ParticleList>
void checkParticleListFinal( const ParticleList plist )
{
    // Check the modification.
    auto& aosoa = plist.aosoa();
    auto aosoa_host =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto px_h = Cabana::slice<0>( aosoa_host );
    auto pm_h = Cabana::slice<1>( aosoa_host );
    auto pc_h = Cabana::slice<2>( aosoa_host );
    auto pf_h = Cabana::slice<3>( aosoa_host );
    for ( std::size_t p = 0; p < aosoa_host.size(); ++p )
    {
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( px_h( p, d ), 1.23 + p + d );

        EXPECT_DOUBLE_EQ( pm_h( p ), 3.3 + p );

        EXPECT_EQ( pc_h( p ), 5 + p );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( pf_h( p, i, j ), -1.2 + p + i + j );
    }
}

//---------------------------------------------------------------------------//
template <class ListType>
void particleListTest( ListType plist )
{
    setParticleListTestData( plist );

    checkParticleListLabels( plist );
    checkParticleListInitial( plist );

    // Locally modify.
    Kokkos::parallel_for(
        "modify", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, plist.size() ),
        KOKKOS_LAMBDA( const int p ) {
            auto particle = plist.getParticle( p );

            for ( int d = 0; d < 3; ++d )
                get( particle, Cabana::Field::Position<3>(), d ) += p + d;

            get( particle, Foo() ) += p;

            get( particle, CommRank() ) += p;

            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    get( particle, Bar(), i, j ) += p + i + j;

            plist.setParticle( particle, p );
        } );

    // Check the modification.
    checkParticleListFinal( plist );
}

//---------------------------------------------------------------------------//
template <class ListType>
void particleViewTest( ListType plist )
{
    setParticleListTestData( plist );

    checkParticleListLabels( plist );
    checkParticleListInitial( plist );

    // Locally modify.
    Kokkos::parallel_for(
        "modify", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, plist.size() ),
        KOKKOS_LAMBDA( const int p ) {
            auto particle = plist.getParticleView( p );

            for ( int d = 0; d < 3; ++d )
                get( particle, Cabana::Field::Position<3>(), d ) += p + d;

            get( particle, Foo() ) += p;

            get( particle, CommRank() ) += p;

            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    get( particle, Bar(), i, j ) += p + i + j;
        } );

    // Check the modification.
    checkParticleListFinal( plist );
}

//---------------------------------------------------------------------------//

} // end namespace Test
