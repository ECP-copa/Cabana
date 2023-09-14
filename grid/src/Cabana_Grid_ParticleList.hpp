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

/*!
  \file Cabana_Grid_ParticleList.hpp
  \brief Application-level particle/mesh storage and single particle access.
*/
#ifndef CABANA_GRID_PARTICLELIST_HPP
#define CABANA_GRID_PARTICLELIST_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberTypes.hpp>
#include <Cabana_ParticleList.hpp>
#include <Cabana_SoA.hpp>
#include <Cabana_Tuple.hpp>
#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <Cabana_Grid_ParticleGridDistributor.hpp>

#include <memory>
#include <string>
#include <type_traits>

namespace Cabana
{
namespace Grid
{

//---------------------------------------------------------------------------//
//! List of particle fields stored in AoSoA with associated Cajita mesh.
template <class MemorySpace, int VectorLength, class... FieldTags>
class ParticleList
    : public Cabana::ParticleList<MemorySpace, VectorLength, FieldTags...>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Base Cabana particle list type.
    using base = Cabana::ParticleList<memory_space, VectorLength, FieldTags...>;

    //! Particle AoSoA member types.
    using traits = typename base::traits;
    //! AoSoA type.
    using aosoa_type = typename base::aosoa_type;
    //! Particle tuple type.
    using tuple_type = typename base::tuple_type;
    /*!
      \brief Single field slice type.
      \tparam M AoSoA field index.
    */
    template <std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;
    //! Single particle type.
    using particle_type = typename base::particle_type;
    //! Single SoA type.
    using particle_view_type = typename base::particle_view_type;

    //! Default constructor.
    ParticleList( const std::string& label )
        : base( label )
    {
    }

    /*!
      \brief Redistribute particles to new owning grids.
      \return Whether the particles were actually redistributed.
    */
    template <class LocalGridType>
    bool redistribute( const LocalGridType& local_grid,
                       const bool force_redistribute = false )
    {
        return redistribute(
            local_grid, Cabana::Field::Position<LocalGridType::num_space_dim>(),
            force_redistribute );
    }

    /*!
      \brief Redistribute particles to new owning grids with explicit field.
      \tparam PositionFieldTag Field tag for position data.
      \return Whether the particles were actually redistributed.
    */
    template <class PositionFieldTag, class LocalGridType>
    bool redistribute( const LocalGridType& local_grid, PositionFieldTag,
                       const bool force_redistribute = false )
    {
        return particleGridMigrate(
            local_grid, this->slice( PositionFieldTag() ), _aosoa,
            local_grid.haloCellWidth(), force_redistribute );
    }

  protected:
    //! Particle AoSoA.
    using base::_aosoa;
};

template <class>
struct is_particle_list_impl : public std::false_type
{
};

template <class MemorySpace, int VectorLength, class... FieldTags>
struct is_particle_list_impl<
    ParticleList<MemorySpace, VectorLength, FieldTags...>>
    : public std::true_type
{
};

//! ParticleList static type checker.
template <class T>
struct is_particle_list
    : public is_particle_list_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
/*!
  \brief ParticleList creation function.
  \return ParticleList
*/
template <class MemorySpace, int VectorLength, class... FieldTags>
auto createParticleList( const std::string& label,
                         Cabana::ParticleTraits<FieldTags...> )
{
    return ParticleList<MemorySpace, VectorLength, FieldTags...>( label );
}

/*!
  \brief ParticleList creation function with default vector length.
  \return ParticleList
*/
template <class MemorySpace, class... FieldTags>
auto createParticleList( const std::string& label,
                         Cabana::ParticleTraits<FieldTags...> )
{
    return ParticleList<
        MemorySpace,
        Cabana::Impl::PerformanceTraits<
            typename MemorySpace::execution_space>::vector_length,
        FieldTags...>( label );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

namespace Cajita
{
//! \cond Deprecated
template <class MemorySpace, int VectorLength, class... FieldTags>
using ParticleList CAJITA_DEPRECATED =
    Cabana::Grid::ParticleList<MemorySpace, VectorLength, FieldTags...>;

template <class T>
using is_particle_list CAJITA_DEPRECATED = Cabana::Grid::is_particle_list<T>;

// MemorySpace and VectorLength cannot be deduced.
template <class MemorySpace, int VectorLength, class... Args>
CAJITA_DEPRECATED auto createParticleList( Args&&... args )
{
    return Cabana::Grid::createParticleList<MemorySpace, VectorLength>(
        std::forward<Args>( args )... );
}

// MemorySpace cannot be deduced.
template <class MemorySpace, class... Args>
CAJITA_DEPRECATED auto createParticleList( Args&&... args )
{
    return Cabana::Grid::createParticleList<MemorySpace>(
        std::forward<Args>( args )... );
}
//! \endcond
} // namespace Cajita

#endif // end CABANA_GRID_PARTICLELIST_HPP
