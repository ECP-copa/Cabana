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
  \file Cajita_ParticleList.hpp
  \brief Application-level particle/mesh storage and single particle access.
*/
#ifndef CAJITA_PARTICLELIST_HPP
#define CAJITA_PARTICLELIST_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberTypes.hpp>
#include <Cabana_ParticleList.hpp>
#include <Cabana_SoA.hpp>
#include <Cabana_Tuple.hpp>

#include <Cajita_ParticleGridDistributor.hpp>

#include <memory>
#include <string>
#include <type_traits>

namespace Cajita
{

//---------------------------------------------------------------------------//
//! List of particle fields stored in AoSoA with associated Cajita mesh.
template <class MemorySpace, class... FieldTags>
class ParticleList : public Cabana::ParticleList<MemorySpace, FieldTags...>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Base Cabana particle list type.
    using base = Cabana::ParticleList<memory_space, FieldTags...>;

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

      Return true if the particles were actually redistributed.
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
      Return true if the particles were actually redistributed.
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

//---------------------------------------------------------------------------//
//! ParticleList creation function.
template <class MemorySpace, class... FieldTags>
auto createParticleList( const std::string& label,
                         Cabana::ParticleTraits<FieldTags...> )
{
    return ParticleList<MemorySpace, FieldTags...>( label );
}

//---------------------------------------------------------------------------//

} // namespace Cajita

#endif // end CAJITA_PARTICLELIST_HPP
