/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
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
template <class MeshType, class... FieldTags>
class MeshParticleList
    : public Cabana::ParticleList<typename MeshType::memory_space, FieldTags...>
{
  public:
    //! Kokkos memory space.
    using memory_space = typename MeshType::memory_space;
    //! Base Cabana particle list type.
    using base = Cabana::ParticleList<memory_space, FieldTags...>;
    //! Cajita mesh type.
    using mesh_type = MeshType;

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
    MeshParticleList( const std::string& label,
                      const std::shared_ptr<MeshType>& mesh )
        : base( label )
        , _mesh( mesh )
    {
    }

    //! Get the mesh.
    const MeshType& mesh() { return *_mesh; }

    /*!
      \brief Redistribute particles to new owning grids.

      Return true if the particles were actually redistributed.
    */
    bool redistribute( const bool force_redistribute = false )
    {
        return particleGridMigrate(
            *( _mesh->localGrid() ),
            this->slice( Cabana::Field::Position<mesh_type::num_space_dim>() ),
            _aosoa, _mesh->minimumHaloWidth(), force_redistribute );
    }

  protected:
    //! Particle AoSoA.
    using base::_aosoa;
    //! Cajita mesh.
    std::shared_ptr<MeshType> _mesh;
};

//---------------------------------------------------------------------------//
//! MeshParticleList creation function.
template <class Mesh, class... FieldTags>
std::shared_ptr<MeshParticleList<Mesh, FieldTags...>>
createMeshParticleList( const std::string& label,
                        const std::shared_ptr<Mesh>& mesh,
                        Cabana::ParticleTraits<FieldTags...> )
{
    return std::make_shared<MeshParticleList<Mesh, FieldTags...>>( label,
                                                                   mesh );
}

//---------------------------------------------------------------------------//

} // namespace Cajita

#endif // end CAJITA_PARTICLELIST_HPP
