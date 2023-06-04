#ifndef CABANA_TRANSPOSE_IN_PLACE_HPP
#define CABANA_TRANSPOSE_IN_PLACE_HPP

#include <Cabana_AoSoA.hpp>
#include <Kokkos_Core.hpp>

namespace Cabana
{

template<class AoSoA_t>
void transpose_particles_from_AoS_to_AoSoA(AoSoA_t& aosoa, int ioffset, int n_vec){
    const std::size_t PTL_N_DBL = sizeof( Cabana::Tuple<AoSoA_t::data_type>)/8;

    // Pointer to local particles
    VecParticlesSimple<PTL_N_DBL>* vptl = (VecParticlesSimple<PTL_N_DBL>*)aosoa.data() + ioffset;

#ifdef USE_GPU
    int team_size = AoSoA_t::vector_length;
#else
    int team_size = 1;
#endif
    int league_size = n_vec;

    typedef Kokkos::View<double[AoSoA_t::vector_length*PTL_N_DBL],ExSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> PtlVec;
    size_t shmem_size = PtlVec::shmem_size(); // Could halve this, but may worsen memory access
    Kokkos::parallel_for("transpose_particles_from_AoS_to_AoSoA",
                         Kokkos::TeamPolicy<ExSpace> (league_size, team_size).set_scratch_size(KOKKOS_TEAM_SCRATCH_OPT,Kokkos::PerTeam(shmem_size)),
                         KOKKOS_LAMBDA (Kokkos::TeamPolicy<ExSpace>::member_type team_member){
        // Locally shared: global index in population
        PtlVec shmem_ptl_vec(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));

        int ivec = team_member.league_rank(); // vector assigned to this team
        int ith = team_member.team_rank(); // This thread's rank in the team

        // Copy vector to shared memory
        for(int idbl = ith; idbl<AoSoA_t::vector_length*PTL_N_DBL; idbl+=team_size){ // Loop through doubles
            shmem_ptl_vec(idbl) = vptl[ivec].data[idbl];
        }

        // Write transposed particles back to particle array
        for(int iptl = ith; iptl<AoSoA_t::vector_length; iptl+=team_size){ // Loop through particles 
            for(int iprop = 0; iprop<PTL_N_DBL; iprop++){ // Loop through properties
                vptl[ivec].data[iprop*AoSoA_t::vector_length + iptl] = shmem_ptl_vec(iptl*PTL_N_DBL + iprop);
            }
        }
    });

    Kokkos::fence();
}

template<class AoSoA_t>
void transpose_particles_from_AoSoA_to_AoS(AoSoA_t& aosoa, int ioffset, int n_vec){
    const std::size_t PTL_N_DBL = sizeof( Cabana::Tuple<AoSoA_t::data_type>)/8;

    // Pointer to local particles
    VecParticlesSimple<PTL_N_DBL>* vptl = (VecParticlesSimple<PTL_N_DBL>*)aosoa.data() + ioffset;

#ifdef USE_GPU
    int team_size = AoSoA_t::vector_length;
#else
    int team_size = 1;
#endif
    int league_size = n_vec;

    typedef Kokkos::View<double[AoSoA_t::vector_length*PTL_N_DBL],ExSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> PtlVec;
    size_t shmem_size = PtlVec::shmem_size(); // Could halve this, but may worsen memory access
    Kokkos::parallel_for("transpose_particles_from_AoS_to_AoSoA",
                         Kokkos::TeamPolicy<ExSpace> (league_size, team_size).set_scratch_size(KOKKOS_TEAM_SCRATCH_OPT,Kokkos::PerTeam(shmem_size)),
                         KOKKOS_LAMBDA (Kokkos::TeamPolicy<ExSpace>::member_type team_member){
        // Locally shared: global index in population
        PtlVec shmem_ptl_vec(team_member.team_scratch(KOKKOS_TEAM_SCRATCH_OPT));

        int ivec = team_member.league_rank(); // vector assigned to this team
        int ith = team_member.team_rank(); // This thread's rank in the team

        // Transpose into shared memory
        for(int iptl = ith; iptl<AoSoA_t::vector_length; iptl+=team_size){ // Loop through particles
            for(int iprop = 0; iprop<PTL_N_DBL; iprop++){ // Loop through properties
                shmem_ptl_vec(iptl*PTL_N_DBL + iprop) = vptl[ivec].data[iprop*AoSoA_t::vector_length + iptl];
            }
        }

        // Copy transposed particles back to particle array
        for(int idbl = ith; idbl<AoSoA_t::vector_length*PTL_N_DBL; idbl+=team_size){ // Loop through doubles
            vptl[ivec].data[idbl] = shmem_ptl_vec(idbl);
        }
    });

    Kokkos::fence();
}

}

#endif
