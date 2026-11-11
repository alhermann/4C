// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_PD_NEIGHBOR_PAIRS_HPP
#define FOUR_C_PARTICLE_INTERACTION_PD_NEIGHBOR_PAIRS_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_pd_neighbor_pair_struct.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleEngineInterface;
  class ParticleContainerBundle;
  class WallHandlerInterface;
}  // namespace Particle

namespace Core::IO
{
  class DiscretizationReader;
}

/*---------------------------------------------------------------------------*
 | type definitions                                                          |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  using PDParticlePairData = std::vector<Particle::PDParticlePair>;
  using PDParticleWallPairData = std::vector<Particle::PDParticleWallPair>;
  using PDContactCandidateData = std::vector<Particle::PDContactCandidate>;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class PDNeighborPairs final
  {
   public:
    //! constructor
    explicit PDNeighborPairs(const MPI_Comm& comm, const Teuchos::ParameterList& params_pd);

    //! setup neighbor pair handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface);

    //! get reference to particle pair data
    inline const PDParticlePairData& get_ref_to_particle_pair_data() const
    {
      return particlepairdata_;
    };

    //! collect non-bonded particle pairs with a gap below the skin as contact candidates in each
    //! evaluation of the particle pairs (multi-rate integration; skin <= 0: no candidates)
    void set_contact_candidate_skin(const double skin) { contact_candidate_skin_ = skin; }

    //! contact candidates of the last evaluation of the particle pairs
    inline const PDContactCandidateData& get_ref_to_contact_candidates() const
    {
      return contactcandidates_;
    }

    //! collect the pairs of an owned PD particle and a boundary particle within the radius (plus
    //! the skin of the contact candidates) separately for the contact with the smooth boundary
    //! surface; these pairs are then no regular contact pairs or candidates (radius <= 0: off)
    void set_boundary_contact_radius(const double radius) { boundary_contact_radius_ = radius; }

    //! pairs of an owned PD particle (particle i) and a boundary particle (particle j) of the last
    //! evaluation of the particle pairs (contact with the smooth boundary surface)
    inline const PDContactCandidateData& get_ref_to_boundary_contact_pairs() const
    {
      return boundarycontactpairs_;
    }

    //! true once the particle pairs were evaluated (bond list resolved, contact candidates set)
    inline bool have_evaluated_particle_pairs() const { return haveevaluatedparticlepairs_; }

    //! set pd bond list
    void set_bond_list(const std::shared_ptr<
        std::vector<std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>>>
            bondlist)
    {
      bondlist_ = bondlist;
    }

    //! evaluate neighbor pairs
    void evaluate_neighbor_pairs();

    //! communicate bond list
    void communicate_bond_list(const std::vector<std::vector<int>>& particletargets);

    //! write restart
    void write_restart() const;

    //! read restart
    void read_restart(const std::shared_ptr<Core::IO::DiscretizationReader> reader);

    //! write the local bond list of this processor to a text file (consistency checks)
    void write_bond_list(const std::string& filename) const;

   private:
    //! number of intact bonds of every owned PD particle from the local bond list
    void count_bonds_of_owned_particles() const;

    //! evaluate particle pairs
    void evaluate_particle_pairs();

    //! create the hash keys for pd bond pairs
    void setup_peridynamic_pair_hashes();

    //! evaluate particle-wall pairs
    void evaluate_particle_wall_pairs();

    //! pack bond list in the buffer
    void pack_bond_list_pairs(Core::Communication::PackBuffer& buffer) const;

    //! unpack peridynamic bond list data
    void unpack_peridynamic_bond_list_data(
        const std::vector<char>& buffer, std::set<long>& known_bond_hashes);

    //! reference to bond list
    std::shared_ptr<
        std::vector<std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>>>
        bondlist_;

    //! map the hashkey to its corresponding entry in the bond list
    std::unordered_map<long, size_t> map_hashkey_to_bondlist_entry_;

    //! particle pair data with evaluated quantities
    PDParticlePairData particlepairdata_;

    //! particle-wall pair data with evaluated quantities
    PDParticleWallPairData particlewallpairdata_;

    //! contact candidates (non-bonded particle pairs with a gap below the skin)
    PDContactCandidateData contactcandidates_;

    //! skin of the contact candidates
    double contact_candidate_skin_ = 0.0;

    //! pairs of an owned PD particle and a boundary particle (contact with the smooth surface)
    PDContactCandidateData boundarycontactpairs_;

    //! radius of the pairs of PD and boundary particles for the smooth boundary surface
    double boundary_contact_radius_ = 0.0;

    //! flag indicating an evaluation of the particle pairs
    bool haveevaluatedparticlepairs_ = false;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface_;

    //! communicator
    const MPI_Comm& comm_;

    //! peridynamic grid spacing
    const double peridynamic_grid_spacing_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
