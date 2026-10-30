// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_PERIDYNAMIC_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_PERIDYNAMIC_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_io_pstream.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_sph.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class MaterialHandler;
  class ParticleContainerBundle;
  class ParticleEngineInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHPeridynamic final
  {
   public:
    //! constructor
    explicit SPHPeridynamic(const Teuchos::ParameterList& params);

    //! init peridynamic handler
    void init(const std::shared_ptr<PDNeighborPairs> neighborpairs_pd);

    //! setup peridynamic handler
    void setup(const std::shared_ptr<ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<MaterialHandler> particlematerial, MPI_Comm comm);

    //! insert peridynamic evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<Type, std::set<Particle::State>>& particlestatestotypes) const;

    //! setup peridynamic bond list
    void init_peridynamic_bondlist();

    //! compute peridynamic forces
    void add_acceleration_contribution();

    //! number of velocity Verlet substeps of the peridynamic phase per time step
    int pd_substeps() const { return pd_substeps_; }

    //! current time step size (zero in the evaluation of the initial states)
    void set_current_step_size(const double dt) { dt_ = dt; }

    /*!
     * \brief multi-rate integration of the peridynamic phase from t_n to t_n+1
     *
     * Impulse multiple time stepping (r-RESPA): the force of the fluid on the PD particles
     * (coupling force of the evaluation at t_n) acts as a half kick dt/2, the bond, contact,
     * gravity and viscous damping forces are integrated with n velocity Verlet substeps h = dt/n.
     * The force evaluation of the last substep is the evaluation of all interactions at t_n+1,
     * followed by complete_substeps(). For slow motion the scheme is identical to velocity Verlet
     * with dt; it is stable as long as omega_max dt < pi with the highest frequency omega_max of
     * the PD bonds (checked at the first call).
     */
    void integrate_substeps(double dt, const std::vector<double>& gravity);

    //! closing half kick of the multi-rate integration at t_n+1: h/2 times the acceleration of the
    //! evaluation at t_n+1 plus the remainder (dt - h)/2 of the half kick with the coupling force
    void complete_substeps(double dt) const;

    //! correction of the time integral over the last time step of the force on the boundary
    //! particles (summed as for sum_boundary_forces()) for the contact forces of the substeps, to
    //! be added to dt times the force of the evaluation at t_n+1; resets the accumulated substep
    //! contributions
    void take_boundary_impulse_correction(
        double dt, std::vector<double>& impulse_dbc_funct, std::vector<double>& impulse_other);

    //! check valid peridynamic bond
    bool check_valid_peridynamic_bond_entry(
        const int localid, const int globalid, ParticleContainer* container) const;

    //! damage evaluation of peridynamic body
    void damage_evaluation();

    //! remove the PD particles crushed at the smooth boundary surface; called before the next
    //! update of the connectivity (end of the substeps, or after the step with a single rate),
    //! which rebuilds ghosting, neighbor pairs and the bond list (bonds to removed particles are
    //! dropped)
    void remove_crushed_particles();

    //! clear force state of boundary particles (reaction of fluid and contact forces on walls)
    void clear_boundary_forces() const;

    //! sum the forces on owned boundary particles, split into particles with a per-particle
    //! Dirichlet function (e.g. a moving structure) and all other boundary particles
    void sum_boundary_forces(
        std::vector<double>& force_dbc_funct, std::vector<double>& force_other) const;

   private:
    //! calculate peridynamic volume correction factor
    double calculate_volume_correction_factor(double xi) const;

    //! check that every owned PD particle has a positive surface correction factor
    void check_surface_correction_factors() const;

    //! outward surface normals of the boundary particles for the contact with PD particles (once)
    void compute_boundary_contact_normals();

    //! contact of the owned PD particles with the smooth surface of the boundary particles: forces
    //! on the PD particles, reactions to the boundary force sums of the evaluation of all
    //! interactions (substep false) or to the contact impulse of the substeps (substep true, weight
    //! h)
    void compute_smooth_boundary_contact_forces(bool substep, double weight);

    //! Coulomb friction with an elastic tangential spring (Cundall-Strack) on particle i of a
    //! contact (direction e from i to j, normal force factor fac <= 0, stored tangential
    //! displacement xi of i relative to j, incremented with the relative drift velocity over the
    //! contact interval; nullptr: viscous regularization); returns false without friction
    bool spring_friction_force(double mu, const double* vel_i, const double* vel_j,
        const double* drift_i, const double* drift_j, const double* e, double fac, double* xi,
        double* f_t) const;

    //! friction of a contact pair (spring friction with history for PD-PD pairs if enabled)
    bool pair_friction(Particle::Type type_i, Particle::Status status_i,
        ParticleContainer* container_i, int particle_i, int globalid_i, Particle::Type type_j,
        Particle::Status status_j, ParticleContainer* container_j, int particle_j, int globalid_j,
        const double* vel_i, const double* vel_j, const double* e, double fac, double* f_t);

    //! history slot (tangential displacement) of the contact of an owned PD particle with a partner
    //! (global id); a free slot is taken if create; nullptr if not found or all slots are taken
    double* contact_history_slot(
        ParticleContainer* container, int particle, int globalid_partner, bool create);

    //! release the history slot of a contact (if any)
    void release_contact_history_slot(
        ParticleContainer* container, int particle, int globalid_partner);

    //! start / finish a contact evaluation: time since the previous contact evaluation, flag for
    //! the drift velocity of the PD particles in the state SubstepVelocity (evaluation at t_n+1 of
    //! the multi-rate integration, where Velocity is the mean velocity of the step); history slots
    //! of contacts not evaluated are released at the end
    void begin_contact_history(double interval, bool driftsubstepvelocity);
    void end_contact_history();

    //! velocity of a PD particle over the interval since the previous contact evaluation
    const double* drift_velocity(
        const ParticleContainer* container, int particle, const double* velocity) const;

    //! regularized Coulomb friction force on particle i of a contact pair (direction e from i to j,
    //! normal force factor fac <= 0); returns false (and a zero force) without friction
    bool friction_force(const Particle::Type type_i, const Particle::Type type_j,
        const double* vel_i, const double* vel_j, const double* e, const double fac,
        double* f_t) const;

    //! add a force acting on a boundary particle to the boundary force sums
    void add_boundary_force_vector(const ParticleContainer* container, int particle,
        const double* force, double weight, std::vector<double>& sum_dbc_funct,
        std::vector<double>& sum_other) const;

    //! direction of the contact force of a pair (from particle i to j): the line between the
    //! particle centres or, for PD-boundary pairs, the boundary surface normal
    void contact_direction(const Particle::Type type_i,
        const Particle::ParticleContainer* container_i, const int particle_i,
        const Particle::Type type_j, const Particle::ParticleContainer* container_j,
        const int particle_j, const double* e_ji, double* e) const;

    //! calculate force for rigid phase
    void compute_interaction_forces();

    //! calculate the bond forces and break bonds beyond the critical stretch
    void compute_bond_forces() const;

    //! calculate the contact forces of the contact candidates at the current positions (substeps)
    void compute_substep_contact_forces(double weight);

    //! add a contact force on an owned boundary particle to the sums of the boundary force output
    void add_boundary_contact_force(const ParticleContainer* container, int particle, double fac,
        const double* e_ji, double weight, std::vector<double>& sum_dbc_funct,
        std::vector<double>& sum_other) const;

    //! set the acceleration of the owned PD particles in a substep and clear their force
    void set_substep_acceleration(const std::vector<double>& gravity) const;

    //! check the time step sizes of the multi-rate integration against the highest bond frequency
    void check_substep_stability(double dt) const;

    //! calculate acceleration for rigid phase
    void compute_acceleration() const;

    //! communicator
    MPI_Comm comm_ = MPI_COMM_NULL;

    //! interface to particle engine
    std::shared_ptr<ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! particle material handler
    std::shared_ptr<MaterialHandler> particlematerial_;

    //! neighbor pair handler for PD and for DEM like interaction
    std::shared_ptr<PDNeighborPairs> neighborpairs_pd_;

    //! bond list for PD bodies
    std::shared_ptr<std::vector<std::pair<LocalGlobalIndexTuple, LocalGlobalIndexTuple>>> bondlist_;

    //! peridynamic interaction horizon
    const double horizon_pd_;

    //! peridynamic grid spacing
    const double dx_pd_;

    //! contact stiffness
    const double stiff_;

    //! contact damping parameter
    const double damp_;

    //! peridynamic dimension
    const PeridynamicDimension peridynamic_dimension_;

    //! surface correction of the bond micromoduli by per-particle factors (bond: mean of both)
    const bool surface_correction_;

    //! flag indicating the check of the surface correction factors
    mutable bool checkedsurfacecorrection_ = false;

    //! number of velocity Verlet substeps of the peridynamic phase per time step
    const int pd_substeps_;

    //! viscous damping factor (applied in the substeps as by the viscous damping handler)
    const double viscdampfac_;

    //! skin of the contact candidates of the substeps
    const double contact_candidate_skin_;

    //! contact force between PD and boundary particles along the boundary surface normal
    const bool boundary_contact_normal_;

    //! Coulomb friction coefficients of PD-PD and PD-boundary contact, tangential regularization
    const double friction_pd_;
    const double friction_boundary_;
    const double friction_damp_;

    //! contact between PD particles and the smooth boundary surface, radius of its weight function
    const bool boundary_contact_smooth_;
    const double smooth_surface_radius_;

    //! Coulomb friction with an elastic tangential spring: flag, tangential stiffness
    const bool friction_spring_;
    const double friction_kt_;

    //! crushing at the smooth boundary surface: crushing pressure sigma_p (0: off) and the limit
    //! sigma_p dx^(d-1) of the normal contact force on a PD particle (2D: per unit thickness)
    const double crushing_pressure_;
    const double crushing_force_;

    //! crushing: global ids of the owned PD particles crushed in the last evaluation of all
    //! interactions (removed before the next update of the connectivity), totals on this processor
    //! of the removed particles, their mass and kinetic energy and the plastic work of the contact
    std::vector<int> crushedglobalids_;
    int crushedcount_ = 0;
    double crushedmass_ = 0.0;
    double crushedkinetic_ = 0.0;
    double plasticwork_ = 0.0;

    //! current time step size, time since the previous contact evaluation and drift velocity flag
    //! of the current contact evaluation (begin_contact_history)
    double dt_ = 0.0;
    double contact_interval_ = 0.0;
    bool contact_drift_substep_ = false;

    //! history slots of the owned PD particles evaluated in the current contact evaluation (bit
    //! mask)
    std::vector<std::uint16_t> historytouched_;

    //! reaction of the contact with the smooth boundary surface in the last evaluation of all
    //! interactions (owned PD particles of this processor), part of the boundary force sums
    std::vector<double> smoothcontactforce_dbc_funct_ = std::vector<double>(3, 0.0);
    std::vector<double> smoothcontactforce_other_ = std::vector<double>(3, 0.0);

    //! sums of the smooth boundary surface of the owned PD particles near the boundary (reused)
    std::vector<int> smoothslot_;
    std::vector<std::array<double, 15>> smoothsums_;

    //! flag indicating the computation of the boundary surface normals
    bool computedboundarynormals_ = false;

    //! flag indicating the stability check of the multi-rate integration
    bool checkedsubstepstability_ = false;

    //! time integral of the contact forces on boundary particles over the substeps of this step
    std::vector<double> substepcontactimpulse_dbc_funct_ = std::vector<double>(3, 0.0);
    std::vector<double> substepcontactimpulse_other_ = std::vector<double>(3, 0.0);

    //! contact force on boundary particles of the last evaluation of all interactions
    std::vector<double> contactforce_dbc_funct_ = std::vector<double>(3, 0.0);
    std::vector<double> contactforce_other_ = std::vector<double>(3, 0.0);

    //! pre-crack line segments: each is {x1, y1, z1, x2, y2, z2}
    std::vector<std::array<double, 6>> pre_crack_lines_;

    //! pre-crack parallelogram patches: each is {p0x,p0y,p0z, p1x,p1y,p1z, p2x,p2y,p2z}
    std::vector<std::array<double, 9>> pre_crack_planes_;

    //! check if the bond from pos_i to pos_j crosses any pre-crack line segment or plane
    bool bond_crosses_pre_crack(const double* pos_i, const double* pos_j) const;
  };

}  // namespace Particle
FOUR_C_NAMESPACE_CLOSE

#endif
