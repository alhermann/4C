// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_peridynamic.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_dem_neighbor_pairs.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_pd_neighbor_pairs.hpp"
#include "4C_particle_interaction_sph.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>

FOUR_C_NAMESPACE_OPEN

// forward declarations of file-local geometry helpers
namespace
{
  bool intervals_overlap(double min_a, double max_a, double min_b, double max_b);
  bool bond_aabb_overlaps_segment(
      const double* pos_i, const double* pos_j, const std::array<double, 6>& segment);
  bool bond_aabb_overlaps_patch(
      const double* pos_i, const double* pos_j, const std::array<double, 9>& patch);

  // unit eigenvector of the smallest eigenvalue of the symmetric 3x3 matrix a (a00, a11, a22, a01,
  // a02, a12); false for points on a line (middle eigenvalue not separated from zero) or an
  // isotropic or degenerate matrix
  bool smallest_principal_axis(const std::array<double, 6>& a, double* axis)
  {
    const double p1 = a[3] * a[3] + a[4] * a[4] + a[5] * a[5];
    const double q = (a[0] + a[1] + a[2]) / 3.0;
    const double p2 =
        (a[0] - q) * (a[0] - q) + (a[1] - q) * (a[1] - q) + (a[2] - q) * (a[2] - q) + 2.0 * p1;
    if (p2 <= 0.0) return false;
    const double p = std::sqrt(p2 / 6.0);
    const double b00 = (a[0] - q) / p, b11 = (a[1] - q) / p, b22 = (a[2] - q) / p;
    const double b01 = a[3] / p, b02 = a[4] / p, b12 = a[5] / p;
    const double detb = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) +
                        b02 * (b01 * b12 - b11 * b02);
    const double phi = std::acos(std::clamp(0.5 * detb, -1.0, 1.0)) / 3.0;
    const double lmax = q + 2.0 * p * std::cos(phi);
    const double lmin = q + 2.0 * p * std::cos(phi + 2.0 * std::numbers::pi / 3.0);
    const double lmid = 3.0 * q - lmax - lmin;
    if (lmid <= 1.0e-6 * lmax) return false;

    // eigenvector of lmin: largest cross product of two rows of a - lmin I
    const double r[3][3] = {
        {a[0] - lmin, a[3], a[4]}, {a[3], a[1] - lmin, a[5]}, {a[4], a[5], a[2] - lmin}};
    double best = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      const double* u = r[k];
      const double* v = r[(k + 1) % 3];
      const double c[3] = {
          u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
      const double norm = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
      if (norm > best)
      {
        best = norm;
        for (int dim = 0; dim < 3; ++dim) axis[dim] = c[dim] / norm;
      }
    }
    return best > 1.0e-12 * lmax * lmax;
  }
}  // namespace

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHPeridynamic::SPHPeridynamic(const Teuchos::ParameterList& particle_params)
    : bondlist_(std::make_shared<std::vector<
              std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>>>()),
      horizon_pd_(particle_params.sublist("PD").get<double>("INTERACTION_HORIZON")),
      dx_pd_(particle_params.sublist("PD").get<double>("PERIDYNAMIC_GRID_SPACING")),
      stiff_(particle_params.sublist("PD").get<double>("NORMAL_STIFF")),
      damp_(particle_params.sublist("PD").get<double>("NORMAL_DAMP")),
      peridynamic_dimension_(Teuchos::getIntegralValue<Particle::PeridynamicDimension>(
          particle_params.sublist("PD"), "PD_DIMENSION")),
      surface_correction_(particle_params.sublist("PD").get<bool>("SURFACE_CORRECTION")),
      pd_substeps_(particle_params.sublist("PD").get<int>("PD_SUBSTEPS")),
      viscdampfac_(particle_params.get<double>("VISCOUS_DAMPING")),
      contact_candidate_skin_(0.1 * dx_pd_),
      boundary_contact_normal_(
          particle_params.sublist("PD").get<bool>("BOUNDARY_CONTACT_SURFACE_NORMAL")),
      friction_pd_(particle_params.sublist("PD").get<double>("FRICTION_COEFF")),
      friction_boundary_(particle_params.sublist("PD").get<double>("FRICTION_COEFF_BOUNDARY")),
      friction_damp_(particle_params.sublist("PD").get<double>("FRICTION_TANGENTIAL_DAMP")),
      boundary_contact_smooth_(
          particle_params.sublist("PD").get<bool>("BOUNDARY_CONTACT_SMOOTH_SURFACE")),
      smooth_surface_radius_(2.0 * dx_pd_)
{
  FOUR_C_ASSERT_ALWAYS(friction_pd_ >= 0.0 and friction_boundary_ >= 0.0 and friction_damp_ >= 0.0,
      "PD contact: FRICTION_COEFF, FRICTION_COEFF_BOUNDARY and FRICTION_TANGENTIAL_DAMP must not "
      "be negative!");
  FOUR_C_ASSERT_ALWAYS((friction_pd_ == 0.0 and friction_boundary_ == 0.0) or friction_damp_ > 0.0,
      "PD contact: Coulomb friction requires FRICTION_TANGENTIAL_DAMP > 0 (regularization)!");

  FOUR_C_ASSERT_ALWAYS(not particle_params.get<bool>("RIGID_BODY_MOTION"),
      "Peridynamic interaction is not available in combination with rigid body motion!");

  // A mismatch produces physically inconsistent forces
  const double dx_sph = particle_params.sublist("SPH").get<double>("INITIALPARTICLESPACING");
  FOUR_C_ASSERT_ALWAYS(abs(dx_pd_ - dx_sph) <= 1.0e-10 * dx_pd_,
      "PERIDYNAMIC_GRID_SPACING ({:.10f}) must equal INITIALPARTICLESPACING ({:.10f}). ", dx_pd_,
      dx_sph);

  auto normalcontacttype = Teuchos::getIntegralValue<Particle::NormalContact>(
      particle_params.sublist("PD"), "NORMALCONTACTLAW");

  if (normalcontacttype == Particle::NormalLinSpring and damp_ != 0.0)
    FOUR_C_THROW(
        "NORMAL_DAMP in PARTICLE DYNAMIC/PD section needs to be 0.0 if NORMALCONTACTLAW is set "
        "to NormalLinearSpring!");

  FOUR_C_ASSERT_ALWAYS(
      horizon_pd_ > 0.0, "Peridynamic INTERACTION_HORIZON must be greater than zero!");

  // to ensure that slightly stretched bonds are still found properly in the neighbor search, the
  // interaction horizon needs to be smaller than the minimum bin size of the binning strategy
  // The factor 1.1 is physics motivated as usual critical stretch values are (way) smaller
  Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
  const double minimum_bin_size = binning_params.get<double>("BIN_SIZE_LOWER_BOUND");
  FOUR_C_ASSERT_ALWAYS(horizon_pd_ * 1.1 < minimum_bin_size,
      "Peridynamic INTERACTION_HORIZON * 1.1(safety factor to account for stretch) must be smaller "
      "than BIN_SIZE_LOWER_BOUND!");
  FOUR_C_ASSERT_ALWAYS(not boundary_contact_smooth_ or
                           smooth_surface_radius_ + contact_candidate_skin_ < minimum_bin_size,
      "BOUNDARY_CONTACT_SMOOTH_SURFACE: the radius 2 PERIDYNAMIC_GRID_SPACING of the smooth "
      "surface plus the skin of the contact candidates must be smaller than BIN_SIZE_LOWER_BOUND!");

  // parse pre-crack line segments (3D): each entry has START and END (3 doubles each)
  {
    const Teuchos::ParameterList& pd_params = particle_params.sublist("PD");
    if (pd_params.isParameter("PRE_CRACK_LINES"))
    {
      const auto& lines = pd_params.get<std::vector<Teuchos::ParameterList>>("PRE_CRACK_LINES");
      for (const auto& entry : lines)
      {
        const auto& s = entry.get<std::vector<double>>("START");
        const auto& e = entry.get<std::vector<double>>("END");
        pre_crack_lines_.push_back({s[0], s[1], s[2], e[0], e[1], e[2]});
      }
      Core::IO::cout << "Number of pre-crack line segments: " << pre_crack_lines_.size()
                     << Core::IO::endl;
    }
  }

  // parse pre-crack parallelogram patches (3D): each entry has P0, P1, P2 (3 doubles each)
  {
    const Teuchos::ParameterList& pd_params = particle_params.sublist("PD");
    if (pd_params.isParameter("PRE_CRACK_PLANES"))
    {
      const auto& planes = pd_params.get<std::vector<Teuchos::ParameterList>>("PRE_CRACK_PLANES");
      for (const auto& entry : planes)
      {
        const auto& p0 = entry.get<std::vector<double>>("P0");
        const auto& p1 = entry.get<std::vector<double>>("P1");
        const auto& p2 = entry.get<std::vector<double>>("P2");
        pre_crack_planes_.push_back(
            {p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]});
      }
      Core::IO::cout << "Number of pre-crack planes: " << pre_crack_planes_.size()
                     << Core::IO::endl;
    }
  }

  // checks for the dimensionality of the problem
  {
    const auto constraint_type = particle_params.sublist("INITIAL AND BOUNDARY CONDITIONS")
                                     .get<Particle::Constraint>("CONSTRAINT");

    if (peridynamic_dimension_ == PeridynamicDimension::Peridynamic_2DPlaneStress ||
        peridynamic_dimension_ == PeridynamicDimension::Peridynamic_2DPlaneStrain)
    {
      FOUR_C_ASSERT_ALWAYS(constraint_type == Particle::Projection2D,
          "Plane stress or plane strain for peridynamic requested. CONSTRAINT must be set to "
          "Projection2D!");
    }

    if (constraint_type == Particle::Projection2D)
    {
      FOUR_C_ASSERT_ALWAYS(peridynamic_dimension_ != PeridynamicDimension::Peridynamic_3D,
          "Projection2D CONSTRAINT is active. Choose 2D_PlaneStress or 2D_PlaneStrain as "
          "PD_DIMENSION!");
    }
  }
}

void Particle::SPHPeridynamic::init(
    const std::shared_ptr<Particle::PDNeighborPairs> neighborpairs_pd)
{
  // set bond list to pd neighbor pairs here because read_restart requires already bond list
  neighborpairs_pd->set_bond_list(bondlist_);

  // multi-rate integration: contact candidates for the substeps
  if (pd_substeps_ > 1) neighborpairs_pd->set_contact_candidate_skin(contact_candidate_skin_);

  // contact with the smooth boundary surface: pairs of PD and boundary particles within its radius
  if (boundary_contact_smooth_)
    neighborpairs_pd->set_boundary_contact_radius(smooth_surface_radius_);

  neighborpairs_pd_ = neighborpairs_pd;
}

void Particle::SPHPeridynamic::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::MaterialHandler> particlematerial, MPI_Comm comm)
{
  // set communicator
  comm_ = comm;

  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set particle material handler
  particlematerial_ = particlematerial;
}

void Particle::SPHPeridynamic::insert_particle_states_of_particle_types(
    std::map<Particle::Type, std::set<Particle::State>>& particlestatestotypes) const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    if (typeIt.first == Particle::Type::PDPhase)
    {
      // set of particle states for current particle type
      std::set<Particle::State>& particlestates = typeIt.second;

      // set temperature state
      particlestates.insert({Particle::State::Force, Particle::State::PDBodyId,
          Particle::State::ReferencePosition, Particle::State::Young,
          Particle::State::CriticalStretch, Particle::State::InitialConnectedBonds,
          Particle::State::CurrentConnectedBonds, Particle::State::PDDamageVariable});

      // force of the fluid on the PD particles for the half kicks of the multi-rate integration
      if (pd_substeps_ > 1)
        particlestates.insert({Particle::State::CouplingForce, Particle::State::SubstepVelocity});

      // surface correction factors
      if (surface_correction_) particlestates.insert(Particle::State::PDSurfaceCorrection);
    }
    else if (typeIt.first == Particle::Type::BoundaryPhase)
    {
      // force on boundary particles: reaction of the fluid and of PD contact (output only)
      typeIt.second.insert(Particle::State::Force);

      // outward surface normal for the contact with PD particles
      if (boundary_contact_normal_ or boundary_contact_smooth_)
        typeIt.second.insert(Particle::State::BoundaryContactNormal);
    }
  }
}

void Particle::SPHPeridynamic::init_peridynamic_bondlist()
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  // get material for peridynamic phase
  const Mat::PAR::ParticleMaterialBase* material =
      particlematerial_->get_ptr_to_particle_mat_parameter(Particle::Type::PDPhase);

  // (initial) radius of current phase
  const double initradius = material->initRadius_;
#endif

  // important: bin size must be large enough to cover at least peridynamic horizon
  const Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // iterate over potential particle neighbors
  for (const auto& potentialneighbors :
      particleengineinterface_->get_potential_particle_neighbors())
  {
    // access values of local index tuples of particle i and j
    Particle::Type type_i;
    Particle::Status status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

    Particle::Type type_j;
    Particle::Status status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = potentialneighbors.second;

    // only peridynamic phase particles can undergo peridynamic interaction
    if (type_i != Particle::Type::PDPhase || type_j != Particle::Type::PDPhase) continue;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle->get_specific_container(type_i, status_i);
    Particle::ParticleContainer* container_j =
        particlecontainerbundle->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(Particle::State::Position, particle_i);
    const double* pdbodyid_i = container_i->get_ptr_to_state(Particle::State::PDBodyId, particle_i);
    double* initialconnectedbonds_i =
        container_i->get_ptr_to_state_writable(Particle::State::InitialConnectedBonds, particle_i);
    double* currentconnectedbonds_i =
        container_i->get_ptr_to_state_writable(Particle::State::CurrentConnectedBonds, particle_i);

    const double* pos_j = container_j->get_ptr_to_state(Particle::State::Position, particle_j);
    const double* pdbodyid_j = container_j->get_ptr_to_state(Particle::State::PDBodyId, particle_j);
    double* initialconnectedbonds_j =
        container_j->get_ptr_to_state_writable(Particle::State::InitialConnectedBonds, particle_j);
    double* currentconnectedbonds_j =
        container_j->get_ptr_to_state_writable(Particle::State::CurrentConnectedBonds, particle_j);

    // vector from particle i to j
    double r_ji[3];
    // distance between particles considering periodic boundaries
    particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);
    // absolute distance between particles
    const double absdist = ParticleUtils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (absdist < (1.0e-10 * initradius) or absdist < (1.0e-10 * initradius))
      FOUR_C_THROW("absolute distance %f between particles close to zero!", absdist);
#endif

    // neighboring particles within interaction distance
    if (absdist <= horizon_pd_)
    {
      // initialize particle pair
      const int id_i = std::round(pdbodyid_i[0]);
      const int id_j = std::round(pdbodyid_j[0]);
      if (id_j == id_i)
      {
        // always count towards initial bonds (including pre-cracked)
        initialconnectedbonds_i[0] += 1.0;
        initialconnectedbonds_j[0] += 1.0;

        // bonds crossing a pre-crack are initially broken
        if (bond_crosses_pre_crack(pos_i, pos_j)) continue;

        // count active (non-pre-cracked) bonds as current
        currentconnectedbonds_i[0] += 1.0;
        currentconnectedbonds_j[0] += 1.0;

        // get global id of particle i
        const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);
        const int* globalid_j = container_j->get_ptr_to_global_id(particle_j);

        // creat the content of the bond between particles i & j
        Particle::LocalGlobalIndexTuple tuple_i =
            std::make_tuple(type_i, status_i, particle_i, globalid_i[0]);
        Particle::LocalGlobalIndexTuple tuple_j =
            std::make_tuple(type_j, status_j, particle_j, globalid_j[0]);

        bondlist_->push_back(std::make_pair(tuple_i, tuple_j));
      }
    }
  }

  Core::IO::cout << "Number of initialized peridynamic bonds on this proc: " << bondlist_->size()
                 << Core::IO::endl;
}

void Particle::SPHPeridynamic::check_surface_correction_factors() const
{
  // the factors are read with the particles (per-particle input SURFACE_CORRECTION) or from a
  // restart; a missing factor (zero) would switch its bonds off
  Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
      Particle::Type::PDPhase, Particle::Status::Owned);
  for (int i = 0; i < container->particles_stored(); ++i)
  {
    const double factor = container->get_ptr_to_state(Particle::State::PDSurfaceCorrection, i)[0];
    if (not(factor > 0.0))
      FOUR_C_THROW(
          "PD particle with global id {} has the surface correction factor {}: SURFACE_CORRECTION "
          "requires a positive factor for every PD particle (particle input SURFACE_CORRECTION)!",
          container->get_ptr_to_global_id(i)[0], factor);
  }
}

void Particle::SPHPeridynamic::add_acceleration_contribution()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::add_acceleration_contribution");

  // multi-rate integration: the force state holds the force of the fluid at this point
  if (pd_substeps_ > 1)
    particlecontainerbundle_
        ->get_specific_container(Particle::Type::PDPhase, Particle::Status::Owned)
        ->update_state(0.0, Particle::State::CouplingForce, 1.0, Particle::State::Force);

  // calculate peridynamic and short range DEM interaction forces
  compute_interaction_forces();

  // calculate acceleration from forces
  compute_acceleration();

  // clear force of peridynamic phase particles
  Particle::ParticleContainer* container =
      particleengineinterface_->get_particle_container_bundle()->get_specific_container(
          Particle::Type::PDPhase, Particle::Status::Owned);
  container->clear_state(Particle::State::Force);
}

void Particle::SPHPeridynamic::compute_interaction_forces()
{
  // calculate the bond forces
  compute_bond_forces();

  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::contact_forces");

  if ((boundary_contact_normal_ or boundary_contact_smooth_) and not computedboundarynormals_)
    compute_boundary_contact_normals();

  const Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // multi-rate integration: contact force on boundary particles of this evaluation
  if (pd_substeps_ > 1)
  {
    contactforce_dbc_funct_.assign(3, 0.0);
    contactforce_other_.assign(3, 0.0);
  }

  const PDParticlePairData& pd_neighbor_pairs = neighborpairs_pd_->get_ref_to_particle_pair_data();
  for (const auto& particlepair : pd_neighbor_pairs)
  {
    // access values of local index tuples of particle i and j
    Particle::Type type_i;
    Particle::Status status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    Particle::Type type_j;
    Particle::Status status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* vel_i = container_i->get_ptr_to_state(Particle::State::Velocity, particle_i);
    double* force_i =
        container_i->try_get_ptr_to_state_writable(Particle::State::Force, particle_i);

    // get pointer to particle states
    const double* vel_j = container_j->get_ptr_to_state(Particle::State::Velocity, particle_j);

    double* force_j = nullptr;
    if (status_j == Particle::Status::Owned)
      force_j = container_j->try_get_ptr_to_state_writable(Particle::State::Force, particle_j);

    // direction of the contact force (from i to j)
    double e[3];
    contact_direction(
        type_i, container_i, particle_i, type_j, container_j, particle_j, particlepair.e_ji_, e);

    // compute normal gap and rate of normal gap
    const double gap = particlepair.gap_;
    const double gapdot =
        -1.0 * (ParticleUtils::vec_dot(vel_i, e) - ParticleUtils::vec_dot(vel_j, e));

    // magnitude of peridynamic particle contact force
    const double fac = std::min(0.0, (stiff_ * gap + damp_ * gapdot));

    // regularized Coulomb friction force on particle i (none without friction)
    double f_t[3];
    const bool friction = friction_force(type_i, type_j, vel_i, vel_j, e, fac, f_t);

    // add contributions
    if (force_i) ParticleUtils::vec_add_scale(force_i, fac, e);
    if (force_j) ParticleUtils::vec_add_scale(force_j, -fac, e);
    if (friction)
    {
      if (force_i) ParticleUtils::vec_add(force_i, f_t);
      if (force_j) ParticleUtils::vec_sub(force_j, f_t);
    }

    // multi-rate integration: contact force on owned boundary particles
    if (pd_substeps_ > 1)
    {
      double f_t_j[3];
      ParticleUtils::vec_set_scale(f_t_j, -1.0, f_t);
      if (type_i == Particle::Type::BoundaryPhase and status_i == Particle::Status::Owned)
      {
        add_boundary_contact_force(
            container_i, particle_i, fac, e, 1.0, contactforce_dbc_funct_, contactforce_other_);
        if (friction)
          add_boundary_force_vector(
              container_i, particle_i, f_t, 1.0, contactforce_dbc_funct_, contactforce_other_);
      }
      if (type_j == Particle::Type::BoundaryPhase and status_j == Particle::Status::Owned)
      {
        add_boundary_contact_force(
            container_j, particle_j, -fac, e, 1.0, contactforce_dbc_funct_, contactforce_other_);
        if (friction)
          add_boundary_force_vector(
              container_j, particle_j, f_t_j, 1.0, contactforce_dbc_funct_, contactforce_other_);
      }
    }
  }

  // contact of PD particles with the smooth boundary surface
  if (boundary_contact_smooth_) compute_smooth_boundary_contact_forces(false, 1.0);
}

void Particle::SPHPeridynamic::compute_bond_forces() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::compute_bond_forces");

  if (surface_correction_ and not checkedsurfacecorrection_)
  {
    check_surface_correction_factors();
    checkedsurfacecorrection_ = true;
  }

  const Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  size_t iter = 0;
  while (iter < bondlist_->size())
  {
    const auto& particlepair = (*bondlist_)[iter];

    // access values of local index tuples of particle i and j
    Particle::Type type_i;
    Particle::Status status_i;
    int particle_i, globalid_i;
    std::tie(type_i, status_i, particle_i, globalid_i) = particlepair.first;

    Particle::Type type_j;
    Particle::Status status_j;
    int particle_j, globalid_j;
    std::tie(type_j, status_j, particle_j, globalid_j) = particlepair.second;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle->get_specific_container(type_j, status_j);

    //  get pointer to particle states
    const double* ref_pos_i =
        container_i->get_ptr_to_state(Particle::State::ReferencePosition, particle_i);

    const double* pos_i = container_i->get_ptr_to_state(Particle::State::Position, particle_i);
    const double* young_i = container_i->get_ptr_to_state(Particle::State::Young, particle_i);
    double* force_i =
        container_i->try_get_ptr_to_state_writable(Particle::State::Force, particle_i);
    const double* critical_stretch_i =
        container_i->get_ptr_to_state(Particle::State::CriticalStretch, particle_i);

    const double* ref_pos_j =
        container_j->get_ptr_to_state(Particle::State::ReferencePosition, particle_j);
    const double* pos_j = container_j->get_ptr_to_state(Particle::State::Position, particle_j);
    const double* young_j = container_j->get_ptr_to_state(Particle::State::Young, particle_j);
    double* force_j = nullptr;
    if (status_j == Particle::Status::Owned)
      force_j = container_j->get_ptr_to_state_writable(Particle::State::Force, particle_j);

    const double* critical_stretch_j =
        container_j->get_ptr_to_state(Particle::State::CriticalStretch, particle_j);
    // calculate the bond between two particles
    double xi[3];
    ParticleUtils::vec_set(xi, ref_pos_j);
    ParticleUtils::vec_sub(xi, ref_pos_i);

    // calculate the relative position of the pair
    double xi_eta[3];
    ParticleUtils::vec_set(xi_eta, pos_j);
    ParticleUtils::vec_sub(xi_eta, pos_i);

    // calculate the required norms of the pair
    const double xi_norm = ParticleUtils::vec_norm_two(xi);
    const double xi_eta_norm = ParticleUtils::vec_norm_two(xi_eta);

    // calculate the bond stretch
    double stretch = (xi_eta_norm - xi_norm) / xi_norm;

    // check the stretch
    const double critical_stretch = 0.5 * (critical_stretch_i[0] + critical_stretch_j[0]);

    // if critical stretch is not reached
    if (stretch < critical_stretch)
    {
      double m[3];
      ParticleUtils::vec_set_scale(m, 1.0 / xi_eta_norm, xi_eta);

      // partial volume factor of the bond; with the surface correction the mean of the factors of
      // both particles (operator-matching weights, which contain the correction of the integration
      // near the horizon edge, so the partial volume factor is not applied in addition)
      const double volume_factor =
          surface_correction_ ? 0.5 * (container_i->get_ptr_to_state(
                                           Particle::State::PDSurfaceCorrection, particle_i)[0] +
                                          container_j->get_ptr_to_state(
                                              Particle::State::PDSurfaceCorrection, particle_j)[0])
                              : calculate_volume_correction_factor(xi_norm);

      // calculate the bond force of the pair
      double fac;
      switch (peridynamic_dimension_)
      {
        case PeridynamicDimension::Peridynamic_3D:
          fac = (12.0 * (young_i[0] + young_j[0]) * 0.5) /
                (std::numbers::pi * horizon_pd_ * horizon_pd_ * horizon_pd_ * horizon_pd_) *
                stretch * volume_factor * dx_pd_ * dx_pd_ * dx_pd_ * dx_pd_ * dx_pd_ * dx_pd_;
          break;
        case PeridynamicDimension::Peridynamic_2DPlaneStress:
          fac = (9.0 * (young_i[0] + young_j[0]) * 0.5) /
                (std::numbers::pi * horizon_pd_ * horizon_pd_ * horizon_pd_) * stretch *
                volume_factor * dx_pd_ * dx_pd_ * dx_pd_ * dx_pd_;
          break;
        case PeridynamicDimension::Peridynamic_2DPlaneStrain:
          fac = (48.0 * (young_i[0] + young_j[0]) * 0.5) /
                (std::numbers::pi * 5.0 * horizon_pd_ * horizon_pd_ * horizon_pd_) * stretch *
                volume_factor * dx_pd_ * dx_pd_ * dx_pd_ * dx_pd_;
          break;
        default:
          std23::unreachable();
      }

      // add bond force contribution
      ParticleUtils::vec_add_scale(force_i, fac, m);
      if (force_j) ParticleUtils::vec_add_scale(force_j, -fac, m);

      ++iter;
    }
    else
    {
      double* currentconnectedbonds_i = container_i->get_ptr_to_state_writable(
          Particle::State::CurrentConnectedBonds, particle_i);
      double* currentconnectedbonds_j = container_j->get_ptr_to_state_writable(
          Particle::State::CurrentConnectedBonds, particle_j);

      currentconnectedbonds_i[0] -= 1.0;
      currentconnectedbonds_j[0] -= 1.0;

      (*bondlist_)[iter] = std::move(bondlist_->back());
      bondlist_->pop_back();
    }
  }
}

void Particle::SPHPeridynamic::compute_acceleration() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::compute_acceleration");

  // get container of owned particles of current particle type
  Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
      Particle::Type::PDPhase, Particle::Status::Owned);

  // get number of particles stored in container
  const int particlestored = container->particles_stored();

  // no owned particles of current particle type
  if (particlestored <= 0) return;

  // get particle state dimension
  const int statedim = container->get_state_dim(Particle::State::Acceleration);

  // get pointer to particle states
  const double* radius = container->get_ptr_to_state(Particle::State::Radius, 0);
  const double* mass = container->get_ptr_to_state(Particle::State::Mass, 0);
  const double* force = container->get_ptr_to_state(Particle::State::Force, 0);
  const double* moment = container->try_get_ptr_to_state(Particle::State::Moment, 0);
  double* acc = container->get_ptr_to_state_writable(Particle::State::Acceleration, 0);
  double* angacc =
      container->try_get_ptr_to_state_writable(Particle::State::AngularAcceleration, 0);

  // compute acceleration
  for (int i = 0; i < particlestored; ++i)
  {
    ParticleUtils::vec_add_scale(&acc[statedim * i], (1.0 / mass[i]), &force[statedim * i]);
  }

  // compute angular acceleration
  if (angacc and moment)
  {
    for (int i = 0; i < particlestored; ++i)
      ParticleUtils::vec_add_scale(&angacc[statedim * i],
          (5.0 / (2.0 * mass[i] * ParticleUtils::pow<2>(radius[i]))), &moment[statedim * i]);
  }
}

void Particle::SPHPeridynamic::damage_evaluation()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::damage_evaluation");

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();
  // get container of owned particles of peridynamic phase
  Particle::ParticleContainer* container = particlecontainerbundle->get_specific_container(
      Particle::Type::PDPhase, Particle::Status::Owned);

  // loop over particles in container
  for (int particle_i = 0; particle_i < container->particles_stored(); ++particle_i)
  {
    const double* initialconnectedbonds_i =
        container->get_ptr_to_state(Particle::State::InitialConnectedBonds, particle_i);

    const double* currentconnectedbonds_i =
        container->get_ptr_to_state(Particle::State::CurrentConnectedBonds, particle_i);

    double* pddamagevariable_i =
        container->get_ptr_to_state_writable(Particle::State::PDDamageVariable, particle_i);

    pddamagevariable_i[0] = 1.0 - currentconnectedbonds_i[0] / initialconnectedbonds_i[0];
  }
}

// the beta correction volume function in peridynamic
double Particle::SPHPeridynamic::calculate_volume_correction_factor(const double xi) const
{
  if (xi <= horizon_pd_ - dx_pd_ * 0.5)
  {
    return 1.0;
  }
  else if (xi <= horizon_pd_ + dx_pd_ * 0.5)
  {
    return (horizon_pd_ + dx_pd_ * 0.5 - xi) / (dx_pd_);
  }
  else
  {
    return 0.0;
  }
}

bool Particle::SPHPeridynamic::bond_crosses_pre_crack(
    const double* pos_i, const double* pos_j) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::bond_crosses_pre_crack");

  // --- line segments (3D) ---
  // Two segments in 3D are generally skew; a bond crosses a crack segment if the
  // minimum distance between the two segments is below a tolerance and both
  // parameters t (along bond) and u (along crack segment) lie in [0, 1].
  // Closest-approach parametric form:
  //   bond:   P(t) = pos_i + t*ab,  t in [0,1]
  //   crack:  Q(u) = c    + u*cd,   u in [0,1]
  //   w = pos_i - c;  A = dot(ab,ab);  B = dot(ab,cd);  C = dot(cd,cd)
  //   D = dot(ab,w);  E = dot(cd,w);   denom = A*C - B*B
  //   t = (B*E - C*D)/denom;  u = (A*E - B*D)/denom
  //   distance = |P(t) - Q(u)|  (must be zero for a true 3D intersection)

  for (const auto& crack_segment : pre_crack_lines_)
  {
    // fast AABB rejection
    if (!bond_aabb_overlaps_segment(pos_i, pos_j, crack_segment)) continue;

    // bond vector ab = pos_j - pos_i
    double ab[3] = {pos_j[0] - pos_i[0], pos_j[1] - pos_i[1], pos_j[2] - pos_i[2]};
    // crack segment vector cd = d - c
    double cd[3] = {crack_segment[3] - crack_segment[0], crack_segment[4] - crack_segment[1],
        crack_segment[5] - crack_segment[2]};
    // offset vector w = pos_i - c
    double w[3] = {
        pos_i[0] - crack_segment[0], pos_i[1] - crack_segment[1], pos_i[2] - crack_segment[2]};

    const double A = ParticleUtils::vec_dot(ab, ab);
    const double B = ParticleUtils::vec_dot(ab, cd);
    const double C = ParticleUtils::vec_dot(cd, cd);
    const double D = ParticleUtils::vec_dot(ab, w);
    const double E = ParticleUtils::vec_dot(cd, w);

    const double denom = A * C - B * B;
    if (std::abs(denom) < 1.0e-14) continue;  // segments parallel

    const double t = (B * E - C * D) / denom;
    const double u = (A * E - B * D) / denom;

    if (t < 0.0 || t > 1.0 || u < 0.0 || u > 1.0) continue;

    // distance vector between closest points: g = w + t*ab - u*cd
    double g[3] = {
        w[0] + t * ab[0] - u * cd[0], w[1] + t * ab[1] - u * cd[1], w[2] + t * ab[2] - u * cd[2]};

    // bond crosses crack if the two lines touch
    if (ParticleUtils::vec_dot(g, g) <= 1.0e-20) return true;
  }

  // --- parallelogram patches (3D) ---
  // Patch corners: p0, p1, p2; edges e1 = p1-p0, e2 = p2-p0; normal n = e1 x e2.
  // 1. Find bond-plane intersection: pos_i + t*(pos_j - pos_i), t in [0,1].
  // 2. Express intersection point q in local coords: q - p0 = s*e1 + r*e2, s,r in [0,1].
  for (const auto& patch : pre_crack_planes_)
  {
    // fast AABB rejection
    if (!bond_aabb_overlaps_patch(pos_i, pos_j, patch)) continue;

    const double p0[3] = {patch[0], patch[1], patch[2]};
    // edge vectors
    double e1[3] = {patch[3] - p0[0], patch[4] - p0[1], patch[5] - p0[2]};
    double e2[3] = {patch[6] - p0[0], patch[7] - p0[1], patch[8] - p0[2]};

    // normal n = e1 x e2
    double n[3];
    ParticleUtils::vec_set_cross(n, e1, e2);

    // bond direction d = pos_j - pos_i
    double d[3] = {pos_j[0] - pos_i[0], pos_j[1] - pos_i[1], pos_j[2] - pos_i[2]};

    const double denom = ParticleUtils::vec_dot(d, n);
    if (std::abs(denom) < 1.0e-14) continue;  // bond parallel to patch plane

    // offset from p0 to pos_i
    double p0_to_i[3] = {p0[0] - pos_i[0], p0[1] - pos_i[1], p0[2] - pos_i[2]};

    // parameter t along bond where it crosses the plane
    const double t = ParticleUtils::vec_dot(p0_to_i, n) / denom;
    if (t < 0.0 || t > 1.0) continue;  // intersection outside bond

    // intersection point q relative to p0: q = pos_i + t*d - p0
    double q[3] = {
        pos_i[0] + t * d[0] - p0[0], pos_i[1] + t * d[1] - p0[1], pos_i[2] + t * d[2] - p0[2]};

    // solve q = s*e1 + r*e2 via Gram matrix (works for non-orthogonal edges)
    const double e1e1 = ParticleUtils::vec_dot(e1, e1);
    const double e2e2 = ParticleUtils::vec_dot(e2, e2);
    const double e1e2 = ParticleUtils::vec_dot(e1, e2);
    const double qe1 = ParticleUtils::vec_dot(q, e1);
    const double qe2 = ParticleUtils::vec_dot(q, e2);

    const double det = e1e1 * e2e2 - e1e2 * e1e2;
    if (std::abs(det) < 1.0e-14) continue;  // degenerate patch

    const double s = (qe1 * e2e2 - qe2 * e1e2) / det;
    const double r = (qe2 * e1e1 - qe1 * e1e2) / det;

    if (s >= 0.0 && s <= 1.0 && r >= 0.0 && r <= 1.0) return true;
  }

  return false;
}

// -----------------------------------------------------------------------
// File-local geometry helper implementations
// -----------------------------------------------------------------------
namespace
{
  /*!
   * \brief Check whether two 1D intervals [min_a, max_a] and [min_b, max_b] overlap.
   *
   * Used as a per-dimension AABB rejection test before the full geometric intersection
   * calculation in bond_crosses_pre_crack().
   */
  bool intervals_overlap(
      const double min_a, const double max_a, const double min_b, const double max_b)
  {
    return min_a <= max_b && max_a >= min_b;
  }

  /*!
   * \brief Check whether the AABB of bond [pos_i, pos_j] overlaps the AABB of a crack
   * line segment stored as {c0x,c0y,c0z, c1x,c1y,c1z} in \p segment.
   */
  bool bond_aabb_overlaps_segment(
      const double* pos_i, const double* pos_j, const std::array<double, 6>& segment)
  {
    for (int dim = 0; dim < 3; ++dim)
    {
      if (!intervals_overlap(std::min(pos_i[dim], pos_j[dim]), std::max(pos_i[dim], pos_j[dim]),
              std::min(segment[dim], segment[dim + 3]), std::max(segment[dim], segment[dim + 3])))
        return false;
    }
    return true;
  }

  /*!
   * \brief Check whether the AABB of bond [pos_i, pos_j] overlaps the AABB of a crack
   * patch stored as {p0x,p0y,p0z, p1x,p1y,p1z, p2x,p2y,p2z} in \p patch.
   */
  bool bond_aabb_overlaps_patch(
      const double* pos_i, const double* pos_j, const std::array<double, 9>& patch)
  {
    for (int dim = 0; dim < 3; ++dim)
    {
      const double patch_min = std::min({patch[dim], patch[dim + 3], patch[dim + 6]});
      const double patch_max = std::max({patch[dim], patch[dim + 3], patch[dim + 6]});
      if (!intervals_overlap(std::min(pos_i[dim], pos_j[dim]), std::max(pos_i[dim], pos_j[dim]),
              patch_min, patch_max))
        return false;
    }
    return true;
  }
}  // namespace

void Particle::SPHPeridynamic::clear_boundary_forces() const
{
  if (not particlecontainerbundle_->get_particle_types().contains(Particle::Type::BoundaryPhase))
    return;

  particlecontainerbundle_
      ->get_specific_container(Particle::Type::BoundaryPhase, Particle::Status::Owned)
      ->clear_state(Particle::State::Force);
}

void Particle::SPHPeridynamic::sum_boundary_forces(
    std::vector<double>& force_dbc_funct, std::vector<double>& force_other) const
{
  // reaction of the contact with the smooth boundary surface, summed by the owners of the PD
  // particles (not part of the force state of the boundary particles)
  force_dbc_funct = smoothcontactforce_dbc_funct_;
  force_other = smoothcontactforce_other_;

  if (not particlecontainerbundle_->get_particle_types().contains(Particle::Type::BoundaryPhase))
    return;

  Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
      Particle::Type::BoundaryPhase, Particle::Status::Owned);

  const int particlestored = container->particles_stored();
  if (particlestored <= 0) return;

  const int statedim = container->get_state_dim(Particle::State::Force);
  const double* force = container->get_ptr_to_state(Particle::State::Force, 0);
  const double* funct_id =
      container->have_stored_state(Particle::State::DirichletFunctionId)
          ? container->get_ptr_to_state(Particle::State::DirichletFunctionId, 0)
          : nullptr;

  for (int i = 0; i < particlestored; ++i)
  {
    std::vector<double>& sum =
        (funct_id and static_cast<int>(funct_id[i]) > 0) ? force_dbc_funct : force_other;
    for (int dim = 0; dim < statedim; ++dim) sum[dim] += force[statedim * i + dim];
  }
}

void Particle::SPHPeridynamic::integrate_substeps(
    const double dt, const std::vector<double>& gravity)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::integrate_substeps");

  // after a restart the bond list and the contact candidates are set by a first pair evaluation
  if (not neighborpairs_pd_->have_evaluated_particle_pairs())
    neighborpairs_pd_->evaluate_neighbor_pairs();

  if (not checkedsubstepstability_)
  {
    check_substep_stability(dt);
    checkedsubstepstability_ = true;
  }

  const int n = pd_substeps_;
  const double h = dt / n;

  const bool haveboundary =
      particlecontainerbundle_->get_particle_types().contains(Particle::Type::BoundaryPhase);

  // ghosted boundary particles at t_n+1 (structures moved by Dirichlet functions)
  // (and their velocities for the contact damping and friction)
  if (haveboundary)
  {
    if (damp_ != 0.0 or friction_pd_ > 0.0 or friction_boundary_ > 0.0)
      particleengineinterface_->refresh_particles_of_specific_states_and_types(
          {{Particle::Type::BoundaryPhase,
              {Particle::State::Position, Particle::State::Velocity}}});
    else
      particleengineinterface_->refresh_particles_of_specific_states_and_types(
          {{Particle::Type::BoundaryPhase, {Particle::State::Position}}});
  }

  // states of ghosted PD particles needed for the force evaluation of a substep
  StatesOfTypesToRefresh pdstatestorefresh;
  if (damp_ != 0.0 or friction_pd_ > 0.0 or friction_boundary_ > 0.0)
    pdstatestorefresh.push_back(
        {Particle::Type::PDPhase, {Particle::State::Position, Particle::State::Velocity}});
  else
    pdstatestorefresh.push_back({Particle::Type::PDPhase, {Particle::State::Position}});

  Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
      Particle::Type::PDPhase, Particle::Status::Owned);
  const int particlestored = container->particles_stored();
  const int statedim = container->get_state_dim(Particle::State::Position);

  // particles with a per-particle Dirichlet function keep their prescribed state of t_n+1
  const double* functid =
      (particlestored > 0 and container->have_stored_state(Particle::State::DirichletFunctionId))
          ? container->get_ptr_to_state(Particle::State::DirichletFunctionId, 0)
          : nullptr;
  auto prescribed = [functid](const int i) { return functid and static_cast<int>(functid[i]) > 0; };

  // remainder (dt - h)/2 of the half kick with the coupling force of the evaluation at t_n (the
  // acceleration of that evaluation, used in the first substep, contains its part h/2)
  double maxprescribeddisplacement = 0.0;
  if (particlestored > 0)
  {
    const double* mass = container->get_ptr_to_state(Particle::State::Mass, 0);
    const double* couplingforce = container->get_ptr_to_state(Particle::State::CouplingForce, 0);
    double* vel = container->get_ptr_to_state_writable(Particle::State::Velocity, 0);
    for (int i = 0; i < particlestored; ++i)
    {
      if (prescribed(i))
      {
        maxprescribeddisplacement = std::max(
            maxprescribeddisplacement, dt * ParticleUtils::vec_norm_two(&vel[statedim * i]));
        continue;
      }
      ParticleUtils::vec_add_scale(
          &vel[statedim * i], 0.5 * (dt - h) / mass[i], &couplingforce[statedim * i]);
    }
  }

  // positions at t_n for the completeness check of the contact candidates
  std::vector<double> pos_n(statedim * particlestored);
  if (particlestored > 0)
  {
    const double* pos = container->get_ptr_to_state(Particle::State::Position, 0);
    std::copy(pos, pos + statedim * particlestored, pos_n.begin());
  }
  double maxdisplacement = 0.0;

  for (int k = 0; k < n; ++k)
  {
    // half kick and drift
    if (particlestored > 0)
    {
      double* pos = container->get_ptr_to_state_writable(Particle::State::Position, 0);
      double* vel = container->get_ptr_to_state_writable(Particle::State::Velocity, 0);
      const double* acc = container->get_ptr_to_state(Particle::State::Acceleration, 0);

      for (int i = 0; i < particlestored; ++i)
      {
        if (prescribed(i)) continue;
        ParticleUtils::vec_add_scale(&vel[statedim * i], 0.5 * h, &acc[statedim * i]);
        ParticleUtils::vec_add_scale(&pos[statedim * i], h, &vel[statedim * i]);

        double disp[3];
        ParticleUtils::vec_set(disp, &pos[statedim * i]);
        ParticleUtils::vec_sub(disp, &pos_n[statedim * i]);
        maxdisplacement = std::max(maxdisplacement, ParticleUtils::vec_norm_two(disp));
      }
    }

    // the force evaluation of the last substep is the evaluation of all interactions at t_n+1
    // (preceded by the update of the connectivity, which refreshes all ghosted particles)
    if (k == n - 1) break;

    // refresh ghosted PD particles
    particleengineinterface_->refresh_particles_of_specific_states_and_types(pdstatestorefresh);

    // bond and contact forces at the current positions
    compute_bond_forces();
    compute_substep_contact_forces(h);

    // acceleration from bond, contact, gravity and viscous damping forces, then half kick
    set_substep_acceleration(gravity);
    if (particlestored > 0)
    {
      double* vel = container->get_ptr_to_state_writable(Particle::State::Velocity, 0);
      const double* acc = container->get_ptr_to_state(Particle::State::Acceleration, 0);
      for (int i = 0; i < particlestored; ++i)
        if (not prescribed(i))
          ParticleUtils::vec_add_scale(&vel[statedim * i], 0.5 * h, &acc[statedim * i]);
    }
  }

  if (particlestored > 0)
  {
    const double* pos = container->get_ptr_to_state(Particle::State::Position, 0);
    double* vel = container->get_ptr_to_state_writable(Particle::State::Velocity, 0);
    double* substepvel = container->get_ptr_to_state_writable(Particle::State::SubstepVelocity, 0);
    double* acc = container->get_ptr_to_state_writable(Particle::State::Acceleration, 0);
    for (int i = 0; i < particlestored; ++i)
    {
      if (prescribed(i)) continue;
      for (int dim = 0; dim < statedim; ++dim)
      {
        const int idx = statedim * i + dim;

        // in the evaluation at t_n+1 the fluid sees the PD particles with the velocity of the time
        // level of its own, the mean velocity (x_n+1 - x_n) / dt over the step (identical to the
        // half-step velocity of single-rate velocity Verlet); the velocity after the substeps is
        // kept for the closing kick
        substepvel[idx] = vel[idx];
        vel[idx] = (pos[idx] - pos_n[idx]) / dt;

        // the evaluation at t_n+1 adds to the acceleration (as after the pre-interaction routine
        // of the single-rate scheme); the prescribed acceleration of Dirichlet particles is kept
        acc[idx] = 0.0;
      }
    }
  }

  // the contact candidates of t_n are complete if no pair approached by more than the skin: PD
  // particles moved at most maxdisplacement, boundary and prescribed PD particles at most |v| dt
  double maxboundarydisplacement = maxprescribeddisplacement;
  if (haveboundary)
  {
    Particle::ParticleContainer* boundarycontainer =
        particlecontainerbundle_->get_specific_container(
            Particle::Type::BoundaryPhase, Particle::Status::Owned);
    const int boundarystored = boundarycontainer->particles_stored();
    const int boundarystatedim = boundarycontainer->get_state_dim(Particle::State::Velocity);
    if (boundarystored > 0)
    {
      // all boundary particles: moved by per-particle or per-type Dirichlet conditions
      const double* vel = boundarycontainer->get_ptr_to_state(Particle::State::Velocity, 0);
      for (int i = 0; i < boundarystored; ++i)
        maxboundarydisplacement = std::max(
            maxboundarydisplacement, dt * ParticleUtils::vec_norm_two(&vel[boundarystatedim * i]));
    }
  }
  const std::array<double, 2> maxdisp = Core::Communication::max_all(
      std::array<double, 2>{maxdisplacement, maxboundarydisplacement}, comm_);
  if (2.0 * maxdisp[0] + maxdisp[1] >= contact_candidate_skin_)
    FOUR_C_THROW(
        "PD substeps: particles moved by up to {} (PD) and {} (boundary) within one time step; the "
        "contact candidates (skin {}) may be incomplete, reduce TIMESTEP!",
        maxdisp[0], maxdisp[1], contact_candidate_skin_);
}

void Particle::SPHPeridynamic::complete_substeps(const double dt) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::complete_substeps");

  const double h = dt / pd_substeps_;

  Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
      Particle::Type::PDPhase, Particle::Status::Owned);
  const int particlestored = container->particles_stored();
  if (particlestored <= 0) return;
  const int statedim = container->get_state_dim(Particle::State::Velocity);

  const double* mass = container->get_ptr_to_state(Particle::State::Mass, 0);
  double* acc = container->get_ptr_to_state_writable(Particle::State::Acceleration, 0);
  const double* couplingforce = container->get_ptr_to_state(Particle::State::CouplingForce, 0);
  const double* substepvel = container->get_ptr_to_state(Particle::State::SubstepVelocity, 0);
  double* vel = container->get_ptr_to_state_writable(Particle::State::Velocity, 0);
  const double* functid = container->have_stored_state(Particle::State::DirichletFunctionId)
                              ? container->get_ptr_to_state(Particle::State::DirichletFunctionId, 0)
                              : nullptr;

  for (int i = 0; i < particlestored; ++i)
  {
    // particles with a per-particle Dirichlet function keep their prescribed velocity
    if (functid and static_cast<int>(functid[i]) > 0) continue;

    // the viscous damping of the evaluation at t_n+1 used the mean velocity of the step seen by the
    // fluid: replace it by the damping with the velocity after the substeps
    if (viscdampfac_ > 0.0)
    {
      double dv[3];
      ParticleUtils::vec_set(dv, &substepvel[statedim * i]);
      ParticleUtils::vec_sub(dv, &vel[statedim * i]);
      ParticleUtils::vec_add_scale(&acc[statedim * i], -viscdampfac_, dv);
    }

    // velocity after the substeps (the evaluation at t_n+1 used the mean velocity of the step)
    ParticleUtils::vec_set(&vel[statedim * i], &substepvel[statedim * i]);
    ParticleUtils::vec_add_scale(&vel[statedim * i], 0.5 * h, &acc[statedim * i]);
    ParticleUtils::vec_add_scale(
        &vel[statedim * i], 0.5 * (dt - h) / mass[i], &couplingforce[statedim * i]);
  }
}

void Particle::SPHPeridynamic::take_boundary_impulse_correction(
    const double dt, std::vector<double>& impulse_dbc_funct, std::vector<double>& impulse_other)
{
  // contact impulse of the step: h times the contact force of every force evaluation (substeps and
  // evaluation at t_n+1), the caller adds dt times the force of the evaluation at t_n+1
  const double h = dt / pd_substeps_;
  impulse_dbc_funct.assign(3, 0.0);
  impulse_other.assign(3, 0.0);
  for (int dim = 0; dim < 3; ++dim)
  {
    impulse_dbc_funct[dim] =
        substepcontactimpulse_dbc_funct_[dim] - (dt - h) * contactforce_dbc_funct_[dim];
    impulse_other[dim] = substepcontactimpulse_other_[dim] - (dt - h) * contactforce_other_[dim];
  }
  substepcontactimpulse_dbc_funct_.assign(3, 0.0);
  substepcontactimpulse_other_.assign(3, 0.0);
}

void Particle::SPHPeridynamic::compute_substep_contact_forces(const double weight)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::compute_substep_contact_forces");

  if ((boundary_contact_normal_ or boundary_contact_smooth_) and not computedboundarynormals_)
    compute_boundary_contact_normals();

  for (const auto& candidate : neighborpairs_pd_->get_ref_to_contact_candidates())
  {
    // access values of local index tuples of particle i and j
    Particle::Type type_i;
    Particle::Status status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = candidate.tuple_i_;

    Particle::Type type_j;
    Particle::Status status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = candidate.tuple_j_;

    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);
    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // local indices are valid within a time step (no transfer of particles during the substeps)
    FOUR_C_ASSERT_ALWAYS(
        particle_i < container_i->particles_stored() and
            particle_j < container_j->particles_stored() and
            container_i->get_ptr_to_global_id(particle_i)[0] == candidate.globalid_i_ and
            container_j->get_ptr_to_global_id(particle_j)[0] == candidate.globalid_j_,
        "PD substeps: invalid contact candidate (global ids {} and {})!", candidate.globalid_i_,
        candidate.globalid_j_);

    const double* pos_i = container_i->get_ptr_to_state(Particle::State::Position, particle_i);
    const double* pos_j = container_j->get_ptr_to_state(Particle::State::Position, particle_j);

    // vector from particle i to j considering periodic boundaries
    double r_ji[3];
    particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);
    const double absdist = ParticleUtils::vec_norm_two(r_ji);

    // gap calculation based on initial spacing
    const double gap = absdist - dx_pd_;
    if (gap >= 0.0) continue;

    // versor from particle i to j and direction of the contact force
    double e_ji[3];
    ParticleUtils::vec_set_scale(e_ji, 1.0 / absdist, r_ji);
    double e[3];
    contact_direction(type_i, container_i, particle_i, type_j, container_j, particle_j, e_ji, e);

    // rate of normal gap
    const bool needvelocity = (damp_ != 0.0 or friction_pd_ > 0.0 or friction_boundary_ > 0.0);
    const double* vel_i = needvelocity
                              ? container_i->get_ptr_to_state(Particle::State::Velocity, particle_i)
                              : nullptr;
    const double* vel_j = needvelocity
                              ? container_j->get_ptr_to_state(Particle::State::Velocity, particle_j)
                              : nullptr;
    double gapdot = 0.0;
    if (damp_ != 0.0)
      gapdot = -1.0 * (ParticleUtils::vec_dot(vel_i, e) - ParticleUtils::vec_dot(vel_j, e));

    // magnitude of peridynamic particle contact force
    const double fac = std::min(0.0, (stiff_ * gap + damp_ * gapdot));

    // regularized Coulomb friction force on particle i (none without friction)
    double f_t[3] = {0.0, 0.0, 0.0};
    const bool friction =
        needvelocity and friction_force(type_i, type_j, vel_i, vel_j, e, fac, f_t);

    // add contributions
    double* force_i =
        container_i->try_get_ptr_to_state_writable(Particle::State::Force, particle_i);
    double* force_j = nullptr;
    if (status_j == Particle::Status::Owned)
      force_j = container_j->try_get_ptr_to_state_writable(Particle::State::Force, particle_j);

    if (force_i) ParticleUtils::vec_add_scale(force_i, fac, e);
    if (force_j) ParticleUtils::vec_add_scale(force_j, -fac, e);
    if (friction)
    {
      if (force_i) ParticleUtils::vec_add(force_i, f_t);
      if (force_j) ParticleUtils::vec_sub(force_j, f_t);
    }

    // contact impulse on owned boundary particles for the boundary force output
    double f_t_j[3];
    ParticleUtils::vec_set_scale(f_t_j, -1.0, f_t);
    if (type_i == Particle::Type::BoundaryPhase and status_i == Particle::Status::Owned)
    {
      add_boundary_contact_force(container_i, particle_i, fac, e, weight,
          substepcontactimpulse_dbc_funct_, substepcontactimpulse_other_);
      if (friction)
        add_boundary_force_vector(container_i, particle_i, f_t, weight,
            substepcontactimpulse_dbc_funct_, substepcontactimpulse_other_);
    }
    if (type_j == Particle::Type::BoundaryPhase and status_j == Particle::Status::Owned)
    {
      add_boundary_contact_force(container_j, particle_j, -fac, e, weight,
          substepcontactimpulse_dbc_funct_, substepcontactimpulse_other_);
      if (friction)
        add_boundary_force_vector(container_j, particle_j, f_t_j, weight,
            substepcontactimpulse_dbc_funct_, substepcontactimpulse_other_);
    }
  }

  // contact of PD particles with the smooth boundary surface
  if (boundary_contact_smooth_) compute_smooth_boundary_contact_forces(true, weight);

  // (the force state of boundary particles is cleared before the evaluation at t_n+1)
}

void Particle::SPHPeridynamic::compute_boundary_contact_normals()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::compute_boundary_contact_normals");

  // no boundary particles: nothing to do (the particle types are the same on all processors)
  computedboundarynormals_ = true;
  if (not particlecontainerbundle_->get_particle_types().contains(Particle::Type::BoundaryPhase))
    return;

  // outward normal of a boundary particle: opposite to the weighted sum of the distance vectors to
  // its boundary neighbours within the horizon (a colour-function gradient); interior particles of
  // thick or solid structures get no normal (vanishing sum) and keep the centre line direction
  Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
      Particle::Type::BoundaryPhase, Particle::Status::Owned);
  const int particlestored = container->particles_stored();
  container->clear_state(Particle::State::BoundaryContactNormal);
  std::vector<double> sumdist(particlestored, 0.0);

  for (const auto& potentialneighbors :
      particleengineinterface_->get_potential_particle_neighbors())
  {
    Particle::Type type_i;
    Particle::Status status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

    Particle::Type type_j;
    Particle::Status status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = potentialneighbors.second;

    if (type_i != Particle::Type::BoundaryPhase or type_j != Particle::Type::BoundaryPhase)
      continue;

    const Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);
    const Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    const double* pos_i = container_i->get_ptr_to_state(Particle::State::Position, particle_i);
    const double* pos_j = container_j->get_ptr_to_state(Particle::State::Position, particle_j);

    double r_ji[3];
    particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);
    const double absdist = ParticleUtils::vec_norm_two(r_ji);
    if (absdist >= horizon_pd_) continue;
    const double w = 1.0 - absdist / horizon_pd_;

    if (status_i == Particle::Status::Owned)
    {
      ParticleUtils::vec_add_scale(
          container->get_ptr_to_state_writable(Particle::State::BoundaryContactNormal, particle_i),
          -w, r_ji);
      sumdist[particle_i] += w * absdist;
    }
    if (status_j == Particle::Status::Owned)
    {
      ParticleUtils::vec_add_scale(
          container->get_ptr_to_state_writable(Particle::State::BoundaryContactNormal, particle_j),
          w, r_ji);
      sumdist[particle_j] += w * absdist;
    }
  }

  int numwithnormal = 0;
  for (int i = 0; i < particlestored; ++i)
  {
    double* normal =
        container->get_ptr_to_state_writable(Particle::State::BoundaryContactNormal, i);
    const double norm = ParticleUtils::vec_norm_two(normal);
    // a one-sided neighbourhood (surface): |sum| is a sizeable part of the summed distances
    if (norm > 0.1 * sumdist[i] and norm > 0.0)
    {
      ParticleUtils::vec_scale(normal, 1.0 / norm);
      ++numwithnormal;
    }
    else
      ParticleUtils::vec_clear(normal);
  }

  // smooth boundary surface: only the outermost layer keeps its normal (a boundary particle within
  // 1.5 dx lying more than dx/2 above a particle along its normal marks an inner layer, whose
  // tangent plane would lie dx below the surface)
  if (boundary_contact_smooth_)
  {
    std::vector<char> inner(particlestored, 0);
    for (const auto& potentialneighbors :
        particleengineinterface_->get_potential_particle_neighbors())
    {
      Particle::Type type_i;
      Particle::Status status_i;
      int particle_i;
      std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

      Particle::Type type_j;
      Particle::Status status_j;
      int particle_j;
      std::tie(type_j, status_j, particle_j) = potentialneighbors.second;

      if (type_i != Particle::Type::BoundaryPhase or type_j != Particle::Type::BoundaryPhase)
        continue;

      const double* pos_i = particlecontainerbundle_->get_specific_container(type_i, status_i)
                                ->get_ptr_to_state(Particle::State::Position, particle_i);
      const double* pos_j = particlecontainerbundle_->get_specific_container(type_j, status_j)
                                ->get_ptr_to_state(Particle::State::Position, particle_j);

      double r_ji[3];
      particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);
      if (ParticleUtils::vec_norm_two(r_ji) >= 1.5 * dx_pd_) continue;

      if (status_i == Particle::Status::Owned and
          ParticleUtils::vec_dot(r_ji,
              container->get_ptr_to_state(Particle::State::BoundaryContactNormal, particle_i)) >
              0.5 * dx_pd_)
        inner[particle_i] = 1;
      if (status_j == Particle::Status::Owned and
          -ParticleUtils::vec_dot(r_ji,
              container->get_ptr_to_state(Particle::State::BoundaryContactNormal, particle_j)) >
              0.5 * dx_pd_)
        inner[particle_j] = 1;
    }
    for (int i = 0; i < particlestored; ++i)
    {
      double* normal =
          container->get_ptr_to_state_writable(Particle::State::BoundaryContactNormal, i);
      if (inner[i] and ParticleUtils::vec_norm_two(normal) > 0.5)
      {
        ParticleUtils::vec_clear(normal);
        --numwithnormal;
      }
    }
  }

  // smooth boundary surface: refined normals of the outermost layer from a plane fit of the
  // outermost particles within 2 dx whose normals agree within 30 degrees (smallest principal axis
  // of their positions, oriented like the colour-function normal; at least 3 points not on a
  // line). The colour-function normal is smeared over the horizon: next to a concave corner of two
  // walls it tilts by ~25 degrees and the tangent planes cut into the corner (a PD particle
  // touching both walls had a gap of -0.1 dx, i.e. a force of 0.1 k dx at rest). A fit that turns
  // the normal by more than 30 degrees while the particle lies more than 0.3 dx (in the fitted
  // plane) from the centroid of its points is rejected: it belongs to a neighbouring face (edge
  // rows of thin walls, whose own face is too narrow to be included)
  int numrefined = 0;
  if (boundary_contact_smooth_)
  {
    // normals (and thus outermost flags) of the ghosted boundary particles
    particleengineinterface_->refresh_particles_of_specific_states_and_types(
        {{Particle::Type::BoundaryPhase, {Particle::State::BoundaryContactNormal}}});

    constexpr double cos_similar = 0.86602540378443865;
    const double fitradius = 2.0 * dx_pd_ * (1.0 + 1.0e-6);  // lattice neighbours at 2 dx included

    // points of the fit of each owned outermost particle relative to its position: number (the
    // particle itself included), sum of r and of r r^T (xx, yy, zz, xy, xz, yz)
    std::vector<int> npoints(particlestored, 1);
    std::vector<std::array<double, 9>> moments(particlestored, std::array<double, 9>{});
    auto add_point = [&](const int i, const double* r)
    {
      ++npoints[i];
      std::array<double, 9>& m = moments[i];
      for (int dim = 0; dim < 3; ++dim) m[dim] += r[dim];
      m[3] += r[0] * r[0];
      m[4] += r[1] * r[1];
      m[5] += r[2] * r[2];
      m[6] += r[0] * r[1];
      m[7] += r[0] * r[2];
      m[8] += r[1] * r[2];
    };

    for (const auto& potentialneighbors :
        particleengineinterface_->get_potential_particle_neighbors())
    {
      Particle::Type type_i;
      Particle::Status status_i;
      int particle_i;
      std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

      Particle::Type type_j;
      Particle::Status status_j;
      int particle_j;
      std::tie(type_j, status_j, particle_j) = potentialneighbors.second;

      if (type_i != Particle::Type::BoundaryPhase or type_j != Particle::Type::BoundaryPhase)
        continue;

      const Particle::ParticleContainer* container_i =
          particlecontainerbundle_->get_specific_container(type_i, status_i);
      const Particle::ParticleContainer* container_j =
          particlecontainerbundle_->get_specific_container(type_j, status_j);

      const double* normal_i =
          container_i->get_ptr_to_state(Particle::State::BoundaryContactNormal, particle_i);
      const double* normal_j =
          container_j->get_ptr_to_state(Particle::State::BoundaryContactNormal, particle_j);
      if (ParticleUtils::vec_norm_two(normal_i) < 0.5 or
          ParticleUtils::vec_norm_two(normal_j) < 0.5 or
          ParticleUtils::vec_dot(normal_i, normal_j) <= cos_similar)
        continue;

      double r_ji[3];
      particleengineinterface_->distance_between_particles(
          container_i->get_ptr_to_state(Particle::State::Position, particle_i),
          container_j->get_ptr_to_state(Particle::State::Position, particle_j), r_ji);
      if (ParticleUtils::vec_norm_two(r_ji) >= fitradius) continue;

      if (status_i == Particle::Status::Owned) add_point(particle_i, r_ji);
      if (status_j == Particle::Status::Owned)
      {
        double r_ij[3];
        ParticleUtils::vec_set_scale(r_ij, -1.0, r_ji);
        add_point(particle_j, r_ij);
      }
    }

    std::vector<double> refined(3 * particlestored, 0.0);
    for (int i = 0; i < particlestored; ++i)
    {
      const double* normal = container->get_ptr_to_state(Particle::State::BoundaryContactNormal, i);
      ParticleUtils::vec_set(&refined[3 * i], normal);
      if (ParticleUtils::vec_norm_two(normal) < 0.5 or npoints[i] < 3) continue;

      // covariance of the points
      const std::array<double, 9>& m = moments[i];
      const double n = static_cast<double>(npoints[i]);
      const double mean[3] = {m[0] / n, m[1] / n, m[2] / n};
      const std::array<double, 6> cov = {m[3] / n - mean[0] * mean[0], m[4] / n - mean[1] * mean[1],
          m[5] / n - mean[2] * mean[2], m[6] / n - mean[0] * mean[1], m[7] / n - mean[0] * mean[2],
          m[8] / n - mean[1] * mean[2]};
      double axis[3];
      if (not smallest_principal_axis(cov, axis)) continue;
      double offset[3];
      ParticleUtils::vec_set(offset, mean);
      ParticleUtils::vec_add_scale(offset, -ParticleUtils::vec_dot(mean, axis), axis);
      if (std::abs(ParticleUtils::vec_dot(axis, normal)) < cos_similar and
          ParticleUtils::vec_norm_two(offset) > 0.3 * dx_pd_)
        continue;
      const double sign = ParticleUtils::vec_dot(axis, normal) < 0.0 ? -1.0 : 1.0;
      ParticleUtils::vec_set_scale(&refined[3 * i], sign, axis);
      ++numrefined;
    }
    for (int i = 0; i < particlestored; ++i)
      ParticleUtils::vec_set(
          container->get_ptr_to_state_writable(Particle::State::BoundaryContactNormal, i),
          &refined[3 * i]);
  }

  // number of boundary particles with a surface normal (once, for checking the structure surfaces)
  const int globalwithnormal = Core::Communication::sum_all(numwithnormal, comm_);
  const int globalstored = Core::Communication::sum_all(particlestored, comm_);
  const int globalrefined = Core::Communication::sum_all(numrefined, comm_);
  if (Core::Communication::my_mpi_rank(comm_) == 0)
  {
    std::cout << "PD contact: " << globalwithnormal << " of " << globalstored
              << " boundary particles have a surface normal";
    if (boundary_contact_smooth_)
      std::cout << " (outermost layer; smooth boundary surface, radius " << smooth_surface_radius_
                << "; " << globalrefined << " normals from a plane fit)";
    std::cout << std::endl;
  }

  // ghosted boundary particles (contact pairs with ghosted boundary particles)
  particleengineinterface_->refresh_particles_of_specific_states_and_types(
      {{Particle::Type::BoundaryPhase, {Particle::State::BoundaryContactNormal}}});
}

void Particle::SPHPeridynamic::compute_smooth_boundary_contact_forces(
    const bool substep, const double weight)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::compute_smooth_boundary_contact_forces");

  if (not substep)
  {
    smoothcontactforce_dbc_funct_.assign(3, 0.0);
    smoothcontactforce_other_.assign(3, 0.0);
  }

  Particle::ParticleContainer* container_i = particlecontainerbundle_->get_specific_container(
      Particle::Type::PDPhase, Particle::Status::Owned);
  const int particlestored = container_i->particles_stored();
  if (particlestored <= 0) return;

  const bool needvelocity = (damp_ != 0.0 or friction_boundary_ > 0.0);
  const double radius2 = smooth_surface_radius_ * smooth_surface_radius_;
  const double visibility_depth = 0.5 * dx_pd_;

  // smooth onset of the contact with the weight sum S: the weighted mean f does not decay with the
  // weights, so near convex edges and at the border of the support it would give a gap of a PD
  // particle far from the boundary; potential phi(S) k g^2 / 2 with phi = 0 for S <= S_0, 1 for
  // S >= S_1 (smoothstep); a PD particle at the distance dx from an outermost boundary particle has
  // S >= W(dx) = 27/64 > S_1, so contacts at the surface are not affected
  constexpr double gate_lower = 0.05;
  constexpr double gate_upper = 0.30;

  // implicit moving least squares surface of the outermost boundary particles (positions x_j,
  // outward normals n_j) at an owned PD particle x (Kolluri 2008): f(x) = A / S with
  // S = sum_j W_j, A = sum_j W_j (x - x_j).n_j, W_j = (1 - |x - x_j|^2 / R^2)^3 h_j (h_j:
  // visibility, below), and
  // grad f = (sum_j W_j n_j + sum_j ((x - x_j).n_j - f) grad W_j) / S; sums of each PD particle:
  // [0] S, [1] A, [2-4] sum W_j n_j, [5-7] sum (x - x_j).n_j grad W_j, [8-10] sum grad W_j,
  // [11-13] sum W_j v_j, [14] sum W_j of boundary particles with a per-particle Dirichlet function
  smoothslot_.assign(particlestored, -1);
  smoothsums_.clear();

  for (const auto& pair : neighborpairs_pd_->get_ref_to_boundary_contact_pairs())
  {
    const int particle_i = std::get<2>(pair.tuple_i_);
    Particle::Type type_j;
    Particle::Status status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = pair.tuple_j_;
    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // local indices are valid within a time step (no transfer of particles during the substeps)
    FOUR_C_ASSERT_ALWAYS(
        particle_i < particlestored and particle_j < container_j->particles_stored() and
            container_i->get_ptr_to_global_id(particle_i)[0] == pair.globalid_i_ and
            container_j->get_ptr_to_global_id(particle_j)[0] == pair.globalid_j_,
        "smooth boundary surface: invalid pair (global ids {} and {})!", pair.globalid_i_,
        pair.globalid_j_);

    // inner boundary particles do not form the surface
    const double* normal =
        container_j->get_ptr_to_state(Particle::State::BoundaryContactNormal, particle_j);
    if (ParticleUtils::vec_norm_two(normal) < 0.5) continue;

    // vector from the boundary particle to the PD particle
    double r[3];
    particleengineinterface_->distance_between_particles(
        container_j->get_ptr_to_state(Particle::State::Position, particle_j),
        container_i->get_ptr_to_state(Particle::State::Position, particle_i), r);
    const double dist2 = ParticleUtils::vec_dot(r, r);
    if (dist2 >= radius2) continue;

    // weight W_r(|x - x_j|) h((x - x_j).n_j): visibility h fades the contribution of a boundary
    // particle from its tangent plane to the depth c dx behind it (smoothstep), so that tangent
    // planes behind the PD particle (around convex edges) do not count; contacts with a penetration
    // below dx are not affected (on a flat wall the contact holds up to a penetration of ~1.35 dx,
    // ~0.85 k dx^2 of energy; beyond, the weight sum falls below the onset gate)
    const double rn = ParticleUtils::vec_dot(r, normal);
    const double u = std::clamp((rn + visibility_depth) / visibility_depth, 0.0, 1.0);
    if (u <= 0.0) continue;
    const double h = u * u * (3.0 - 2.0 * u);
    const double dh = 6.0 * u * (1.0 - u) / visibility_depth;
    const double q = 1.0 - dist2 / radius2;
    const double wr = q * q * q;
    const double dwrfac = -6.0 * q * q / radius2;
    const double w = wr * h;
    double gradw[3];
    for (int dim = 0; dim < 3; ++dim) gradw[dim] = h * dwrfac * r[dim] + wr * dh * normal[dim];

    int& slot = smoothslot_[particle_i];
    if (slot < 0)
    {
      slot = static_cast<int>(smoothsums_.size());
      smoothsums_.emplace_back();
      smoothsums_.back().fill(0.0);
    }
    std::array<double, 15>& sums = smoothsums_[slot];
    sums[0] += w;
    sums[1] += w * rn;
    for (int dim = 0; dim < 3; ++dim)
    {
      sums[2 + dim] += w * normal[dim];
      sums[5 + dim] += rn * gradw[dim];
      sums[8 + dim] += gradw[dim];
    }
    if (needvelocity)
    {
      const double* vel_j = container_j->get_ptr_to_state(Particle::State::Velocity, particle_j);
      for (int dim = 0; dim < 3; ++dim) sums[11 + dim] += w * vel_j[dim];
    }
    if (container_j->have_stored_state(Particle::State::DirichletFunctionId) and
        static_cast<int>(
            container_j->get_ptr_to_state(Particle::State::DirichletFunctionId, particle_j)[0]) > 0)
      sums[14] += w;
  }

  for (int i = 0; i < particlestored; ++i)
  {
    const int slot = smoothslot_[i];
    if (slot < 0) continue;
    const std::array<double, 15>& sums = smoothsums_[slot];

    // gap to the surface (contact at the distance dx of the particle centres, as for a lattice)
    const double f = sums[1] / sums[0];
    const double gap = f - dx_pd_;
    if (gap >= 0.0) continue;

    const double t = std::clamp((sums[0] - gate_lower) / (gate_upper - gate_lower), 0.0, 1.0);
    if (t <= 0.0) continue;
    const double phi = t * t * (3.0 - 2.0 * t);
    const double dphi = 6.0 * t * (1.0 - t) / (gate_upper - gate_lower);

    // spring force -grad(phi k g^2 / 2) = -k g (phi grad f + g phi' grad S / 2)
    double grad[3];
    double fspring[3];
    for (int dim = 0; dim < 3; ++dim)
    {
      grad[dim] = (sums[2 + dim] + sums[5 + dim] - f * sums[8 + dim]) / sums[0];
      fspring[dim] = -stiff_ * gap * (phi * grad[dim] + 0.5 * gap * dphi * sums[8 + dim]);
    }

    // outward unit normal of the surface (where |grad f| is small: the mean normal)
    double normal[3];
    const double gradnorm = ParticleUtils::vec_norm_two(grad);
    if (gradnorm > 0.1)
      ParticleUtils::vec_set_scale(normal, 1.0 / gradnorm, grad);
    else
    {
      const double meannorm = ParticleUtils::vec_norm_two(&sums[2]);
      if (meannorm <= 0.0) continue;
      ParticleUtils::vec_set_scale(normal, 1.0 / meannorm, &sums[2]);
    }

    // direction from the PD particle into the surface (as e from particle i to j of a pair)
    double e[3];
    ParticleUtils::vec_set_scale(e, -1.0, normal);

    // velocity of the surface (weighted mean of the boundary particles) and rate of the gap
    double vel_b[3] = {0.0, 0.0, 0.0};
    const double* vel_i =
        needvelocity ? container_i->get_ptr_to_state(Particle::State::Velocity, i) : nullptr;
    if (needvelocity)
      for (int dim = 0; dim < 3; ++dim) vel_b[dim] = sums[11 + dim] / sums[0];
    double gapdot = 0.0;
    if (damp_ != 0.0)
      gapdot = -1.0 * (ParticleUtils::vec_dot(vel_i, e) - ParticleUtils::vec_dot(vel_b, e));

    // normal force (spring and damping, compressive only) along e, tangential part of the spring
    const double fspring_n = ParticleUtils::vec_dot(fspring, normal);
    const double fac = std::min(0.0, -fspring_n + phi * damp_ * gapdot);
    if (fac == 0.0) continue;

    double force[3];
    ParticleUtils::vec_set(force, fspring);
    ParticleUtils::vec_add_scale(force, -fspring_n, normal);
    ParticleUtils::vec_add_scale(force, fac, e);
    double f_t[3];
    if (needvelocity and friction_force(Particle::Type::PDPhase, Particle::Type::BoundaryPhase,
                             vel_i, vel_b, e, fac, f_t))
      ParticleUtils::vec_add(force, f_t);

    ParticleUtils::vec_add(
        container_i->get_ptr_to_state_writable(Particle::State::Force, i), force);

    // reaction on the boundary, split by the weights of the boundary particles with a Dirichlet
    // function (moving structure) and without
    const double share = sums[14] / sums[0];
    for (int dim = 0; dim < 3; ++dim)
    {
      const double reaction_dbc_funct = -share * force[dim];
      const double reaction_other = -(1.0 - share) * force[dim];
      if (substep)
      {
        substepcontactimpulse_dbc_funct_[dim] += weight * reaction_dbc_funct;
        substepcontactimpulse_other_[dim] += weight * reaction_other;
      }
      else
      {
        smoothcontactforce_dbc_funct_[dim] += reaction_dbc_funct;
        smoothcontactforce_other_[dim] += reaction_other;
        if (pd_substeps_ > 1)
        {
          contactforce_dbc_funct_[dim] += reaction_dbc_funct;
          contactforce_other_[dim] += reaction_other;
        }
      }
    }
  }
}

void Particle::SPHPeridynamic::contact_direction(const Particle::Type type_i,
    const Particle::ParticleContainer* container_i, const int particle_i,
    const Particle::Type type_j, const Particle::ParticleContainer* container_j,
    const int particle_j, const double* e_ji, double* e) const
{
  ParticleUtils::vec_set(e, e_ji);
  if (not boundary_contact_normal_) return;

  // the PD particle is pushed along the outward normal of the boundary particle: e (from i to j)
  // is the negative normal of a boundary particle j and the normal of a boundary particle i
  const double* normal = nullptr;
  double sign = 1.0;
  if (type_j == Particle::Type::BoundaryPhase and type_i == Particle::Type::PDPhase)
  {
    normal = container_j->get_ptr_to_state(Particle::State::BoundaryContactNormal, particle_j);
    sign = -1.0;
  }
  else if (type_i == Particle::Type::BoundaryPhase and type_j == Particle::Type::PDPhase)
    normal = container_i->get_ptr_to_state(Particle::State::BoundaryContactNormal, particle_i);

  if (normal and ParticleUtils::vec_norm_two(normal) > 0.5)
    ParticleUtils::vec_set_scale(e, sign, normal);
}

bool Particle::SPHPeridynamic::friction_force(const Particle::Type type_i,
    const Particle::Type type_j, const double* vel_i, const double* vel_j, const double* e,
    const double fac, double* f_t) const
{
  ParticleUtils::vec_clear(f_t);
  const bool boundary =
      (type_i == Particle::Type::BoundaryPhase or type_j == Particle::Type::BoundaryPhase);
  const double mu = boundary ? friction_boundary_ : friction_pd_;
  if (mu <= 0.0 or fac >= 0.0) return false;

  // tangential relative velocity of particle j with respect to particle i
  double v_t[3];
  ParticleUtils::vec_set(v_t, vel_j);
  ParticleUtils::vec_sub(v_t, vel_i);
  ParticleUtils::vec_add_scale(v_t, -ParticleUtils::vec_dot(v_t, e), e);
  const double v_t_norm = ParticleUtils::vec_norm_two(v_t);
  if (v_t_norm <= 0.0) return false;

  // regularized Coulomb friction on particle i, along the relative motion of j (opposing the
  // sliding of i relative to j): magnitude min(mu |f_n|, c_t |v_t|)
  const double magnitude = std::min(mu * (-fac), friction_damp_ * v_t_norm);
  ParticleUtils::vec_set_scale(f_t, magnitude / v_t_norm, v_t);
  return true;
}

void Particle::SPHPeridynamic::add_boundary_force_vector(const ParticleContainer* container,
    const int particle, const double* force, const double weight,
    std::vector<double>& sum_dbc_funct, std::vector<double>& sum_other) const
{
  const bool dbc_funct = container->have_stored_state(Particle::State::DirichletFunctionId) and
                         static_cast<int>(container->get_ptr_to_state(
                             Particle::State::DirichletFunctionId, particle)[0]) > 0;
  std::vector<double>& sum = dbc_funct ? sum_dbc_funct : sum_other;
  for (int dim = 0; dim < 3; ++dim) sum[dim] += weight * force[dim];
}

void Particle::SPHPeridynamic::add_boundary_contact_force(const ParticleContainer* container,
    const int particle, const double fac, const double* e_ji, const double weight,
    std::vector<double>& sum_dbc_funct, std::vector<double>& sum_other) const
{
  const bool dbc_funct = container->have_stored_state(Particle::State::DirichletFunctionId) and
                         static_cast<int>(container->get_ptr_to_state(
                             Particle::State::DirichletFunctionId, particle)[0]) > 0;
  std::vector<double>& sum = dbc_funct ? sum_dbc_funct : sum_other;
  for (int dim = 0; dim < 3; ++dim) sum[dim] += weight * fac * e_ji[dim];
}

void Particle::SPHPeridynamic::set_substep_acceleration(const std::vector<double>& gravity) const
{
  Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
      Particle::Type::PDPhase, Particle::Status::Owned);
  const int particlestored = container->particles_stored();
  if (particlestored <= 0) return;
  const int statedim = container->get_state_dim(Particle::State::Acceleration);

  const double* mass = container->get_ptr_to_state(Particle::State::Mass, 0);
  const double* force = container->get_ptr_to_state(Particle::State::Force, 0);
  const double* vel = container->get_ptr_to_state(Particle::State::Velocity, 0);
  double* acc = container->get_ptr_to_state_writable(Particle::State::Acceleration, 0);
  const double* functid = container->have_stored_state(Particle::State::DirichletFunctionId)
                              ? container->get_ptr_to_state(Particle::State::DirichletFunctionId, 0)
                              : nullptr;

  // as in the evaluation of all interactions: gravity, interaction forces, viscous damping
  // (particles with a per-particle Dirichlet function keep their prescribed acceleration)
  for (int i = 0; i < particlestored; ++i)
  {
    if (functid and static_cast<int>(functid[i]) > 0) continue;
    for (int dim = 0; dim < statedim; ++dim) acc[statedim * i + dim] = gravity[dim];
    ParticleUtils::vec_add_scale(&acc[statedim * i], (1.0 / mass[i]), &force[statedim * i]);
    if (viscdampfac_ > 0.0)
      ParticleUtils::vec_add_scale(&acc[statedim * i], -viscdampfac_, &vel[statedim * i]);
  }

  container->clear_state(Particle::State::Force);
}

void Particle::SPHPeridynamic::check_substep_stability(const double dt) const
{
  // bound of the highest frequency of the bonds: for the stiffness K = sum_bonds k b b^T with
  // b = (e_i - e_j) (x) xi_hat, u^T K u <= 2 sum_i u_i^T K_i u_i with K_i = sum_j k_ij xi_hat
  // xi_hat^T, hence omega_max^2 <= max_i 2 lambda_max(K_i) / m_i
  Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
      Particle::Type::PDPhase, Particle::Status::Owned);
  const int particlestored = container->particles_stored();
  std::vector<std::array<double, 6>> stiffness(particlestored, std::array<double, 6>{});

  double stiffness_factor = 0.0;
  switch (peridynamic_dimension_)
  {
    case PeridynamicDimension::Peridynamic_3D:
      stiffness_factor = 12.0 / (std::numbers::pi * std::pow(horizon_pd_, 4)) * std::pow(dx_pd_, 6);
      break;
    case PeridynamicDimension::Peridynamic_2DPlaneStress:
      stiffness_factor = 9.0 / (std::numbers::pi * std::pow(horizon_pd_, 3)) * std::pow(dx_pd_, 4);
      break;
    case PeridynamicDimension::Peridynamic_2DPlaneStrain:
      stiffness_factor =
          48.0 / (std::numbers::pi * 5.0 * std::pow(horizon_pd_, 3)) * std::pow(dx_pd_, 4);
      break;
    default:
      std23::unreachable();
  }

  for (const auto& [first, second] : *bondlist_)
  {
    const Particle::Type type_i = std::get<0>(first);
    const Particle::Status status_i = std::get<1>(first);
    const int particle_i = std::get<2>(first);
    const Particle::Type type_j = std::get<0>(second);
    const Particle::Status status_j = std::get<1>(second);
    const int particle_j = std::get<2>(second);
    const Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);
    const Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    double xi[3];
    ParticleUtils::vec_set(
        xi, container_j->get_ptr_to_state(Particle::State::ReferencePosition, particle_j));
    ParticleUtils::vec_sub(
        xi, container_i->get_ptr_to_state(Particle::State::ReferencePosition, particle_i));
    const double xi_norm = ParticleUtils::vec_norm_two(xi);
    const double young =
        0.5 * (container_i->get_ptr_to_state(Particle::State::Young, particle_i)[0] +
                  container_j->get_ptr_to_state(Particle::State::Young, particle_j)[0]);
    const double volume_factor =
        surface_correction_ ? 0.5 * (container_i->get_ptr_to_state(
                                         Particle::State::PDSurfaceCorrection, particle_i)[0] +
                                        container_j->get_ptr_to_state(
                                            Particle::State::PDSurfaceCorrection, particle_j)[0])
                            : calculate_volume_correction_factor(xi_norm);
    const double k = stiffness_factor * young * volume_factor / xi_norm;

    const double* x = xi;
    const double s = k / (xi_norm * xi_norm);
    const std::array<double, 6> kxx = {s * x[0] * x[0], s * x[1] * x[1], s * x[2] * x[2],
        s * x[0] * x[1], s * x[0] * x[2], s * x[1] * x[2]};
    if (status_i == Particle::Status::Owned)
      for (int c = 0; c < 6; ++c) stiffness[particle_i][c] += kxx[c];
    if (status_j == Particle::Status::Owned)
      for (int c = 0; c < 6; ++c) stiffness[particle_j][c] += kxx[c];
  }

  // largest eigenvalue of a symmetric 3x3 matrix (a00, a11, a22, a01, a02, a12)
  auto lambda_max = [](const std::array<double, 6>& a)
  {
    const double p1 = a[3] * a[3] + a[4] * a[4] + a[5] * a[5];
    if (p1 == 0.0) return std::max({a[0], a[1], a[2]});
    const double q = (a[0] + a[1] + a[2]) / 3.0;
    const double p2 =
        (a[0] - q) * (a[0] - q) + (a[1] - q) * (a[1] - q) + (a[2] - q) * (a[2] - q) + 2.0 * p1;
    const double p = std::sqrt(p2 / 6.0);
    const double b00 = (a[0] - q) / p, b11 = (a[1] - q) / p, b22 = (a[2] - q) / p;
    const double b01 = a[3] / p, b02 = a[4] / p, b12 = a[5] / p;
    const double detb = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) +
                        b02 * (b01 * b12 - b11 * b02);
    const double r = std::clamp(0.5 * detb, -1.0, 1.0);
    return q + 2.0 * p * std::cos(std::acos(r) / 3.0);
  };

  double omega2_bond = 0.0;
  double mass_min = std::numeric_limits<double>::max();
  if (particlestored > 0)
  {
    const double* mass = container->get_ptr_to_state(Particle::State::Mass, 0);
    for (int i = 0; i < particlestored; ++i)
    {
      omega2_bond = std::max(omega2_bond, 2.0 * lambda_max(stiffness[i]) / mass[i]);
      mass_min = std::min(mass_min, mass[i]);
    }
  }
  const double omega_bond = std::sqrt(Core::Communication::max_all(omega2_bond, comm_));
  mass_min = Core::Communication::min_all(mass_min, comm_);

  // frequency of one contact spring of a PD particle with a fixed partner; the check assumes a
  // bonded particle with up to four such springs (a PD-PD contact counts twice)
  const double omega_contact = stiff_ > 0.0 ? std::sqrt(stiff_ / mass_min) : 0.0;
  const int contacts = 4;
  const double omega_max =
      std::sqrt(omega_bond * omega_bond + contacts * omega_contact * omega_contact);

  // a velocity Verlet substep rotates a mode of frequency omega by 2 asin(omega h / 2); the first
  // parametric resonance of the half kicks with the coupling force is reached when n substeps
  // rotate by pi: omega h < 2 sin(pi / (2 n)) (n = 1: stability limit omega h < 2); margin for
  // damping and the geometric stiffness of stretched bonds
  const double h = dt / pd_substeps_;
  const double limit = 0.95 * 2.0 * std::sin(std::numbers::pi / (2.0 * pd_substeps_));

  if (Core::Communication::my_mpi_rank(comm_) == 0)
    Core::IO::cout << "PD substeps " << pd_substeps_ << ": time step " << dt << ", substep " << h
                   << "; highest frequency (bound) of the bonds " << omega_bond
                   << " 1/s, of one contact spring " << omega_contact << " 1/s, of a particle with "
                   << contacts << " contacts " << omega_max << " 1/s: omega h = " << omega_max * h
                   << " (limit " << limit << ")" << Core::IO::endl;

  // explicit stability of the tangential friction regularization: a particle sliding against up to
  // eight fixed contact partners is damped by 8 c_t; the update is monotonic for 8 c_t h / m <= 1
  // (stable for free PD partners); the regularization has no true stick: a sustained tangential
  // load F_t below the Coulomb limit creeps with F_t / c_t
  if (friction_pd_ > 0.0 or friction_boundary_ > 0.0)
  {
    const double ratio = 8.0 * friction_damp_ * h / mass_min;
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      Core::IO::cout << "PD contact friction: mu " << friction_pd_ << " (PD), "
                     << friction_boundary_ << " (boundary), c_t " << friction_damp_
                     << ": 8 c_t h / m = " << ratio << " (limit 1)" << Core::IO::endl;
    if (ratio > 1.0)
      FOUR_C_THROW(
          "PD contact friction: FRICTION_TANGENTIAL_DAMP {} too large for the substep {} "
          "(8 c_t h / m = {} > 1); use at most {}!",
          friction_damp_, h, ratio, mass_min / (8.0 * h));
  }

  if (omega_max * h >= limit)
    FOUR_C_THROW(
        "PD substeps: substep {} exceeds the limit {} of the multi-rate integration (resonance "
        "of the half kicks with the coupling force, bound of the highest PD frequency {} 1/s "
        "including {} contact springs); reduce TIMESTEP to at most {} or change PD_SUBSTEPS!",
        h, limit / omega_max, omega_max, contacts, pd_substeps_ * limit / omega_max);
}

FOUR_C_NAMESPACE_CLOSE
