// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_spec_validators.hpp"
FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | set the particle parameters                                               |
 *---------------------------------------------------------------------------*/
std::vector<Core::IO::InputSpec> Particle::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  using namespace Core::IO::InputSpecBuilders::Validators;

  /*-------------------------------------------------------------------------*
   | general control parameters for particle simulations                     |
   *-------------------------------------------------------------------------*/

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("PARTICLE DYNAMIC",
      {

          // type of particle time integration
          deprecated_selection<DynamicType>("DYNAMICTYPE",
              {
                  {"SemiImplicitEuler", Particle::dyna_semiimpliciteuler},
                  {"VelocityVerlet", Particle::dyna_velocityverlet},
              },
              {.description = "type of particle time integration",
                  .default_value = Particle::dyna_velocityverlet}),

          // type of particle interaction
          deprecated_selection<InteractionType>("INTERACTION",
              {
                  {"None", Particle::interaction_none},
                  {"SPH", Particle::interaction_sph},
                  {"DEM", Particle::interaction_dem},
              },
              {.description = "type of particle interaction",
                  .default_value = Particle::interaction_none}),

          // output type
          parameter<int>("RESULTSEVERY",
              {.description = "write particle runtime output every RESULTSEVERY steps",
                  .default_value = 1}),
          parameter<int>(
              "RESTARTEVERY", {.description = "write restart possibility every RESTARTEVERY steps",
                                  .default_value = 1}),

          // write ghosted particles
          parameter<bool>("WRITE_GHOSTED_PARTICLES",
              {.description = "write ghosted particles (debug feature)", .default_value = false}),

          // time loop control
          parameter<double>("TIMESTEP", {.description = "time step size", .default_value = 0.01}),

          parameter<int>(
              "NUMSTEP", {.description = "maximum number of steps", .default_value = 100}),

          parameter<double>("MAXTIME", {.description = "maximum time", .default_value = 1.0}),

          // gravity acceleration control
          parameter<std::string>("GRAVITY_ACCELERATION",
              {.description = "acceleration due to gravity", .default_value = "0.0 0.0 0.0"}),
          parameter<int>("GRAVITY_RAMP_FUNCT",
              {.description = "number of function governing gravity ramp", .default_value = -1}),

          // viscous damping factor
          parameter<double>("VISCOUS_DAMPING",
              {.description =
                      "apply viscous damping force to determine static equilibrium solutions",
                  .default_value = -1.0}),

          // transfer particles to new bins every time step
          parameter<bool>(
              "TRANSFER_EVERY", {.description = "transfer particles to new bins every time step",
                                    .default_value = false}),

          // considered particle phases with dynamic load balance weighting factor
          parameter<std::string>("PHASE_TO_DYNLOADBALFAC",
              {.description =
                      "considered particle phases with dynamic load balance weighting factor",
                  .default_value = "none"}),

          // coordinate directions of the cuts of the recursive coordinate bisection of the bins
          parameter<std::string>("DYNLOADBAL_CUT_DIRECTIONS",
              {.description = "coordinate directions in which the bins are cut by the recursive "
                              "coordinate bisection of the dynamic load balancing (any of x, y, "
                              "z); xy partitions shallow domains into columns along z",
                  .default_value = "xyz"}),

          // PD work as a separate criterion of the dynamic load balancing
          parameter<bool>("DYNLOADBAL_SEPARATE_PD_WEIGHT",
              {.description = "weight the work of PD particles (pairs of PD particles with PD and "
                              "boundary particles) as a second criterion of the dynamic load "
                              "balancing (multicriteria recursive coordinate bisection: the cuts "
                              "balance the sum of both normalized criteria, the norm decides the "
                              "cut directions)",
                  .default_value = false}),

          // norm of the multicriteria recursive coordinate bisection
          parameter<int>("DYNLOADBAL_MULTICRITERIA_NORM",
              {.description = "norm combining the criteria of DYNLOADBAL_SEPARATE_PD_WEIGHT in the "
                              "recursive coordinate bisection (Zoltan RCB_MULTICRITERIA_NORM: 1, 2 "
                              "or 3 = max norm, which favours the worse balanced criterion)",
                  .default_value = 1,
                  .validator = in_range(1, 3)}),

          // relate particle phase to material id
          parameter<std::string>("PHASE_TO_MATERIAL_ID",
              {.description = "relate particle phase to material id", .default_value = "none"}),

          // amplitude of noise added to initial position for each spatial direction
          parameter<std::string>("INITIAL_POSITION_AMPLITUDE",
              {.description =
                      "amplitude of noise added to initial position for each spatial direction",
                  .default_value = "0.0 0.0 0.0"}),

          // type of particle wall source
          parameter<ParticleWallSource>(
              "PARTICLE_WALL_SOURCE", {.description = "type of particle wall source",
                                          .default_value = Particle::NoParticleWall}),

          // material id for particle wall from bounding box source
          parameter<int>("PARTICLE_WALL_MAT",
              {.description = "material id for particle wall from bounding box source",
                  .default_value = -1}),

          // flags defining considered states of particle wall
          parameter<bool>("PARTICLE_WALL_MOVING",
              {.description = "consider a moving particle wall", .default_value = false}),
          parameter<bool>("PARTICLE_WALL_LOADED",
              {.description = "consider loading on particle wall", .default_value = false}),

          // consider rigid body motion
          parameter<bool>("RIGID_BODY_MOTION",
              {.description = "consider rigid body motion", .default_value = false}),

          parameter<double>("RIGID_BODY_PHASECHANGE_RADIUS",
              {.description = "search radius for neighboring rigid bodies in case of phase change",
                  .default_value = -1.0}),
          parameter<bool>("PD_BODY_INTERACTION",
              {.description = "consider peridynamic body interaction", .default_value = false})},
      {.required = false}));

  auto all_dirichlet_boundaries = all_of({
      parameter<int>("BOUNDARY"),
      one_of(
          std::vector<Core::IO::InputSpec>{

              group("NoDirichletOpenBoundary", {}, {.description = "no dirichlet open boundary"}),
              group("DirichletNormalToPlane",
                  {
                      parameter<int>(
                          "FUNCT", {.description = "number of function governing velocity "
                                                   "condition on dirichlet open boundary"}),
                      parameter<std::vector<double>>("OUTWARD_NORMAL",
                          {.description = "direction of outward normal on dirichlet open boundary",
                              .size = 3}),
                      parameter<std::vector<double>>("PLANE_POINT",
                          {.description = "point on dirichlet open boundary plane", .size = 3}),
                  },
                  {.description = "dirichlet normal to plane open boundary"})},
          store_index_as<Particle::DirichletOpenBoundaryType>("_boundary_type",
              {Particle::NoDirichletOpenBoundary, Particle::DirichletNormalToPlane})),
  });

  auto all_neumann_boundaries = all_of({
      parameter<int>("BOUNDARY"),
      one_of(
          std::vector<Core::IO::InputSpec>{

              group("NoNeumannOpenBoundary", {}, {.description = "no neumann open boundary"}),
              group("NeumannNormalToPlane",
                  {
                      parameter<int>(
                          "FUNCT", {.description = "number of function governing pressure "
                                                   "condition on neumann open boundary",
                                       .default_value = -1}),
                      parameter<std::vector<double>>("OUTWARD_NORMAL",
                          {.description = "direction of outward normal on neumann open boundary",
                              .size = 3}),
                      parameter<std::vector<double>>("PLANE_POINT",
                          {.description = "point on neumann open boundary plane", .size = 3}),
                  },
                  {.description = "neumann normal to plane open boundary"})},
          store_index_as<Particle::NeumannOpenBoundaryType>(
              "_boundary_type", {Particle::NoNeumannOpenBoundary, Particle::NeumannNormalToPlane})),
  });

  /*-------------------------------------------------------------------------*
 | control parameters for initial/boundary conditions |
 *-------------------------------------------------------------------------*/
  specs.push_back(group("PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS",
      {

          // initial temperature field of particle phase given by function
          parameter<std::string>("INITIAL_TEMP_FIELD",
              {.description = "Refer to the function ID describing the initial "
                              "temperature field of particle phase",
                  .default_value = "none"}),

          // initial velocity field of particle phase given by function
          parameter<std::string>(
              "INITIAL_VELOCITY_FIELD", {.description = "Refer to the function ID describing the "
                                                        "initial velocity field of particle phase",
                                            .default_value = "none"}),

          // initial angular velocity field of particle phase given by function
          parameter<std::string>("INITIAL_ANGULAR_VELOCITY_FIELD",
              {.description =
                      "Refer to the function ID describing the initial angular velocity field of "
                      "rigid body  phase/DEM particle",
                  .default_value = "none"}),


          // initial acceleration field of particle phase given by function
          parameter<std::string>("INITIAL_ACCELERATION_FIELD",
              {.description = "Refer to the function ID describing the "
                              "initial acceleration field of particle phase",
                  .default_value = "none"}),

          // initial angular acceleration field of particle phase given by function
          parameter<std::string>("INITIAL_ANGULAR_ACCELERATION_FIELD",
              {.description =
                      "Refer to the function ID describing the initial angular acceleration "
                      "field of rigid body  phase/DEM particle",
                  .default_value = "none"}),


          // Dirichlet functions depending on time only
          parameter<bool>("DIRICHLET_FUNCT_TIME_ONLY",
              {.description = "the Dirichlet functions (per particle type and per particle) depend "
                              "on time only, e.g. a translating structure: each function is "
                              "evaluated once per evaluation instead of once per particle",
                  .default_value = false}),

          // particle types that may carry a per-particle Dirichlet function id
          parameter<std::optional<std::vector<std::string>>>("DIRICHLET_BOUNDARY_CONDITION_FLAGGED",
              {.description = "Particle type names for which individual particles may carry a "
                              "DIRICHLET_FUNCT id (per-particle Dirichlet BC), e.g. '[phase1, "
                              "phase2]'. \nNote: Individually flagged particles override a "
                              "potentially set condition via DIRICHLET_BOUNDARY_CONDITION."}),

          // dirichlet boundary condition of particle phase given by function
          parameter<std::string>("DIRICHLET_BOUNDARY_CONDITION",
              {.description =
                      "Refer to the function ID describing the dirichlet boundary condition of "
                      "particle phase",
                  .default_value = "none"}),

          // temperature boundary condition of particle phase given by function
          parameter<std::string>("TEMPERATURE_BOUNDARY_CONDITION",
              {.description =
                      "Refer to the function ID describing the temperature boundary condition of "
                      "particle phase",
                  .default_value = "none"}),

          // kinematic constraint
          parameter<Constraint>(
              "CONSTRAINT", {.description = "type of kinematic constraint imposed",
                                .default_value = Particle::NoConstraint}),
      },
      {.required = false}));

  specs.push_back(group("PARTICLE DYNAMIC/OPEN BOUNDARIES",
      {

          // open dirichlet boundaries
          list("DIRICHLET_BOUNDARIES", {all_dirichlet_boundaries}, {.required = false}),
          // open neumann boundaries
          list("NEUMANN_BOUNDARIES", {all_neumann_boundaries}, {.required = false})},
      {.required = false}));

  /*-------------------------------------------------------------------------*
   | smoothed particle hydrodynamics (SPH) specific control parameters       |
   *-------------------------------------------------------------------------*/
  specs.push_back(group("PARTICLE DYNAMIC/SPH",
      {

          // write particle-wall interaction output
          parameter<bool>("WRITE_PARTICLE_WALL_INTERACTION",
              {.description = "write particle-wall interaction output", .default_value = false}),

          // type of smoothed particle hydrodynamics kernel
          parameter<KernelType>(
              "KERNEL", {.description = "type of smoothed particle hydrodynamics kernel",
                            .default_value = Particle::CubicSpline}),

          // kernel space dimension number
          parameter<KernelSpaceDimension>(
              "KERNEL_SPACE_DIM", {.description = "kernel space dimension number",
                                      .default_value = Particle::KernelSpaceDimension::Kernel3D}),

          parameter<double>("INITIALPARTICLESPACING",
              {.description = "initial spacing of particles", .default_value = 0.0}),

          // type of smoothed particle hydrodynamics equation of state
          parameter<EquationOfStateType>("EQUATIONOFSTATE",
              {.description = "type of smoothed particle hydrodynamics equation of state",
                  .default_value = Particle::GenTait}),

          // type of smoothed particle hydrodynamics momentum formulation
          parameter<MomentumFormulationType>("MOMENTUMFORMULATION",
              {.description = "type of smoothed particle hydrodynamics momentum formulation",
                  .default_value = Particle::AdamiMomentumFormulation}),

          // type of density evaluation scheme
          parameter<DensityEvaluationScheme>(
              "DENSITYEVALUATION", {.description = "type of density evaluation scheme",
                                       .default_value = Particle::DensitySummation}),

          // type of density correction scheme
          parameter<DensityCorrectionScheme>(
              "DENSITYCORRECTION", {.description = "type of density correction scheme",
                                       .default_value = Particle::NoCorrection}),

          // type of boundary particle formulation
          parameter<BoundaryParticleFormulationType>("BOUNDARYPARTICLEFORMULATION",
              {.description = "type of boundary particle formulation",
                  .default_value = Particle::NoBoundaryFormulation}),

          // type of boundary particle interaction
          parameter<BoundaryParticleInteraction>("BOUNDARYPARTICLEINTERACTION",
              {.description = "type of boundary particle interaction",
                  .default_value = Particle::NoSlipBoundaryParticle}),

          // type of wall formulation
          parameter<WallFormulationType>(
              "WALLFORMULATION", {.description = "type of wall formulation",
                                     .default_value = Particle::NoWallFormulation}),

          // type of transport velocity formulation
          parameter<TransportVelocityFormulation>("TRANSPORTVELOCITYFORMULATION",
              {.description = "type of transport velocity formulation",
                  .default_value = Particle::NoTransportVelocity}),

          // reduced dimension scale factor for computing SPH forces
          parameter<double>("REDUCED_DIMENSION_SCALE_FACTOR",
              {.description =
                      "scaling factor for the wall contact force to consider reduced "
                      "dimensional modeling from 1D or 2D particle field to 3D structural field",
                  .default_value = 1.0}),

          // type of temperature evaluation scheme
          parameter<TemperatureEvaluationScheme>(
              "TEMPERATUREEVALUATION", {.description = "type of temperature evaluation scheme",
                                           .default_value = Particle::NoTemperatureEvaluation}),

          parameter<bool>("TEMPERATUREGRADIENT",
              {.description = "evaluate temperature gradient", .default_value = false}),

          // type of heat source
          parameter<HeatSourceType>("HEATSOURCETYPE",
              {.description = "type of heat source", .default_value = Particle::NoHeatSource}),

          parameter<int>("HEATSOURCE_FUNCT",
              {.description = "number of function governing heat source", .default_value = -1}),

          parameter<std::string>("HEATSOURCE_DIRECTION",
              {.description = "direction of surface heat source", .default_value = "0.0 0.0 0.0"}),

          // evaporation induced heat loss
          parameter<bool>("VAPOR_HEATLOSS",
              {.description = "evaluate evaporation induced heat loss", .default_value = false}),
          parameter<double>("VAPOR_HEATLOSS_LATENTHEAT",
              {.description = "latent heat in heat loss formula", .default_value = 0.0}),
          parameter<double>("VAPOR_HEATLOSS_ENTHALPY_REFTEMP",
              {.description = "enthalpy reference temperature in heat loss formula",
                  .default_value = 0.0}),
          parameter<double>("VAPOR_HEATLOSS_PFAC",
              {.description = "pressure factor in heat loss formula", .default_value = 0.0}),
          parameter<double>("VAPOR_HEATLOSS_TFAC",
              {.description = "temperature factor in heat loss formula", .default_value = 0.0}),

          // evaporation induced recoil pressure
          parameter<bool>(
              "VAPOR_RECOIL", {.description = "evaluate evaporation induced recoil pressure",
                                  .default_value = false}),
          parameter<double>("VAPOR_RECOIL_BOILINGTEMPERATURE",
              {.description = "boiling temperature in recoil pressure formula",
                  .default_value = 0.0}),
          parameter<double>("VAPOR_RECOIL_PFAC",
              {.description = "pressure factor in recoil pressure formula", .default_value = 0.0}),
          parameter<double>(
              "VAPOR_RECOIL_TFAC", {.description = "temperature factor in recoil pressure formula",
                                       .default_value = 0.0}),

          // type of surface tension formulation
          parameter<SurfaceTensionFormulation>(
              "SURFACETENSIONFORMULATION", {.description = "type of surface tension formulation",
                                               .default_value = Particle::NoSurfaceTension}),

          parameter<int>("SURFACETENSION_RAMP_FUNCT",
              {.description = "number of function governing surface tension ramp",
                  .default_value = -1}),

          parameter<double>("SURFACETENSIONCOEFFICIENT",
              {.description = "constant part of surface tension coefficient",
                  .default_value = -1.0}),
          parameter<double>("SURFACETENSIONMINIMUM",
              {.description =
                      "minimum surface tension coefficient in case of temperature dependence",
                  .default_value = 0.0}),
          parameter<double>("SURFACETENSIONTEMPFAC",
              {.description = "factor of dependence of surface tension coefficient on temperature",
                  .default_value = 0.0}),
          parameter<double>("SURFACETENSIONREFTEMP",
              {.description = "reference temperature for surface tension coefficient",
                  .default_value = 0.0}),

          // wetting
          parameter<double>("STATICCONTACTANGLE",
              {.description = "static contact angle in degree with wetting effects",
                  .default_value = 0.0}),
          parameter<double>("TRIPLEPOINTNORMAL_CORR_CF_LOW",
              {.description = "triple point normal correction wall color field low",
                  .default_value = 0.0}),
          parameter<double>("TRIPLEPOINTNORMAL_CORR_CF_UP",
              {.description = "triple point normal correction wall color field up",
                  .default_value = 0.0}),

          // interface viscosity
          parameter<bool>("INTERFACE_VISCOSITY",
              {.description = "evaluate interface viscosity", .default_value = false}),
          parameter<double>("INTERFACE_VISCOSITY_LIQUIDGAS",
              {.description = "artificial viscosity on liquid-gas interface",
                  .default_value = 0.0}),
          parameter<double>("INTERFACE_VISCOSITY_SOLIDLIQUID",
              {.description = "artificial viscosity on solid-liquid interface",
                  .default_value = 0.0}),

          // barrier force
          parameter<bool>(
              "BARRIER_FORCE", {.description = "evaluate barrier force", .default_value = false}),
          parameter<double>("BARRIER_FORCE_DISTANCE",
              {.description = "barrier force distance", .default_value = 0.0}),
          parameter<double>("BARRIER_FORCE_TEMPSCALE",
              {.description = "barrier force temperature scaling", .default_value = 0.0}),
          parameter<double>("BARRIER_FORCE_STIFF_HEAVY",
              {.description = "barrier force stiffness of heavy phase", .default_value = -1.0}),
          parameter<double>("BARRIER_FORCE_DAMP_HEAVY",
              {.description = "barrier force damping parameter of heavy phase",
                  .default_value = 0.0}),
          parameter<double>("BARRIER_FORCE_STIFF_GAS",
              {.description = "barrier force stiffness of gas phase", .default_value = -1.0}),
          parameter<double>("BARRIER_FORCE_DAMP_GAS",
              {.description = "barrier force damping parameter of gas phase",
                  .default_value = 0.0}),

          // linear transition in surface tension evaluation
          parameter<double>("TRANS_REF_TEMPERATURE",
              {.description = "transition reference temperature", .default_value = 0.0}),
          parameter<double>("TRANS_DT_SURFACETENSION",
              {.description = "transition temperature difference for surface tension evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_MARANGONI",
              {.description = "transition temperature difference for marangoni evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_CURVATURE",
              {.description = "transition temperature difference for curvature evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_WETTING",
              {.description = "transition temperature difference for wetting evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_INTVISC",
              {.description =
                      "transition temperature difference for interface viscosity evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_BARRIER",
              {.description = "transition temperature difference for barrier force evaluation",
                  .default_value = 0.0}),

          // type of phase change
          parameter<PhaseChangeType>("PHASECHANGETYPE",
              {.description = "type of phase change", .default_value = Particle::NoPhaseChange}),

          // definition of phase change
          parameter<std::string>("PHASECHANGEDEFINITION",
              {.description = "phase change definition", .default_value = "none"}),

          // type of rigid particle contact
          parameter<RigidParticleContactType>(
              "RIGIDPARTICLECONTACTTYPE", {.description = "type of rigid particle contact",
                                              .default_value = Particle::NoRigidParticleContact}),

          parameter<double>("RIGIDPARTICLECONTACTSTIFF",
              {.description = "rigid particle contact stiffness", .default_value = -1.0}),
          parameter<double>("RIGIDPARTICLECONTACTDAMP",
              {.description = "rigid particle contact damping parameter", .default_value = 0.0})},
      {.required = false}));

  /*-------------------------------------------------------------------------*
   | discrete element method (DEM) specific control parameters               |
   *-------------------------------------------------------------------------*/
  specs.push_back(group("PARTICLE DYNAMIC/DEM",
      {

          // write particle energy output
          parameter<bool>("WRITE_PARTICLE_ENERGY",
              {.description = "write particle energy output", .default_value = false}),

          // write particle-wall interaction output
          parameter<bool>("WRITE_PARTICLE_WALL_INTERACTION",
              {.description = "write particle-wall interaction output", .default_value = false}),

          // type of normal contact law
          deprecated_selection<NormalContact>("NORMALCONTACTLAW",
              {
                  {"NormalLinearSpring", Particle::NormalLinSpring},
                  {"NormalLinearSpringDamp", Particle::NormalLinSpringDamp},
                  {"NormalHertz", Particle::NormalHertz},
                  {"NormalLeeHerrmann", Particle::NormalLeeHerrmann},
                  {"NormalKuwabaraKono", Particle::NormalKuwabaraKono},
                  {"NormalTsuji", Particle::NormalTsuji},
              },
              {.description = "normal contact law for particles",
                  .default_value = Particle::NormalLinSpring}),

          // type of tangential contact law
          parameter<TangentialContact>(
              "TANGENTIALCONTACTLAW", {.description = "tangential contact law for particles",
                                          .default_value = Particle::NoTangentialContact}),

          // type of rolling contact law
          parameter<RollingContact>(
              "ROLLINGCONTACTLAW", {.description = "rolling contact law for particles",
                                       .default_value = Particle::NoRollingContact}),

          // type of normal adhesion law
          parameter<AdhesionLaw>(
              "ADHESIONLAW", {.description = "type of adhesion law for particles",
                                 .default_value = Particle::NoAdhesion}),

          // type of (random) surface energy distribution
          parameter<SurfaceEnergyDistribution>("ADHESION_SURFACE_ENERGY_DISTRIBUTION",
              {.description = "type of (random) surface energy distribution",
                  .default_value = SurfaceEnergyDistribution::Constant}),

          parameter<std::optional<double>>(
              "MIN_RADIUS", {.description = "minimum allowed particle radius: used as lower cutoff "
                                            "for a radius distribution",
                                .validator = null_or(positive<double>())}),
          parameter<double>("MAX_RADIUS",
              {.description = "maximum allowed particle radius: used as upper cutoff for a radius "
                              "distribution and to calculate the contact stiffness.",
                  .default_value = 0.0}),
          parameter<double>("MAX_VELOCITY",
              {.description = "maximum expected particle velocity", .default_value = -1.0}),

          // type of initial particle radius assignment
          parameter<InitialRadiusAssignment>(
              "INITIAL_RADIUS", {.description = "type of initial particle radius assignment",
                                    .default_value = Particle::RadiusFromParticleMaterial}),

          parameter<std::optional<double>>("RADIUSDISTRIBUTION_SIGMA",
              {
                  .description = "Standard deviation sigma of random particle radius distribution",
                  .validator = null_or(positive<double>()),
              }),

          parameter<double>("REL_PENETRATION",
              {.description = "maximum allowed relative penetration", .default_value = -1.0}),
          parameter<double>(
              "NORMAL_STIFF", {.description = "normal contact stiffness", .default_value = -1.0}),
          parameter<double>("NORMAL_DAMP",
              {.description = "normal contact damping parameter", .default_value = -1.0}),
          parameter<double>("COEFF_RESTITUTION",
              {.description = "coefficient of restitution", .default_value = -1.0}),
          parameter<double>("DAMP_REG_FAC",
              {.description = "linearly regularized damping normal force in the interval "
                              "$|g| < (\\text{DAMP_REG_FAC} \\cdot r_{\\min})$",
                  .default_value = -1.0}),
          parameter<bool>(
              "TENSION_CUTOFF", {.description = "evaluate tension cutoff of normal contact force",
                                    .default_value = true}),


          parameter<double>(
              "POISSON_RATIO", {.description = "poisson ratio", .default_value = -1.0}),
          parameter<double>(
              "YOUNG_MODULUS", {.description = "young's modulus", .default_value = -1.0}),

          parameter<double>(
              "FRICT_COEFF_TANG", {.description = "friction coefficient for tangential contact",
                                      .default_value = -1.0}),
          parameter<double>("FRICT_COEFF_ROLL",
              {.description = "friction coefficient for rolling contact", .default_value = -1.0}),

          parameter<double>(
              "ADHESION_DISTANCE", {.description = "adhesion distance between interacting surfaces",
                                       .default_value = -1.0}),

          parameter<double>("ADHESION_MAX_CONTACT_PRESSURE",
              {.description = "adhesion maximum contact pressure", .default_value = 0.0}),
          parameter<double>("ADHESION_MAX_CONTACT_FORCE",
              {.description = "adhesion maximum contact force", .default_value = 0.0}),
          parameter<bool>("ADHESION_USE_MAX_CONTACT_FORCE",
              {.description = "use maximum contact force instead of maximum contact pressure",
                  .default_value = false}),

          parameter<bool>("ADHESION_VDW_CURVE_SHIFT",
              {.description = "shifts van-der-Waals-curve to g = 0", .default_value = false}),

          parameter<double>(
              "ADHESION_HAMAKER", {.description = "hamaker constant of van-der-Waals interaction",
                                      .default_value = -1.0}),

          parameter<double>("ADHESION_SURFACE_ENERGY",
              {.description = "adhesion surface energy for the calculation of the pull-out force",
                  .default_value = -1.0}),
          parameter<double>("ADHESION_SURFACE_ENERGY_DISTRIBUTION_STDDEV",
              {.description = "standard deviation of adhesion surface energy distribution",
                  .default_value = -1.0}),

          parameter<double>("ADHESION_SURFACE_ENERGY_DISTRIBUTION_CUTOFF_FACTOR",
              {.description = "adhesion surface energy distribution limited by multiple of "
                              "standard deviation",
                  .default_value = -1.0}),
          parameter<double>("ADHESION_SURFACE_ENERGY_FACTOR",
              {.description = "factor to calculate minimum adhesion surface energy",
                  .default_value = 1.0})},
      {.required = false}));

  /*-------------------------------------------------------------------------*
   | peridynamics (PD) specific control parameters                           |
   *-------------------------------------------------------------------------*/
  specs.push_back(group("PARTICLE DYNAMIC/PD",
      {

          // compact support in peridynamics
          parameter<double>(
              "INTERACTION_HORIZON", {.description = "peridynamic horizon", .default_value = 0.0}),

          // the spatial grid spacing of the peridynamic grid
          parameter<double>("PERIDYNAMIC_GRID_SPACING",
              {.description = "peridynamic grid spacing", .default_value = 0.0}),

          // type of normal contact law
          deprecated_selection<NormalContact>("NORMALCONTACTLAW",
              {
                  {"NormalLinearSpring", Particle::NormalLinSpring},
                  {"NormalLinearSpringDamp", Particle::NormalLinSpringDamp},
              },
              {.description = "normal contact law for particles between peridynamic bodies",
                  .default_value = Particle::NormalLinSpring}),

          parameter<double>(
              "NORMAL_STIFF", {.description = "normal contact stiffness", .default_value = 0.0}),
          parameter<double>("NORMAL_DAMP",
              {.description = "normal contact damping parameter", .default_value = 0.0}),

          // runtime csv output of the resultant force on boundary particles
          parameter<bool>("WRITE_BOUNDARY_FORCE",
              {.description = "write the resultant force on boundary particles (current and "
                              "time-averaged since the last output) to a csv file",
                  .default_value = false}),

          // force history on boundary particles
          parameter<int>("BOUNDARY_FORCE_EVERY",
              {.description = "with WRITE_BOUNDARY_FORCE: additionally write the resultant force "
                              "on boundary particles every this many time steps to the csv file "
                              "boundary-force-history (mean force since the previous row); 0 = off",
                  .default_value = 0}),

          // surface correction of the bond stiffness
          parameter<bool>("SURFACE_CORRECTION",
              {.description =
                      "surface correction of the bond micromoduli by per-particle factors w (bond "
                      "factor (w_i + w_j) / 2, replacing the partial volume factor of the bond), "
                      "given for every PD particle in the particle input (SURFACE_CORRECTION w), "
                      "e.g. operator-matching weights (perifit) computed with full particle "
                      "volumes",
                  .default_value = false}),

          // critical stretch per particle (e.g. varying through the thickness of a plate)
          parameter<bool>("CRITICAL_STRETCH_PER_PARTICLE",
              {.description =
                      "critical stretch given for every PD particle in the particle input "
                      "(CRITICAL_STRETCH s); a bond fails when its stretch reaches the mean "
                      "critical stretch (s_i + s_j) / 2 of its two particles. The material "
                      "CRITICAL_STRETCH is still required but not used. On restart the values "
                      "come from the restart file",
                  .default_value = false}),

          // elastic-plastic bonds (bond-based plasticity with linear hardening, failure at the
          // total stretch)
          parameter<bool>("BOND_PLASTICITY",
              {.description =
                      "elastic-plastic bonds: one plastic stretch s_p per bond, bond force "
                      "c (s - s_p); tension yields at s - s_p = s_Y + h max(s_p, 0) with "
                      "s_Y = BOND_YIELD_STRETCH, h = eta / (1 - eta), eta = "
                      "BOND_HARDENING_RATIO (tangent eta c); compression yields perfectly "
                      "plastically at s - s_p = -BOND_COMPRESSIVE_YIELD_STRETCH (0: elastic); "
                      "unloading with c (a bond with a compressive set carries c s_Y in tension "
                      "until s = s_Y); a bond fails at the total stretch (critical stretch) and, "
                      "if given, at the compressive total stretch "
                      "-BOND_COMPRESSIVE_FAILURE_STRETCH. Regular PD lattices, horizon of up to "
                      "4 dx",
                  .default_value = false}),
          parameter<double>("BOND_YIELD_STRETCH",
              {.description = "tensile yield stretch s_Y of the elastic-plastic bonds (> 0)",
                  .default_value = 0.0}),
          parameter<double>("BOND_HARDENING_RATIO",
              {.description = "ratio eta of the bond tangent after tensile yield to the elastic "
                              "micromodulus, 0 <= eta < 1 (continuum tangent eta E)",
                  .default_value = 0.0}),
          parameter<double>("BOND_COMPRESSIVE_YIELD_STRETCH",
              {.description = "compressive yield stretch s_Yc of the elastic-plastic bonds "
                              "(magnitude, perfectly plastic; 0: elastic in compression)",
                  .default_value = 0.0}),
          parameter<double>("BOND_COMPRESSIVE_FAILURE_STRETCH",
              {.description = "elastic-plastic bonds: magnitude of the compressive total "
                              "stretch at which a bond fails (crushed; the pair then interacts by "
                              "contact); 0: no compressive failure",
                  .default_value = 0.0}),

          // Coulomb friction of the short-range contact (regularized by a tangential damping)
          parameter<double>("FRICTION_COEFF",
              {.description = "Coulomb friction coefficient of the contact between PD particles "
                              "without an intact bond; the tangential force opposes the tangential "
                              "relative velocity v_t with magnitude min(mu |f_n|, "
                              "FRICTION_TANGENTIAL_DAMP |v_t|) (viscous regularization of the "
                              "stick state); 0 = frictionless",
                  .default_value = 0.0}),
          parameter<double>("FRICTION_COEFF_BOUNDARY",
              {.description = "Coulomb friction coefficient of the contact between PD particles "
                              "and boundary particles (as FRICTION_COEFF); 0 = frictionless",
                  .default_value = 0.0}),
          parameter<double>("FRICTION_TANGENTIAL_DAMP",
              {.description = "regularization coefficient c_t of the Coulomb friction (tangential "
                              "force c_t |v_t| below the Coulomb limit; with "
                              "FRICTION_TANGENTIAL_SPRING the tangential damping); required if a "
                              "friction coefficient is positive",
                  .default_value = 0.0}),
          parameter<bool>("FRICTION_TANGENTIAL_SPRING",
              {.description =
                      "Coulomb friction with an elastic tangential spring (Cundall-Strack): the "
                      "tangential displacement xi of each contact is stored (PD-PD contacts: 12 "
                      "slots per particle; contact with the smooth boundary surface: one per "
                      "particle), f_t = -k_t xi - c_t v_t capped at mu |f_n| (slip sets xi to the "
                      "Coulomb force), so that loaded contacts stick instead of creeping (viscous "
                      "regularization, also used for a PD-PD contact if all slots of the particle "
                      "are taken); boundary friction requires BOUNDARY_CONTACT_SMOOTH_SURFACE",
                  .default_value = false}),
          parameter<double>("FRICTION_TANGENTIAL_STIFF_RATIO",
              {.description = "k_t / NORMAL_STIFF of the tangential spring (0.5 as in DEM ice "
                              "models, e.g. Ji, Di & Liu 2015)",
                  .default_value = 0.5}),
          parameter<double>("BOUNDARY_CONTACT_CRUSHING_PRESSURE",
              {.description =
                      "crushing pressure sigma_p of the contact between PD particles and the "
                      "smooth boundary surface, local crushing as a contact property (Hopkins "
                      "1992; Paavilainen et al. 2009; Polojarvi 2022): the normal contact force on "
                      "a PD particle is limited to sigma_p dx^2 (2D: sigma_p dx per unit "
                      "thickness), the overlap beyond the elastic "
                      "part at the limit is plastic (unloading releases only the elastic part), "
                      "and a particle whose overlap reaches dx is crushed and removed (the removed "
                      "volume equals the volume swept by the boundary); sigma_p: measured "
                      "compressive strength; 0 = off; requires BOUNDARY_CONTACT_SMOOTH_SURFACE",
                  .default_value = 0.0}),

          // direction of the contact force between PD and boundary particles
          parameter<bool>("BOUNDARY_CONTACT_SURFACE_NORMAL",
              {.description =
                      "the contact force between a PD particle and a boundary particle acts along "
                      "the outward surface normal of the boundary (computed once from the "
                      "boundary particle distribution; translating structures only) instead of "
                      "the line between the particle centres, so that PD bodies slide without "
                      "friction along discretized (staircase) boundary surfaces; the magnitude is "
                      "unchanged",
                  .default_value = false}),

          // contact between PD particles and the smooth surface of the boundary particles
          parameter<bool>("BOUNDARY_CONTACT_SMOOTH_SURFACE",
              {.description =
                      "one contact of a PD particle with the boundary: gap g = f - dx with the "
                      "implicit moving least squares surface f(x) = sum_j W_j (x - x_j).n_j / "
                      "sum_j W_j of the outermost boundary particles and their outward normals "
                      "(plane fit of the outermost neighbours within 2 dx; W_j = (1 - |x - x_j|^2 "
                      "/ R^2)^3 h_j, R = 2 PERIDYNAMIC_GRID_SPACING, visibility h_j fading behind "
                      "the tangent plane of x_j over dx/2); spring force from the potential "
                      "phi(S) k g^2 / 2 with a smooth onset phi of the weight sum S = sum_j W_j "
                      "(conservative, smooth along the surface; sharp convex edges are rounded by "
                      "~0.3 dx; resists penetrations up to ~1.35 dx), damping and friction along "
                      "the normal and "
                      "tangent of the surface; the PD contact reaction is part of the boundary "
                      "force sums, "
                      "not of the force state of the boundary particles; translating structures "
                      "only",
                  .default_value = false}),

          // multi-rate time integration of the peridynamic phase
          parameter<int>("PD_SUBSTEPS",
              {.description =
                      "number of velocity Verlet substeps of the peridynamic phase per time step "
                      "(bond, contact, gravity and damping forces); the force of the fluid on the "
                      "PD particles is applied as half kicks at the ends of the time step "
                      "(impulse multiple time stepping); 1 = single rate; a restart requires a "
                      "restart file written with substeps",
                  .default_value = 1,
                  .validator = positive<int>()}),

          // peridynamic dimension
          parameter<PeridynamicDimension>(
              "PD_DIMENSION", {.description = "dimension of peridynamic problem",
                                  .default_value = PeridynamicDimension::Peridynamic_3D}),

          // peridynamic initial bond breakage via line definition
          list("PRE_CRACK_LINES",
              all_of({
                  parameter<std::vector<double>>(
                      "START", {.description = "start point of crack line segment", .size = 3}),
                  parameter<std::vector<double>>(
                      "END", {.description = "end point of crack line segment", .size = 3}),
              }),
              {.description =
                      "Pre-crack line segments (3D): peridynamic bonds whose reference "
                      "configuration crosses any of these segments are broken at initialization. "
                      "Each entry is defined by a start point (START) and an end point (END).",
                  .required = false}),

          // peridynamic initial bond breakage via plane definition
          list("PRE_CRACK_PLANES",
              all_of({
                  parameter<std::vector<double>>(
                      "P0", {.description = "corner point 0 of parallelogram patch", .size = 3}),
                  parameter<std::vector<double>>(
                      "P1", {.description = "corner point 1 of parallelogram patch", .size = 3}),
                  parameter<std::vector<double>>(
                      "P2", {.description = "corner point 2 of parallelogram patch", .size = 3}),
              }),
              {.description =
                      "Pre-crack parallelogram patches (3D): peridynamic bonds whose reference "
                      "configuration crosses any of these patches are broken at initialization. "
                      "Each patch is defined by three corner points P0, P1, P2; the parallelogram "
                      "is spanned by edges e1 = P1-P0 and e2 = P2-P0",
                  .required = false})},
      {.required = false}));
  return specs;
}

/*---------------------------------------------------------------------------*
 | set the particle conditions                                               |
 *---------------------------------------------------------------------------*/
void Particle::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*-------------------------------------------------------------------------*
   | particle wall condition                                                 |
   *-------------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition surfpartwall("DESIGN SURFACE PARTICLE WALL", "ParticleWall",
      "Wall for particle interaction with (optional) material definition",
      Core::Conditions::ParticleWall, true, Core::Conditions::geometry_type_surface);

  surfpartwall.add_component(parameter<int>("MAT"));

  condlist.push_back(surfpartwall);
}

template <>
std::string Particle::open_boundary_type_to_string(Particle::DirichletOpenBoundaryType type)
{
  switch (type)
  {
    case Particle::NoDirichletOpenBoundary:
      return "NoDirichletOpenBoundary";
    case Particle::DirichletNormalToPlane:
      return "DirichletNormalToPlane";
    default:
      FOUR_C_THROW("Unknown Dirichlet open boundary type");
  }
}

template <>
std::string Particle::open_boundary_type_to_string(Particle::NeumannOpenBoundaryType type)
{
  switch (type)
  {
    case Particle::NoNeumannOpenBoundary:
      return "NoNeumannOpenBoundary";
    case Particle::NeumannNormalToPlane:
      return "NeumannNormalToPlane";
    default:
      FOUR_C_THROW("Unknown Neumann open boundary type");
  }
}

FOUR_C_NAMESPACE_CLOSE
