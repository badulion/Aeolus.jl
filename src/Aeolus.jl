module Aeolus

# Aeolus.jl — a Julia-native large-eddy simulation framework.
#
# Design intent: the spatial discretization is behind an `AbstractMesh`
# interface. Phase 0/1 ships a finite-volume backend on a staggered
# Cartesian grid (RegularGrid). A future spectral-element backend can be
# added without changing user code — it plugs in via multiple dispatch on
# the mesh type inside the Operators layer.

include("Grids.jl")
include("Fields.jl")
include("BoundaryConditions.jl")
include("Operators.jl")
include("TimeSteppers.jl")
include("PressureSolvers.jl")
include("Simulations.jl")
include("IO.jl")
include("Microphysics.jl")
include("IBM.jl")
include("Parallel.jl")
include("Nesting.jl")
include("Backends.jl")
include("SpectralElement.jl")
include("Radiation.jl")
include("Chemistry.jl")
include("Particles.jl")
include("ML.jl")
include("DataAssimilation.jl")

using .Grids
using .Fields
using .BoundaryConditions
using .Operators
using .TimeSteppers
using .PressureSolvers
using .Simulations
using .IO
using .Microphysics
using .IBM
using .Parallel
using .Nesting
using .Backends
using .SpectralElement
using .Radiation
using .Chemistry
using .Particles
using .ML
using .DataAssimilation

# Re-exports for the user-facing API.
export AbstractMesh, RegularGrid
export Field, CellCenter, FaceX, FaceY, FaceZ
export Periodic, NoFlux, ValueBC, FluxBC, FieldBoundaryConditions, fill_halos!
export ∂x, ∂y, ∂z, divergence, laplacian, advect_centered!, advect_upwind!,
       interpolate_to_face!, interpolate_to_center!, momentum_advect!,
       add_buoyancy!, add_coriolis!, compute_smagorinsky_νt!,
       add_eddy_viscosity_u!, add_eddy_viscosity_v!, add_eddy_viscosity_w!,
       add_eddy_diffusion_scalar!,
       add_surface_flux!, add_sponge!,
       advect_weno5!
export RK3, step!
export AbstractPressureSolver, FFTPoissonSolver, CosineFFTPoissonSolver,
       CGPoissonSolver, solve!
export ImpermeableWall
export BoussinesqSimulation, run!, step_rk3!, kinetic_energy, max_divergence,
       max_νt, mean_θ_profile,
       horizontal_mean, horizontal_variance, resolved_flux_wθ
export write_snapshot, read_snapshot, checkpoint!, restart!, load_config
export apply_kessler_step!
export SolidMask, mark_solid!, apply_solid_mask!, add_ibm_penalty!,
       implicit_ibm_correction!
export AbstractPartition, SerialPartition, PencilPartition,
       nranks, is_serial, exchange_halos_distributed!
export NestSpec, apply_one_way_forcing!, apply_two_way_feedback!
export AbstractBackend, CPUBackend, GPUBackend, allocate, to_host, backend_name
export SpectralElementMesh, gll_nodes, gll_weights
export apply_radiative_cooling!, SlabLand, step_land!, surface_temperature
export apply_linear_chemistry!
export ParticleSet, add_particles!, advance_particles!, positions
export MLSGS, apply_mlsgs!
export Observation, assimilate_nudging!

end # module
