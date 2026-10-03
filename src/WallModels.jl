"""
    WallModels

Era VI, Phase 57: the modelled wall stress of wall-modelled LES. **Owner
of every modelled wall stress in the code base**: the channel harness,
the stable-boundary-layer cases, the TKE surface value and the Berlin
ground drag all call this module rather than writing their own.

The parts, each in its own file under `src/wall_models/` so the later
Phase 57 issues (matching height and filtering, TBLE, facades, dynamic
roughness) only add files and include lines:

  1. **Laws** (`laws.jl`) — [`AbstractWallLaw`](@ref) and the generics
     [`friction_velocity`](@ref), [`wall_stress`](@ref) and
     [`wall_drag`](@ref), with three equilibrium laws:
     [`SpaldingLaw`](@ref) (smooth wall, viscous sublayer to log layer
     in one formula), [`RoughLogLaw`](@ref) (fully rough, neutral) and
     [`MOSTWallLaw`](@ref) (rough, Monin–Obukhov stability correction).
     κ of the rough laws and the stability functions are *delegated* to
     the Era V contract `SurfaceExchange`; nothing is restated.
  2. **The operator** (`operator.jl`) — [`apply_wall_stress!`](@ref),
     which applies the stress of a law to the first velocity level next
     to the z-walls of a `RegularGrid`, implicitly (stable for any Δt),
     in the local (Schumann 1975 / Grötzbach 1987), plane-mean
     (Schumann 1975) or Moeng (1984) formulation. It is a
     `post_predictor!` hook; halos and the finite-volume operators are
     untouched, so the wall model is the only momentum flux through a
     free-slip (`ImpermeableWall`) wall face.
  3. **The wall-model input** (`matching.jl`, P57-b #183) —
     [`MatchingInput`](@ref): the velocity the law is evaluated with,
     sampled at a matching height off the first cell (Kawai & Larsson
     2012) and filtered in time (Yang, Park & Moin 2017) and along the
     wall (Bou-Zeid, Meneveau & Parlange 2005), against log-layer
     mismatch.
  4. **The benchmark cases** (`channel_cases.jl`, P54-d #174) —
     [`half_channel_case`](@ref) (the neutral ABL half channel of
     Porté-Agel et al. 2000 / Bou-Zeid et al. 2005) and
     [`channel_case`](@ref) (the two-wall channel at a given Re_τ), with
     [`advance_channel!`](@ref) and [`run_channel!`](@ref): the shared
     wall-bounded harness every Era VI closure is validated on. They add
     no wall physics; the diagnostics they report (Φ_m, the stress budget,
     E11) are `Operators.phi_m_profile`, `total_stress_profile` and
     `streamwise_spectrum`.

# References

  * Spalding, D. B. (1961). A single formula for the "law of the wall".
    *J. Appl. Mech.*, 28, 455-458.
  * Schumann, U. (1975). Subgrid scale model for finite difference
    simulations of turbulent flows in plane channels and annuli.
    *J. Comput. Phys.*, 18, 376-404.
  * Grötzbach, G. (1987). Direct numerical and large eddy simulation of
    turbulent channel flows. In *Encyclopedia of Fluid Mechanics*, vol. 6,
    1337-1391, Gulf Publishing.
  * Moeng, C.-H. (1984). A large-eddy-simulation model for the study of
    planetary boundary-layer turbulence. *J. Atmos. Sci.*, 41, 2052-2062.
  * Piomelli, U. & Balaras, E. (2002). Wall-layer models for large-eddy
    simulations. *Annu. Rev. Fluid Mech.*, 34, 349-374.
  * Kawai, S. & Larsson, J. (2012). Wall-modeling in large eddy
    simulation: length scales, grid resolution, and accuracy.
    *Phys. Fluids*, 24, 015105.
  * Yang, X. I. A., Park, G. I. & Moin, P. (2017). Log-layer mismatch
    and modeling of the fluctuating wall stress in wall-modeled
    large-eddy simulations. *Phys. Rev. Fluids*, 2, 104601.
  * Bou-Zeid, E., Meneveau, C. & Parlange, M. B. (2005). A
    scale-dependent Lagrangian dynamic model for large eddy simulation
    of complex turbulent flows. *Phys. Fluids*, 17, 025105.
  * Lee, M. & Moser, R. D. (2015). Direct numerical simulation of
    turbulent channel flow up to Re_τ ≈ 5200. *J. Fluid Mech.*, 774,
    395-415.
"""
module WallModels

using ..Grids
using ..Grids: RegularGrid
using ..BoundaryConditions: fill_halos!, Periodic, FieldBoundaryConditions
using ..Operators: AbstractSGSClosure, StaticSmagorinsky, compute_sgs!, box_filter!
using ..Fields: Field
import ..Operators: phi_m_profile, total_stress_profile
using ..Simulations: BoussinesqSimulation, step_rk3!
using ..TimeSteppers: step!
using Random
using ..SurfaceExchange
using ..SurfaceExchange: AbstractStabilityFunctions, BusingerDyerBeljaarsHoltslag,
                         SurfaceConstants, DEFAULT_SURFACE_CONSTANTS, von_karman

# Each file exports its own names, so a later phase adds a file and an
# include line here and nothing else.
include("wall_models/laws.jl")
include("wall_models/operator.jl")
include("wall_models/matching.jl")        # P57-b (#183): matching height, input filters
include("wall_models/tble.jl")            # P57-c (#189): non-equilibrium (TBLE) wall model, dynamic κ
include("wall_models/channel_cases.jl")   # P54-d (#174): half channel, channel

end # module
