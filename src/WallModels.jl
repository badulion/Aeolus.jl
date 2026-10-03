"""
    WallModels

Era VI, Phase 57: the modelled wall stress of wall-modelled LES. **Owner
of every modelled wall stress in the code base**: the channel harness,
the stable-boundary-layer cases, the TKE surface value and the Berlin
ground drag all call this module rather than writing their own.

Two parts, each in its own file under `src/wall_models/` so the later
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
  * Lee, M. & Moser, R. D. (2015). Direct numerical simulation of
    turbulent channel flow up to Re_τ ≈ 5200. *J. Fluid Mech.*, 774,
    395-415.
"""
module WallModels

using ..Grids: RegularGrid
using ..BoundaryConditions: fill_halos!, Periodic
using ..SurfaceExchange
using ..SurfaceExchange: AbstractStabilityFunctions, BusingerDyerBeljaarsHoltslag,
                         SurfaceConstants, DEFAULT_SURFACE_CONSTANTS, von_karman

# Each file exports its own names, so a later phase adds a file and an
# include line here and nothing else.
include("wall_models/laws.jl")
include("wall_models/operator.jl")
include("wall_models/immersed.jl")

end # module
