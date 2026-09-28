# The surface bus: per-column state that more than one coupling reads or
# writes. Couplings never keep a private copy of these quantities; they
# read and write them here, so the radiation, the surface schemes, the
# buildings and the fog/dew all see the same numbers.
#
# This is the I1 bus: it holds exactly what the pre-refactor loop shared
# (the slab temperature, the irradiances and the combined surface-loss
# term). The surface-exchange issue extends it with the MOST fields
# (T_air, q_air, U, z_ref, roof-aware k_air, H, LE, E, τx, τy, G).

# ---------------------------------------------------------------------------
# Legacy constants of the slab scheme.
#
# These duplicate numbers that live in src (RadiationCore.STEFAN_BOLTZMANN
# = 5.670374419e-8, MoistThermodynamics latent heat). They stay until the
# slab scheme is replaced by SurfaceExchange / SoilColumn / UrbanFacet
# skins, because changing them changes the answer and the refactor has to
# be bit-for-bit.
# ---------------------------------------------------------------------------
const STEFAN = 5.67e-8
const _L_v   = 2.45e6    # J kg⁻¹ latent heat of vaporization

# ---------------------------------------------------------------------------
# Per-class physical parameters
# ---------------------------------------------------------------------------

"""
    SurfaceParams(C_s, albedo, emissivity)

Per-surface-class thermal and optical parameters of the slab scheme,
indexed by the `SURFACE_*` class code: areal heat capacity `C_s`
(J m⁻² K⁻¹), `albedo`, `emissivity`.
"""
struct SurfaceParams
    C_s::Vector{Float64}        # areal heat capacity [J/m²/K]
    albedo::Vector{Float64}
    emissivity::Vector{Float64}
end

"""
    default_surface_params(; μ₀ = 0.7) -> SurfaceParams

The five urban classes (1 street, 2 roof, 3 park, 4 water, 5 wall).

Albedo and emissivity are NOT hand-entered here. They come from
`Aeolus.Radiosity.default_urban_materials`, which derives them by
band-averaging the Phase 36 spectral reflectance library over a mosaic of
real materials per class. `μ₀` (cosine solar zenith) only affects the
water class, whose Fresnel reflectance runs from ~0.03 overhead to ~0.29
at grazing incidence.

Heat capacity stays here: it is a bulk thermal property of the slab, not
an optical one. Walls have a high effective heat capacity (thick
masonry), water an enormous one.
"""
function default_surface_params(; μ₀::Real = 0.7)
    mats = default_urban_materials(; μ₀ = μ₀)
    C_s  = [3e5, 1.8e5, 5e5, 8e6, 4e5]
    return SurfaceParams(C_s, copy(mats.albedo), copy(mats.emissivity))
end

# ---------------------------------------------------------------------------
# The bus
# ---------------------------------------------------------------------------

"""
    SurfaceBus

Per-column surface state shared between couplings (all `(Nx, Ny)`):

* `Ts` — surface (slab skin) temperature, K;
* `sw` — shortwave irradiance at the surface, W m⁻² (per unit surface);
* `lw` — downwelling longwave at the surface, W m⁻²;
* `sensible` — the combined surface loss term of the slab scheme, W m⁻²
  (sensible + latent − dew heating + window transmission − window sun on
  walls). Recomputed every step by the surface-flux stage;
* `dew` — surface dew reservoir, kg m⁻²;
* `sw_beam` — clear-sky direct beam on a horizontal plane at the last
  radiation update, W m⁻²;
* `params` — the per-class [`SurfaceParams`](@ref);
* `swsol`, `lwsol` — the last radiosity solutions (`swsol === nothing`
  at night);
* `L_sky` — downwelling longwave from the sky at the surface, W m⁻², set
  by the `:column_radiation` slot at every radiation update (#130);
* `dni`, `dhi` — direct-normal and diffuse-horizontal shortwave for the
  radiosity solve, W m⁻², set by the `:column_radiation` slot; `NaN`
  means "use the radiosity coupling's own clear-sky model" (#130).
"""
mutable struct SurfaceBus
    Ts::Matrix{Float64}
    sw::Matrix{Float64}
    lw::Matrix{Float64}
    sensible::Matrix{Float64}
    dew::Matrix{Float64}
    sw_beam::Float64
    params::SurfaceParams
    swsol::Any
    lwsol::Any
    L_sky::Float64
    dni::Float64
    dhi::Float64
end

function SurfaceBus(Nx::Integer, Ny::Integer; Ts0::Real = 285.0)
    return SurfaceBus(fill(Float64(Ts0), Nx, Ny), zeros(Nx, Ny), fill(320.0, Nx, Ny),
                      zeros(Nx, Ny), zeros(Nx, Ny), 0.0, default_surface_params(),
                      nothing, nothing, NaN, NaN, NaN)
end
