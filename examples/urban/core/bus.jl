# The surface bus: per-column state that more than one coupling reads or
# writes. Couplings never keep a private copy of these quantities; they
# read and write them here, so the radiation, the surface schemes, the
# buildings and the fog/dew all see the same numbers.
#
# The I1 part holds what the pre-refactor loop shared (the surface
# temperature, the irradiances and a combined surface-loss term). The
# surface-exchange issue (#129) added the Monin–Obukhov exchange: the
# per-column surface implementers (`AbstractSurface`, built by per-class
# factories that couplings register), the roof-aware air level, the
# forcing each surface last saw and the fluxes it returned.

# ---------------------------------------------------------------------------
# Constants still used by couplings that later issues replace.
# ---------------------------------------------------------------------------
"""
Stefan–Boltzmann constant, delegated to `RadiationCore.STEFAN_BOLTZMANN`
(the example used to carry its own `5.67e-8`). Used by the sky longwave
of `radiosity_sun` until the column-radiation issue (#130).
"""
const STEFAN = Aeolus.RadiationCore.STEFAN_BOLTZMANN
# J kg⁻¹, the latent heat buildings_citymean converts its latent release
# with (and books to the ledger with). The surface exchange uses
# MoistThermodynamics.latent_heat_vapor(T); the per-building zones issue
# (#144) replaces the city-mean building and this constant.
const _L_v   = 2.45e6

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

Per-column surface state shared between couplings (all `(Nx, Ny)`).

Radiation and the other couplings:

* `Ts` — surface (skin) temperature, K;
* `sw` — shortwave irradiance at the surface, W m⁻² (per unit surface);
* `lw` — downwelling longwave at the surface, W m⁻²;
* `sensible` — **extra** energy the surface loses this step to couplings
  other than the surface exchange, W m⁻² (positive = the surface loses):
  window sun leaving the wall cells, sunlight transmitted by glazing,
  minus the latent heat of dew. Not the turbulent sensible flux (that is
  `H`). The surface exchange collects it every step and delivers it to
  the surfaces as `Q_extra` at their next step, then zeroes it;
* `dew` — surface dew reservoir, kg m⁻²;
* `sw_beam` — clear-sky direct beam on a horizontal plane at the last
  radiation update, W m⁻²;
* `params` — the per-class [`SurfaceParams`](@ref) (albedo, emissivity;
  `C_s` is the heat capacity of the slab surfaces);
* `swsol`, `lwsol` — the last radiosity solutions (`swsol === nothing`
  at night).

Surface exchange (#129):

* `surface` — the `AbstractSurface` of every column (`nothing` where the
  column is solid up to the lid), built by the surface exchange from the
  per-class `factories`;
* `factories` — `class => (owner, f)`: `f(case, I) -> AbstractSurface`
  builds the surface of column `I`; see [`register_surface_factory!`](@ref);
* `k_air` — the first air level above each column (1 on open ground, one
  above the roof on building columns, `Nz + 1` for a column solid to the
  lid): every surface flux goes into this cell;
* `z_ref` — height of that cell's centre above the surface, m;
* `T_air`, `q_air`, `U` — air temperature (K), specific humidity and
  wind speed (m s⁻¹) the surface last saw;
* `Rn`, `H`, `LE`, `G` — net radiation, sensible, latent and ground /
  storage heat flux of the last surface step, W m⁻² (Rn and G into the
  surface, H and LE upward), held until the next one;
* `E` — evaporation, kg m⁻² s⁻¹ (negative: dew onto the surface);
* `u_star` — friction velocity, m s⁻¹;
* `dew_mask` — the columns whose surface has no vapour exchange of its
  own (moisture availability 0), where the RadiationFog dew reservoir
  acts; elsewhere dew is the negative evaporation of the surface model.
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
    # surface exchange (#129)
    surface::Matrix{Union{Nothing,AbstractSurface}}
    factories::Dict{UInt8,Tuple{Symbol,Any}}
    k_air::Matrix{Int}
    z_ref::Matrix{Float64}
    T_air::Matrix{Float64}
    q_air::Matrix{Float64}
    U::Matrix{Float64}
    Rn::Matrix{Float64}
    H::Matrix{Float64}
    LE::Matrix{Float64}
    G::Matrix{Float64}
    E::Matrix{Float64}
    u_star::Matrix{Float64}
    dew_mask::BitMatrix
end

function SurfaceBus(Nx::Integer, Ny::Integer; Ts0::Real = 285.0)
    z() = zeros(Nx, Ny)
    return SurfaceBus(fill(Float64(Ts0), Nx, Ny), z(), fill(320.0, Nx, Ny),
                      z(), z(), 0.0, default_surface_params(), nothing, nothing,
                      Matrix{Union{Nothing,AbstractSurface}}(nothing, Nx, Ny),
                      Dict{UInt8,Tuple{Symbol,Any}}(), ones(Int, Nx, Ny), z(),
                      z(), z(), z(), z(), z(), z(), z(), z(), z(), trues(Nx, Ny))
end

"""
    register_surface_factory!(case, class, owner, f) -> case

Make coupling `owner` responsible for the surfaces of class `class`
(`SURFACE_*`): the surface exchange builds the surface of every column of
that class with `f(case, I::CartesianIndex{2}) -> AbstractSurface`. Call
it from the coupling's `build` (every `build` runs before any `init!`,
and the surfaces are built in the surface exchange's `init!`). Two
owners for one class are an error. Classes without a factory get the
surface exchange's `SlabLand` fallback.
"""
function register_surface_factory!(case, class::Integer, owner::Symbol, f)
    fs = case.bus.factories
    c = UInt8(class)
    haskey(fs, c) && fs[c][1] !== owner &&
        error("surface class $c already has a factory (from :$(fs[c][1])); :$owner cannot register another")
    fs[c] = (owner, f)
    return case
end

"""
    surface_store_energy(s) -> J m⁻² or `nothing`

Energy held by surface `s` relative to a fixed reference, less what left
it through its base, so that its change over a step equals the `G` that
`step_surface!` returned times the step (the per-column closure check of
the surface exchange). `nothing` for an implementer without a method
(the check then skips it).
"""
surface_store_energy(::AbstractSurface) = nothing
surface_store_energy(s::SlabLand) = s.C_s * s.T

"""
    skin_has_storage(s) -> Bool

Whether the `G` that `step_surface!(s, …)` returns is stored in the
surface layer itself (a slab: the ledger books it as `ΔS`) rather than
conducted into a substrate below a massless skin (a soil column: `G`).
"""
skin_has_storage(::AbstractSurface) = false
skin_has_storage(::SlabLand) = true
