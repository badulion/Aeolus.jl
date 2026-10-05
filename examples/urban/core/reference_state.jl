# The reference state of the case: a hydrostatic p̄(z), ρ̄(z), T̄(z), Π̄(z).
#
# The Boussinesq core carries neither pressure nor density, but the moist
# physics needs both: saturation is a function of (T, p), a flux in W m⁻²
# becomes a mixing-ratio or θ increment through ρ, and θ becomes T through
# the Exner function. Before #131 every coupling made its own choice
# (an isothermal 8 km scale height from 1013.25 hPa for fog and dew,
# ρ = 1 in the surface, building and ledger conversions, p = 1013.25 hPa
# for evaporation). This file is the one place they now come from.
#
# The profile is `Aeolus.Anelastic.reference_from_profile` — the
# discretely hydrostatic base state of the anelastic core — integrated
# from the initial θ(z) of the case and `cfg.p_surface` at the ground,
# with the dry-air constants of `MoistThermodynamics` (R_d, c_pd, p_ref =
# 1000 hPa) so that T = θ Π̄ here is the same T the fog adjustment and
# the saturation functions see, and with the gravity of the dynamics
# (`cfg.g`). It is fixed for the run.

"""
    CaseReference

Hydrostatic reference state of a case (see `core/reference_state.jl`).

* `p̄`, `ρ̄`, `θ̄`, `T̄`, `π̄` — cell-centred profiles, length `Nz`
  (Pa, kg m⁻³, K, K, 1), copied from `state` so that couplings see
  concrete vectors;
* `state` — the `Aeolus.Anelastic.ReferenceState` itself (face profiles,
  `max_hydrostatic_residual`);
* `p_surface` — pressure at the ground (the bottom wall face), Pa;
* `cp` — the specific heat of dry air the conversions use, J kg⁻¹ K⁻¹.

Accessors: [`ref_pressure`](@ref), [`ref_density`](@ref),
[`ref_exner`](@ref), [`surface_pressure`](@ref), [`ref_ρcp`](@ref),
[`ref_θ_per_heat`](@ref).
"""
struct CaseReference
    p̄::Vector{Float64}
    ρ̄::Vector{Float64}
    θ̄::Vector{Float64}
    T̄::Vector{Float64}
    π̄::Vector{Float64}
    state::Any
    p_surface::Float64
    cp::Float64
end

"""
    initial_θ(cfg) -> (x, y, z) -> θ

The initial potential temperature of the case,
``θ(z) = θ_s + \\max(0, z - z_i)Γ`` (`cfg.θ_surface`, `cfg.θ_inversion_z`,
`cfg.θ_lapse`). The one definition: `build_case` sets `sim.θ` from it and
[`build_reference`](@ref) integrates the reference state from it.
"""
function initial_θ(cfg::CaseConfig)
    θs, zi, Γ = cfg.θ_surface, cfg.θ_inversion_z, cfg.θ_lapse
    return (x, y, z) -> θs + max(0, z - zi) * Γ
end

"""
    build_reference(cfg, mesh) -> CaseReference

The hydrostatic reference state of the case: θ̄(z) is the initial
potential-temperature profile [`initial_θ`](@ref) of `cfg` sampled at
the cell centres, integrated from `cfg.p_surface` with
`Aeolus.Anelastic.reference_from_profile`, the dry-air constants of
`Aeolus.MoistThermodynamics` and `g = cfg.g`.
"""
function build_reference(cfg::CaseConfig, mesh)
    c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
    R_d = c.gas_constant / c.molmass_dry
    θ0 = initial_θ(cfg)
    state = Aeolus.Anelastic.reference_from_profile(mesh, z -> θ0(0.0, 0.0, z),
                                                    cfg.p_surface;
                                                    R_d = R_d, cp = c.cp_dry, p₀ = c.p_ref,
                                                    g_accel = cfg.g)
    return CaseReference(copy(state.p̄), copy(state.ρ̄), copy(state.θ̄), copy(state.T̄),
                         copy(state.π̄), state, cfg.p_surface, Float64(c.cp_dry))
end

"Reference pressure (Pa) at the centre of level `k`, or the whole profile."
ref_pressure(case, k::Integer) = case.ref.p̄[k]
ref_pressure(case) = case.ref.p̄
"Reference density (kg m⁻³) at the centre of level `k`, or the whole profile."
ref_density(case, k::Integer) = case.ref.ρ̄[k]
ref_density(case) = case.ref.ρ̄
"Reference Exner function ``Π̄ = (p̄/p_{ref})^{R_d/c_{pd}}`` at level `k` (T = θ Π̄)."
ref_exner(case, k::Integer) = case.ref.π̄[k]
"Pressure at the ground (Pa)."
surface_pressure(case) = case.ref.p_surface
"``ρ̄_k c_{pd}`` (J m⁻³ K⁻¹): the heat that raises the *temperature* of level `k` by 1 K."
ref_ρcp(case, k::Integer) = case.ref.ρ̄[k] * case.ref.cp

"""
    ref_θ_per_heat(case, k; cp = case.ref.cp) -> K m³ J⁻¹

The potential-temperature increment of level `k` per J m⁻³ of heat,
``1 / (ρ̄_k c_p Π̄_k)``. Heat raises the temperature by ``q / (ρ̄ c_p)``
and, since ``T = θ Π̄``, θ by that over ``Π̄``. A flux `F` (W m⁻²) into a
cell of thickness `Δz` over `τ` s raises θ by
`F * τ / Δz * ref_θ_per_heat(case, k)`. The one conversion every
coupling that heats the air uses, and the inverse of the weight with
which the air-heat ledger (`core/air_heat.jl`) counts θ, so a flux `F`
appears in the ledger as exactly `F τ`. (Before #253 the couplings
divided by ``ρ̄ c_p`` only and the air received ``Π̄_k F`` — 0.4 % too
much at the ground, where ``Π̄ ≈ 1.004``.) `cp` defaults to the case's
dry-air value; the surface exchange passes the moist heat capacity of
the first air cell.
"""
ref_θ_per_heat(case, k::Integer; cp::Real = case.ref.cp) =
    1 / (case.ref.ρ̄[k] * cp * case.ref.π̄[k])

"""
    hydrostatic_check(cfg, ref, mesh; nsub = 1000) -> max relative |p̄ − p_hyd| / p_hyd

The reference pressure against an independent integration of the
continuous hydrostatic equation ``dp/dz = -g p / (R_d θ(z) Π(p))`` by
RK4 with `nsub` sub-steps per level, from the same surface pressure and
θ(z) (the one used for `ref.state`, evaluated at the exact heights, not
interpolated). The discrete balance of the ReferenceState is exact on
the grid by construction; this measures how far that grid balance is
from the continuous atmosphere.
"""
function hydrostatic_check(cfg::CaseConfig, ref::CaseReference, mesh; nsub::Int = 1000)
    c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS
    R_d = c.gas_constant / c.molmass_dry
    κ = R_d / c.cp_dry
    θ0 = initial_θ(cfg)
    θ(z) = θ0(0.0, 0.0, z)
    f(z, p) = -cfg.g * p / (R_d * θ(z) * (p / c.p_ref)^κ)
    err = 0.0
    p, z = ref.p_surface, 0.0
    zc = [(k - 0.5) * mesh.dz for k in 1:mesh.Nz]
    for k in 1:mesh.Nz
        h = (zc[k] - z) / nsub
        for _ in 1:nsub
            k1 = f(z, p); k2 = f(z + h / 2, p + h / 2 * k1)
            k3 = f(z + h / 2, p + h / 2 * k2); k4 = f(z + h, p + h * k3)
            p += h / 6 * (k1 + 2k2 + 2k3 + k4); z += h
        end
        err = max(err, abs(ref.p̄[k] - p) / p)
    end
    return err
end
