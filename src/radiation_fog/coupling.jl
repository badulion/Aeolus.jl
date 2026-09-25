# Phase 44 (#9) — droplet-radiative coupling of the fog column.
#
# Included into `RadiationFog`. Nothing here computes an optical property
# or solves a transfer equation: the droplet field (LWC, N_d, r_eff) is
# written into a RadiationCore `AtmosphericColumn` and handed to the Era III
# stack — `GasOptics` for the gases, `TwoStream.MieCloudOptics` for the
# droplets, `RadiationCore.combine_optics!` to merge them and
# `TwoStream.solve_lw!` / `solve_sw!` for the fluxes. The only physics this
# file adds is the bookkeeping of *which* droplets the radiation sees
# (the `droplet_feedback` switch) and the Wærsted et al. (2017)
# radiation-driven condensation diagnostic.

# =========================================================================
# Droplet → optics inputs
# =========================================================================

"""
    fog_effective_radius(ql, ρ, N_d; k = 0.8, r_min = 1.0) -> μm

Effective radius of the fog droplets, in the μm that
`TwoStream.MieCloudOptics` consumes, for liquid specific humidity `ql`
(kg/kg) in air of density `ρ` (kg m⁻³) with `N_d` droplets per m³.

Delegates to `AerosolCloud.droplet_effective_radius` (the Martin et al.
1994 relation `r_eff = (3 LWC / (4π ρ_w N_d k))^{1/3}`, returned there in
metres); `k` is the spectral-shape factor. `r_min` only guards the
`ql → 0` limit, where the value is irrelevant because the optical depth
`τ = 1.5 LWP / r_eff ∝ ql^{2/3}` vanishes anyway.
"""
fog_effective_radius(ql::Real, ρ::Real, N_d::Real; k::Real = 0.8,
                     r_min::Real = 1.0) =
    max(1e6 * droplet_effective_radius(max(float(ql), 0.0) * float(ρ),
                                       float(N_d); k = k), float(r_min))

# H2O specific humidity → volume mixing ratio, the conversion
# `RadiationCore.column_from_simulation` uses for the LES coupling.
const _VMR_PER_Q = 28.9647 / 18.01528

# =========================================================================
# The radiation object
# =========================================================================

"""
    FogRadiation

Era III radiation configured for a 1-D fog column: longwave and
shortwave RRTMG-band grids, correlated-k gas optics, the Phase 33
`MieCloudOptics` for the droplets and a two-stream solver for each, plus
a fixed upper atmosphere above the model top so that the downwelling
longwave at the fog top is that of a real sky rather than of space.

Build with [`fog_radiation`](@ref); evaluate with
[`fog_radiative_transfer!`](@ref).

Fields of interest after a call: `flux_lw`, `flux_sw` (RadiationCore
`RadiativeFluxes`, levels bottom-up over the **full** column), and
`nz`, the number of model layers at the bottom of it.
"""
mutable struct FogRadiation{T<:AbstractFloat, GL, GS, CL, CS, SL, SS}
    lw::SpectralGrid{T}
    sw::SpectralGrid{T}
    gas_lw::GL
    gas_sw::GS
    cloud_lw::CL
    cloud_sw::CS
    solver_lw::SL
    solver_sw::SS
    nz::Int
    upper::AtmosphericColumn{T}
    gas_opt_lw::OpticalProperties{T, 2}
    gas_opt_sw::OpticalProperties{T, 2}
    opt_lw::OpticalProperties{T, 2}
    opt_sw::OpticalProperties{T, 2}
    cld_lw::OpticalProperties{T, 2}
    cld_sw::OpticalProperties{T, 2}
    flux_lw::RadiativeFluxes{T, 1}
    flux_sw::RadiativeFluxes{T, 1}
    gas_valid::Bool
end

"""
    fog_radiation(nz, z_top, p_top, T_top, qv_top; z_toa = 30e3,
                  n_std = 30, humid = :subarctic_winter, ngpt_per_band = 2)
        -> FogRadiation

Radiation for a fog column of `nz` model layers whose top interface is at
height `z_top` (m), pressure `p_top` (Pa), temperature `T_top` (K) and
specific humidity `qv_top`.

Above the model top the atmosphere is `GasOptics.us_standard_column`
(`n_std` equal layers to `z_toa`), shifted so that its temperature and
pressure are continuous with the model top and its water vapour scaled to
the model-top value. It is held fixed: a fog night does not change the
free troposphere's emission to first order, and the upper air is not
the subject here. `z_top` must be one of the standard column's level
heights (a multiple of `z_toa / n_std`, 1000 m by default).
"""
function fog_radiation(nz::Int, z_top::Real, p_top::Real, T_top::Real,
                       qv_top::Real; z_toa::Real = 30e3, n_std::Int = 30,
                       humid::Symbol = :subarctic_winter,
                       ngpt_per_band::Int = 2)
    T = Float64
    Δ = z_toa / n_std
    k0 = round(Int, z_top / Δ)
    (k0 >= 1 && isapprox(k0 * Δ, z_top; atol = 1e-6)) ||
        throw(ArgumentError("fog_radiation: z_top = $z_top m must be a " *
                            "positive multiple of z_toa/n_std = $Δ m"))
    # Standard-atmosphere lapse rate below the tropopause is 6.5 K/km;
    # choose its surface value so the column passes through T_top at z_top.
    std = us_standard_column(T; nz = n_std, z_top = z_toa,
                             T_sfc = T_top + 6.5e-3 * z_top, humid = humid)
    sel = (k0 + 1):n_std
    fp  = p_top / std.p_lev[k0 + 1]
    vmr_ratio = qv_top * _VMR_PER_Q / std.vmr[:h2o][k0 + 1]
    vmr = Dict{Symbol, Vector{T}}()
    for (g, v) in std.vmr
        vmr[g] = g === :h2o ? max.(v[sel] .* vmr_ratio, 4e-6) : v[sel]
    end
    nu = length(sel)
    upper = AtmosphericColumn{T}(std.p_lay[sel] .* fp, std.T_lay[sel],
                                 std.p_lev[(k0 + 1):(n_std + 1)] .* fp,
                                 std.T_lev[(k0 + 1):(n_std + 1)],
                                 std.dz[sel], std.ρ[sel] .* fp, vmr,
                                 zeros(T, nu), zeros(T, nu),
                                 fill(T(10), nu), fill(T(30), nu), T(T_top))
    lw = longwave_bands(T; ngpt_per_band = ngpt_per_band)
    sw = shortwave_bands(T; ngpt_per_band = ngpt_per_band)
    ntot = nz + nu
    return FogRadiation(lw, sw, correlated_k_optics(lw), correlated_k_optics(sw),
                        mie_cloud_optics(lw), mie_cloud_optics(sw),
                        two_stream_solver(lw), two_stream_solver(sw),
                        nz, upper,
                        allocate_optics(T, (ntot,), lw),
                        allocate_optics(T, (ntot,), sw),
                        allocate_optics(T, (ntot,), lw),
                        allocate_optics(T, (ntot,), sw),
                        allocate_optics(T, (ntot,), lw),
                        allocate_optics(T, (ntot,), sw),
                        allocate_fluxes(T, (), ntot + 1),
                        allocate_fluxes(T, (), ntot + 1), false)
end

"""
    fog_atmospheric_column(rad, T_lay, p_lay, p_lev, dz, ρ, qv, ql, T_sfc;
                           N_d, k = 0.8, droplet_feedback = true)
        -> AtmosphericColumn

RadiationCore column for the model layers (bottom-up) stacked under the
fixed upper atmosphere of `rad`. Water vapour enters the gas optics
through `vmr[:h2o]`; the fog water enters the cloud optics through `ql`
and `r_eff_liq = fog_effective_radius(ql, ρ, N_d; k)`.

With `droplet_feedback = false` the column carries `ql = 0`: the fog
still exists thermodynamically (its water and latent heat are there) but
is transparent to radiation. That is the switch the feedback tests use
to isolate the droplet-radiative effect.
"""
function fog_atmospheric_column(rad::FogRadiation, T_lay, p_lay, p_lev, dz,
                                ρ, qv, ql, T_sfc::Real; N_d::Real,
                                k::Real = 0.8, droplet_feedback::Bool = true)
    nz = rad.nz; up = rad.upper
    length(T_lay) == nz || throw(DimensionMismatch("T_lay has $(length(T_lay)) layers, radiation expects $nz"))
    T_levm = similar(T_lay, nz + 1)
    T_levm[1] = T_lay[1]
    for k2 in 2:nz
        # thickness-weighted interpolation to the interface
        w = dz[k2 - 1] / (dz[k2 - 1] + dz[k2])
        T_levm[k2] = (1 - w) * T_lay[k2 - 1] + w * T_lay[k2]
    end
    T_levm[nz + 1] = up.T_lev[1]
    qlr  = droplet_feedback ? max.(ql, 0.0) : zeros(nz)
    reff = [fog_effective_radius(qlr[i], ρ[i], N_d; k = k) for i in 1:nz]
    vmr = Dict{Symbol, Vector{Float64}}()
    for (g, v) in up.vmr
        vmr[g] = g === :h2o ? vcat(max.(qv, 0.0) .* _VMR_PER_Q, v) :
                              vcat(fill(v[1], nz), v)
    end
    return AtmosphericColumn{Float64}(
        vcat(p_lay, up.p_lay), vcat(T_lay, up.T_lay),
        vcat(p_lev[1:nz], up.p_lev), vcat(T_levm[1:nz], up.T_lev),
        vcat(dz, up.dz), vcat(ρ, up.ρ), vmr,
        vcat(qlr, up.ql), vcat(zeros(nz), up.qi),
        vcat(reff, up.r_eff_liq), vcat(fill(30.0, nz), up.r_eff_ice),
        float(T_sfc))
end

"""
    fog_radiative_transfer!(rad, col; μ₀ = 0, solar_scale = 1,
                            albedo = 0.22, emissivity = 0.97,
                            update_gas = true) -> rad

Longwave and (if `μ₀ > 0`) shortwave fluxes through `col`, written to
`rad.flux_lw` and `rad.flux_sw`.

Gas optics are the expensive part and change only with temperature and
humidity, so they are recomputed only when `update_gas = true` (or on the
first call) and reused otherwise; droplet optics, which carry the
feedback, are recomputed on every call. `solar_scale` multiplies the
shortwave fluxes (the solution is linear in the top-of-atmosphere
irradiance), e.g. by the Earth–Sun distance factor.

The defaults `albedo = 0.22` and `emissivity = 0.97` are the SIRTA values
of Wærsted et al. (2017, Sect. 2.3): median observed albedo 0.221,
longwave emissivity 0.97.
"""
function fog_radiative_transfer!(rad::FogRadiation, col::AtmosphericColumn;
                                 μ₀::Real = 0.0, solar_scale::Real = 1.0,
                                 albedo = 0.22, emissivity = 0.97,
                                 update_gas::Bool = true)
    if update_gas || !rad.gas_valid
        compute_optics!(rad.gas_opt_lw, rad.gas_lw, col)
        compute_optics!(rad.gas_opt_sw, rad.gas_sw, col)
        rad.gas_valid = true
    end
    # Longwave: gases + droplets
    _copy_optics!(rad.opt_lw, rad.gas_opt_lw)
    compute_optics!(rad.cld_lw, rad.cloud_lw, col)
    combine_optics!(rad.opt_lw, rad.cld_lw)
    solve_lw!(rad.flux_lw, rad.solver_lw, rad.opt_lw, col;
              sfc_emissivity = emissivity)
    # Shortwave
    if μ₀ > 0
        _copy_optics!(rad.opt_sw, rad.gas_opt_sw)
        compute_optics!(rad.cld_sw, rad.cloud_sw, col)
        combine_optics!(rad.opt_sw, rad.cld_sw)
        solve_sw!(rad.flux_sw, rad.solver_sw, rad.opt_sw; μ₀ = μ₀,
                  sfc_albedo_direct = albedo)
        if solar_scale != 1
            f = rad.flux_sw
            f.flux_up .*= solar_scale; f.flux_dn .*= solar_scale
            f.flux_dn_dir .*= solar_scale; f.flux_net .*= solar_scale
        end
    else
        zero_fluxes!(rad.flux_sw)
    end
    return rad
end

function _copy_optics!(dst::OpticalProperties, src::OpticalProperties)
    copyto!(dst.τ, src.τ); copyto!(dst.ssa, src.ssa); copyto!(dst.g, src.g)
    return dst
end

"""
    fog_radiative_heating!(Q, rad, col) -> Q

Radiative energy input of each **model** layer, W m⁻³ (LW + SW),
`Q_k = (F_net(k+1) − F_net(k))/Δz_k = +∂F_net/∂z` with `F_net`
positive downward (energy entering through the top minus energy leaving
through the bottom), from `RadiationCore.flux_divergence!` on the last
fluxes in `rad`.

Summed over the model layers, `Σ Q_k Δz_k = F_net(model top) − F_net(sfc)`
exactly (the divergence telescopes); the column energy test relies on it.
"""
function fog_radiative_heating!(Q::AbstractVector, rad::FogRadiation,
                                col::AtmosphericColumn)
    ntot = length(col.dz)
    div = zeros(ntot)
    for f in (rad.flux_lw, rad.flux_sw)
        flux_divergence!(div, f, col.dz)
        @inbounds for k in 1:rad.nz
            Q[k] = (f === rad.flux_lw ? 0.0 : Q[k]) + div[k]
        end
    end
    return Q
end

"""
    fog_top_flux_jump(rad, k_bot, k_top) -> (ΔF_lw, ΔF_sw) W m⁻²

Net-flux convergence of the model layers `k_bot:k_top`
(`F_net(k_top+1) − F_net(k_bot)`): the energy the layer gains per unit
area. Negative in the longwave for a fog layer at night — its radiative
cooling. In an opaque fog it is the net LW flux at the fog top
(`rad.flux_lw.flux_net[k_top+1]`, the quantity Wærsted et al. 2017,
Fig. 6b, report: −73 W m⁻² on 2 and 8 Nov 2015, −100 W m⁻² on
28 Oct 2014) minus the small net LW at the fog base (≈ −10 W m⁻² at
the ground), so its magnitude is ≈ 10 W m⁻² smaller than theirs.
"""
function fog_top_flux_jump(rad::FogRadiation, k_bot::Int, k_top::Int)
    Fl = rad.flux_lw.flux_net; Fs = rad.flux_sw.flux_net
    return (Fl[k_top + 1] - Fl[k_bot], Fs[k_top + 1] - Fs[k_bot])
end

# =========================================================================
# Radiation-driven condensation (Wærsted et al. 2017, Eq. 5)
# =========================================================================

"""
    radiative_condensation_rate(Q, T, ρ; c = DEFAULT_CONSTANTS) -> kg m⁻³ s⁻¹

Condensation rate a radiative energy input `Q` (W m⁻³; negative for
cooling) forces in saturated air at temperature `T` and density `ρ`,
assuming the air stays saturated and nothing else acts — Wærsted et al.
(2017), Eqs. (1)-(5):

```math
C_\\mathrm{rad} = -\\frac{\\frac{dρ_s}{dT}\\left(\\frac{dT}{dt}\\right)_\\mathrm{rad}}
                        {1 + \\frac{L_v}{ρ c_p}\\frac{dρ_s}{dT}},\\qquad
\\frac{dρ_s}{dT} = \\frac{e_s}{R_v T^2}\\left(\\frac{L_v}{R_v T} - 1\\right),
```

with `(dT/dt)_rad = Q/(ρ c_p)`. `e_s`, `L_v`, `R_v` and `c_p` come from
the MoistThermodynamics contract (Wærsted used Bolton's `e_s`; the two
agree to ~0.1 % at fog temperatures). Integrated over the fog layer this
is their `C_LW` (from the longwave `Q`) or `−E_SW` (shortwave), which
they measured at 40–70 g m⁻² h⁻¹ for opaque fog (LWP ≳ 30 g m⁻²) at
SIRTA.
"""
function radiative_condensation_rate(Q::Real, T::Real, ρ::Real;
                                     c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T)
    Rv = c.gas_constant / c.molmass_water
    es = saturation_vapor_pressure(Tk, c)
    Lv = latent_heat_vapor(Tk, c)
    dρs = es / (Rv * Tk^2) * (Lv / (Rv * Tk) - 1)
    dTdt = float(Q) / (float(ρ) * c.cp_dry)
    return -dρs * dTdt / (1 + Lv / (float(ρ) * c.cp_dry) * dρs)
end
