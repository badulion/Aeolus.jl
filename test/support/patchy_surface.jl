# =========================================================================
# Patchy-surface LES driver — resolved heterogeneity vs tiled aggregation
#
# Phase 53 (#43). A small Boussinesq LES (Smagorinsky, SSP-RK3 with a
# pressure projection per stage) over a surface of strips of different
# skin temperature, the classical idealised heterogeneous convective
# boundary layer (Avissar & Schmidt 1998; Patton, Sullivan & Moeng 2005).
#
# Every surface column is an Era V `AbstractSurface` and its fluxes come
# from the contract's `turbulent_fluxes` (Monin–Obukhov through
# `SurfaceExchange.solve_surface_layer`), so the LES and the tile mosaic
# of `SurfaceTiles` compute fluxes with *the same code*; they differ only
# in which air each surface sees:
#
#   :resolved   — each column its own surface, forced by its own
#                 first-level air (the heterogeneity is resolved);
#   :tiled      — every column holds a `SurfaceMosaic` of all surface
#                 types at their domain area fractions, forced by that
#                 column's air (the heterogeneity is sub-grid, tiled);
#   :parameter  — every column holds `parameter_aggregate(mosaic)`: one
#                 surface with the area-mean skin temperature.
#
# This file only defines functions (prefix `patchy_`) and the coarse grid
# `HS_GRID`; it is included by `test/heterogeneous_surface.jl` (the
# coupling's conservation and code-path identity) and
# `validation/heterogeneous_surface.jl` (the heterogeneous runs). (A test harness that runs its own
# LES, moved from examples/validation/ by ONE-g, #456: the example becomes
# a configuration once the model has a heterogeneous-surface component.)
# =========================================================================

using Aeolus
using Aeolus.Fields: set!

# Coarse version of the example's setup: 100 m cells, 3.2 × 1.6 × 1.6 km.
const HS_GRID = (Nx = 32, Ny = 16, Nz = 16, Lx = 3200.0, Ly = 1600.0, Lz = 1600.0)
using Aeolus.SurfaceTiles
using Aeolus.MoistThermodynamics: PhasePartition, air_density, heat_capacity_moist,
                                  dry_adiabatic_lapse_rate
import Aeolus.SurfaceExchange: surface_temperature, roughness_lengths, surface_albedo,
                               surface_emissivity, step_surface!
using Random

"""
    PatchSkin(T; z0m = 0.1, z0h = z0m/10)

A surface with a prescribed skin temperature `T` (K): no prognostic state,
fluxes from the contract's default `turbulent_fluxes`. Prescribing the
skin temperature (rather than the flux, as Avissar & Schmidt 1998 did)
keeps the flux interactive — it responds to the local air temperature
and wind — which is where tiled and resolved fluxes can differ.
"""
struct PatchSkin <: AbstractSurface
    T::Float64
    z0m::Float64
    z0h::Float64
end
PatchSkin(T::Real; z0m::Real = 0.1, z0h::Real = z0m / 10) = PatchSkin(T, z0m, z0h)
surface_temperature(s::PatchSkin) = s.T
roughness_lengths(s::PatchSkin) = (z0m = s.z0m, z0h = s.z0h, z0q = s.z0h)
surface_albedo(::PatchSkin) = 0.2
surface_emissivity(::PatchSkin) = 1.0
step_surface!(s::PatchSkin, f::SurfaceForcing, dt::Real; kwargs...) = turbulent_fluxes(s, f; kwargs...)

"""
    patchy_strip_classes(Nx, Ny, dx, λ; f_hot = 0.5) -> Matrix{Int}

Class map of strips normal to x with wavelength `λ` (m): class 1 (hot) on
the first `f_hot λ` of every wavelength, class 2 (cool) on the rest.
"""
function patchy_strip_classes(Nx, Ny, dx, λ; f_hot = 0.5)
    x = ((1:Nx) .- 0.5) .* dx
    c = [mod(xi, λ) < f_hot * λ ? 1 : 2 for xi in x]
    return repeat(c, 1, Ny)
end

mutable struct PatchyLES
    sim::Any
    mode::Symbol
    λ::Float64                          # strip wavelength (m)
    f_hot::Float64
    classes::Matrix{Int}
    types::Vector{PatchSkin}            # one surface per class
    mosaic::SurfaceMosaic               # the tiled cell (domain fractions)
    aggregate::ParameterAggregatedSurface
    surfaces::Matrix{AbstractSurface}   # what each column sees
    p_sfc::Float64
    Ug::Float64
    # per-column kinematic fluxes of the last evaluation
    wθ::Matrix{Float64}                 # K m s⁻¹
    τx::Matrix{Float64}                 # m² s⁻² (cell centre, along x)
    τy::Matrix{Float64}
    # conservation book-keeping: ∫ mean kinematic heat flux dt actually applied
    applied_heat::Float64
    stage_weight::Float64
end

"""
    patchy_les(; Nx, Ny, Nz, Lx, Ly, Lz, λ, T_hot, T_cool, mode, θ0, z_inv, Γ,
               Ug, f_cor, z0m, seed) -> PatchyLES

Build the LES. The initial state is a mixed layer of potential
temperature `θ0` up to `z_inv` capped by a lapse rate `Γ` (K m⁻¹), with
small random perturbations in the lowest 100 m, and a uniform wind `Ug`
along x. With `f_cor > 0` the wind is held by a geostrophic forcing
``(U_g, 0)``; with `f_cor = 0` and `Ug = 0` the case is free convection.
"""
function patchy_les(; Nx = 64, Ny = 32, Nz = 32, Lx = 3200.0, Ly = 1600.0, Lz = 1600.0,
                    λ = 800.0, f_hot = 0.5, T_hot = 310.0, T_cool = 300.0, mode::Symbol = :resolved,
                    θ0 = 300.0, z_inv = 600.0, Γ = 0.006, Ug = 0.0, f_cor = 0.0,
                    z0m = 0.1, C_s = 0.17, seed = 1, p_sfc = 1e5)
    mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (Lx, Ly, Lz), halo = (2, 2, 2))
    sim = BoussinesqSimulation(mesh; θ₀ = θ0, g_accel = 9.81, C_s = C_s, Pr_t = 1 / 3,
                               f_coriolis = f_cor,
                               sponge_α_max = 0.01, sponge_z_start = 0.75Lz,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    rng = Random.MersenneTwister(seed)
    set!(sim.θ, (x, y, z) -> (z < z_inv ? θ0 : θ0 + Γ * (z - z_inv)) +
                             (z < 100 ? 0.2 * (rand(rng) - 0.5) : 0.0))
    set!(sim.u, (x, y, z) -> Ug)
    set!(sim.v, (x, y, z) -> 0.0)
    set!(sim.w, (x, y, z) -> 0.0)

    classes = patchy_strip_classes(Nx, Ny, mesh.dx, λ; f_hot = f_hot)
    types = [PatchSkin(T_hot; z0m = z0m), PatchSkin(T_cool; z0m = z0m)]
    fr = class_fractions(classes, 2)
    mosaic = SurfaceMosaic(AbstractSurface[types...], fr)
    agg = parameter_aggregate(mosaic)
    surfaces = Matrix{AbstractSurface}(undef, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        surfaces[i, j] = mode === :resolved  ? types[classes[i, j]] :
                         mode === :tiled     ? mosaic :
                         mode === :parameter ? agg :
                         throw(ArgumentError("patchy_les: mode must be :resolved, :tiled or :parameter"))
    end
    z = zeros(Nx, Ny)
    return PatchyLES(sim, mode, Float64(λ), Float64(f_hot), classes, types, mosaic, agg, surfaces, p_sfc, Ug,
                     copy(z), copy(z), copy(z), 0.0, 1.0)
end

_lapse0() = dry_adiabatic_lapse_rate(PhasePartition(0.0))

# Cell-centre wind components and θ of the first level of column (i, j).
@inline function _first_level(sim, i, j)
    m = sim.mesh
    I, J, K = m.Hx + i, m.Hy + j, m.Hz + 1
    uc = 0.5 * (sim.u.data[I, J, K] + sim.u.data[I + 1, J, K])
    vc = 0.5 * (sim.v.data[I, J, K] + sim.v.data[I, J + 1, K])
    return uc, vc, sim.θ.data[I, J, K]
end

"""
    patchy_forcing(pl, U, θ) -> SurfaceForcing

The contract forcing at the first LES level ``z_1 = Δz/2`` for wind speed
`U` and potential temperature `θ` (the LES θ is referred to the surface
pressure, so ``T = θ - Γ_d z_1``). Dry air.
"""
function patchy_forcing(pl::PatchyLES, U, θ; z = pl.sim.mesh.dz / 2)
    return SurfaceForcing(sw_down = 0.0, lw_down = 0.0, T_air = θ - _lapse0() * z,
                          q_air = 0.0, U = U, p = pl.p_sfc, z_ref = z, rain = 0.0, snow = 0.0)
end

# ρ c_p and ρ of the dry air of a forcing, as `turbulent_fluxes` uses them.
function _rho_cp(f::SurfaceForcing)
    q = PhasePartition(f.q_air)
    ρ = air_density(f.T_air, f.p, q)
    return ρ, ρ * heat_capacity_moist(q)
end

"""
    patchy_column_fluxes!(pl) -> pl

Evaluate every column's surface fluxes from the current first-level
state through the contract's `turbulent_fluxes` and store them as
kinematic fluxes (`wθ = H/(ρc_p)`, `τ = τ/ρ` split along the local wind).
"""
function patchy_column_fluxes!(pl::PatchyLES)
    sim = pl.sim
    Nx, Ny = size(pl.classes)
    @inbounds for j in 1:Ny, i in 1:Nx
        uc, vc, θ = _first_level(sim, i, j)
        U = sqrt(uc^2 + vc^2)
        f = patchy_forcing(pl, U, θ)
        fl = turbulent_fluxes(pl.surfaces[i, j], f)
        ρ, ρcp = _rho_cp(f)
        pl.wθ[i, j] = fl.H / ρcp
        # Stress along the local wind, from the effective wind the surface
        # layer used (U_min floor): components u/U_e, v/U_e.
        Ue = max(U, 0.1)
        pl.τx[i, j] = fl.τ / ρ * uc / Ue
        pl.τy[i, j] = fl.τ / ρ * vc / Ue
    end
    return pl
end

# Body force: surface fluxes into the first level, geostrophic forcing.
function _patchy_body_force!(pl::PatchyLES, sim, Δt)
    patchy_column_fluxes!(pl)
    m = sim.mesh
    Nx, Ny = size(pl.classes)
    K = m.Hz + 1
    inv_dz = 1 / m.dz
    @inbounds for j in 1:Ny, i in 1:Nx
        I, J = m.Hx + i, m.Hy + j
        sim.tθ.data[I, J, K] += pl.wθ[i, j] * inv_dz
        im = i == 1 ? Nx : i - 1          # periodic neighbours for the faces
        jm = j == 1 ? Ny : j - 1
        sim.tu.data[I, J, K] -= 0.5 * (pl.τx[im, j] + pl.τx[i, j]) * inv_dz
        sim.tv.data[I, J, K] -= 0.5 * (pl.τy[i, jm] + pl.τy[i, j]) * inv_dz
    end
    if sim.f_coriolis != 0 && pl.Ug != 0
        # du/dt = f v, dv/dt = -f (u - Ug): the geostrophic pressure gradient
        # (+f Ug on the v tendency), from LargeScaleForcing (#171).
        apply_forcing!(sim, GeostrophicPressureGradient(pl.Ug, 0.0), sim.t, Δt)
    end
    pl.applied_heat += pl.stage_weight * Δt * sum(pl.wθ) / length(pl.wθ)
    return nothing
end

"""
    patchy_step!(pl, dt) -> pl

One Shu–Osher SSP-RK3 step (three projected Euler stages, as
`step_rk3!`) with the surface fluxes applied as a body force at every
stage. The stage weights of the RK3 combination for the tendency are
1/6, 1/6, 2/3; `applied_heat` accumulates the domain-mean kinematic heat
flux with those weights, so that ``Δ\\int θ\\,dz = `` `applied_heat` is a
discrete conservation statement.
"""
function patchy_step!(pl::PatchyLES, dt::Real)
    sim = pl.sim
    bf!(s, Δt) = _patchy_body_force!(pl, s, Δt)
    u0 = copy(sim.u.data); v0 = copy(sim.v.data); w0 = copy(sim.w.data); θ0 = copy(sim.θ.data)
    pl.stage_weight = 1 / 6
    step!(sim, dt; body_force! = bf!)
    pl.stage_weight = 1 / 6
    step!(sim, dt; body_force! = bf!)
    @. sim.u.data = 0.75 * u0 + 0.25 * sim.u.data
    @. sim.v.data = 0.75 * v0 + 0.25 * sim.v.data
    @. sim.w.data = 0.75 * w0 + 0.25 * sim.w.data
    @. sim.θ.data = 0.75 * θ0 + 0.25 * sim.θ.data
    pl.stage_weight = 2 / 3
    step!(sim, dt; body_force! = bf!)
    @. sim.u.data = u0 / 3 + 2 / 3 * sim.u.data
    @. sim.v.data = v0 / 3 + 2 / 3 * sim.v.data
    @. sim.w.data = w0 / 3 + 2 / 3 * sim.w.data
    @. sim.θ.data = θ0 / 3 + 2 / 3 * sim.θ.data
    sim.t -= 2dt
    return pl
end

"Column integral ``\\sum_k θ Δz`` averaged over the domain (K m)."
function patchy_heat_content(pl::PatchyLES)
    m = pl.sim.mesh
    ii, jj, kk = Aeolus.Grids.interior(m)
    return sum(@view pl.sim.θ.data[ii, jj, kk]) * m.dz / (m.Nx * m.Ny)
end

"Boundary-layer depth (m): height of the largest gradient of the mean θ profile."
function patchy_zi(pl::PatchyLES)
    θm = mean_θ_profile(pl.sim)
    k = argmax(diff(θm))
    return k * pl.sim.mesh.dz
end

"Deardorff convective velocity ``w_* = (g/θ_0\\, \\overline{w'θ'}_0\\, z_i)^{1/3}``."
patchy_wstar(pl::PatchyLES, wθ0, zi) = cbrt(max(pl.sim.g_accel / pl.sim.θ₀ * wθ0 * zi, 0.0))

# -------------------------------------------------------------------------
# Running and sampling
# -------------------------------------------------------------------------

# Kinematic heat flux of a surface (or mosaic) under a forcing.
function _wθ(s, f::SurfaceForcing; kwargs...)
    _, ρcp = _rho_cp(f)
    return turbulent_fluxes(s, f; kwargs...).H / ρcp
end

"""
    patchy_sample(pl) -> NamedTuple

One sample of the resolved surface flux and of the tiled estimates made
from the same LES state. With ``\\overline{\\cdot}`` the domain mean over
the first level:

  * `les` — ``\\overline{w'θ'_0}``, the resolved flux (mean of the column fluxes);
  * `tile_vec` — the mosaic forced by ``(|\\bar{\\mathbf u}|, \\bar θ)``: what a
    grid cell the size of the domain would compute from its resolved state;
  * `tile_speed` — the mosaic forced by the mean *speed* ``\\overline{|\\mathbf u|}``;
  * `tile_gust` — `tile_vec` with the Beljaars (1995) gustiness ``1.2 w_*``;
  * `tile_patch` — each tile forced by the mean air over its own patches;
  * `param` — one surface of mean skin temperature forced like `tile_speed`;
  * `blended` — a `BlendedMosaic` with the blending height at the first
    level, forced by the domain-mean speed and θ at the level nearest
    `z_blend` (NaN if the blended solve fails);
  * `ustar2` — domain-mean kinematic stress; `wstar`, `zi`.
"""
function patchy_sample(pl::PatchyLES; z_blend = 125.0)
    sim = pl.sim
    patchy_column_fluxes!(pl)
    Nx, Ny = size(pl.classes)
    n = Nx * Ny
    su = sv = ss = sθ = 0.0
    nk = zeros(2); sk = zeros(2); θk = zeros(2)
    for j in 1:Ny, i in 1:Nx
        uc, vc, θ = _first_level(sim, i, j)
        sp = sqrt(uc^2 + vc^2)
        su += uc; sv += vc; ss += sp; sθ += θ
        c = pl.classes[i, j]
        nk[c] += 1; sk[c] += sp; θk[c] += θ
    end
    Uvec, Uspd, θm = sqrt(su^2 + sv^2) / n, ss / n, sθ / n
    les = sum(pl.wθ) / n
    zi = patchy_zi(pl)
    ws = patchy_wstar(pl, les, zi)
    tile_vec = _wθ(pl.mosaic, patchy_forcing(pl, Uvec, θm))
    tile_speed = _wθ(pl.mosaic, patchy_forcing(pl, Uspd, θm))
    tile_gust = _wθ(pl.mosaic, patchy_forcing(pl, Uvec, θm); gustiness = 1.2ws)
    fr = mosaic_fractions(pl.mosaic)
    tile_patch = sum(fr[c] * _wθ(pl.types[c], patchy_forcing(pl, sk[c] / nk[c], θk[c] / nk[c]))
                     for c in 1:2 if nk[c] > 0)
    param = _wθ(pl.aggregate, patchy_forcing(pl, Uspd, θm))
    blended = _blended_estimate(pl, z_blend)
    ustar2 = sum(sqrt.(pl.τx .^ 2 .+ pl.τy .^ 2)) / n
    return (les = les, tile_vec = tile_vec, tile_speed = tile_speed, tile_gust = tile_gust,
            tile_patch = tile_patch, param = param, blended = blended, ustar2 = ustar2, wstar = ws, zi = zi,
            U_vec = Uvec, U_speed = Uspd)
end

# Blending-height estimate: the tiles see the blended state at the first
# level, obtained by matching the aggregate flux to the domain-mean speed
# and θ at a higher level through one Monin–Obukhov layer.
function _blended_estimate(pl::PatchyLES, z_blend)
    m = pl.sim.mesh
    k = clamp(round(Int, z_blend / m.dz + 0.5), 2, m.Nz)
    ii, jj, _ = Aeolus.Grids.interior(m)
    K = m.Hz + k
    sp = 0.0
    for J in jj, I in ii
        uc = 0.5 * (pl.sim.u.data[I, J, K] + pl.sim.u.data[I + 1, J, K])
        vc = 0.5 * (pl.sim.v.data[I, J, K] + pl.sim.v.data[I, J + 1, K])
        sp += sqrt(uc^2 + vc^2)
    end
    sp /= m.Nx * m.Ny
    θk = mean_θ_profile(pl.sim)[k]
    f = patchy_forcing(pl, sp, θk; z = (k - 0.5) * m.dz)
    b = BlendedMosaic(pl.mosaic, FixedBlendingHeight(m.dz / 2))
    try
        fl = turbulent_fluxes(b, f)
        last_blending(b).converged || return NaN
        return fl.H / _rho_cp(patchy_forcing(pl, sp, θk))[2]
    catch err
        (err isa ArgumentError || nameof(typeof(err)) === :SingularException) || rethrow()
        return NaN
    end
end

"""
    patchy_run!(pl; t_end, dt, t_avg, every = 10, z_mid = 400) -> NamedTuple

Integrate to `t_end` (s) with steps `dt` and average over `t ≥ t_avg`,
sampling every `every` steps: time means of every field of
[`patchy_sample`](@ref), the phase-averaged (over y, time and strip
wavelengths) first-level wind `u_p`, surface flux `wθ_p` and vertical
velocity `w_p` at the level nearest `z_mid`, the mean θ profile at the
end, and `heat_error`, the relative mismatch between the change of the
column heat content and the heat flux applied.
"""
function patchy_run!(pl::PatchyLES; t_end = 7200.0, dt = 2.0, t_avg = 3600.0,
                     every = 10, z_mid = 400.0)
    sim = pl.sim
    m = sim.mesh
    Nx, Ny = size(pl.classes)
    nλ = max(1, round(Int, pl.λ / m.dx))
    nph = min(nλ, Nx)
    up = zeros(nph); wθp = zeros(nph); wp = zeros(nph); cnt = zeros(nph)
    kmid = clamp(round(Int, z_mid / m.dz + 0.5), 1, m.Nz)
    h0 = patchy_heat_content(pl)
    samples = NamedTuple[]
    nsteps = round(Int, t_end / dt)
    for s in 1:nsteps
        patchy_step!(pl, dt)
        if sim.t >= t_avg - 1e-9 && s % every == 0
            smp = patchy_sample(pl)
            push!(samples, smp)
            for j in 1:Ny, i in 1:Nx
                ϕ = mod(i - 1, nph) + 1
                uc, _, _ = _first_level(sim, i, j)
                I, J, K = m.Hx + i, m.Hy + j, m.Hz + kmid
                up[ϕ] += uc; wθp[ϕ] += pl.wθ[i, j]
                wp[ϕ] += 0.5 * (sim.w.data[I, J, K] + sim.w.data[I, J, K + 1])
                cnt[ϕ] += 1
            end
            s % (20every) == 0 &&
                @info("patchy LES: t = $(round(Int, sim.t)) s  zi = $(smp.zi) m  wθ = $(round(smp.les, digits = 4))",
                      subsystem = :patchy, domain = current_domain(), t = sim.t, zi = smp.zi, wθ = smp.les)
        end
    end
    mean_of(k) = sum(getfield(x, k) for x in samples) / length(samples)
    keys_ = (:les, :tile_vec, :tile_speed, :tile_gust, :tile_patch, :param, :blended, :ustar2, :wstar, :zi,
             :U_vec, :U_speed)
    means = NamedTuple{keys_}(Tuple(mean_of(k) for k in keys_))
    heat_error = (patchy_heat_content(pl) - h0 - pl.applied_heat) / pl.applied_heat
    return merge(means, (u_p = up ./ cnt, wθ_p = wθp ./ cnt, w_p = wp ./ cnt,
                         x_p = ((1:nph) .- 0.5) .* m.dx, z_mid = (kmid - 0.5) * m.dz,
                         θ_profile = mean_θ_profile(sim), z = ((1:m.Nz) .- 0.5) .* m.dz,
                         heat_error = heat_error, nsamples = length(samples),
                         series = samples))
end

"Root-mean-square about the mean of a phase-averaged profile."
patchy_amplitude(x) = sqrt(sum(abs2, x .- sum(x) / length(x)) / length(x))
