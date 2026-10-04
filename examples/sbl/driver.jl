# =========================================================================
# Shared stable-boundary-layer (SBL) driver — P55-f (#191)
#
# A horizontally homogeneous, periodic Boussinesq LES of a boundary layer
# over a cooled surface, built only from package physics:
#
#   * the geostrophic pressure gradient is `LargeScaleForcing`'s
#     `GeostrophicPressureGradient` (P58-c, #171), applied with
#     `apply_forcing!` from the step's `body_force!` hook, together with the
#     core's own Coriolis term (`f_coriolis`);
#   * the surface temperature is a `SurfaceTemperatureSeries` (#171), read
#     with `surface_temperature(series, t)` — a prescribed cooling rate is a
#     two-point series;
#   * the surface heat flux is `SurfaceExchange.solve_surface_layer` (the
#     Era V Monin–Obukhov solver), column by column, from the first-level
#     wind and θ: w'θ'_0 = -u_* θ_*;
#   * the surface stress is `WallModels.MOSTWallLaw` (P57-a, #169) with the
#     Obukhov length of that same solve, applied implicitly by
#     `apply_wall_stress!` in every substage (`post_predictor!`); with the
#     same stability functions and L the law returns the solver's u_* bit
#     for bit;
#   * the SGS closure is any `AbstractEddyViscosityClosure`, wrapped in
#     `Operators.StabilityCorrected` (P55-e, #187) unless
#     `sgs_stability = :none`, with the stress-form momentum divergence
#     (`sgs_stress = :symmetric`, P55-b), so that the corrected vertical
#     viscosity reaches the momentum equation;
#   * the top Rayleigh sponge is the core's (`sponge_α_max`,
#     `sponge_z_start`), retargeted at (U_g, V_g) with the core's own
#     `add_sponge!(…; f_ref)` profile, as the Berlin `:geostrophic` coupling
#     does.
#
# The wall-resolved stratified channel of the same validation
# (`stratified_channel_run`, at the end) is the P54-d `channel_case` with
# `bottom_bc = :noslip` and θ walls; P59-e (#195) reuses that builder.
#
# There is no private log law, no private geostrophic forcing and no
# private stability function family beyond the case's prescribed
# coefficients (`SBLLogLinear`, an implementer of the SurfaceExchange
# contract). This file only defines functions (prefix `sbl_`); it is
# included by `examples/sbl/gabls1.jl` and `test/era6_gabls1.jl`. P58-d
# (#194) extends it for GABLS2/3/4 and CASES-99 (time-dependent U_g,
# advective tendencies and an observed T_s series are already accepted).
# =========================================================================

using Aeolus
using Aeolus.Fields: set!
using Aeolus.Operators: add_sponge!, total_stress_profile, interior_view
import Aeolus.SurfaceExchange: phi_m, phi_h, psi_m, psi_h
using Random
using Statistics: mean

"""
    SBLLogLinear(; β_m = 4.8, β_h = 7.8)

The log-linear Monin–Obukhov functions with separate momentum and heat
coefficients, ``φ_m = 1 + β_m ζ``, ``φ_h = 1 + β_h ζ``,
``ψ = -β ζ`` for ``ζ ≥ 0``; the unstable branch is delegated to
`SurfaceExchange.BusingerDyerBeljaarsHoltslag`. The GABLS1 protocol
prescribes ``β_m = 4.8``, ``β_h = 7.8`` (Beare et al. 2006; quoted by
Basu & Porté-Agel 2006, JAS 63, 2074, §4a: "the constants b_m and b_h were
set to 4.8 and 7.8"). An implementer of the SurfaceExchange stability
contract, so `solve_surface_layer` and `MOSTWallLaw` use it unchanged.
"""
Base.@kwdef struct SBLLogLinear <: Aeolus.SurfaceExchange.AbstractStabilityFunctions
    β_m::Float64 = 4.8
    β_h::Float64 = 7.8
end
const _SBL_UNSTABLE = BusingerDyerBeljaarsHoltslag()
phi_m(s::SBLLogLinear, ζ::Real) = ζ < 0 ? phi_m(_SBL_UNSTABLE, ζ) : 1 + s.β_m * ζ
phi_h(s::SBLLogLinear, ζ::Real) = ζ < 0 ? phi_h(_SBL_UNSTABLE, ζ) : 1 + s.β_h * ζ
psi_m(s::SBLLogLinear, ζ::Real) = ζ < 0 ? psi_m(_SBL_UNSTABLE, ζ) : -s.β_m * ζ
psi_h(s::SBLLogLinear, ζ::Real) = ζ < 0 ? psi_h(_SBL_UNSTABLE, ζ) : -s.β_h * ζ

mutable struct SBLCase
    sim::Any
    closure::Any
    sgs_stability::Symbol
    forcing::Any                      # AbstractLargeScaleForcing or a Tuple of them
    Ts::Any                           # SurfaceTemperatureSeries (or anything with surface_temperature)
    sf::Any                           # SurfaceExchange stability functions
    z0m::Float64
    z0h::Float64
    U_g::Float64
    V_g::Float64
    laws::Matrix{MOSTWallLaw}         # the per-column wall law of the current step
    wθ::Matrix{Float64}               # per-column kinematic surface heat flux (K m s⁻¹, upward +)
    ustar::Matrix{Float64}            # per-column u_* of the surface-layer solve
    Lob::Matrix{Float64}              # per-column Obukhov length
    zero_u::Any                       # zero fields for the sponge retargeting
    zero_v::Any
    τx::Float64                       # plane-mean kinematic surface stress the last step applied
    τy::Float64
    heat_applied::Float64             # ∫ ⟨w'θ'⟩_0 dt actually applied (K m)
    nsteps::Int
    wall_time::Float64
end

"""
    sbl_case(; N = (32, 32, 32), extent = (400, 400, 400), U_g = 8, V_g = 0,
             f = 1.39e-4, z0m = 0.1, z0h = z0m, θ_ml = 265, z_ml = 100,
             Γ = 0.01, θ₀ = 263.5, T_s0 = 265, cooling = 0.25/3600,
             T_s = nothing, base = StaticSmagorinsky(0.1),
             sgs_stability = :brown94, stability = SBLLogLinear(),
             sponge_z = 300, sponge_α = 0.01, perturbation = 0.1,
             z_pert = 50, seed = 1, extra_forcing = ()) -> SBLCase

Build the SBL LES. Defaults are the GABLS1 protocol (Beare et al. 2006,
BLM 118, 247; set-up as quoted by Basu & Porté-Agel 2006, JAS 63, 2074):
400 m cube, U_g = 8 m s⁻¹, f = 1.39e-4 s⁻¹ (73° N), z0 = 0.1 m,
θ = 265 K to 100 m then +0.01 K m⁻¹, the surface cooled at 0.25 K h⁻¹
from 265 K, log-linear surface functions β_m = 4.8, β_h = 7.8, a Rayleigh
damping layer above 300 m, random θ perturbations of ±`perturbation` K
below `z_pert`. `T_s` (a `SurfaceTemperatureSeries`) overrides the linear
cooling. `θ₀` is the Boussinesq reference temperature of the buoyancy.
`sgs_stability` is `:brown94` (Brown et al. 1994, the Berlin default),
`:lilly`, or `:none` (the uncorrected `base`).
"""
function sbl_case(; N = (32, 32, 32), extent = (400.0, 400.0, 400.0),
                  U_g = 8.0, V_g = 0.0, f = 1.39e-4, z0m = 0.1, z0h = z0m,
                  θ_ml = 265.0, z_ml = 100.0, Γ = 0.01, θ₀ = 263.5,
                  T_s0 = 265.0, cooling = 0.25 / 3600, T_s = nothing,
                  base = StaticSmagorinsky(0.1), sgs_stability::Symbol = :brown94,
                  stability = SBLLogLinear(), sponge_z = 300.0, sponge_α = 0.01,
                  perturbation = 0.1, z_pert = 50.0, seed = 1, extra_forcing = ())
    Nx, Ny, Nz = N
    mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = Float64.(extent))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(mesh; θ₀ = θ₀, g_accel = 9.81, f_coriolis = f,
                               sponge_α_max = sponge_α, sponge_z_start = sponge_z,
                               bcs = bcs,
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    rng = Random.Xoshiro(seed)
    set!(sim.θ, (x, y, z) -> (z <= z_ml ? θ_ml : θ_ml + Γ * (z - z_ml)) +
                             (z < z_pert ? perturbation * (2rand(rng) - 1) : 0.0))
    set!(sim.u, (x, y, z) -> U_g)
    set!(sim.v, (x, y, z) -> V_g)
    set!(sim.w, (x, y, z) -> 0.0)
    closure = sgs_stability === :none ? base :
              StabilityCorrected(base, mesh; scheme = _sbl_scheme(sgs_stability))
    Ts = T_s === nothing ? SurfaceTemperatureSeries([0.0, 86400.0], [T_s0, T_s0 - cooling * 86400]) : T_s
    F = (GeostrophicPressureGradient(U_g, V_g), extra_forcing...)
    laws = [MOSTWallLaw(stability, z0m, Inf) for _ in 1:Nx, _ in 1:Ny]
    z = zeros(Nx, Ny)
    zu = Aeolus.Fields.Field(Aeolus.Grids.FaceX(), mesh)
    zv = Aeolus.Fields.Field(Aeolus.Grids.FaceY(), mesh)
    return SBLCase(sim, closure, sgs_stability, F, Ts, stability, Float64(z0m), Float64(z0h),
                   Float64(U_g), Float64(V_g), laws, copy(z), copy(z), fill(Inf, Nx, Ny), zu, zv,
                   0.0, 0.0, 0.0, 0, 0.0)
end

_sbl_scheme(s::Symbol) = s === :brown94 ? BrownEtAl1994() :
                         s === :lilly   ? LillyCutoff() :
                         s === :long_tail ? LongTail() :
                         s === :short_tail ? ShortTail() :
                         throw(ArgumentError("sgs_stability must be :brown94, :lilly, :long_tail, :short_tail or :none, got :$s"))

# cell-centre wind and θ of the first level of column (i, j)
@inline function _sbl_first_level(sim, i, j)
    m = sim.mesh
    I, J, K = m.Hx + i, m.Hy + j, m.Hz + 1
    uc = 0.5 * (sim.u.data[I, J, K] + sim.u.data[I+1, J, K])
    vc = 0.5 * (sim.v.data[I, J, K] + sim.v.data[I, J+1, K])
    return uc, vc, sim.θ.data[I, J, K]
end

"""
    sbl_surface!(c, t) -> c

Solve Monin–Obukhov similarity (`SurfaceExchange.solve_surface_layer`)
in every column between the first-level wind and θ (at Δz/2) and the
surface temperature `surface_temperature(c.Ts, t)`; store the kinematic
heat flux ``w'θ'_0 = -u_*θ_*``, ``u_*``, ``L``, and the column's
`MOSTWallLaw(sf, z0m, L)` for the momentum.
"""
function sbl_surface!(c::SBLCase, t)
    sim = c.sim
    m = sim.mesh
    z1 = m.dz / 2
    θs = surface_temperature(c.Ts, t)
    @inbounds for j in 1:m.Ny, i in 1:m.Nx
        uc, vc, θ = _sbl_first_level(sim, i, j)
        sol = solve_surface_layer(c.sf; U = sqrt(uc^2 + vc^2), z_m = z1, θ_a = θ, θ_s = θs,
                                  z0m = c.z0m, z0h = c.z0h)
        c.wθ[i, j] = -sol.u_star * sol.θ_star
        c.ustar[i, j] = sol.u_star
        c.Lob[i, j] = sol.L
        c.laws[i, j] = MOSTWallLaw(c.sf, c.z0m, sol.L)
    end
    return c
end

"""
    sbl_dt(c; cfl = 1.2, c_diff = 0.5) -> Δt

A stable step: the advective limit on Σ|u_i|/Δx_i and the diffusive limit
on the largest eddy viscosity of the last evaluation.
"""
function sbl_dt(c::SBLCase; cfl = 1.2, c_diff = 0.5)
    s = c.sim; g = s.mesh
    ii, jj, kk = Aeolus.Grids.interior(g)
    adv = maximum(abs, view(s.u.data, ii, jj, kk)) / g.dx +
          maximum(abs, view(s.v.data, ii, jj, kk)) / g.dy +
          maximum(abs, view(s.w.data, ii, jj, kk)) / g.dz
    νmax = maximum(view(s.νt.data, ii, jj, kk))
    dif = νmax * (1 / g.dx^2 + 1 / g.dy^2 + 1 / g.dz^2)
    dt = cfl / adv
    dif > 0 && (dt = min(dt, c_diff / dif))
    return dt
end

const _SBL_RK3_WEIGHTS = (1 / 6, 1 / 6, 2 / 3)

"""
    sbl_step!(c, dt) -> c

One SSP-RK3 step (`step_rk3!`). The surface layer is solved once from the
start-of-step state at the mid-step surface temperature; the heat flux is
held over the step (its RK3 weights sum to one, so exactly
``⟨w'θ'_0⟩Δt`` enters the column), the wall law is applied implicitly in
every substage with the start-of-step L. The geostrophic forcing is the
exact step mean of `c.forcing`.
"""
function sbl_step!(c::SBLCase, dt::Real)
    s = c.sim; m = s.mesh
    t0 = s.t
    sbl_surface!(c, t0 + dt / 2)
    A = m.Lx * m.Ly
    stage = Ref(0)
    τx = Ref(0.0); τy = Ref(0.0)
    wm = WallModel(c.laws)
    z_top = m.z0 + m.Lz
    force! = function (sim, Δt)
        apply_forcing!(sim, c.forcing, t0, dt)
        K = m.Hz + 1
        @inbounds for j in 1:m.Ny, i in 1:m.Nx
            sim.tθ.data[m.Hx+i, m.Hy+j, K] += c.wθ[i, j] / m.dz
        end
        if sim.has_sponge
            add_sponge!(sim.tu, c.zero_u, sim.sponge_α_max, sim.sponge_z_start, z_top; f_ref = c.U_g)
            add_sponge!(sim.tv, c.zero_v, sim.sponge_α_max, sim.sponge_z_start, z_top; f_ref = c.V_g)
        end
        return nothing
    end
    wall! = function (sim, Δt)
        stage[] += 1
        w = _SBL_RK3_WEIGHTS[mod1(stage[], 3)]
        F = apply_wall_stress!(sim, wm, Δt; side = :bottom)
        τx[] += w * (-F.Fx) / A
        τy[] += w * (-F.Fy) / A
        return nothing
    end
    tw = time()
    step_rk3!(s, dt; sgs = c.closure, sgs_stress = :symmetric,
              body_force! = force!, post_predictor! = wall!)
    c.wall_time += time() - tw
    c.τx = τx[]; c.τy = τy[]
    c.heat_applied += mean(c.wθ) * dt
    c.nsteps += 1
    return c
end

"""
    sbl_column_heat(c) -> K m

The plane-mean column integral ``∫_0^{L_z} ⟨θ⟩ dz``.
"""
function sbl_column_heat(c::SBLCase)
    s = c.sim; m = s.mesh
    return sum(interior_view(s.θ)) * m.dz / (m.Nx * m.Ny)
end

"""
    sbl_profiles(c) -> NamedTuple

Instantaneous plane-mean profiles: `z` (cell centres), `U`, `V`, speed
`S`, `θ`; at the w-faces `zf`: the total (resolved + SGS) kinematic
momentum fluxes `τx`, `τy` (positive downward, with the surface stress of
the last step at z = 0, from `Operators.total_stress_profile` with the
closure's vertical viscosity) and their magnitude `τ`, and the total
heat flux `wθ` (resolved ⟨w θ⟩ + SGS −⟨κ_v ∂θ/∂z⟩, the surface flux at
z = 0).
"""
function sbl_profiles(c::SBLCase)
    s = c.sim; m = s.mesh
    Nx, Ny, Nz, Hx, Hy, Hz = m.Nx, m.Ny, m.Nz, m.Hx, m.Hy, m.Hz
    fill_halos!(s.u, s.bcs); fill_halos!(s.v, s.bcs); fill_halos!(s.w, s.bcs)
    fill_halos!(s.θ, s.θ_bcs)
    f = compute_sgs!(c.closure, s)
    νv = f.νv; κv = f.κv
    fill_halos!(νv, s.bcs); fill_halos!(κv, s.bcs)
    px = total_stress_profile(s.u, s.v, s.w, νv; ν = 0.0, τ_wall_bottom = c.τx)
    py = total_stress_profile(s.u, s.v, s.w, νv; ν = 0.0, τ_wall_bottom = c.τy, component = :y)
    U = zeros(Nz); V = zeros(Nz); θ = zeros(Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        U[k] += 0.5 * (s.u.data[I, J, K] + s.u.data[I+1, J, K])
        V[k] += 0.5 * (s.v.data[I, J, K] + s.v.data[I, J+1, K])
        θ[k] += s.θ.data[I, J, K]
    end
    n = Nx * Ny
    U ./= n; V ./= n; θ ./= n
    wθ = zeros(Nz + 1)
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        θf = 0.5 * (s.θ.data[I, J, K] + s.θ.data[I, J, K-1])
        κf = 0.5 * (κv.data[I, J, K] + κv.data[I, J, K-1])
        wθ[k] += s.w.data[I, J, K] * θf - κf * (s.θ.data[I, J, K] - s.θ.data[I, J, K-1]) / m.dz
    end
    wθ ./= n
    wθ[1] = mean(c.wθ)
    τ = sqrt.(px.total .^ 2 .+ py.total .^ 2)
    return (z = [(k - 0.5) * m.dz for k in 1:Nz], U = U, V = V, S = sqrt.(U .^ 2 .+ V .^ 2),
            θ = θ, zf = px.z, τx = px.total, τy = py.total, τ = τ, wθ = wθ)
end

"""
    sbl_bl_height(zf, τ; fraction = 0.05) -> h (m)

The SBL height of Kosović & Curry (2000) and Beare et al. (2006): the
height where the mean total stress first falls to `fraction` (5 %) of its
surface value, divided by 0.95 (linear interpolation between faces).
"""
function sbl_bl_height(zf, τ; fraction = 0.05)
    τ0 = τ[1]
    for k in 2:length(τ)
        if τ[k] <= fraction * τ0
            z = zf[k-1] + (fraction * τ0 - τ[k-1]) / (τ[k] - τ[k-1]) * (zf[k] - zf[k-1])
            return z / 0.95
        end
    end
    return zf[end] / 0.95
end

"""
    sbl_run!(c, t_end; cfl = 1.2, average_from = Inf, sample_every = 10,
             callback = nothing) -> stats

Advance to `t_end` (s) with [`sbl_dt`](@ref). From `average_from` on,
every `sample_every`-th step is sampled: the returned `stats` hold the
mean profiles (`U`, `V`, `S`, `θ`, `τ`, `wθ`), the BL height of the mean
stress profile `h`, the mean `u_star` (√ of the mean surface stress
magnitude), the mean surface heat flux `wθ0` (K m s⁻¹) and `H0`
(W m⁻², ρc_p = 1.3·1005 J m⁻³ K⁻¹ near 263 K), the low-level-jet maximum
`S_max` and its height `z_jet`, the mean Obukhov length `L`, and
`nsteps`, `wall_time`.
"""
function sbl_run!(c::SBLCase, t_end; cfl = 1.2, average_from = Inf, sample_every = 10,
                  callback = nothing)
    s = c.sim
    acc = nothing; ns = 0; τs = 0.0; wθs = 0.0; Ls = 0.0
    n0 = c.nsteps
    while s.t < t_end - 1e-9
        dt = min(sbl_dt(c; cfl = cfl), t_end - s.t)
        sbl_step!(c, dt)
        callback === nothing || callback(c)
        if s.t >= average_from && (c.nsteps - n0) % sample_every == 0
            p = sbl_profiles(c)
            acc === nothing ? (acc = Dict(k => copy(getfield(p, k)) for k in (:U, :V, :S, :θ, :τ, :wθ, :τx, :τy))) :
                              (for k in keys(acc); acc[k] .+= getfield(p, k); end)
            τs += sqrt(c.τx^2 + c.τy^2); wθs += mean(c.wθ)
            Ls += mean(c.ustar)^3 / (0.4 * 9.81 / s.θ₀ * max(-mean(c.wθ), 1e-12))
            ns += 1
        end
    end
    ns == 0 && return (nsamples = 0, nsteps = c.nsteps - n0, wall_time = c.wall_time)
    for k in keys(acc); acc[k] ./= ns; end
    m = s.mesh
    z = [(k - 0.5) * m.dz for k in 1:m.Nz]
    zf = [(k - 1) * m.dz for k in 1:m.Nz+1]
    τm = sqrt.(acc[:τx] .^ 2 .+ acc[:τy] .^ 2)
    kj = argmax(acc[:S])
    wθ0 = wθs / ns
    return (z = z, zf = zf, U = acc[:U], V = acc[:V], S = acc[:S], θ = acc[:θ],
            τ = τm, wθ = acc[:wθ], h = sbl_bl_height(zf, τm),
            u_star = sqrt(τs / ns), wθ0 = wθ0, H0 = 1.3 * 1005 * wθ0,
            S_max = acc[:S][kj], z_jet = z[kj], L = Ls / ns,
            nsamples = ns, nsteps = c.nsteps - n0, wall_time = c.wall_time)
end

"""
    sbl_read_csv(path) -> (; header, columns, rows)

Read a reference-data CSV of `examples/sbl/data/`: the leading `#` lines
are its provenance header (returned verbatim), the first other line the
column names, the rest the rows (numbers parsed where possible).
"""
function sbl_read_csv(path)
    header = String[]; columns = String[]; rows = Vector{Any}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        if startswith(line, "#")
            push!(header, line)
        elseif isempty(columns)
            columns = strip.(split(line, ","))
        else
            push!(rows, [something(tryparse(Float64, strip(x)), strip(x)) for x in split(line, ",")])
        end
    end
    return (header = header, columns = columns, rows = rows)
end

"The value of column `col` in the row whose first column is `key`."
function sbl_csv_value(tab, key, col)
    j = findfirst(==(col), tab.columns)
    for r in tab.rows
        r[1] == key && return r[j]
    end
    throw(KeyError(key))
end

# =========================================================================
# The wall-resolved, stably stratified channel (García-Villalba & del
# Álamo 2011) on the P54-d builder — shared by examples/sbl/
# stratified_channel.jl and test/era6_stratified_channel.jl.
# =========================================================================

"""
    stratified_channel_case(Ri, stab; N, seed = 1) -> WallBoundedCase

The wall-resolved channel at Re_τ = 180 on 2πδ × πδ × 2δ: no-slip walls,
θ walls at `stratified_channel_walls(Ri)` (for `Ri = 0` θ is passive: walls
(0, 1) and g = 0), Pr = 0.71, the plane-averaged dynamic Smagorinsky
closure (Germano et al. 1991), wrapped in `StabilityCorrected` (Brown et
al. 1994, anisotropic length scales: the Berlin default) for
`stab = :brown94`, stress-form momentum divergence.
"""
function stratified_channel_case(Ri, stab; N = (32, 32, 48), seed = 1)
    mesh = RegularGrid(; size = N, extent = (2π, π, 2.0))
    base = DynamicSmagorinsky(mesh)
    closure = stab === :none ? base : StabilityCorrected(base, mesh)
    walls = Ri == 0 ? (0.0, 1.0) : stratified_channel_walls(Ri)
    return channel_case(; Re_τ = 180, N = N, closure = closure, bottom_bc = :noslip,
                        sgs_stress = :symmetric, θ_walls = walls, Pr = 0.71,
                        g_accel = Ri == 0 ? 0.0 : 1.0, seed = seed)
end

"""
    stratified_channel_spinup(; N = (32, 32, 48), T_spin = 10, seed = 1) -> snapshot

The neutral spin-up the stratified runs start from (García-Villalba & del
Álamo 2011 switch the stratification on in a developed neutral channel):
the uncorrected dynamic closure with θ walls (0, 1) and no buoyancy, from
the law-of-the-wall profile plus noise, for `T_spin` δ/u_τ. Returns the
fields `u, v, w, θ` (normalised θ, 0 at the bottom wall and 1 at the top)
and the time `t`.
"""
function stratified_channel_spinup(; N = (32, 32, 48), T_spin = 10.0, seed = 1)
    c = stratified_channel_case(0.0, :none; N = N, seed = seed)
    run_channel!(c, T_spin)
    s = c.sim
    return (u = copy(s.u.data), v = copy(s.v.data), w = copy(s.w.data), θ = copy(s.θ.data), t = s.t)
end

"""
    stratified_channel_run(Ri, stab; N = (32, 32, 48), T_end = 40, T_avg = 20, seed = 1,
                           spinup = nothing) -> NamedTuple

One wall-resolved channel ([`stratified_channel_case`](@ref)) at
Re_τ = 180 and `Ri`, run to `T_end` and averaged from `T_avg` (δ/u_τ),
started from `spinup` (a [`stratified_channel_spinup`](@ref) snapshot, θ
rescaled to the walls) or, with `spinup = nothing`, from noise. Returns the time-averaged mean profile `U` (cell
centres `zc`), bulk velocity `Ub` (u_τ units), Nusselt number `Nu` (wall
heat flux over the laminar ``κΔθ/2δ``), mean wall stress, the averaged
total stress profile and the momentum-budget residual
``\\max_z|τ(z) - τ(0) - ∫_0^z(∂⟨U⟩/∂t - G)dz'|``.
"""
function stratified_channel_run(Ri, stab; N = (32, 32, 48), T_end = 40.0, T_avg = 20.0,
                                seed = 1, spinup = nothing)
    c = stratified_channel_case(Ri, stab; N = N, seed = seed)
    walls = c.θ_walls
    if spinup !== nothing          # start from the developed neutral channel
        s = c.sim
        s.u.data .= spinup.u; s.v.data .= spinup.v; s.w.data .= spinup.w
        s.θ.data .= walls[1] .+ (walls[2] - walls[1]) .* spinup.θ
        s.t = spinup.t
    end
    run_channel!(c, T_avg)
    U0 = Aeolus.WallModels._uv_means(c.sim)[1]; t0 = c.sim.t
    q = Ref(0.0); nq = Ref(0)
    cb = c -> (q[] += -(c.q_wall_bottom + c.q_wall_top) / 2; nq[] += 1)
    st = run_channel!(c, T_end; average_from = T_avg, callback = cb)
    U1 = Aeolus.WallModels._uv_means(c.sim)[1]
    dUdt = (U1 .- U0) ./ (c.sim.t - t0)
    C = [0.0; cumsum((dUdt .- c.forcing) .* c.sim.mesh.dz)]
    budget = maximum(abs, st.stress.total .- (st.stress.total[1] .+ C))
    q_lam = (c.sim.ν / c.Pr) * (walls[2] - walls[1]) / 2
    return (Ri = Ri, stab = stab, N = N, U = st.U, zc = st.zc, Ub = mean(st.U),
            Nu = q[] / nq[] / q_lam, τ_wall = st.τ_wall, stress = st.stress, budget = budget,
            ΔUb = mean(U1) - mean(U0), nsteps = c.nsteps, wall_time = st.wall_time)
end

