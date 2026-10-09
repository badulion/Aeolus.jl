"""
    Aeolus.RunDrivers.CBLDriver

The `cbl` run driver (EX-d, #441): free convection over a heated
free-slip floor with the Deardorff (1980) / Moeng (1984) prognostic SGS
TKE closure, horizontally averaged over the last part of the run
([`tke_cbl`](@ref), formerly the harness `test/tke_cbl_driver.jl` of
P56-b, #182, which `test/era6_tke.jl` runs at 24³). With `[inflow_statistics]`
the run is also a convective nest parent: every step of the averaging
window is folded into a `Transition.ParentResolvedStatistics` (the
boundary statistics a refined child's synthetic inflow needs,
N-inflow-b, #258).

Configs: `examples/sgs/configs/cbl_tke.toml` (64³, against Lenschow,
Wyngaard & Pennell 1980) and `examples/nesting/configs/cbl_inflow_statistics.toml`.
`results.toml`: the summary scalars and profiles of [`tke_cbl`](@ref), and
`[inflow]` the boundary statistics per level.
"""
module CBLDriver

using ...Grids: RegularGrid
using ...Fields: interior_view, set!
using ...BoundaryConditions: FieldBoundaryConditions
using ...Simulations: BoussinesqSimulation, step_rk3!
import ...Simulations
using ...TimeIntegration: CFLConstraints, adaptive_dt
using ...IBM: SolidMask, wall_distance, first_level_cells
using ...PrognosticSGS: TKEClosure, initialize_tke!, tke_statistics
using ...TurbulenceStability: boundary_layer_height
using ...Transition: ParentResolvedStatistics, update_inflow_statistics!, inflow_stress_parts
using ...IO: RunFolder, CheckpointSchedule
using ...Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, section, option!,
                 register_driver!, write_results, no_checkpoints
using Printf: @sprintf
using Random: Xoshiro, rand

export CBLDriverType, CBLRun, tke_cbl, tke_cbl_summary

"""
    tke_cbl(N; Δ = 50.0, Δz = Δ, Q = 0.1, T_end = 3600.0, avg = 900.0, z_i0 = 600.0,
            Γ = 0.003, θ₀ = 300.0, seed = 1, wall_cap = false, surface_factor = 3.9,
            C_advective = 0.4, progress = false, on_step = nothing) -> NamedTuple

Run the CBL on an `N × N × N` grid with horizontal spacing `Δ` and vertical
`Δz` and return the averaged profiles (`zc` cell centres, `zf` interior w
faces): `e` (SGS TKE), `kres` (resolved TKE), `w2` (resolved ``σ_w^2``),
`flux` (total heat flux / Q at the w faces: resolved ``⟨w'θ'⟩`` plus the SGS
``−⟨κ_t ∂θ/∂z⟩``), `sgs_flux`, `θ`; and the scalars `zi` (height of the
minimum heat flux, `boundary_layer_height(zf, flux; method = :min_flux)`), `min_flux_ratio`, `max_sgs_fraction` (of
``e/(e + k_{res})`` over ``0.2 < z/z_i < 0.8``), `z_epeak`, `w_star`
(``(g Q z_i/θ_0)^{1/3}``), `w2max_norm` (``\\max σ_w^2/w_*^2``),
`z_w2max_norm` (its ``z/z_i``), `clips`, `min_e`, `steps`, `wall` (s).

Set-up (after Moeng 1984 and Sullivan & Patton 2011, simplified): free
convection (no mean wind) over a free-slip floor heated with a constant
kinematic flux Q, periodic laterally, a lid with a Rayleigh sponge in the
top quarter. θ is uniform up to z_i0 and increases at Γ above, with a
small random perturbation in the lowest 200 m. SSP-RK3 (`step_rk3!`)
with the CFL step of `adaptive_dt` (advective number `C_advective`), the
closure passed as `sgs`. Statistics are horizontal means averaged over
every step of the last `avg` seconds.

`on_step(sim, closure, Δt)`, if given, is called after every step (e.g. to
fold the state into `Transition.ParentResolvedStatistics`, N-inflow-b).

`wall_cap = true` caps the mixing length at ``κ`` × `IBM.wall_distance`
(the floor); the default is Moeng's (1984) closure, ``l ≤ Δ`` with
``c_ε × 3.9`` (`surface_factor`) at the first level.
"""
function tke_cbl(N::Integer; Δ = 50.0, Δz = Δ, Q = 0.1, T_end = 3600.0, avg = 900.0,
                 z_i0 = 600.0, Γ = 0.003, θ₀ = 300.0, seed = 1, wall_cap::Bool = false,
                 surface_factor = 3.9, C_advective = 0.4, progress::Bool = false,
                 on_step = nothing)
    rng = Xoshiro(seed)
    L, H = N * Δ, N * Δz
    m = RegularGrid(; size = (N, N, N), extent = (L, L, H), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; θ₀ = θ₀, θ_flux_bottom = Q, sponge_α_max = 0.01,
                               sponge_z_start = 0.75H,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    set!(sim.θ, (x, y, z) -> (z < z_i0 ? θ₀ : θ₀ + Γ * (z - z_i0)) +
                             (z < 200 ? 0.1 * (rand(rng) - 0.5) : 0.0))
    sm = SolidMask(m)
    c = TKEClosure(m; wall_distance = wall_cap ? wall_distance(sm; bottom = true) : nothing,
                   first_level = first_level_cells(sm; bottom = true),
                   surface_factor = surface_factor)
    initialize_tke!(c, sim; e_min = 1e-3)          # e = 0 is a fixed point: seed it
    cfl = CFLConstraints(; C_advective = C_advective, dt_max = 20.0)
    acc = nothing; nacc = 0; dt = nothing; steps = 0; min_e = Inf
    wall = @elapsed while sim.t < T_end - 1e-9
        h = min(adaptive_dt(sim, cfl; dt_prev = dt, sgs = c), T_end - sim.t)
        step_rk3!(sim, h; sgs = c)
        dt = h; steps += 1
        on_step === nothing || on_step(sim, c, h)
        min_e = min(min_e, minimum(interior_view(c.e)))
        if sim.t > T_end - avg
            p = _cbl_profiles(sim, c, Q)
            acc = acc === nothing ? p : map(+, acc, p)
            nacc += 1
        end
        progress && steps % 200 == 0 &&
            @info "cbl" t = round(sim.t; digits = 1) dt = round(h; digits = 3) max_e = maximum(interior_view(c.e))
    end
    P = map(x -> x ./ nacc, acc)
    zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    zf = [k * m.dz for k in 1:m.Nz-1]
    # the one z_i diagnostic: the height of the minimum total heat flux
    zi = boundary_layer_height(zf, P.flux; method = :min_flux)
    frac = P.e ./ (P.e .+ P.kres)
    sel = (zc .> 0.2zi) .& (zc .< 0.8zi)
    w_star = cbrt(sim.g_accel / θ₀ * Q * zi)
    kw = argmax(P.w2)
    st = tke_statistics(c)
    return (; P..., zc, zf, zi, min_flux_ratio = minimum(P.flux),
              max_sgs_fraction = maximum(frac[sel]), sgs_fraction = frac,
              z_epeak = zc[argmax(P.e)], w_star, w2max_norm = P.w2[kw] / w_star^2,
              z_w2max_norm = zc[kw] / zi, clips = st.clips, min_e, steps, wall, N,
              Δ, Δz, Q, T_end)
end

# horizontal-mean profiles of one state
function _cbl_profiles(sim, c, Q)
    m = sim.mesh
    Hx, Hy, Hz, Nx, Ny, Nz = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny, m.Nz
    u = interior_view(sim.u); v = interior_view(sim.v); θ = interior_view(sim.θ)
    e = interior_view(c.e); κ = interior_view(sim.κt)
    wd = sim.w.data[Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz+1]       # all w faces, walls included
    uc = 0.5 .* (u .+ circshift(u, (-1, 0, 0)))
    vc = 0.5 .* (v .+ circshift(v, (0, -1, 0)))
    wc = 0.5 .* (wd[:, :, 1:end-1] .+ wd[:, :, 2:end])
    # the plane mean, as Statistics.mean(a; dims = (1, 2)) computes it (the
    # sum times 1//n; Statistics is not a dependency of the package)
    pm(a) = vec(sum(a; dims = (1, 2)) .* (1 // (size(a, 1) * size(a, 2))))
    pv(a) = pm(a .^ 2) .- pm(a) .^ 2
    θf = 0.5 .* (θ[:, :, 1:end-1] .+ θ[:, :, 2:end])
    wf = wd[:, :, 2:end-1]
    κf = 0.5 .* (κ[:, :, 1:end-1] .+ κ[:, :, 2:end])
    res = pm(wf .* θf) .- pm(wf) .* pm(θf)
    sgs = pm(.-κf .* (θ[:, :, 2:end] .- θ[:, :, 1:end-1]) ./ m.dz)
    return (e = pm(e), kres = 0.5 .* (pv(uc) .+ pv(vc) .+ pv(wc)), w2 = pv(wc),
            flux = (res .+ sgs) ./ Q, sgs_flux = sgs ./ Q, θ = pm(θ))
end

"""
    tke_cbl_summary(r) -> String

One line of the numbers of a [`tke_cbl`](@ref) result.
"""
tke_cbl_summary(r) =
    @sprintf("CBL %d³ (Δ = %.0f m, Δz = %.0f m), %d steps, %.1f s: z_i = %.0f m, min flux ratio %.3f, max SGS TKE fraction (0.2–0.8 z_i) %.3f, e peak at z = %.0f m (%.2f z_i), σ_w² max %.2f w*² at z/z_i = %.2f, clips %d",
             r.N, r.Δ, r.Δz, r.steps, r.wall, r.zi, r.min_flux_ratio, r.max_sgs_fraction,
             r.z_epeak, r.z_epeak / r.zi, r.w2max_norm, r.z_w2max_norm, r.clips)

# ---------------------------------------------------------------------------
# the run driver
# ---------------------------------------------------------------------------

"""
    CBLDriverType()

The `cbl` driver: `[run] model = "cbl"`. Sections `[grid]`, `[cbl]`
(the case and the averaging) and `[inflow_statistics]` (the parent's
boundary statistics, off by default); `[run] seed` seeds the initial θ
noise. `aeolus options cbl` prints them.
"""
struct CBLDriverType <: AbstractRunDriver end

Config.driver_name(::CBLDriverType) = "cbl"
Config.config_dir(::CBLDriverType) = normpath(joinpath(@__DIR__, "..", "..", "examples", "sgs", "configs"))
Config.driver_defaults(::CBLDriverType) = ("run.seed" => 1,)

function Config.driver_sections(::CBLDriverType)
    grid = section("grid"; doc = "An N × N × N box: N Δ wide, N Δz deep.") do s
        option!(s, "n", Int; default = 64, range = (4, 4096), doc = "cells per side N")
        option!(s, "spacing", Float64; default = 50.0, range = (0, Inf), open_lower = true, units = "m",
                doc = "horizontal spacing Δ")
        option!(s, "dz", Float64; default = 25.0, range = (0, Inf), open_lower = true, units = "m",
                doc = "vertical spacing Δz")
    end
    cbl = section("cbl"; doc = "The convective boundary layer and its averaging (tke_cbl).") do s
        option!(s, "heat_flux", Float64; default = 0.1, range = (0, Inf), open_lower = true, units = "K m s^-1",
                doc = "kinematic surface heat flux Q")
        option!(s, "z_i0", Float64; default = 800.0, range = (0, Inf), units = "m",
                doc = "initial mixed-layer depth (θ uniform below)")
        option!(s, "lapse", Float64; default = 0.003, range = (0, Inf), units = "K m^-1",
                doc = "θ gradient Γ above z_i0")
        option!(s, "theta0", Float64; default = 300.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "reference and mixed-layer θ")
        option!(s, "wall_cap", Bool; default = false,
                doc = "cap the mixing length at κ × the distance to the floor (default: Moeng's l ≤ Δ with c_ε × surface_factor at the first level)")
        option!(s, "surface_factor", Float64; units = "1", default = 3.9, range = (0, Inf), open_lower = true,
                doc = "Moeng's (1984) factor of c_ε at the first level")
        option!(s, "cfl", Float64; units = "1", default = 0.4, range = (0, Inf), open_lower = true,
                doc = "advective Courant number of the adaptive step (dt ≤ 20 s)")
        option!(s, "end_time", Float64; default = 7200.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "simulated time")
        option!(s, "average", Float64; default = 1800.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "the profiles are averaged over every step of the last `average` seconds")
    end
    inf = section("inflow_statistics"; doc = "The run as a nest parent: ParentResolvedStatistics over the averaging window (west face, whole plane, scale-gap mode).") do s
        option!(s, "enabled", Bool; default = false, doc = "fold every step of the window into the statistics")
        option!(s, "tau_avg", Float64; default = 600.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "running-mean time scale τ_avg")
        option!(s, "refine", NTuple{3,Int}; default = (2, 2, 4), range = (1, 64),
                doc = "the child's refinement (sets the band k_c < k ≤ r k_c)")
        option!(s, "depth", Float64; default = 500.0, range = (0, Inf), open_lower = true, units = "m",
                doc = "the levels below this height (at most N)")
    end
    return Section[grid, cbl, inf]
end

function _checks(cfg)
    e = no_checkpoints(cfg)
    c = cfg.cbl
    c.average <= c.end_time ||
        push!(e, ConfigError("cbl.average", "the averaging window is longer than the run (end_time)"))
    c.z_i0 < cfg.grid.n * cfg.grid.dz ||
        push!(e, ConfigError("cbl.z_i0", "the mixed layer is deeper than the box"))
    i = cfg.inflow_statistics
    (!i.enabled && (i.tau_avg != 600.0 || i.refine != (2, 2, 4) || i.depth != 500.0)) &&
        push!(e, ConfigError("inflow_statistics.enabled", "[inflow_statistics] keys are set but it is off"))
    return e
end
Config.driver_checks(::CBLDriverType) = Any[
    """`[checkpoint]` stays off (the run is one call)
    `cbl.average` ≤ `cbl.end_time`; `cbl.z_i0` below the lid
    `[inflow_statistics]` keys only with `inflow_statistics.enabled`""" => _checks]

"""
    CBLRun

What the `cbl` driver builds and runs: the resolved `cfg`, then `result`
(the NamedTuple of [`tke_cbl`](@ref)), `stats` (the
`ParentResolvedStatistics`, `nothing` when off), and `nupdates`,
`update_seconds` of its folding.
"""
mutable struct CBLRun
    cfg::RunConfig
    result::Any
    stats::Any
    nupdates::Int
    update_seconds::Float64
end

Config.build(::CBLDriverType, cfg::RunConfig) = CBLRun(cfg, nothing, nothing, 0, 0.0)

function Simulations.run!(::CBLDriverType, r::CBLRun, rf::RunFolder, ::CheckpointSchedule; callback = nothing)
    cfg = r.cfg; g = cfg.grid; c = cfg.cbl; inf = cfg.inflow_statistics
    N = g.n
    nlev = min(N, round(Int, inf.depth / g.dz))
    steps = Ref(0)
    function fold!(sim, closure, h)
        steps[] += 1
        callback === nothing || callback(r, steps[])
        inf.enabled && sim.t > c.end_time - c.average || return
        if r.stats === nothing
            r.stats = ParentResolvedStatistics(sim; closure = closure, τ_avg = inf.tau_avg, refine = inf.refine,
                                               levels = 1:nlev)
        end
        r.update_seconds += @elapsed update_inflow_statistics!(r.stats, h)
        r.nupdates += 1
    end
    res = tke_cbl(N; Δ = g.spacing, Δz = g.dz, Q = c.heat_flux, z_i0 = c.z_i0, Γ = c.lapse, θ₀ = c.theta0,
                  T_end = c.end_time, avg = c.average, seed = cfg.run.seed, wall_cap = c.wall_cap,
                  surface_factor = c.surface_factor, C_advective = c.cfl, progress = true, on_step = fold!)
    r.result = res
    @info tke_cbl_summary(res)
    out = Dict{String,Any}(
        "zi" => res.zi, "w_star" => res.w_star, "min_flux_ratio" => res.min_flux_ratio,
        "max_sgs_fraction" => res.max_sgs_fraction, "z_epeak" => res.z_epeak, "w2max_norm" => res.w2max_norm,
        "z_w2max_norm" => res.z_w2max_norm, "clips" => res.clips, "min_e" => res.min_e, "steps" => res.steps,
        "wall_seconds" => res.wall,
        "profiles" => Dict{String,Any}("zc" => res.zc, "zf" => res.zf, "e" => res.e, "kres" => res.kres,
                                       "w2" => res.w2, "theta" => res.θ, "flux" => res.flux,
                                       "sgs_flux" => res.sgs_flux, "sgs_fraction" => res.sgs_fraction))
    if r.stats !== nothing
        s = r.stats; p = inflow_stress_parts(s)
        out["inflow"] = Dict{String,Any}(
            "z" => p.z, "updates" => r.nupdates, "wall_ms_per_update" => 1e3 * r.update_seconds / r.nupdates,
            "lag_z" => s.lag_z, "band_min" => s.c_gap * s.Δh / s.refine, "band_max" => s.c_gap * s.Δh,
            # components 1..6: uu, vv, ww, uv, uw, vw (face frame); the missing part = generated
            "uu_resolved" => p.resolved[1, :], "uu_band" => p.band[1, :], "uu_sgs" => p.sgs[1, :],
            "uu_generated" => p.generated[1, :], "ww_resolved" => p.resolved[3, :], "ww_generated" => p.generated[3, :],
            "uw_resolved" => p.resolved[5, :], "uw_generated" => p.generated[5, :], "e_sgs" => p.e_sgs,
            "L_z_full_u" => p.L_z_full[1, :], "L_z_u" => p.L_z[1, :], "L_s_u" => p.L_s[1, :])
    end
    write_results(rf, out)
    return r
end

"""
    state_fields(r::CBLRun) -> NamedTuple

The averaged profiles (the run keeps no 3-D state).
"""
Config.state_fields(r::CBLRun) = (e = r.result.e, kres = r.result.kres, w2 = r.result.w2, flux = r.result.flux,
                                  theta = r.result.θ)

register_driver!(CBLDriverType())

end # module
