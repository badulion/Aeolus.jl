"""
    Aeolus.RunDrivers.InflowFetchDriver

The `inflow_fetch` run driver (EX-d, #441): the development length behind
a synthetic inflow (N-inflow-d, #266), on the periodic, neutral,
wall-modelled half channel of P54-d. A precursor (the statistics source
and the mean inflow profile) and a reference on the child grid (its own
equilibrium) are spun up and averaged ([`fetch_precursor`](@ref)); then
each child run opens the channel in x and is fed by one generator
([`fetch_child`](@ref)): `off` (the mean profile only), `df` (the digital
filter of Xie & Castro 2008 with the precursor's statistics), `fsd` (the
random-mode `FreeStreamInflow`) and `df_same` (the digital filter with
the reference's statistics and mean). Formerly
`examples/nesting/inflow_fetch.jl` and the harness
`test/inflow_fetch_harness.jl`, which `test/era6_inflow_fetch.jl` runs.

`results.toml`: the precursor and reference profiles; per child run the
bulk ratio, the extra pressure gradient, the injected KE flux, the
development lengths (10 % and 20 %, against its own source and against
the reference) at the two `heights`, and `x`, TKE(x), −u'w'(x) there
(normalised by the reference). Config: `examples/nesting/configs/inflow_fetch.toml`.
"""
module InflowFetchDriver

using ...Grids: RegularGrid
using ...BoundaryConditions: ProfileBoundaryData
using ...WallModels: half_channel_case, run_channel!, channel_dt, advance_channel!
using ...Operators: Operators, total_stress_profile
using ...Simulations: Simulations
using ...Transition: ParentResolvedStatistics, update_inflow_statistics!, FetchStatistics, accumulate_fetch!,
                     fetch_profiles, InflowTurbulence, free_stream_generator, added_mass_flux, injected_ke_flux,
                     development_length
using ...IO: RunFolder, CheckpointSchedule
using ...Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, section, option!,
                 register_driver!, write_results, no_checkpoints

export InflowFetchDriverType, InflowFetchRun, fetch_precursor, fetch_child, pressure_gradient_bias, nearest_level

# Units: δ = H = 1, u* = 1, so times are in δ/u*.

"""
    fetch_precursor(; N, extent, t_spin, t_avg, seed) -> NamedTuple

Spin up a periodic half channel on `N` cells and `extent = (Lx, Ly)`,
then average for `t_avg` (δ/u*): the generator statistics `src`
(ParentResolvedStatistics, :full, τ_avg = Inf), the mean profile `U`
(cell centres), the x- and time-averaged `FetchStatistics` profiles
`tke`, `uw` (resolved), and the total-stress profile `stress`.
"""
function fetch_precursor(; N, extent, t_spin, t_avg, seed = 1, z0_over_H = 1e-4)
    case = half_channel_case(; N = N, extent = extent, seed = seed, z0_over_H = z0_over_H)
    t0 = time()
    run_channel!(case, t_spin)
    src = ParentResolvedStatistics(case.sim; closure = case.closure, face = :west,
                                   mode = :full, τ_avg = Inf)
    fs = FetchStatistics(case.sim)
    t_end = t_spin + t_avg
    nz = N[3]
    stress = zeros(nz + 1); W = 0.0
    while case.sim.t < t_end - 1e-12
        h = min(channel_dt(case), t_end - case.sim.t)
        advance_channel!(case, h)
        update_inflow_statistics!(src, h)
        accumulate_fetch!(fs, case.sim, h)
        stress .+= h .* total_stress_profile(case).total
        W += h
    end
    p = fetch_profiles(fs)
    nx = size(p.U, 1)
    xmean(A) = vec(sum(A; dims = 1)) ./ nx
    return (case = case, src = src, z = p.z, U = xmean(p.U), tke = xmean(p.tke),
            uw = xmean(p.uw), uu = xmean(p.uu), vv = xmean(p.vv), ww = xmean(p.ww),
            stress = stress ./ W, z_faces = [(k - 1) * case.sim.mesh.dz for k in 1:nz+1],
            wall_time = time() - t0)
end

# the precursor's mean profile on the storage levels of a child mesh with
# the same vertical grid (halo levels: the nearest interior level)
function _storage_profile(U::AbstractVector, m)
    nz = length(U)
    m.Nz == nz || throw(DimensionMismatch("child has $(m.Nz) levels, the profile $nz"))
    return [U[clamp(k - m.Hz, 1, nz)] for k in 1:nz+2m.Hz]
end

"""
    fetch_child(pre; generator = :df, N, extent, t_spin, t_avg, x_stress = 4.0,
                lateral = :open, y_margin = 0.5) -> NamedTuple

The inflow–outflow half channel fed by the precursor `pre`
(`generator` = `:off`, `:df` (DigitalFilterGenerator) or `:fsd`
(FreeStreamInflow)), spun up for `t_spin` and averaged for `t_avg`
(δ/u*). Returns the `fetch_profiles` (`p`), the time-mean bulk velocity
of every x-face column (`bulk`, m/s), the time- and plane-mean pressure
per column (`pmean`), the total stress at `x = x_stress` (three columns,
`stress`), the largest relative added mass flux of the generator
(`added`), the injected KE flux and the wall time. The bulk velocity,
pressure and stress are sampled every `sample_every` steps.
`lateral = :open` (default) opens the south and north faces
(`half_channel_case(lateral = :open)`); the fetch statistics, pressure
and stress are then averaged over the cells farther than `y_margin`
(δ) from them (`jrange`, returned). The bulk velocity is always the
volume flux of the whole face.
"""
function fetch_child(pre; generator::Symbol = :df, N, extent, t_spin, t_avg,
                     x_stress::Real = 4.0, nmodes::Integer = 200, seed::Integer = 1,
                     sample_every::Integer = 1, lateral::Symbol = :open,
                     lateral_outflow::Symbol = :dirichlet, y_margin::Real = 0.5)
    probe = RegularGrid(size = N, extent = (extent[1], extent[2], 1.0))
    data = ProfileBoundaryData(u = _storage_profile(pre.U, probe), v = 0.0, w = 0.0, θ = 0.0)
    case = half_channel_case(; N = N, extent = extent, inflow = data, noise = 0.0,
                             lateral = lateral, lateral_outflow = lateral_outflow)
    sim = case.sim; g = sim.mesh
    jr = lateral === :open ?
         (findfirst(j -> (j - 0.5) * g.dy >= y_margin, 1:g.Ny):
          findlast(j -> g.Ly - (j - 0.5) * g.dy >= y_margin, 1:g.Ny)) : (1:g.Ny)
    it = nothing
    if generator === :df
        it = InflowTurbulence(sim; statistics = pre.src, faces = (:west,), seed = seed)
    elseif generator === :fsd
        # the isotropic random-mode field of P59-d, one length (the face-mean
        # streamwise L_s of the statistics), swept at the bulk velocity
        Ub = sum(pre.U) / length(pre.U)
        gen = free_stream_generator(; U∞ = Ub, nmodes = nmodes)
        it = InflowTurbulence(sim; statistics = pre.src, faces = (:west,), generator = gen,
                              seed = seed)
    else
        generator === :off || throw(ArgumentError("generator must be :off, :df or :fsd"))
    end
    t0 = time()
    run_channel!(case, t_spin)
    fs = FetchStatistics(sim; jrange = jr)
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    bulk = zeros(Nx + 1); pm = zeros(Nx); stress = zeros(Nz + 1)
    is = round(Int, x_stress / g.dx)
    W = 0.0; Ws = 0.0; added = 0.0; ke = 0.0; n = 0
    t_end = t_spin + t_avg
    while sim.t < t_end - 1e-12
        h = min(channel_dt(case), t_end - sim.t)
        advance_channel!(case, h)
        accumulate_fetch!(fs, sim, h)
        if it !== nothing
            added = max(added, added_mass_flux(it).west.relative)
            ke += h * injected_ke_flux(it).total
        end
        W += h
        (n += 1) % sample_every == 0 || continue
        # bulk velocity, pressure and the stress band: every `sample_every` steps
        Ws += h
        @inbounds for i in 1:Nx+1
            s = 0.0
            for k in 1:Nz, j in 1:Ny
                s += sim.u.data[Hx+i, Hy+j, Hz+k]
            end
            bulk[i] += h * s / (Ny * Nz)
        end
        @inbounds for i in 1:Nx
            s = 0.0
            for k in 1:Nz, j in jr
                s += sim.p.data[Hx+i, Hy+j, Hz+k]
            end
            pm[i] += h * s / (length(jr) * Nz)
        end
        νt = Operators.compute_sgs!(case.closure, sim).νv
        stress .+= h .* total_stress_profile(sim.u, sim.v, sim.w, νt; irange = is-1:is+1,
                                              jrange = jr).total
    end
    return (case = case, inflow = it, p = fetch_profiles(fs), jrange = jr, bulk = bulk ./ Ws, pmean = pm ./ Ws,
            stress = stress ./ Ws, x_stress = (is - 0.5) * g.dx, added = added,
            ke_flux = ke / W, wall_time = time() - t0,
            maxdiv = Simulations.max_divergence(sim))
end

"""
    pressure_gradient_bias(child, G; x_range = (1.0, 4.0)) -> b

The least-squares slope of the time- and plane-mean pressure over
`x_range` (δ), divided by the driving gradient `G` (the body force of
the precursor, u*²/δ): the extra mean pressure gradient the developing
flow needs on top of the precursor's forcing.
"""
function pressure_gradient_bias(child, G; x_range = (1.0, 4.0))
    x = child.p.x
    sel = findall(xx -> x_range[1] <= xx <= x_range[2], x)
    xs = x[sel]; ps = child.pmean[sel]
    xm = sum(xs) / length(xs); pm = sum(ps) / length(ps)
    slope = sum((xs .- xm) .* (ps .- pm)) / sum((xs .- xm) .^ 2)
    return -slope / G            # dp/dx < 0 adds to the drive
end

"The index of the level of `z` nearest to the height `h`."
nearest_level(z::AbstractVector, h::Real) = findmin(abs.(z .- h))[2]

# ---------------------------------------------------------------------------
# the run driver
# ---------------------------------------------------------------------------

"""
    InflowFetchDriverType()

The `inflow_fetch` driver: `[run] model = "inflow_fetch"`. Sections
`[channel]` (the half channel), `[precursor]` (the statistics source and
the times of both precursors), `[reference]` (the child-grid
reference) and `[child]` (the open runs and what is measured on them).
"""
struct InflowFetchDriverType <: AbstractRunDriver end

Config.driver_name(::InflowFetchDriverType) = "inflow_fetch"
Config.config_dir(::InflowFetchDriverType) = normpath(joinpath(@__DIR__, "..", "..", "examples", "nesting", "configs"))

const _GENERATORS = Dict("off" => :off, "df" => :df, "fsd" => :fsd, "df_same" => :df)

function Config.driver_sections(::InflowFetchDriverType)
    ch = section("channel"; doc = "The half channel (δ = u* = 1).") do s
        option!(s, "width", Float64; default = 2.5, range = (0, Inf), open_lower = true, units = "1",
                doc = "spanwise width L_y (δ)")
        option!(s, "precursor_length", Float64; default = 5.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "streamwise length of the periodic precursors (δ)")
        option!(s, "z0_over_h", Float64; default = 1e-4, range = (0, 1), open_lower = true, units = "1",
                doc = "roughness length of the wall model (δ)")
    end
    pre = section("precursor"; doc = "The statistics source: a periodic half channel spun up, then averaged.") do s
        option!(s, "size", NTuple{3,Int}; default = (24, 12, 16), range = (1, 4096), doc = "cells")
        option!(s, "t_spin", Float64; units = "1", default = 20.0, range = (0, Inf), doc = "spin-up (δ/u*), both precursors")
        option!(s, "t_avg", Float64; units = "1", default = 10.0, range = (0, Inf), open_lower = true,
                doc = "averaging time (δ/u*), both precursors")
        option!(s, "seed", Int; default = 1, range = (0, typemax(Int)), doc = "seed of the initial noise")
    end
    ref = section("reference"; doc = "The periodic channel on the child grid: its own equilibrium.") do s
        option!(s, "size", NTuple{3,Int}; default = (48, 24, 16), range = (1, 4096), doc = "cells (the child's spacing)")
        option!(s, "seed", Int; default = 2, range = (0, typemax(Int)), doc = "seed of the initial noise")
    end
    cd = section("child"; doc = "The channel opened in x, one run per generator.") do s
        option!(s, "size", NTuple{3,Int}; default = (144, 24, 16), range = (1, 4096), doc = "cells")
        option!(s, "length", Float64; default = 15.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "streamwise length L_x (δ)")
        option!(s, "t_spin", Float64; units = "1", default = 2.0, range = (0, Inf), doc = "spin-up (δ/u*)")
        option!(s, "t_avg", Float64; units = "1", default = 10.0, range = (0, Inf), open_lower = true,
                doc = "averaging time (δ/u*)")
        option!(s, "sample_every", Int; default = 4, range = (1, typemax(Int)),
                doc = "bulk velocity, pressure and the stress band every this many steps")
        option!(s, "lateral", Symbol; default = :periodic, allowed = [:periodic, :open],
                doc = "the spanwise faces: periodic, or open with the mean profile as Dirichlet data (statistics over the span centre, 0.5δ from the sides)")
        option!(s, "runs", Vector{String}; default = ["off", "df", "fsd", "df_same"],
                allowed = sort!(collect(keys(_GENERATORS))),
                doc = "off (mean profile only), df (digital filter, precursor statistics), fsd (FreeStreamInflow, precursor statistics), df_same (digital filter, reference statistics and mean)")
        option!(s, "heights", Vector{Float64}; default = [0.2, 0.5], range = (0, 1), units = "1",
                doc = "the levels (nearest to these heights, δ) where the fetch is measured")
        option!(s, "outflow_margin", Float64; default = 0.5, range = (0, Inf), units = "1",
                doc = "the last part of the child (δ) left out of the development lengths: the columns next to the east face lose variance to the radiation condition")
        option!(s, "sample_x", Vector{Float64}; default = [1.0, 3.0, 5.0, 10.0, 14.0], range = (0, Inf), units = "1",
                doc = "positions (δ) where the ratios to the reference are reported")
    end
    return Section[ch, pre, ref, cd]
end

function _checks(cfg)
    e = no_checkpoints(cfg)
    isempty(cfg.child.runs) && push!(e, ConfigError("child.runs", "no child run"))
    allunique(cfg.child.runs) || push!(e, ConfigError("child.runs", "a run is listed twice"))
    cfg.precursor.size[3] == cfg.reference.size[3] == cfg.child.size[3] ||
        push!(e, ConfigError("child.size", "the precursor, the reference and the child share the vertical grid"))
    cfg.child.outflow_margin < cfg.child.length ||
        push!(e, ConfigError("child.outflow_margin", "the margin is longer than the child"))
    return e
end
Config.driver_checks(::InflowFetchDriverType) = Any[
    """`[checkpoint]` stays off (the run is one call)
    `child.runs` is not empty and has no repeats; the three grids share their vertical levels; `child.outflow_margin` < `child.length`""" => _checks]

"""
    InflowFetchRun

What the `inflow_fetch` driver builds and runs: the resolved `cfg`, the
precursor `pre` and reference `ref` ([`fetch_precursor`](@ref)) and the
child runs `runs` (name => [`fetch_child`](@ref) result).
"""
mutable struct InflowFetchRun
    cfg::RunConfig
    pre::Any
    ref::Any
    runs::Vector{Pair{String,Any}}
end

Config.build(::InflowFetchDriverType, cfg::RunConfig) = InflowFetchRun(cfg, nothing, nothing, Pair{String,Any}[])

function Simulations.run!(::InflowFetchDriverType, r::InflowFetchRun, rf::RunFolder, ::CheckpointSchedule;
                          callback = nothing)
    cfg = r.cfg; ch = cfg.channel; c = cfg.child; P = cfg.precursor
    ext = (ch.precursor_length, ch.width)
    r.pre = pre = fetch_precursor(; N = P.size, extent = ext, t_spin = P.t_spin, t_avg = P.t_avg, seed = P.seed,
                                  z0_over_H = ch.z0_over_h)
    r.ref = ref = fetch_precursor(; N = cfg.reference.size, extent = ext, t_spin = P.t_spin, t_avg = P.t_avg,
                                  seed = cfg.reference.seed, z0_over_H = ch.z0_over_h)
    @info "precursors" precursor_s = round(pre.wall_time; digits = 1) reference_s = round(ref.wall_time; digits = 1)
    out = Dict{String,Any}("precursor_wall_seconds" => pre.wall_time, "reference_wall_seconds" => ref.wall_time,
        "profiles" => Dict{String,Any}("z" => pre.z, "z_faces" => pre.z_faces, "U_pre" => pre.U, "U_ref" => ref.U,
                                       "tke_pre" => pre.tke, "tke_ref" => ref.tke, "uw_pre" => pre.uw,
                                       "uw_ref" => ref.uw, "stress_pre" => pre.stress))
    ks = [nearest_level(pre.z, h) for h in c.heights]
    runs = Dict{String,Any}()
    for (n, name) in enumerate(c.runs)
        src = name == "df_same" ? ref : pre
        res = fetch_child(src; generator = _GENERATORS[name], N = c.size, extent = (c.length, ch.width),
                          t_spin = c.t_spin, t_avg = c.t_avg, sample_every = c.sample_every, lateral = c.lateral)
        push!(r.runs, name => res)
        Ub = sum(src.U) / length(src.U)
        bias = pressure_gradient_bias(res, src.case.forcing)
        @info "child run" name wall_s = round(res.wall_time; digits = 1) bulk_ratio = res.bulk[1] / Ub dpdx_bias = bias
        x = res.p.x
        keep = findall(<=(c.length - c.outflow_margin), x)
        d = Dict{String,Any}("bulk_ratio" => res.bulk[1] / Ub, "dpdx_bias" => bias, "ke_flux" => res.ke_flux,
                             "added_mass_flux" => res.added, "max_div" => res.maxdiv, "wall_seconds" => res.wall_time,
                             "x" => x)
        for (j, k) in enumerate(ks)
            L(f, T, fr) = development_length((x[keep], getfield(res.p, f)[keep, k]), T; x0 = 0.0, frac = fr)
            at(f, T) = [getfield(res.p, f)[nearest_level(x, xx), k] / T for xx in c.sample_x if xx < c.length]
            d["level_$j"] = Dict{String,Any}(
                "z" => pre.z[k],
                "L10_tke_source" => L(:tke, src.tke[k], 0.9), "L10_tke_reference" => L(:tke, ref.tke[k], 0.9),
                "L20_tke_source" => L(:tke, src.tke[k], 0.8), "L20_tke_reference" => L(:tke, ref.tke[k], 0.8),
                "L10_uw_source" => L(:uw, src.uw[k], 0.9), "L10_uw_reference" => L(:uw, ref.uw[k], 0.9),
                "L20_uw_source" => L(:uw, src.uw[k], 0.8), "L20_uw_reference" => L(:uw, ref.uw[k], 0.8),
                "tke_at_sample_x" => at(:tke, ref.tke[k]), "uw_at_sample_x" => at(:uw, ref.uw[k]),
                "tke_over_reference" => res.p.tke[:, k] ./ ref.tke[k], "uw_over_reference" => res.p.uw[:, k] ./ ref.uw[k])
        end
        runs[name] = d
        callback === nothing || callback(r, n)
    end
    out["runs"] = runs
    out["sample_x"] = [xx for xx in c.sample_x if xx < c.length]
    write_results(rf, out)
    return r
end

"""
    state_fields(r::InflowFetchRun) -> NamedTuple

The precursor's and the reference's mean profiles and, per child run, its
TKE(x, z) and −u'w'(x, z).
"""
function Config.state_fields(r::InflowFetchRun)
    out = Pair{Symbol,Any}[:U_pre => r.pre.U, :U_ref => r.ref.U]
    for (name, res) in r.runs
        push!(out, Symbol("tke_", name) => res.p.tke)
        push!(out, Symbol("uw_", name) => res.p.uw)
    end
    return (; out...)
end

register_driver!(InflowFetchDriverType())

end # module
