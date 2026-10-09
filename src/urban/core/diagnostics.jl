# The case's diagnostics and its log events (LOG-g, #338; design in epic #293).
#
# The Berlin case reports through the library's diagnostics registry
# (`Aeolus.RunLog`): every coupling declares its scalars with
# `diagnostics(c, case)` (and its vertical profiles with
# `profile_diagnostics(c, case)`), as it declares `output_fields`; the
# core declares the frame quantities, the ledgers and the stage timing.
# The same declarations feed the two consumers:
#
#   * the run monitor's `diagnostics.csv` / `diagnostics.nc`
#     ([`case_diagnostics`](@ref), [`case_profiles`](@ref));
#   * the log: one `Info` event per output frame, one per coupling with
#     diagnostics, and the end-of-run events (run, stage timing, ledgers),
#     all `subsystem = :urban` with the values as key-values
#     ([`log_frame`](@ref), [`log_run_end`](@ref)).
#
# There is no other text: what a frame line shows is what the files hold.

"""
    per_state(f, case) -> () -> f()

`f()` evaluated at most once per model state (the step count and the
simulated time): the diagnostics of a coupling that share one costly
computation (a level mean, a reference closure) call the returned
function, and only the first of them in a row computes.
"""
function per_state(f, case)
    key = Ref((-1, NaN))
    val = Ref{Any}(nothing)
    return function ()
        k = (case.clock.n, Float64(case.sim.t))
        k === key[] || (val[] = f(); key[] = k)
        return val[]
    end
end

"""
    per_record(f, case) -> () -> f()

`f()` evaluated at most once per output record (`case.clock.nrecord`),
for the diagnostics that cost more than a step (the SGS a-priori
metrics, the Leighton ratio, the transition summary): they are
evaluated at each output frame, where their `:coupling` event shows
them, and in the first diagnostics row; the costly rows in between
repeat the last record's value. On the 64 × 64 × 64 summer case these
three took 168 ms of the 193 ms that every diagnostic and profile take
together, against a 122 ms step (#338), so evaluating them every 60 s
of a 2 s step would cost about 4.6 % of the run; at the output cadence
(1800 s) it is what their report lines cost before.
"""
function per_record(f, case)
    key = Ref(-1)
    val = Ref{Any}(nothing)
    return function ()
        k = case.clock.nrecord
        k == key[] || (val[] = f(); key[] = k)
        return val[]
    end
end

"Cell-centre heights (m) of the case's levels (stretched or uniform)."
function case_zc(case)
    dz = level_thickness(case.mesh)
    return cumsum(dz) .- dz ./ 2
end

"A registry name from a symbol: `step!` → `step` (names hold letters, digits, `_`, `.`)."
diag_name(s) = replace(string(s), r"[^A-Za-z0-9_.]" => "")

# ---------------------------------------------------------------------------
# the core's declarations
# ---------------------------------------------------------------------------

"""
    frame_diagnostics(case) -> Vector{Diagnostic}

The quantities of the frame line: the local `hour` and the
`solar_elevation` (deg), surface temperature `Ts_min` / `Ts_max` (K),
the shortwave on the surface `sw_min` / `sw_max` (W m⁻²), the first air
level's `theta1_min` / `theta1_max` (K) and `qv1_min` (kg kg⁻¹), the
largest dew on a surface `dew_max` (mm) and the largest fog water of the
first level `ql1_max` (kg kg⁻¹). All on the costly cadence; min and max
are exact and order-independent (NaN-propagating), so thread-invariant.
"""
function frame_diagnostics(case)
    clk, bus = case.clock, case.bus
    lvl1(f) = () -> view(interior_view(f), :, :, 1)
    θ = case.sim.θ
    qv, ql = tracer(case, :qv), tracer(case, :ql)
    return Diagnostic[
        Diagnostic("hour", "h", () -> clk.hour; description = "local hour of day at the step's start"),
        Diagnostic("solar_elevation", "degree",
                   () -> clk.sun === nothing ? NaN : Float64(clk.sun.elevation)),
        Diagnostic("Ts_min", "K", () -> bus.Ts; reduction = :min, description = "surface temperature"),
        Diagnostic("Ts_max", "K", () -> bus.Ts; reduction = :max, description = "surface temperature"),
        Diagnostic("sw_min", "W m-2", () -> bus.sw; reduction = :min, description = "shortwave on the surface"),
        Diagnostic("sw_max", "W m-2", () -> bus.sw; reduction = :max, description = "shortwave on the surface"),
        Diagnostic("theta1_min", "K", lvl1(θ); reduction = :min, description = "θ of the first level"),
        Diagnostic("theta1_max", "K", lvl1(θ); reduction = :max, description = "θ of the first level"),
        Diagnostic("qv1_min", "kg kg-1", lvl1(qv); reduction = :min, description = "qv of the first level"),
        Diagnostic("dew_max", "mm", () -> bus.dew; reduction = :max, description = "dew on a surface"),
        Diagnostic("ql1_max", "kg kg-1", lvl1(ql); reduction = :max,
                   description = "fog (cloud) water of the first level"),
    ]
end

"""
    ledger_diagnostics(case) -> Vector{Diagnostic}

The ledgers, as means over the ledger's interval (the run so far):

* `surface.<class>.<term>` (W m⁻²) for every surface class present and
  the area-weighted `domain`, `term` ∈ `Rn`, `H`, `LE`, `G`, `dS` and
  `closure` ([`surface_budget`](@ref); the extra surface terms are inside
  the closure);
* `water.storage` and `water.residual` (kg m⁻² of domain,
  [`water_budget`](@ref));
* `heat.storage` and `heat.closure` (W m⁻², the whole column,
  [`heat_budget`](@ref)).

`NaN` before the first step. The closures are round-off when every
coupling books what it moves; a growing one is a leak.
"""
function ledger_diagnostics(case)
    l = case.ledger
    rows = per_state(() -> surface_budget(l), case)
    heat = per_state(() -> heat_budget(l), case)
    classes = [LEDGER_CLASS_NAMES[k] for k in 1:length(LEDGER_CLASS_NAMES)
               if count(==(UInt8(k)), case.geom.sfc) > 0]
    push!(classes, "domain")
    function surface(name, term)
        return function ()
            for r in rows()
                r.name == name && return Float64(getfield(r, term))
            end
            return NaN
        end
    end
    ds = Diagnostic[]
    for name in classes, term in (:Rn, :H, :LE, :G, :dS, :closure)
        push!(ds, Diagnostic("surface.$name.$term", "W m-2", surface(name, term);
                             description = "surface energy ledger, $name, mean over the run"))
    end
    perW(x) = (b = heat(); b.elapsed > 0 ? x(b) / b.elapsed : NaN)
    append!(ds, [
        Diagnostic("water.storage", "kg m-2", () -> l.nsteps == 0 ? NaN : water_budget(l).storage;
                   description = "change of the water in the air"),
        Diagnostic("water.residual", "kg m-2", () -> l.nsteps == 0 ? NaN : water_budget(l).residual;
                   description = "water ledger: storage minus the booked sources"),
        Diagnostic("heat.storage", "W m-2", () -> perW(b -> b.storage);
                   description = "change of the heat of the air, whole column"),
        Diagnostic("heat.closure", "W m-2", () -> perW(b -> b.closure);
                   description = "air-heat ledger: storage minus the attributed terms, whole column"),
    ])
    return ds
end

"""
    timing_keys(case) -> Vector{Tuple{Symbol,Symbol}}

Every `(stage, entry)` the case times: each schedule entry, the dynamics
`(:dynamics, :step!)` and the ledgers' own `(:ledger, :ledger)` and
`(:ledger, :air_heat)`.
"""
function timing_keys(case)
    ks = Tuple{Symbol,Symbol}[(:dynamics, :step!), (:ledger, :ledger), (:ledger, :air_heat)]
    for S in ALL_STAGES, e in get(case.schedule, S, ScheduleEntry[])
        k = (S, coupling_name(e.coupling))
        k in ks || push!(ks, k)
    end
    return ks
end

"""
    timing_diagnostics(case) -> Vector{Diagnostic}

The wall time (s, cumulative since the case was built) of every stage
entry ([`timing_keys`](@ref)) as `wall.<stage>.<entry>`. The names start
with `wall`, as the CSV's own wall-clock column, so a comparison of two
runs that must agree bit for bit leaves them out with it.
"""
timing_diagnostics(case) = Diagnostic[
    Diagnostic("wall.$(diag_name(S)).$(diag_name(n))", "s", () -> get(case.timing, (S, n), 0.0);
               description = "wall time of the stage entry, cumulative")
    for (S, n) in timing_keys(case)]

"""
    diagnostics(case::Case, case) -> Vector{Diagnostic}

The core's own diagnostics: [`frame_diagnostics`](@ref), the ledgers
under `ledger.` ([`ledger_diagnostics`](@ref)), the stage timing
([`timing_diagnostics`](@ref)) and the closure the steps run under
`sgs.` (the library's `diagnostics(closure, sim)`).
"""
function diagnostics(case::Case, ::Case)
    ds = frame_diagnostics(case)
    append!(ds, prefixed("ledger", ledger_diagnostics(case)))
    append!(ds, timing_diagnostics(case))
    sgs = get(case.step_kwargs, :sgs, nothing)
    sgs === nothing || append!(ds, prefixed("sgs", diagnostics(sgs, case.sim)))
    return ds
end

"The closure's profiles under `sgs.`."
function profile_diagnostics(case::Case, ::Case)
    sgs = get(case.step_kwargs, :sgs, nothing)
    sgs === nothing && return ProfileDiagnostic[]
    return prefixed("sgs", profile_diagnostics(sgs, case.sim))
end

"The enabled couplings as a `NamedTuple` `name = coupling`, in registry order (the registry's component keys)."
coupling_components(case) = (; (coupling_name(c) => c for c in case.couplings)...)

"""
    case_diagnostics(case) -> Vector{Diagnostic}

Every scalar diagnostic of the run, what the `berlin` run driver hands
the run monitor: the dynamical core's (`diagnostics(case.sim)`:
`ke_mean`, `max_div`, …), the case core's ([`diagnostics`](@ref)`(case,
case)`) and each coupling's `diagnostics(c, case)`, prefixed with the
coupling's name (`pressure_multigrid.cycles`; a coupling's wall-clock
`wall.<x>` becomes `wall.<coupling>.<x>`, [`wall_first`](@ref)). A name
declared twice is an error.
"""
function case_diagnostics(case)
    ds = vcat(collect_diagnostics(case.sim), collect_diagnostics(case, coupling_components(case)))
    ds = Diagnostic[wall_first(d) for d in ds]
    allunique(d.name for d in ds) ||
        throw(ArgumentError("case_diagnostics: a diagnostic name is declared twice"))
    return ds
end

"""
    wall_first(d::Diagnostic) -> Diagnostic

A coupling's wall-clock diagnostic, declared as `wall.<x>` and prefixed
by [`collect_diagnostics`](@ref) to `<coupling>.wall.<x>`, renamed
`wall.<coupling>.<x>`: every wall-clock column of the files starts with
`wall`, as the stage timing and the CSV's own wall column, so a
comparison of two runs that must agree bit for bit drops them all by
the prefix. Any other diagnostic is returned as it is.
"""
function wall_first(d::Diagnostic)
    m = match(r"^([^.]+)\.wall\.(.+)$", d.name)
    m === nothing && return d
    return Diagnostic("wall.$(m[1]).$(m[2])", d.units, d.source; cadence = d.cadence,
                      reduction = d.reduction, description = d.description)
end

"""
    case_profiles(case) -> Vector{ProfileDiagnostic}

Every profile of the run for `diagnostics.nc`: the core's
(`profile_diagnostics(case.sim)`), the closure's (`sgs.`) and each
coupling's `profile_diagnostics(c, case)` under its name.
"""
case_profiles(case) =
    vcat(collect_profiles(case.sim), collect_profiles(case, coupling_components(case)))

# ---------------------------------------------------------------------------
# the log events
# ---------------------------------------------------------------------------

"`name => value` pairs of the diagnostics `ds` evaluated now, values to 4 significant digits."
diagnostic_pairs(ds) =
    [Symbol(d.name) => (v = evaluate_diagnostic(d); isfinite(v) ? round(v; sigdigits = 4) : v) for d in ds]

"""
    log_frame(case)

The log events of an output record, `subsystem = :urban`:

* `event = :frame`: `frame`, `hour`, `wall` (s since `run_case!`
  started) and the values of [`frame_diagnostics`](@ref);
* `event = :coupling`, one per coupling that declares diagnostics:
  `coupling` and the value of each of its `diagnostics(c, case)`.

`Info` records, so they reach the console and `run.log`; nothing is
evaluated when no logger wants them.
"""
function log_frame(case)
    clk = case.clock
    @info("frame $(clk.nrecord) at $(round(clk.hour; digits = 2)) h", subsystem = :urban,
          domain = current_domain(), event = :frame, frame = clk.nrecord,
          wall = round(time() - clk.t_wall0; digits = 1),
          diagnostic_pairs(frame_diagnostics(case))...)
    for c in case.couplings
        ds = diagnostics(c, case)
        isempty(ds) && continue
        @info(string(coupling_name(c)), subsystem = :urban, domain = current_domain(),
              event = :coupling, coupling = coupling_name(c), diagnostic_pairs(ds)...)
    end
    return nothing
end

"""
    log_run_end(case; nsteps, t_wall)

The end-of-run events, `Info`, `subsystem = :urban`:

* `event = :run_end`: `steps`, `wall` (s), `records`, `output` (the
  NetCDF path or `""`), `size_mb`, the peak heat island `peak_uhi`
  (max − min of `Ts`, K) and `max_div` (s⁻¹);
* `event = :timing`, one per stage entry, largest first: `stage`,
  `entry`, `seconds`, `ms_per_step`;
* `event = :ledger`, `ledger = :surface`, one per class (and `domain`):
  `class`, `ncol`, `Rn`, `H`, `LE`, `G`, `dS`, `residual`, every extra
  surface term and `closure` (W m⁻²);
* `event = :ledger`, `ledger = :water`: `storage`, every source and
  `residual` (kg m⁻² of domain);
* `event = :ledger`, `ledger = :heat`, one per layer (`column`, `canopy`,
  `0-100m`): `storage` and `closure` (W m⁻²), `elapsed` (s); then one per
  attributed term, `term = "stage/entry"`, with its W m⁻² per layer.
"""
function log_run_end(case; nsteps::Integer, t_wall::Real)
    cfg, l = case.cfg, case.ledger
    out = cfg.output_path === nothing ? "" : cfg.output_path
    @info(isempty(out) ? "ran $nsteps steps in $(round(t_wall; digits = 1)) s (no output file)" :
                         "ran $nsteps steps in $(round(t_wall; digits = 1)) s, $(case.clock.nrecord) records",
          subsystem = :urban, domain = current_domain(), event = :run_end, steps = nsteps,
          wall = round(t_wall; digits = 2), records = case.clock.nrecord, output = out,
          size_mb = isempty(out) || !isfile(out) ? 0.0 : round(filesize(out) / 1e6; digits = 2),
          peak_uhi = round(maximum(case.bus.Ts) - minimum(case.bus.Ts); sigdigits = 4),
          max_div = max_divergence(case.sim))
    n = max(case.clock.n, 1)
    for ((S, name), t) in sort!(collect(case.timing); by = last, rev = true)
        @info("timing $S/$name", subsystem = :urban, domain = current_domain(), event = :timing,
              stage = S, entry = name, seconds = round(t; digits = 3),
              ms_per_step = round(1e3 * t / n; sigdigits = 4))
    end
    r4(x) = round(x; sigdigits = 4)
    for r in surface_budget(l)
        @info("surface energy ledger: $(r.name)", subsystem = :urban, domain = current_domain(),
              event = :ledger, ledger = :surface, class = r.name, ncol = r.ncol, Rn = r4(r.Rn),
              H = r4(r.H), LE = r4(r.LE), G = r4(r.G), dS = r4(r.dS), residual = r4(r.residual),
              (t => r4(v) for (t, v) in sort!(collect(r.extras)))..., closure = r.closure)
    end
    if l.nsteps > 0
        w = water_budget(l)
        @info("water ledger", subsystem = :urban, domain = current_domain(), event = :ledger,
              ledger = :water, storage = w.storage, (t => v for (t, v) in sort!(collect(w.sources)))...,
              residual = w.residual)
    end
    hl = l.heat
    if hl.meter !== nothing
        layers = Any[:column]
        isfinite(hl.canopy_top) && push!(layers, :canopy)
        push!(layers, (0.0, 100.0))
        lname(x) = x isa Symbol ? x : Symbol("z$(round(Int, x[1]))_$(round(Int, x[2]))m")
        bs = [heat_budget(hl; layer = x) for x in layers]
        el = bs[1].elapsed
        if el > 0
            for (x, b) in zip(layers, bs)
                @info("air-heat ledger: $(lname(x))", subsystem = :urban, domain = current_domain(),
                      event = :ledger, ledger = :heat, layer = lname(x), elapsed = el,
                      storage = r4(b.storage / el), closure = b.closure / el)
            end
            for key in sort!(unique(vcat([collect(keys(b.terms)) for b in bs]...)); by = _heat_key_order)
                @info("air-heat term $(key[1])/$(key[2])", subsystem = :urban,
                      domain = current_domain(), event = :ledger, ledger = :heat_term,
                      term = "$(key[1])/$(key[2])",
                      (lname(x) => r4(get(b.terms, key, 0.0) / el) for (x, b) in zip(layers, bs))...)
            end
        end
    end
    return nothing
end
