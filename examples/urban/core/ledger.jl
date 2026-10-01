# Energy and water budget ledger.
#
# The ledger integrates, step by step, the surface energy balance of every
# surface class and the water budget of the air, from terms the couplings
# report. It is how the case checks that what one coupling takes from the
# surface another one gives to the air, and it is where every clip and
# every non-conservative step becomes a number instead of a silent loss.
#
# Each step the core calls `ledger_begin_step!` before the first stage
# (it snapshots the surface temperature and the water in the air) and
# `ledger_end_step!` after `:sanity` (it calls every coupling's
# `budget!(c, case, ledger)` and books the change of air water). A
# coupling reports only what it did during that step:
#
#   add_surface!(ledger, term, flux, sfc, dt)   W m⁻² per column over dt
#   add_water!(ledger, term, amount)            kg m⁻² (domain mean) into the air
#
# Surface energy. The core terms are `:Rn` (net radiation, into the
# surface), `:H`, `:LE`, `:G` (sensible, latent and ground heat flux, out
# of the surface) and `:dS` (storage change of the surface layer). Any
# other term name is an extra flux INTO the surface (negative = out), e.g.
# the dew's latent heat, sunlight transmitted through glass, energy added
# by a clamp. The ledger reports per class
#
#   residual = Rn − H − LE − G − ΔS                 (the classic imbalance)
#   closure  = residual + Σ extras                  (≈ 0 if nothing is missed)
#
# `residual` is a physical baseline number: what the conventional
# four-term balance misses in this model. `closure` is a bookkeeping
# check: it is round-off only if every coupling that changes a surface
# temperature reports what it did.
#
# Water. `add_water!` terms are sources of water to the air (negative =
# sink): surface evaporation, dew deposition, precipitation reaching the
# ground, clips, the non-conservation of a transport scheme, and the
# `density_redistribution` a transport books through
# `ledger_water_split` (the ledger integrates with the reference density
# ρ̄_k, which the incompressible transport does not conserve). The
# ledger measures the change of water in the air itself, over the tracers
# in `ledger.water_tracers` (default qv, ql, qr; a coupling that adds a
# water species calls `water_tracer!`), and reports
#
#   water residual = ΔW_air − Σ sources.

"Names of the surface classes the ledger reports (class code = index)."
const LEDGER_CLASS_NAMES = ("street", "roof", "park", "water", "wall", "glass")

"The core surface-energy terms; every other surface term is an extra flux into the surface."
const SURFACE_CORE_TERMS = (:Rn, :H, :LE, :G, :dS)

"""
    Ledger

Energy and water budget of a case, integrated over the steps taken since
the ledger was (re)started. See `core/ledger.jl` for the conventions.

* `surface[term]` — per surface class, ``Σ_{columns} ∫ F\\,dt`` in J m⁻²
  (summed over the class's columns);
* `water[term]` — ``∫`` source of water to the air, kg m⁻² of domain;
* `water_air` — change of the water in the air, kg m⁻² of domain;
* `terms[name]` — free-form accumulated diagnostics ([`add_term!`](@ref));
* `ncol` — columns per class, `elapsed` (s), `nsteps`;
* `water_tracers` — tracers counted as water in the air;
* `ρ` — air density per level used for the water integral (kg m⁻³).
  `build_case` sets it to the reference density `case.ref.ρ̄`
  (`core/reference_state.jl`), the density every coupling converts its
  fluxes with; a bare `Ledger()` falls back to 1;
* `Ts_start`, `W_start` — snapshots at the start of the current step;
* `dt` — the length (s) of the current step, set before the `budget!`
  calls. Use it, not `cfg.dt`, to turn a step's energy into a flux: with
  the adaptive time stepper the two differ.
"""
mutable struct Ledger
    terms::Dict{Symbol,Float64}
    surface::Dict{Symbol,Vector{Float64}}
    water::Dict{Symbol,Float64}
    water_air::Float64
    water_tracers::Vector{Symbol}
    ρ::Vector{Float64}
    ncol::Vector{Int}
    elapsed::Float64
    nsteps::Int
    Ts_start::Matrix{Float64}
    W_start::Float64
    dt::Float64
end
Ledger() = Ledger(Dict{Symbol,Float64}(), Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Float64}(),
                  0.0, [:qv, :ql, :qr], Float64[], Int[], 0.0, 0, zeros(0, 0), 0.0, 0.0)

"Add `value` to the free-form ledger term `name` (a diagnostic outside the budgets)."
add_term!(l::Ledger, name::Symbol, value::Real) =
    (l.terms[name] = get(l.terms, name, 0.0) + Float64(value); l)

"""
    water_tracer!(ledger, name) -> ledger

Count tracer `name` as water in the air (call it from the `init!` of a
coupling that adds a water species, e.g. `:qi`). Do not add tracers that
are a *part* of another one (P3's rime mass is inside `qi`).
"""
water_tracer!(l::Ledger, name::Symbol) = (name in l.water_tracers || push!(l.water_tracers, name); l)

"""
    add_surface!(ledger, term, flux, sfc, dt) -> ledger

Book the surface-energy `term` for one step: `flux` is a W m⁻² map on the
`(Nx, Ny)` columns (per unit area of the column), applied over `dt`
seconds, and `sfc` the surface-class map. Core terms (`:Rn`, `:H`,
`:LE`, `:G`, `:dS`) keep their conventional signs; any other term is a
flux into the surface.
"""
function add_surface!(l::Ledger, term::Symbol, flux::AbstractMatrix, sfc::AbstractMatrix, dt::Real)
    size(flux) == size(sfc) || throw(DimensionMismatch("add_surface!: flux $(size(flux)) vs sfc $(size(sfc))"))
    acc = get!(() -> zeros(length(LEDGER_CLASS_NAMES)), l.surface, term)
    @inbounds for I in eachindex(flux, sfc)
        acc[sfc[I]] += flux[I] * dt
    end
    return l
end

"""
    add_water!(ledger, term, amount) -> ledger

Book `amount` kg m⁻² (per unit domain area) of water added to the air
during this step by `term` (negative: removed, e.g. precipitation).
"""
add_water!(l::Ledger, term::Symbol, amount::Real) =
    (l.water[term] = get(l.water, term, 0.0) + Float64(amount); l)

"""
    air_water(case) -> kg m⁻²

Water in the air per unit domain area: ``Σ_{tracers} Σ_{cells} ρ_k q\\,Δz
/ (N_x N_y)`` over `case.ledger.water_tracers` that are registered.
"""
function air_water(case)
    W = 0.0
    for name in case.ledger.water_tracers
        haskey(case.tracers, name) || continue
        W += ledger_water(case, interior_view(tracer(case, name)))
    end
    return W
end

"""
    ledger_water(case, q) -> kg m⁻²

``Σ_k ρ_k Σ_{ij} q\\,Δz / (N_x N_y)`` for an interior-sized array `q`
(a specific humidity, or its increment): the water it represents per unit
domain area.
"""
function ledger_water(case, q::AbstractArray{<:Real,3})
    l, mesh = case.ledger, case.mesh
    isempty(l.ρ) && (l.ρ = ones(mesh.Nz))
    W = 0.0
    @inbounds for k in axes(q, 3)
        W += l.ρ[k] * sum(view(q, :, :, k))
    end
    return W * mesh.dz / (mesh.Nx * mesh.Ny)
end

"""
    ledger_water_split(case, dq, W = nothing) -> (transport, density_redistribution)

Split ``ledger_water(case, dq)`` of a transport increment `dq` into two
parts that sum to it, according to what the transport conserves: ``Σ W
q`` with the per-level weight `W` (length `Nz`), or ``Σ q`` when `W` is
`nothing` (W ≡ 1). With ``c = \\bar ρ_m / \\bar W_m`` (column means of
`ledger.ρ` and `W`) the first part is ``c Σ_k W_k Σ_{ij} dq\\,Δz/(N_xN_y)``,
zero for a flux-form transport that conserves its ``Σ W q``, and the
second ``Σ_k (ρ_k − c W_k) Σ_{ij} dq\\,Δz/(N_xN_y)``, the water the ledger
sees created or destroyed because the transport conserves a different
weight than the ledger's ρ̄. On the Boussinesq core (``∇·u = 0``
conserves ``Σ q``) that is the incompressible transport moving `q`
between levels of different reference density; upward transport of
vapour makes it negative. On the anelastic core the transport is
ρ̄-weighted (`scalar_advection`, #133), `W = ρ̄ = ledger.ρ` and the second
part is exactly 0; on the pseudo-incompressible core `W = ρ̄θ̄`. With
`ledger.ρ ≡ 1` and `W = nothing` it is exactly 0.
"""
function ledger_water_split(case, q::AbstractArray{<:Real,3}, W = nothing)
    l, mesh = case.ledger, case.mesh
    isempty(l.ρ) && (l.ρ = ones(mesh.Nz))
    ρm = sum(l.ρ) / length(l.ρ)
    c = W === nothing ? ρm : ρm / (sum(W) / length(W))
    flat, redis = 0.0, 0.0
    @inbounds for k in axes(q, 3)
        s = sum(view(q, :, :, k))
        w = W === nothing ? 1.0 : W[k]
        flat += c * w * s
        redis += (l.ρ[k] - c * w) * s
    end
    f = mesh.dz / (mesh.Nx * mesh.Ny)
    return flat * f, redis * f
end

"""
    reset_ledger!(case) -> ledger

Zero every accumulated term and start integrating from the current
state (called automatically before the first step).
"""
function reset_ledger!(case)
    l = case.ledger
    empty!(l.terms); empty!(l.surface); empty!(l.water)
    l.water_air = 0.0
    l.elapsed = 0.0
    l.nsteps = 0
    l.ncol = [count(==(UInt8(k)), case.geom.sfc) for k in 1:length(LEDGER_CLASS_NAMES)]
    return l
end

_time_ledger!(case, t0) =
    (case.timing[(:ledger, :ledger)] = get(case.timing, (:ledger, :ledger), 0.0) + (time_ns() - t0) * 1e-9)

"Core hook, before the first stage of a step: snapshot Ts and the water in the air."
function ledger_begin_step!(case)
    t0 = time_ns()
    l = case.ledger
    l.nsteps == 0 && isempty(l.ncol) && reset_ledger!(case)
    size(l.Ts_start) == size(case.bus.Ts) ? copyto!(l.Ts_start, case.bus.Ts) :
                                           (l.Ts_start = copy(case.bus.Ts))
    l.W_start = air_water(case)
    _time_ledger!(case, t0)
    return l
end

"Core hook, after `:sanity`: collect every coupling's `budget!` for this step."
function ledger_end_step!(case, dt::Real)
    t0 = time_ns()
    l = case.ledger
    l.dt = dt
    for c in case.couplings
        budget!(c, case, l)
    end
    l.water_air += air_water(case) - l.W_start
    l.elapsed += dt
    l.nsteps += 1
    _time_ledger!(case, t0)
    return l
end

"""
    surface_budget(ledger) -> Vector{NamedTuple}

Per surface class present, the time- and class-mean terms in W m⁻²:
`(class, name, ncol, Rn, H, LE, G, dS, residual, extras, closure)` with
`residual = Rn − H − LE − G − dS`, `extras::Dict{Symbol,Float64}` the
other terms (into the surface) and `closure = residual + Σ extras`. The
last row, `name = "domain"`, is the area-weighted mean over all columns.
"""
function surface_budget(l::Ledger)
    rows = NamedTuple[]
    l.elapsed > 0 || return rows
    extra_names = sort!([t for t in keys(l.surface) if !(t in SURFACE_CORE_TERMS)])
    function row(k, name, n, get)
        m(t) = n == 0 ? 0.0 : get(t) / (n * l.elapsed)
        Rn, H, LE, G, dS = m(:Rn), m(:H), m(:LE), m(:G), m(:dS)
        res = Rn - H - LE - G - dS
        ex = Dict(t => m(t) for t in extra_names)
        return (class = k, name = name, ncol = n, Rn = Rn, H = H, LE = LE, G = G, dS = dS,
                residual = res, extras = ex, closure = res + sum(values(ex); init = 0.0))
    end
    for k in eachindex(l.ncol)
        l.ncol[k] == 0 && continue
        push!(rows, row(k, LEDGER_CLASS_NAMES[k], l.ncol[k],
                        t -> haskey(l.surface, t) ? l.surface[t][k] : 0.0))
    end
    push!(rows, row(0, "domain", sum(l.ncol),
                    t -> haskey(l.surface, t) ? sum(l.surface[t]) : 0.0))
    return rows
end

"""
    water_budget(ledger) -> NamedTuple

`(storage, sources, residual)` in kg m⁻² of domain over the ledger's
interval: `storage` is the change of water in the air, `sources` the
reported terms, `residual = storage − Σ sources`.
"""
function water_budget(l::Ledger)
    src = copy(l.water)
    return (storage = l.water_air, sources = src,
            residual = l.water_air - sum(values(src); init = 0.0))
end

"""
    ledger_report(case) -> String

The surface energy budget per class (W m⁻², mean over the ledger's
interval) and the water budget of the air (kg m⁻² = mm of water).
"""
function ledger_report(case)
    l = case.ledger
    io = IOBuffer()
    rows = surface_budget(l)
    isempty(rows) && return ""
    @printf(io, "Surface energy ledger over %d steps (%.0f s), W/m² per class:\n", l.nsteps, l.elapsed)
    extra_names = sort!(collect(keys(first(rows).extras)))
    @printf(io, "  %-7s %5s %8s %8s %8s %8s %8s %13s", "class", "cols", "Rn", "H", "LE", "G", "dS",
            "Rn-H-LE-G-dS")
    for t in extra_names
        @printf(io, " %12s", string(t)[1:min(end, 12)])
    end
    @printf(io, " %10s\n", "closure")
    for r in rows
        @printf(io, "  %-7s %5d %8.2f %8.2f %8.2f %8.2f %8.2f %13.3f", r.name, r.ncol,
                r.Rn, r.H, r.LE, r.G, r.dS, r.residual)
        for t in extra_names
            @printf(io, " %12.3f", r.extras[t])
        end
        @printf(io, " %10.2e\n", r.closure)
    end
    w = water_budget(l)
    @printf(io, "Water ledger (kg/m² of domain): ΔW_air %.4e = ", w.storage)
    print(io, join((@sprintf("%s %.4e", t, v) for (t, v) in sort!(collect(w.sources))), " + "))
    @printf(io, "  (residual %.2e)\n", w.residual)
    return String(take!(io))
end
