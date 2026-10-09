# The budgets of a run: one container for every budget (ONE-c, #452).
#
# It merges what were three separate systems: the urban case's energy and
# water ledger, its air-heat ledger and the SBL run type's mean-momentum
# budget. A `Budgets` holds
#
#   * the measured budgets of the air, one `StageBudget` per quantity
#     (`measured[:energy]`: the heat of the air; `:water`, `:momentum_x`,
#     `:momentum_y` and `:tracer` where the run measures them), attributed
#     automatically to the stage entries of every step
#     (budgets/stage_budget.jl);
#   * the surface energy balance of every surface class, from the terms the
#     components report (`add_surface!`), and the water of the air from the
#     sources they report (`add_water!`) against its measured change;
#   * the diagnosed plane-mean momentum budget (`MomentumBudget`,
#     budgets/momentum.jl), whose terms are evaluated from the state;
#   * free-form accumulated diagnostics (`add_term!`).
#
# A component reports only what it did during a step:
#
#   add_surface!(budgets, term, flux, sfc, dt)   W m⁻² per column over dt
#   add_water!(budgets, term, amount)            kg m⁻² (domain mean) into the air
#   add_heat!(budgets, term, J)                  J m⁻² into the air, from inside the step
#
# Surface energy. The core terms are `:Rn` (net radiation, into the
# surface), `:H`, `:LE`, `:G` (sensible, latent and ground heat flux, out
# of the surface) and `:dS` (storage change of the surface layer). Any
# other term name is an extra flux INTO the surface (negative = out), e.g.
# the dew's latent heat, sunlight transmitted through glass, energy added
# by a clamp. The budgets report per class
#
#   residual = Rn − H − LE − G − ΔS                 (the classic imbalance)
#   closure  = residual + Σ extras                  (≈ 0 if nothing is missed)
#
# `residual` is a physical baseline number: what the conventional
# four-term balance misses in this model. `closure` is a bookkeeping
# check: it is round-off only if every component that changes a surface
# temperature reports what it did.
#
# Water. `add_water!` terms are sources of water to the air (negative =
# sink): surface evaporation, dew deposition, precipitation reaching the
# ground, clips, the non-conservation of a transport scheme, and the
# `density_redistribution` a transport books through `water_split` (the
# budget integrates with the reference density ρ̄_k, which the
# incompressible transport does not conserve). The run measures the change
# of water in the air itself, over the tracers in `water_tracers` (default
# qv, ql, qr), and the budgets report
#
#   water residual = ΔW_air − Σ sources.

"Names of the surface classes the budgets report (class code = index)."
const SURFACE_CLASS_NAMES = ("street", "roof", "park", "water", "wall", "glass")

"The core surface-energy terms; every other surface term is an extra flux into the surface."
const SURFACE_CORE_TERMS = (:Rn, :H, :LE, :G, :dS)

"""
    Budgets(; bins = 24, order = ())

The budgets of a run, integrated over the steps taken since they were
(re)started. See `budgets/budgets.jl` for the conventions.

* `measured[q]` — the [`StageBudget`](@ref) of quantity `q` of the air:
  `:energy` (the heat of the air; [`heat_budget`](@ref),
  [`add_heat!`](@ref)) always; `:water`, `:momentum_x`, `:momentum_y`,
  `:tracer` where the run measures them (`bins`, `order`: of these);
* `surface[term]` — per surface class, ``Σ_{columns} ∫ F\\,dt`` in J m⁻²
  (summed over the class's columns);
* `water[term]` — ``∫`` source of water to the air, kg m⁻² of domain;
* `water_air` — change of the water in the air, kg m⁻² of domain;
* `terms[name]` — free-form accumulated diagnostics ([`add_term!`](@ref));
* `momentum` — the diagnosed plane-mean momentum budget
  ([`MomentumBudget`](@ref)), or `nothing`;
* `ncol` — columns per class, `elapsed` (s), `nsteps`;
* `water_tracers` — tracers counted as water in the air;
* `ρ` — air density per level used for the water integral (kg m⁻³), the
  reference density every component converts its fluxes with; empty
  falls back to 1;
* `Ts_start`, `W_start` — snapshots at the start of the current step;
* `dt` — the length (s) of the current step, set before the components'
  `budget!` calls. Use it, not a configured step, to turn a step's energy
  into a flux: with the adaptive time stepper the two differ.
"""
mutable struct Budgets
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
    measured::Dict{Symbol,StageBudget}
    momentum::Union{Nothing,MomentumBudget}
end
Budgets(; bins::Integer = 24, order = ()) =
    Budgets(Dict{Symbol,Float64}(), Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Float64}(),
            0.0, [:qv, :ql, :qr], Float64[], Int[], 0.0, 0, zeros(0, 0), 0.0, 0.0,
            Dict{Symbol,StageBudget}(:energy => StageBudget(; bins = bins, order = order)), nothing)

"The stage budget of the heat of the air (the energy budget of the air)."
energy(b::Budgets) = b.measured[:energy]

"The measured quantities of `b` (`:energy` first, then the others in name order)."
measured_quantities(b::Budgets) =
    vcat([:energy], sort!([q for q in keys(b.measured) if q !== :energy]))

"Add `value` to the free-form term `name` (a diagnostic outside the budgets)."
add_term!(l::Budgets, name::Symbol, value::Real) =
    (l.terms[name] = get(l.terms, name, 0.0) + Float64(value); l)

"""
    water_tracer!(budgets, name) -> budgets

Count tracer `name` as water in the air (call it from the set-up of a
component that adds a water species, e.g. `:qi`). Do not add tracers that
are a *part* of another one (P3's rime mass is inside `qi`).
"""
water_tracer!(l::Budgets, name::Symbol) = (name in l.water_tracers || push!(l.water_tracers, name); l)

"""
    add_surface!(budgets, term, flux, sfc, dt) -> budgets

Book the surface-energy `term` for one step: `flux` is a W m⁻² map on the
`(Nx, Ny)` columns (per unit area of the column), applied over `dt`
seconds, and `sfc` the surface-class map. Core terms (`:Rn`, `:H`,
`:LE`, `:G`, `:dS`) keep their conventional signs; any other term is a
flux into the surface.
"""
function add_surface!(l::Budgets, term::Symbol, flux::AbstractMatrix, sfc::AbstractMatrix, dt::Real)
    size(flux) == size(sfc) || throw(DimensionMismatch("add_surface!: flux $(size(flux)) vs sfc $(size(sfc))"))
    acc = get!(() -> zeros(length(SURFACE_CLASS_NAMES)), l.surface, term)
    # the canonical order (KA-b, #296): the step's per-column contributions
    # flux·dt, summed per class in column-major order, then added to the
    # accumulated term (a threaded flux kernel keeps this ordered fold)
    step = zeros(length(acc))
    @inbounds for I in eachindex(flux, sfc)
        step[sfc[I]] += flux[I] * dt
    end
    acc .+= step
    return l
end

"""
    add_water!(budgets, term, amount) -> budgets

Book `amount` kg m⁻² (per unit domain area) of water added to the air
during this step by `term` (negative: removed, e.g. precipitation).
"""
add_water!(l::Budgets, term::Symbol, amount::Real) =
    (l.water[term] = get(l.water, term, 0.0) + Float64(amount); l)

"""
    add_heat!(budgets, term, J; k = nothing) -> budgets
    add_heat!(budgets, term, J::AbstractVector) -> budgets

Book `J` J m⁻² of heat added to the air by `term` during the current
stage entry: [`book!`](@ref) on the energy budget ([`energy`](@ref)).
"""
add_heat!(l::Budgets, term::Symbol, J::Real; k = nothing) = (book!(energy(l), term, J; k = k); l)
add_heat!(l::Budgets, term::Symbol, J::AbstractVector) = (book!(energy(l), term, J); l)

"""
    heat_budget(budgets; window = nothing, layer = :column) -> NamedTuple

The budget of the heat of the air: [`stage_budget`](@ref) of the energy
budget ([`energy`](@ref)).
"""
heat_budget(l::Budgets; kw...) = stage_budget(energy(l); kw...)

"""
    level_sums(q) -> Vector

``S_k = Σ_{ij} q_{ijk}`` for every level of the 3-D array `q` (an
interior-sized array or an `interior_view`), in the canonical,
thread-invariant order of `plane_partials!` (KA-b, #296). The budgets
build every water and heat integral from these per-level sums, combined
over `k` in ascending order.
"""
function level_sums(q::AbstractArray{<:Real,3})
    S = zeros(size(q, 3))
    # a view into a halo-padded field: reduce the parent over the view's
    # storage range (the same order and bits, without the view's index offsets)
    if q isa SubArray && all(r -> r isa AbstractUnitRange, parentindices(q))
        return plane_partials!(S, identity, map(UnitRange{Int}, parentindices(q)), parent(q))
    end
    return plane_partials!(S, identity, axes(q), q)
end

"""
    water_content(budgets, mesh, q) -> kg m⁻²

``Σ_k ρ_k Σ_{ij} q\\,Δz / (N_x N_y)`` for an interior-sized array `q` (a
specific humidity, or its increment) on `mesh`: the water it represents
per unit domain area, with the budgets' density `ρ`.
"""
function water_content(l::Budgets, mesh, q::AbstractArray{<:Real,3})
    isempty(l.ρ) && (l.ρ = ones(mesh.Nz))
    S = level_sums(q)
    W = 0.0
    @inbounds for k in eachindex(S)
        W += l.ρ[k] * S[k]
    end
    return W * mesh.dz / (mesh.Nx * mesh.Ny)
end

"""
    water_split(budgets, mesh, dq, W = nothing) -> (transport, density_redistribution)

Split [`water_content`](@ref)`(budgets, mesh, dq)` of a transport
increment `dq` into two parts that sum to it, according to what the
transport conserves: ``Σ W q`` with the per-level weight `W` (length
`Nz`), or ``Σ q`` when `W` is `nothing` (W ≡ 1). With ``c = \\bar ρ_m /
\\bar W_m`` (column means of the budgets' `ρ` and `W`) the first part is
``c Σ_k W_k Σ_{ij} dq\\,Δz/(N_xN_y)``, zero for a flux-form transport
that conserves its ``Σ W q``, and the second ``Σ_k (ρ_k − c W_k) Σ_{ij}
dq\\,Δz/(N_xN_y)``, the water the budget sees created or destroyed
because the transport conserves a different weight than the budget's ρ̄.
On the Boussinesq core (``∇·u = 0`` conserves ``Σ q``) that is the
incompressible transport moving `q` between levels of different
reference density; upward transport of vapour makes it negative. On the
anelastic core the transport is ρ̄-weighted (`scalar_advection`, #133),
`W = ρ̄ = budgets.ρ` and the second part is exactly 0; on the
pseudo-incompressible core `W = ρ̄θ̄`. With `ρ ≡ 1` and `W = nothing` it
is exactly 0.
"""
water_split(l::Budgets, mesh, q::AbstractArray{<:Real,3}, W = nothing) =
    water_split_levels(l, mesh, level_sums(q), W)

"""
    water_split_levels(budgets, mesh, S, W = nothing) -> (transport, density_redistribution)

[`water_split`](@ref) of an increment given by its per-level column sums
`S[k] = Σ_ij dq` (e.g. the water an open face moved into each level,
#262).
"""
function water_split_levels(l::Budgets, mesh, S, W = nothing)
    isempty(l.ρ) && (l.ρ = ones(mesh.Nz))
    ρm = sum(l.ρ) / length(l.ρ)
    c = W === nothing ? ρm : ρm / (sum(W) / length(W))
    flat, redis = 0.0, 0.0
    @inbounds for (k, s) in enumerate(S)
        w = W === nothing ? 1.0 : W[k]
        flat += c * w * s
        redis += (l.ρ[k] - c * w) * s
    end
    f = mesh.dz / (mesh.Nx * mesh.Ny)
    return flat * f, redis * f
end

"""
    surface_budget(budgets) -> Vector{NamedTuple}

Per surface class present, the time- and class-mean terms in W m⁻²:
`(class, name, ncol, Rn, H, LE, G, dS, residual, extras, closure)` with
`residual = Rn − H − LE − G − dS`, `extras::Dict{Symbol,Float64}` the
other terms (into the surface) and `closure = residual + Σ extras`. The
last row, `name = "domain"`, is the area-weighted mean over all columns.
"""
function surface_budget(l::Budgets)
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
        push!(rows, row(k, SURFACE_CLASS_NAMES[k], l.ncol[k],
                        t -> haskey(l.surface, t) ? l.surface[t][k] : 0.0))
    end
    push!(rows, row(0, "domain", sum(l.ncol),
                    t -> haskey(l.surface, t) ? sum(l.surface[t]) : 0.0))
    return rows
end

"""
    water_budget(budgets) -> NamedTuple

`(storage, sources, residual)` in kg m⁻² of domain over the budgets'
interval: `storage` is the change of water in the air, `sources` the
reported terms, `residual = storage − Σ sources`.
"""
function water_budget(l::Budgets)
    src = copy(l.water)
    return (storage = l.water_air, sources = src,
            residual = l.water_air - sum(values(src); init = 0.0))
end

"""
    budget_report(budgets) -> String

The surface energy budget per class (W m⁻², mean over the budgets'
interval), the water budget of the air (kg m⁻² = mm of water) and every
measured budget of the air.
"""
function budget_report(l::Budgets)
    io = IOBuffer()
    rows = surface_budget(l)
    if !isempty(rows)
        @printf(io, "Surface energy budget over %d steps (%.0f s), W/m² per class:\n", l.nsteps, l.elapsed)
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
        @printf(io, "Water budget (kg/m² of domain): ΔW_air %.4e = ", w.storage)
        print(io, join((@sprintf("%s %.4e", t, v) for (t, v) in sort!(collect(w.sources))), " + "))
        @printf(io, "  (residual %.2e)\n", w.residual)
    end
    for q in measured_quantities(l)
        title, units = _QUANTITY_LABELS[q in keys(_QUANTITY_LABELS) ? q : :tracer]
        print(io, stage_report(l.measured[q]; title = title, units = units))
    end
    return String(take!(io))
end

"Report titles and units (amount, per second) of the measured quantities."
const _QUANTITY_LABELS = Dict(:energy => ("Air-heat budget", ("J/m²", "W/m²")),
                              :water => ("Water budget of the air", ("kg/m²", "kg/m²/s")),
                              :momentum_x => ("x-momentum budget", ("kg/m/s", "N/m²")),
                              :momentum_y => ("y-momentum budget", ("kg/m/s", "N/m²")),
                              :tracer => ("Tracer budget", ("kg/m²", "kg/m²/s")))
