# Nesting: two-way feedback (N-nest-e, #263).
#
#   * TwoWayFeedback — the configuration of the child → parent feedback:
#     anterpolation (Clark & Farley 1984) with an optional relaxation, the
#     buffer zone, the canopy restriction, the inflow exclusion band,
#     scalar refluxing and the re-projection of the parent;
#   * NestScalarFlux — the transport whose face flux a register books;
#   * NestFluxRegisters — Berger–Colella flux registers on the interface
#     planes, accumulated at every stage of every parent step and child
#     substep through a body-force hook (exactly the fluxes the cores used);
#   * anterpolate!, two_way_feedback!, feedback_mask — the feedback itself.
#
# All operators come from N-nest-a (#250) / N-nest-b (#256): the restriction
# is `restrict_to_parent!` (volume- or mass-weighted, solid-aware), the
# refluxing `apply_flux_correction!`, the face flux of a scalar
# `AdvectionSchemes.advective_face_flux` and the closures'
# `sgs_scalar_face_diffusivity`.

const _SIDE_NAMES = (:west, :east, :south, :north, :bottom, :top)
_side_axis(s::Int) = (s + 1) ÷ 2
_side_high(s::Int) = iseven(s)

# =========================================================================
# Configuration
# =========================================================================

"""
    NestScalarFlux(; scheme = :centered, molecular = false, sgs = :none)

The transport of a cell-centred scalar whose interface flux a two-way
nest books in its flux registers ([`TwoWayFeedback`](@ref)): the
advective face flux of `scheme` (a name of
[`boundary_scalar_flux`](@ref Aeolus.AdvectionSchemes.boundary_scalar_flux) or a reconstruction descriptor; the
Boussinesq and sound-proof cores advect `θ` and `qv` with `:centered`),
plus the molecular diffusion `ν∇ϕ` of the simulation (`molecular = true`,
when the simulation has viscosity) and the SGS flux of the domain's
closure: `sgs = :theta` with θ's diffusivity
([`sgs_scalar_face_diffusivity`](@ref Aeolus.Operators.sgs_scalar_face_diffusivity)),
`sgs = :tracer` with the diffusivity the core gives a tracer
([`sgs_tracer_diffusivity!`](@ref Aeolus.Operators.sgs_tracer_diffusivity!),
as for `qv`), `:none` without.

`θ` and `qv` get their models from the simulations; a tracer of the
field set is refluxed only when its owner declares how it is
transported, `TwoWayFeedback(tracer_fluxes = (:co2 => NestScalarFlux(),))`.
"""
struct NestScalarFlux{R}
    rec::R
    molecular::Bool
    sgs::Symbol
end

function NestScalarFlux(; scheme = :centered, molecular::Bool = false, sgs::Symbol = :none)
    sgs in (:none, :theta, :tracer) ||
        throw(ArgumentError("NestScalarFlux: sgs must be :none, :theta or :tracer, got :$sgs"))
    return NestScalarFlux(_flux_reconstruction(scheme), molecular, sgs)
end

"""
    TwoWayFeedback(; buffer = 2, z_anterp = -Inf, τ = nothing, exclude_width = 0,
                   exclude_sides = :inflow, reflux = true, reproject = true,
                   tracer_fluxes = ())

The child → parent feedback of a two-way [`NestedRun`](@ref)
(`feedback = TwoWayFeedback(…)`, or `feedback = :two_way` for the
defaults), applied once per parent step after the child substeps:

 1. **Anterpolation** (Clark & Farley 1984; Hellsten et al. 2021, §3.5,
    Eq. 15). A covered parent cell takes the conservative child mean
    ``⟨φ_C⟩`` of `θ`, `qv`, the tracers and the velocities: the N-nest-a
    / N-nest-b restriction ([`restrict_to_parent!`](@ref), volume- or
    reference-mass-weighted, solid-aware). With a relaxation time `τ` it
    is relaxed instead,

        φ_P ← φ_P + (Δt_p/τ)(⟨φ_C⟩ − φ_P),

    so a covered-parent deviation decays as ``(1 − Δt_p/τ)^n``.
    `τ = nothing` (default) is direct replacement, as in PALM, and so is
    any `τ ≤ Δt_p`. The relaxation is our extension, not PALM's.
 2. **Buffer zone.** No anterpolation in the outermost `buffer` parent
    cells next to every open (nest) face of the child (PALM's default is
    2): there the child's solution is still dominated by its boundary
    data.
 3. **Canopy restriction.** Only parent cells whose lower face is at or
    above `z_anterp` are anterpolated (PALM leaves the canopy layer out,
    where the coarse parent cannot represent what the child resolves).
 4. **Inflow exclusion.** A band of `exclude_width` parent cells at the
    child's inflow faces is left out too, so synthetic inflow turbulence
    (N-inflow-c) is never fed back. `exclude_sides = :inflow` picks the
    open faces whose mean normal velocity points into the child at the
    time of the feedback; a tuple of side names (`(:west,)`) or an
    `NTuple{6,Bool}` (west, east, south, north, bottom, top) fixes them.
 5. **Scalar refluxing** (Berger & Colella 1989): every parent step and
    child substep books the flux of `θ`, `qv` and the declared tracers
    through the interface planes (advective, molecular and SGS — exactly
    the face fluxes the cores used, at every stage with the stage weights
    of the core) into flux registers; the coarse-minus-fine mismatch is
    put into the parent cells just outside the nest
    ([`apply_flux_correction!`](@ref)). The composite integral
    ``∫φ dV`` (``∫Wφ dV`` for a sound-proof pair) is then conserved to
    round-off. `tracer_fluxes` declares the transport of the tracers
    (`name => NestScalarFlux(…)`); undeclared tracers are not refluxed.
 6. **Re-projection** of the parent after the feedback, so ``∇·u = 0``
    holds in the parent again (anterpolated faces next to the buffer and
    the interface flux correction leave the parent divergent).

The anterpolated velocity faces are those strictly inside the covered
region whose two parent cells are both anterpolated; the interface faces
themselves get the child's flux from the interface flux correction
(`flux_correction` of the run, on the open faces only).

Refluxing needs a core whose stage structure is known: `step!`
(Boussinesq or sound-proof; one stage) or `step_rk3!` (weights 1/6, 1/6,
2/3), and an SGS closure with a two-point face diffusivity. Exact
conservation also needs `sponge = nothing` (a sponge is a source).

References: Clark, T. L. & Farley, R. D. (1984), *J. Atmos. Sci.* 41,
329–350; Berger, M. J. & Colella, P. (1989), *J. Comput. Phys.* 82,
64–84; Hellsten, A. et al. (2021), *Geosci. Model Dev.* 14, 3185–3214,
§3.5 (anterpolation, buffer, canopy) and §4.1 (the spurious secondary
circulation of two-way nests in a convective boundary layer).
"""
struct TwoWayFeedback
    buffer::Int
    z_anterp::Float64
    τ::Union{Nothing,Float64}
    exclude_width::Int
    exclude_sides::Union{Symbol,NTuple{6,Bool}}
    reflux::Bool
    reproject::Bool
    tracer_fluxes::Vector{Pair{Symbol,NestScalarFlux}}
end

function TwoWayFeedback(; buffer::Integer = 2, z_anterp::Real = -Inf,
                        τ::Union{Nothing,Real} = nothing, exclude_width::Integer = 0,
                        exclude_sides = :inflow, reflux::Bool = true, reproject::Bool = true,
                        tracer_fluxes = ())
    buffer >= 0 || throw(ArgumentError("TwoWayFeedback: buffer must be ≥ 0, got $buffer"))
    exclude_width >= 0 ||
        throw(ArgumentError("TwoWayFeedback: exclude_width must be ≥ 0, got $exclude_width"))
    τ === nothing || τ > 0 || throw(ArgumentError("TwoWayFeedback: τ must be > 0, got $τ"))
    ex = _exclude_sides(exclude_sides)
    tf = Pair{Symbol,NestScalarFlux}[]
    for pr in tracer_fluxes
        first(pr) isa Symbol && last(pr) isa NestScalarFlux ||
            throw(ArgumentError("TwoWayFeedback: tracer_fluxes are `name => NestScalarFlux(…)`"))
        first(pr) in (:θ, :qv) &&
            throw(ArgumentError("TwoWayFeedback: θ and qv take their transport from the " *
                                "simulations; tracer_fluxes is for the extra scalars"))
        push!(tf, first(pr) => last(pr))
    end
    return TwoWayFeedback(Int(buffer), Float64(z_anterp),
                          τ === nothing ? nothing : Float64(τ), Int(exclude_width), ex,
                          reflux, reproject, tf)
end

_exclude_sides(s::Symbol) = s === :inflow ? s :
    throw(ArgumentError("TwoWayFeedback: exclude_sides must be :inflow, side names or " *
                        "an NTuple{6,Bool}, got :$s"))
_exclude_sides(t::NTuple{6,Bool}) = t
function _exclude_sides(t::Tuple{Vararg{Symbol}})
    for s in t
        s in _SIDE_NAMES || throw(ArgumentError("TwoWayFeedback: unknown side :$s " *
                                                "(use $(_SIDE_NAMES))"))
    end
    return ntuple(i -> _SIDE_NAMES[i] in t, 6)
end

Base.show(io::IO, f::TwoWayFeedback) =
    print(io, "TwoWayFeedback(buffer=", f.buffer, ", z_anterp=", f.z_anterp, ", τ=",
          f.τ === nothing ? "Δt_p" : f.τ, ", exclude_width=", f.exclude_width,
          ", reflux=", f.reflux, ", reproject=", f.reproject, ")")

# =========================================================================
# Flux registers
# =========================================================================

"""
    NestFluxRegisters

The Berger–Colella flux registers of a two-way [`NestedRun`](@ref): for
every refluxed scalar (`θ`, `qv`, declared tracers) and every interface
plane (the child's open faces), the time integral `∫ F dt` of the face
flux density (`∫ W F dt` for a sound-proof pair) through every parent
face of the plane and every child face of the plane. They are booked by
a hook the run chains in front of each domain's `body_force!`, at every
stage of every accepted step, with the stage weight of the core (a
rejected step's bookings are discarded with it), and consumed by the
refluxing of [`two_way_feedback!`](@ref). Only the planes are stored,
not whole face fields. One more set of planes books the volume (mass)
flux `∫ W u_n dt` of every face, for the volume-consistent refluxing
(see [`two_way_feedback!`](@ref)).
"""
mutable struct NestFluxRegisters
    names::Vector{Symbol}
    models::Vector{NestScalarFlux}
    pfield::Vector{Field}
    cfield::Vector{Field}
    sides::NTuple{6,Bool}
    pacc::Vector{Vector{Matrix{Float64}}}     # committed, per scalar (+ volume), per side
    cacc::Vector{Vector{Matrix{Float64}}}
    ppend::Vector{Vector{Matrix{Float64}}}    # the step being attempted
    cpend::Vector{Vector{Matrix{Float64}}}
    pweights::Tuple
    cweights::Tuple
    pstage::Int
    cstage::Int
    psgs::Any                                  # the domains' (sgs, eddy_viscosity!) keywords
    csgs::Any
    pscratch::Field
    cscratch::Field
end

# Stage weights of a core (the SSP-RK3 combination of the three Euler
# substages is uⁿ + Δt(L₀ + L₁)/6 + 2Δt L₂/3); `nothing` = unknown.
_stage_weights(a::DomainAdvance) = _stage_weights(a.core, a.args)
_stage_weights(core, args) = nothing
_stage_weights(::typeof(step!), args) = (1.0,)
_stage_weights(::typeof(step_rk3!), args) = (1.0 / 6, 1.0 / 6, 2.0 / 3)

_sgs_keywords(a::DomainAdvance) = (get(a.kwargs, :sgs, nothing),
                                   get(a.kwargs, :eddy_viscosity!, nothing))

_domain_closure(sim, kw) = _active_sgs(sim, kw[1], kw[2])

function _plane_sizes(nest::NestedGrid, s::Int)
    ax = _side_axis(s)
    nn = (nest.nx, nest.ny, nest.nz); r = (nest.rx, nest.ry, nest.rz)
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    return (nn[t1], nn[t2]), (nn[t1] * r[t1], nn[t2] * r[t2])
end

function _new_planes(nest, sides, which::Int)
    return [sides[s] ? zeros(_plane_sizes(nest, s)[which]) : zeros(0, 0) for s in 1:6]
end

function _check_register_closure(sim, kw, what)
    c = _domain_closure(sim, kw)
    c === nothing && return nothing
    f = sgs_fields(c, sim)
    sgs_scalar_face_diffusivity(c, f, sim) === nothing &&
        throw(ArgumentError("TwoWayFeedback: the $what's SGS closure $(nameof(typeof(c))) " *
                            "has a tensor scalar flux, which the flux registers cannot " *
                            "book face by face; use TwoWayFeedback(reflux = false)"))
    return nothing
end

function NestFluxRegisters(nest::NestedGrid, parent, child, s::NestInterpState,
                           fb::TwoWayFeedback, ap::DomainAdvance, ac::DomainAdvance)
    wp = _stage_weights(ap); wc = _stage_weights(ac)
    (wp === nothing || wc === nothing) &&
        throw(ArgumentError("TwoWayFeedback: refluxing needs the stage weights of each " *
                            "domain's core (step! or step_rk3!); got " *
                            "$(ap.core) / $(ac.core). Use TwoWayFeedback(reflux = false)"))
    _check_register_closure(parent, _sgs_keywords(ap), "parent")
    _check_register_closure(child, _sgs_keywords(ac), "child")
    names = Symbol[:θ]
    models = NestScalarFlux[NestScalarFlux(; molecular = true, sgs = :theta)]
    pf = Field[parent.θ]; cf = Field[child.θ]
    declared = Dict(fb.tracer_fluxes)
    for i in eachindex(s.names)
        n = s.names[i]
        model = n === :qv && _moist(parent) ? NestScalarFlux(; molecular = true, sgs = :tracer) :
                get(declared, n, nothing)
        model === nothing && continue
        push!(names, n); push!(models, model)
        push!(pf, s.parent[i]); push!(cf, s.child[i])
    end
    for k in keys(declared)
        k in s.names || throw(ArgumentError("TwoWayFeedback: tracer :$k is not a scalar of " *
                                            "the run's field set $(s.names)"))
    end
    sides = nest_open_sides(child)
    _warn_internal_walls(nest, sides)
    mk(which) = [_new_planes(nest, sides, which) for _ in 1:length(names)+1]   # + volume
    return NestFluxRegisters(names, models, pf, cf, sides, mk(1), mk(2), mk(1), mk(2),
                             wp, wc, 0, 0, _sgs_keywords(ap), _sgs_keywords(ac),
                             Field(CellCenter(), nest.parent_mesh),
                             Field(CellCenter(), nest.child_mesh))
end

# A child face that is not open but lies inside the parent (a lid below the
# parent top, a side wall inside a wider parent) blocks the child but not
# the parent: the parent's flux through that plane has no fine counterpart
# and no register, so the composite integral is not conserved across it.
function _warn_internal_walls(nest::NestedGrid, sides::NTuple{6,Bool})
    pm = nest.parent_mesh
    inner = (nest.ix > 1, nest.ix + nest.nx - 1 < pm.Nx,
             nest.iy > 1, nest.iy + nest.ny - 1 < pm.Ny,
             nest.iz > 1, nest.iz + nest.nz - 1 < pm.Nz)
    for s in 1:6
        inner[s] && !sides[s] &&
            @warn "TwoWayFeedback: the child's $(_SIDE_NAMES[s]) face is not open (nest) " *
                  "but lies inside the parent; the parent's flux through that plane has " *
                  "no fine counterpart, so refluxing cannot conserve the composite " *
                  "integral across it. Make it a :nest face." maxlog = 1
    end
    return nothing
end

# --- attempts: begin / commit -----------------------------------------------

function _begin_attempt!(r::NestFluxRegisters, which::Symbol)
    planes = which === :parent ? r.ppend : r.cpend
    for P in planes, M in P
        fill!(M, 0.0)
    end
    which === :parent ? (r.pstage = 0) : (r.cstage = 0)
    return r
end

function _commit_attempt!(r::NestFluxRegisters, which::Symbol)
    acc, pend = which === :parent ? (r.pacc, r.ppend) : (r.cacc, r.cpend)
    for n in eachindex(acc), s in 1:6
        acc[n][s] .+= pend[n][s]
    end
    return r
end

function _clear_registers!(r::NestFluxRegisters)
    for P in (r.pacc, r.cacc, r.ppend, r.cpend), Q in P, M in Q
        fill!(M, 0.0)
    end
    return r
end

# The hook chained in front of a domain's body_force!: it runs inside every
# stage, after the tendencies and before the predictor, when the halos of
# the state, the velocities and the SGS coefficients are those the stage
# used.
struct _RegisterHook <: Function
    regs::NestFluxRegisters
    nest::NestedGrid
    which::Symbol
end

function (h::_RegisterHook)(sim, Δt)
    r = h.regs
    if h.which === :parent
        r.pstage += 1; st = r.pstage; ws = r.pweights; planes = r.ppend
        fields = r.pfield; kw = r.psgs; scratch = r.pscratch
    else
        r.cstage += 1; st = r.cstage; ws = r.cweights; planes = r.cpend
        fields = r.cfield; kw = r.csgs; scratch = r.cscratch
    end
    st <= length(ws) || error("NestFluxRegisters: stage $st of a core with " *
                              "$(length(ws)) stages")
    w = ws[st] * Float64(Δt)
    closure = _domain_closure(sim, kw)
    f = closure === nothing ? nothing : sgs_fields(closure, sim)
    W = _register_weights(sim)
    for n in eachindex(r.names)
        ϕ = fields[n]
        n > 1 && r.names[n] !== :qv && fill_halos!(ϕ, sim.θ_bcs)   # tracers: owner-filled
        K = _face_diffusivity(r.models[n], closure, f, sim, ϕ, scratch)
        ν = r.models[n].molecular && sim.has_viscosity ? Float64(sim.ν) : 0.0
        for s in 1:6
            r.sides[s] || continue
            _book_plane!(planes[n][s], sim, ϕ, r.models[n].rec, K, ν, W, h.nest, s,
                         h.which === :parent, w)
        end
    end
    for s in 1:6                                          # the volume (mass) flux
        r.sides[s] || continue
        _book_plane!(planes[end][s], sim, nothing, nothing, nothing, 0.0, W, h.nest, s,
                     h.which === :parent, w)
    end
    return nothing
end

_register_weights(::BoussinesqSimulation) = nothing
_register_weights(sim::MassWeightedSimulation) = constraint_weights(sim)

function _face_diffusivity(m::NestScalarFlux, closure, f, sim, ϕ, scratch)
    (m.sgs === :none || closure === nothing) && return nothing
    m.sgs === :theta && return sgs_scalar_face_diffusivity(closure, f, sim)
    κ = sgs_tracer_diffusivity!(scratch, closure, f, sim, ϕ)
    return κ === f.κh ? sgs_scalar_face_diffusivity(closure, f, sim) : (κ, κ, 1.0)
end

# The total flux density (advective + molecular + SGS, times W) through the
# face of storage index (i, j, k) normal to `ax`: the face fluxes of
# advect_centered! / weighted_scalar_advect! (advective_face_flux), of the
# Laplacian and of add_eddy_diffusion_scalar! (two-point, coefficient
# averaged linearly to the face).
@inline function _face_total_flux(rec, ϕd, veld, ax::Int, i::Int, j::Int, k::Int, g,
                                  ν::Float64, K, W)
    F = advective_face_flux(rec, ϕd, veld, ax, i, j, k)
    il, jl, kl = ax == 1 ? (i - 1, j, k) : ax == 2 ? (i, j - 1, k) : (i, j, k - 1)
    @inbounds dϕ = ϕd[i, j, k] - ϕd[il, jl, kl]
    invd = ax == 1 ? 1 / g.dx : ax == 2 ? 1 / g.dy : 1 / _dzf(g, k)
    D = ν
    if K !== nothing
        Kd = ax == 3 ? K[2].data : K[1].data
        @inbounds D += K[3] * (0.5 * (Kd[il, jl, kl] + Kd[i, j, k]))
    end
    F -= D * dϕ * invd
    W === nothing && return F
    @inbounds return (ax == 3 ? W[2][k] : W[1][k]) * F
end

# ϕ === nothing books the volume (mass) flux W u_n instead of a scalar flux.
function _book_plane!(M::Matrix{Float64}, sim, ϕ, rec, K, ν, W, nest::NestedGrid,
                      s::Int, parent::Bool, w::Float64)
    ax = _side_axis(s); hi = _side_high(s)
    g = sim.mesh
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    H = (g.Hx, g.Hy, g.Hz)
    veld = ax == 1 ? sim.u.data : ax == 2 ? sim.v.data : sim.w.data
    if parent
        oo = (nest.ix, nest.iy, nest.iz); nn = (nest.nx, nest.ny, nest.nz)
        base = H[ax] + oo[ax] + (hi ? nn[ax] : 0)
        o1 = H[t1] + oo[t1] - 1; o2 = H[t2] + oo[t2] - 1
    else
        N = (g.Nx, g.Ny, g.Nz)
        base = H[ax] + 1 + (hi ? N[ax] : 0)
        o1 = H[t1]; o2 = H[t2]
    end
    idx = [0, 0, 0]
    idx[ax] = base
    if ϕ === nothing
        @inbounds for B in axes(M, 2), A in axes(M, 1)
            idx[t1] = o1 + A; idx[t2] = o2 + B
            k = idx[3]
            Wk = W === nothing ? 1.0 : (ax == 3 ? W[2][k] : W[1][k])
            M[A, B] += w * Wk * veld[idx[1], idx[2], k]
        end
        return M
    end
    ϕd = ϕ.data
    @inbounds for B in axes(M, 2), A in axes(M, 1)
        idx[t1] = o1 + A; idx[t2] = o2 + B
        M[A, B] += w * _face_total_flux(rec, ϕd, veld, ax, idx[1], idx[2], idx[3], g,
                                        ν, K, W)
    end
    return M
end

# coarse − Σ fine on every interface plane, in the units of the registers ×
# the parent face area (a NestFluxMismatch for apply_flux_correction!), for
# the flux of ϕ − ϕ_ref: the booked scalar flux minus ϕ_ref times the booked
# volume flux.
function _register_mismatch(r::NestFluxRegisters, n::Int, nest::NestedGrid,
                            ϕref::Float64 = 0.0)
    rr = (nest.rx, nest.ry, nest.rz)
    out = Matrix{Float64}[]
    for s in 1:6
        ax = _side_axis(s)
        lo, hi_ = 2ax - 1, 2ax
        if !(r.sides[lo] || r.sides[hi_])
            push!(out, zeros(0, 0)); continue
        end
        psz, _ = _plane_sizes(nest, s)
        M = zeros(psz)
        if r.sides[s]
            P = r.pacc[n][s]; C = r.cacc[n][s]
            Pv = r.pacc[end][s]; Cv = r.cacc[end][s]
            t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
            r1, r2 = rr[t1], rr[t2]
            inv = 1 / (r1 * r2)
            @inbounds for B in 1:psz[2], A in 1:psz[1]
                fine = 0.0
                for b in 1:r2, a in 1:r1
                    ia = (A - 1) * r1 + a; ib = (B - 1) * r2 + b
                    fine += C[ia, ib] - ϕref * Cv[ia, ib]
                end
                M[A, B] = ((P[A, B] - ϕref * Pv[A, B]) - fine * inv) *
                          _face_area(nest, nothing, ax, B)
            end
        end
        push!(out, M)
    end
    return NestFluxMismatch{Float64}(out...)
end

# =========================================================================
# Masks
# =========================================================================

# The open faces of the child whose mean normal velocity points inward.
function _inflow_sides(csim, open::NTuple{6,Bool})
    m = csim.mesh
    ii, jj, kk = interior(m)
    flows = (_mean_plane(csim.u.data, first(ii), jj, kk, 1), -_mean_plane(csim.u.data, last(ii) + 1, jj, kk, 1),
             _mean_plane(csim.v.data, first(jj), ii, kk, 2), -_mean_plane(csim.v.data, last(jj) + 1, ii, kk, 2),
             _mean_plane(csim.w.data, first(kk), ii, jj, 3), -_mean_plane(csim.w.data, last(kk) + 1, ii, jj, 3))
    return ntuple(s -> open[s] && flows[s] > 0, 6)
end

function _mean_plane(d, q, ra, rb, ax)
    s = 0.0
    @inbounds for b in rb, a in ra
        s += ax == 1 ? d[q, a, b] : ax == 2 ? d[a, q, b] : d[a, b, q]
    end
    return s / (length(ra) * length(rb))
end

"""
    feedback_mask(nest, fb::TwoWayFeedback, child_sim) -> BitArray{3}
    feedback_mask(run::NestedRun) -> BitArray{3}

The covered parent cells (`nx × ny × nz`, nest-local indices) that the
two-way feedback anterpolates: all covered cells except the `fb.buffer`
outermost cells next to each open (nest) face of the child, the cells
whose lower face lies below `fb.z_anterp`, and the `fb.exclude_width`
cells next to the excluded (inflow) faces. See [`TwoWayFeedback`](@ref).
"""
function feedback_mask(nest::NestedGrid, fb::TwoWayFeedback, csim)
    open = nest_open_sides(csim)
    excl = fb.exclude_sides === :inflow ? _inflow_sides(csim, open) :
           ntuple(s -> open[s] && fb.exclude_sides[s], 6)
    nn = (nest.nx, nest.ny, nest.nz)
    act = trues(nn...)
    for s in 1:6
        open[s] || continue
        ax = _side_axis(s); hi = _side_high(s)
        wd = excl[s] ? max(fb.buffer, fb.exclude_width) : fb.buffer
        for q in 1:min(wd, nn[ax])
            selectdim(act, ax, hi ? nn[ax] - q + 1 : q) .= false
        end
    end
    if isfinite(fb.z_anterp)
        pm = nest.parent_mesh
        zf = _zfaces_storage(pm)
        tol = 1e-9 * max(1.0, abs(fb.z_anterp))
        for K in 1:nest.nz
            zf[pm.Hz + nest.iz + K - 1] >= fb.z_anterp - tol || (act[:, :, K] .= false)
        end
    end
    return act
end

# =========================================================================
# Anterpolation
# =========================================================================

# Whether the parent storage position `I` (1-based within the covered box of
# a field at location `ax`: 0 cells, 1..3 faces) is anterpolated: a cell of
# the mask, or a face strictly inside the covered region between two cells
# of the mask.
@inline function _active(act::BitArray{3}, ax::Int, I::NTuple{3,Int}, nn)
    ax == 0 && return @inbounds act[I[1], I[2], I[3]]
    q = I[ax] - 1                                    # face index 0..n along ax
    (1 <= q <= nn[ax] - 1) || return false
    a = Base.setindex(I, q, ax); b = Base.setindex(I, q + 1, ax)
    return @inbounds act[a[1], a[2], a[3]] && act[b[1], b[2], b[3]]
end

_loc_axis(::Field{CellCenter}) = 0
_loc_axis(::Field{FaceX}) = 1
_loc_axis(::Field{FaceY}) = 2
_loc_axis(::Field{FaceZ}) = 3

function _anterpolate_field!(pf::Field, cf::Field, nest::NestedGrid, act::BitArray{3},
                             α::Float64, meas, χ)
    pm = nest.parent_mesh
    ax = _loc_axis(pf)
    nn = (nest.nx, nest.ny, nest.nz)
    lo = (pm.Hx + nest.ix, pm.Hy + nest.iy, pm.Hz + nest.iz)
    rng = ntuple(d -> lo[d]:lo[d] + nn[d] - (d == ax ? 0 : 1), 3)
    old = pf.data[rng...]
    restrict_to_parent!(pf, cf, nest; measure = meas, solid = χ)
    pd = pf.data
    @inbounds for c in CartesianIndices(old)
        I = Tuple(c)
        i, j, k = rng[1][I[1]], rng[2][I[2]], rng[3][I[3]]
        if !_active(act, ax, I, nn)
            pd[i, j, k] = old[c]
        elseif α != 1.0
            pd[i, j, k] = old[c] + α * (pd[i, j, k] - old[c])
        end
    end
    return pf
end

"""
    anterpolate!(parent_sim, child_sim, nest, state; mask, α = 1, solid = nothing)

Anterpolation of the whole field set (`θ`, the scalars of `state`,
`u, v, w`) from the child onto the covered parent cells of `mask`
([`feedback_mask`](@ref)): the conservative child mean of
[`restrict_to_parent!`](@ref) in the state's measure, solid-aware with
`solid`, relaxed with `α = Δt_p/τ` (`φ_P ← φ_P + α(⟨φ_C⟩ − φ_P)`; `α = 1`
replaces exactly). Parent cells and faces outside the mask are left
bit for bit as they were. Velocity faces are anterpolated when strictly
inside the covered region and both their parent cells are in the mask;
the interface planes are the flux correction's.
"""
function anterpolate!(psim::_NestSim, csim::_NestSim, nest::NestedGrid, s::NestInterpState;
                      mask::BitArray{3}, α::Real = 1.0, solid = nothing)
    size(mask) == (nest.nx, nest.ny, nest.nz) ||
        throw(DimensionMismatch("anterpolate!: mask has size $(size(mask)), expected " *
                                "$((nest.nx, nest.ny, nest.nz))"))
    0 < α <= 1 || throw(ArgumentError("anterpolate!: α = $α must be in (0, 1]"))
    _check_two_way_pair(psim, csim, "anterpolate!", s)
    a = Float64(α)
    meas = s.measure
    χf = _fluid(nest, solid)
    χ = χf === nothing ? nothing : _FluidMask(χf)
    _anterpolate_field!(psim.θ, csim.θ, nest, mask, a, meas, χ)
    for i in eachindex(s.names)
        _anterpolate_field!(s.parent[i], s.child[i], nest, mask, a, meas, χ)
    end
    _anterpolate_field!(psim.u, csim.u, nest, mask, a, meas, χ)
    _anterpolate_field!(psim.v, csim.v, nest, mask, a, meas, χ)
    _anterpolate_field!(psim.w, csim.w, nest, mask, a, meas, χ)
    return psim
end

# =========================================================================
# Re-projection
# =========================================================================

_reproject!(sim::BoussinesqSimulation) = _project_velocity!(sim)
_reproject!(sim::MassWeightedSimulation) = _project_mass_weighted!(sim)

# Keep only the planes of the open sides of a velocity mismatch.
function _only_sides(m::NestFluxMismatch{T}, sides::NTuple{6,Bool}) where {T}
    keep(M, s) = (isempty(M) || sides[s]) ? M : zero(M)
    return NestFluxMismatch{T}(keep(m.west, 1), keep(m.east, 2), keep(m.south, 3),
                               keep(m.north, 4), keep(m.bottom, 5), keep(m.top, 6))
end
