# N-inflow-c (#264) — synthetic turbulence applied on the inflow faces of
# an open-boundary (nest child) domain, with zero added mass flux.
#
# Included inside `module Transition`, after inflow_statistics.jl.
#
# One source of truth:
#   * the unit planes and the Lund transform: N-inflow-a (#252),
#     `unit_fluctuations!`, `apply_lund!`, `lund_coefficients`;
#   * the Reynolds stresses and lengths: N-inflow-b (#258), the
#     `InflowStatistics` contract (`inflow_lund_coefficients`,
#     `inflow_lengths`);
#   * the boundary mass balance: N-open-b (#255),
#     `correct_boundary_mass_flux!`, which runs after this superposition
#     inside `update_open_boundaries!`, untouched.
#
# `InflowTurbulence` is an `AbstractBoundaryData` that wraps the boundary
# data of the child's open faces (a `NestBoundaryData` of N-nest-d, #261,
# or any other implementer): its `boundary_values!` lets the wrapped data
# fill the buffers and then adds the perturbations to them. Nothing is
# copied into the step cores.

"""
    inflow_normal_length(src::InflowStatistics, z::AbstractVector, t) -> L_n

The integral length along the face normal (m), per component (normal,
along, vertical) and height: a 3 × `length(z)` matrix. It sets the
generator's integral time through Taylor's hypothesis,
``T = L_n/\\max(|U_n|, u_{\\min})`` ([`InflowTurbulence`](@ref)).
[`ParentResolvedStatistics`](@ref) measures it (the parent's
autocorrelation along the normal), [`ParametrizedStatistics`](@ref) uses
its single `length_scale`; the fallback for any other source is the
along-face length `L_s` of [`inflow_lengths`](@ref) (horizontal isotropy).
"""
inflow_normal_length(src::InflowStatistics, z::AbstractVector, t::Real) =
    inflow_lengths(src, z, t).L_s
function inflow_normal_length(s::ParentResolvedStatistics, z::AbstractVector, t::Real)
    _check_samples(s)
    L = _lengths(s).L_n
    return [_interp(s.z, @view(L[c, :]), zz) for c in 1:3, zz in z]
end
inflow_normal_length(s::ParametrizedStatistics, z::AbstractVector, t::Real) =
    fill(s.length_scale, 3, length(z))

"""
    set_integral_time!(gen::AbstractInflowGenerator, T) -> gen

Set the integral time (s) of the next [`unit_fluctuations!`](@ref) steps:
`T` is a 3 × `nz` matrix (component, level). Used by
[`InflowTurbulence`](@ref), whose ``T = L_n/\\max(|U_n|, u_{\\min})``
follows the normal velocity of the face. A generator without an integral
time of its own ([`FreeStreamInflow`](@ref), whose time scale is the
sweep of its frozen field) ignores it.
"""
set_integral_time!(gen::AbstractInflowGenerator, T) = gen
function set_integral_time!(gen::DigitalFilterGenerator, T::AbstractMatrix)
    size(T) == size(gen.T) || throw(DimensionMismatch("set_integral_time!: T is $(size(T)), generator has $(size(gen.T))"))
    all(>(0), T) || throw(ArgumentError("set_integral_time!: T must be > 0"))
    copyto!(gen.T, T)
    return gen
end

# =========================================================================
# Per-face state
# =========================================================================

mutable struct _InflowFace{G}
    side::Symbol
    axis::Int                       # 1: west/east, 2: south/north
    s_out::Float64                  # +1 on a high face, −1 on a low face (outward normal)
    gen::G
    stats::Any
    ns::Int
    nz::Int
    Δs::Float64
    dA::Float64                     # area of one face cell (m²)
    a::Vector{Matrix{Float64}}      # Lund amplitude matrices per level
    L_s::Matrix{Float64}            # 3 × nz, the generator's lengths
    L_z::Matrix{Float64}
    L_n::Matrix{Float64}            # normal lengths (Taylor time)
    T::Matrix{Float64}              # current integral time
    P::NTuple{3,Matrix{Float64}}    # face-frame perturbations (normal, along, vertical)
    active::BitMatrix               # faces that receive a perturbation
    U_out::Vector{Float64}          # face-mean outward normal velocity per level
    ramp::Vector{Float64}           # amplitude factor per level, 0 … 1
    t_gen::Float64                  # the generator's clock
    t_prev::Float64                 # the first knot
    Lp::NTuple{3,Matrix{Float64}}   # applied (Lund-transformed) planes at t_prev …
    Lc::NTuple{3,Matrix{Float64}}   # … and at t_gen (zero before the first advance)
    Q::Float64                      # added outward normal flux, last evaluation (m³ s⁻¹)
    Q_level::Float64                # max over levels of |added flux of a level|
    Q_scale::Float64                # Σ |U_n| dA over the fluid faces (m³ s⁻¹)
    ke::Float64                     # injected KE flux, face mean (W m⁻²)
    area::Float64                   # fluid area of the face (m²)
end

"""
    InflowTurbulence(run; generator = DigitalFilterGenerator,
                     statistics = ParentResolvedStatistics, faces = :lateral,
                     u_min = 0.5, ρ_ref = 1.2, seed = 1,
                     length_interval = 3600, statistics_interval = 60,
                     statistics_kwargs = (;))

Synthetic turbulence on the **inflow** parts of the lateral open faces of
a nest child (N-inflow-c, #264): ``u_{b,i} = u_{{\\rm data},i} + a_{ij}ũ_j``,
the unit planes ``ũ_j`` of one generator per face
([`DigitalFilterGenerator`](@ref), Xie & Castro 2008) and the Lund et al.
(1998) amplitudes ``a`` of the statistics source
([`inflow_lund_coefficients`](@ref)). `run` is a [`NestedRun`](@ref)
(its child gets the perturbations; the default statistics are a
[`ParentResolvedStatistics`](@ref) of its parent per face, in scale-gap
mode with the nest's refinement) or a simulation with
[`OpenBoundary`](@ref Aeolus.BoundaryConditions.OpenBoundary) faces (then
`statistics` must be given). The object wraps the open boundary's data
and installs itself as that data, so the perturbations enter **inside
[`update_open_boundaries!`](@ref Aeolus.BoundaryConditions.update_open_boundaries!)**,
after the data fill and **before** the mass-flux correction, at every
stage time of every core. Nothing is copied into the step cores.

**Faces.** `faces = :lateral` takes every open west/east/south/north face
(a tuple of sides selects some). The top face is excluded, as in PALM
(Kadasch et al. 2021, §2.4): the perturbations enter with the mean flow,
which crosses the top of a boundary-layer nest only weakly.

**Inflow detection** (per face and level). ``U_n(z)`` is the face-mean
outward normal velocity of the data over the fluid faces. Perturbations
are multiplied by the ramp ``r = \\min(1, \\max(0, -U_n/u_{\\min}))``: zero
on outflow levels, full where the inflow exceeds `u_min` (default
0.5 m s⁻¹), linear in between, so a wind that veers through a face
switches its turbulence on and off continuously. The integral time is
Taylor's ``T = L_n/\\max(|U_n|, u_{\\min})`` with the normal length of
[`inflow_normal_length`](@ref).

**Time.** The perturbation is piecewise **linear in time** between
*knots*. A knot is a complete perturbation: the generator advanced to the
knot time (always, whatever the wind, so its state does not depend on
the direction), Lund-transformed, ramped with the current ``U_n``,
restricted to the receiving faces and stripped of its per-level mean.
For data with a time window (a [`NestBoundaryData`](@ref): the parent
blended linearly over the parent step ``[t_0, t_0+Δt]``) the knots are
the window ends, so the perturbation is blended exactly like the data
(a stage time past the end extrapolates, as the data does); otherwise
every refresh at a new time makes a knot. The first knot (installation)
is zero. This is what keeps the child's divergence at round-off: a
multistage core combines the step-start state with its stages, and the
open-boundary buffers with them (SSP-RK3:
``u^{n+1} = \tfrac13u^n + \tfrac23u^{(2)}``); boundary values linear in
time make the combined buffers equal to a fresh refresh at the step's
end, so the refresh at the start of the next parent step (the window
update of a [`NestedRun`](@ref)) finds exactly the faces the projection
saw. Planes held constant over a step, a ramp re-evaluated at every
stage, or perturbations switched on at the installation time each left
a divergence of ``O(10^{-3})`` s⁻¹ that the combination only damps by
1/3 per substep (measured, `test/era6_inflow_turbulence.jl`).

**Zero added mass flux.** On every face and level the area-weighted mean
of the normal perturbation over the receiving fluid faces is removed, so
the perturbation adds no mass, whatever the core's constraint weight of
the level (``ρ̄(z)``, ``ρ̄θ̄(z)``). [`added_mass_flux`](@ref) reports the
check. The removal reduces the normal variance by the variance of the
face mean, ``σ_n^2\\,n^{-2}\\sum_{ij}R(|i-j|Δ_s) ≈ 2(L_s/L_{\\rm face})σ_n^2``
for a face of length ``L_{\\rm face} ≫ L_s`` (and ``\\overline{u_n'w'}``
by the same factor); see `docs/src/theory/transition.md`.

**Corners.** The cell next to two open faces is owned by exactly one:
the x-face (west/east) at levels where that face carries turbulence
(ramp > 0, decided at each knot), the y-face otherwise. The y-face's corner column then
receives no perturbation and is excluded from its mean removal. The
along-face (staggered) component of an x-face is not applied at the
slot that is the adjacent open y-face's normal-velocity face, and the
vertical component not at the ground face: those slots belong to the
other face's condition.

**Ledger.** [`injected_ke_flux`](@ref) gives the flux of perturbation
kinetic energy through each face, ``ρ_{\\rm ref}⟨\\tfrac12|u'|^2\\max(-U_n,0)⟩``
(W m⁻², face mean). Solid faces (`fluid = false` in the normal plane,
set by [`NestBoundaryData`](@ref)'s solid mask) receive nothing.

**Statistics.** `statistics` is an [`InflowStatistics`](@ref) (one source
for every face), a function `side -> source`, a `NamedTuple` by side, or
the type `ParentResolvedStatistics` (the default, for a `NestedRun`; one
per face, `statistics_kwargs` passed to its constructor, with the
parent levels the child spans). A source with memory is folded with the
parent's elapsed time once the parent clock has advanced by
`statistics_interval` seconds (default 60 s, against the default
running-mean time of 600 s: one fold costs 65 ms per face on the
192 × 192 Berlin parent, `docs/src/theory/transition.md`). The
stresses (Lund factors) and normal lengths are re-evaluated at every
generator advance; the generators' `L_s`, `L_z` every `length_interval`
seconds ([`refresh_inflow_lengths!`](@ref)), lengths below one grid
spacing raised to one spacing (shorter scales are not representable).

`generator` is the type `DigitalFilterGenerator` (seeded `seed`, `seed+1`,
… per face) or a function `(side; ns, nz, Δs, Δz, L_s, L_z, T, seed) ->
generator` for another [`AbstractInflowGenerator`](@ref).
[`generator_state`](@ref) / [`restore_generator_state!`](@ref) snapshot
and restore everything the next perturbations depend on (bit-identical
continuation). Two-way runs: the interface flux correction leaves the
normal perturbation out of the parent's interface faces
([`synthetic_normal_velocity`](@ref Aeolus.BoundaryConditions.synthetic_normal_velocity), #418),
and the feedback can skip a frame of [`feedback_exclude_width`](@ref)
next to the open faces ([`exclude_inflow_feedback!`](@ref)).

References: Lund, T. S., Wu, X. & Squires, K. D. (1998), *J. Comput.
Phys.* 140, 233-258. Xie, Z.-T. & Castro, I. P. (2008), *Flow Turbul.
Combust.* 81, 449-470. Kadasch, E. et al. (2021), Mesoscale nesting
interface of the PALM model system 6.0, *Geosci. Model Dev.* 14,
5435-5465 (§2.4: the synthetic turbulence generator on the lateral
inflow boundaries, its mean removal and the top exclusion).
"""
mutable struct InflowTurbulence{D, S, P} <: AbstractBoundaryData
    data::D                          # the wrapped boundary data
    sim::S                           # the domain whose faces are perturbed
    parent::P                        # the statistics' parent (or nothing)
    faces::Vector{_InflowFace}
    z::Vector{Float64}
    u_min::Float64
    ρ_ref::Float64
    t_stats::Float64                 # parent clock at the last statistics fold
    length_interval::Float64
    t_lengths::Float64
    statistics_interval::Float64
end

_open_side(bcs, side) = getproperty(bcs, side) isa OpenBoundary

function _resolve_statistics(statistics, side, parent, nest, kwargs)
    if statistics === ParentResolvedStatistics
        parent === nothing && throw(ArgumentError(
            "InflowTurbulence: the default ParentResolvedStatistics needs a NestedRun; " *
            "pass `statistics` for a single domain"))
        (nest.rx > 0 && nest.ry > 0) || throw(ArgumentError(
            "InflowTurbulence: ParentResolvedStatistics needs an integer horizontal " *
            "refinement (the nest has $(refinement_ratio(nest))); pass `statistics`"))
        cz = nest_covered_range(nest)[3]
        lev = max(1, first(cz) - 1):min(nest.parent_mesh.Nz, last(cz) + 1)
        src = ParentResolvedStatistics(parent; face = side, refine = (nest.rx, nest.ry),
                                       levels = lev, kwargs...)
        update_inflow_statistics!(src, 1.0)           # a first sample (S/W is unbiased)
        return src
    elseif statistics isa InflowStatistics
        return statistics
    elseif statistics isa NamedTuple
        return statistics[side]
    elseif statistics isa Function
        return statistics(side)
    end
    throw(ArgumentError("InflowTurbulence: statistics must be an InflowStatistics, a function " *
                        "of the side, a NamedTuple by side or ParentResolvedStatistics"))
end

# the generator's lengths: raised to one grid spacing (not representable below)
function _gen_lengths(src, z, t, Δs, Δz)
    L = inflow_lengths(src, z, t)
    return max.(L.L_s, Δs), max.(L.L_z, Δz), max.(L.T, 1e-12)
end

function _make_generator(generator, side, ns, nz, Δs, Δz, L_s, L_z, T, seed)
    generator === DigitalFilterGenerator &&
        return DigitalFilterGenerator(ns, nz, Δs, Δz; L_s = L_s, L_z = L_z, T = T, seed = seed)
    return generator(side; ns = ns, nz = nz, Δs = Δs, Δz = Δz, L_s = L_s, L_z = L_z, T = T,
                     seed = seed)
end

function InflowTurbulence(run; generator = DigitalFilterGenerator,
                          statistics = ParentResolvedStatistics, faces = :lateral,
                          u_min::Real = 0.5, ρ_ref::Real = 1.2, seed::Integer = 1,
                          length_interval::Real = 3600.0, statistics_interval::Real = 60.0,
                          statistics_kwargs = (;))
    u_min > 0 || throw(ArgumentError("InflowTurbulence: u_min must be > 0"))
    if run isa NestedRun
        sim, parent, nest = run.child, run.parent, run.nest
    else
        sim, parent, nest = run, nothing, nothing
    end
    has_open_boundaries(sim) || throw(ArgumentError("InflowTurbulence: the domain has no open face"))
    sts = open_boundary_states(sim)
    inner = first(sts).data
    all(st -> st.data === inner, sts) ||
        throw(ArgumentError("InflowTurbulence: the open boundary sets carry different data"))
    inner === nothing && throw(ArgumentError("InflowTurbulence: the open faces have no boundary data"))
    inner isa InflowTurbulence && throw(ArgumentError("InflowTurbulence: already installed"))
    m = mesh(sim.u)
    sides = faces === :lateral ? filter(s -> _open_side(sim.bcs, s), [:west, :east, :south, :north]) :
            collect(Symbol, faces)
    for s in sides
        s in (:west, :east, :south, :north) || throw(ArgumentError(
            "InflowTurbulence: faces must be lateral (:west, :east, :south, :north); the top " *
            "face is excluded (PALM), got :$s"))
        _open_side(sim.bcs, s) || throw(ArgumentError("InflowTurbulence: the :$s face is not open"))
    end
    isempty(sides) && throw(ArgumentError("InflowTurbulence: no open lateral face"))
    z = [m.z0 + (k - 0.5) * m.dz for k in 1:m.Nz]
    t = Float64(sim.t)
    fs = _InflowFace[]
    for (q, side) in enumerate(sides)
        ax = side in (:west, :east) ? 1 : 2
        ns = ax == 1 ? m.Ny : m.Nx
        Δs = ax == 1 ? m.dy : m.dx
        src = _resolve_statistics(statistics, side, parent, nest, statistics_kwargs)
        Ls, Lz, Tm = _gen_lengths(src, z, t, Δs, m.dz)
        gen = _make_generator(generator, side, ns, m.Nz, Δs, m.dz, Ls, Lz, Tm, seed + q - 1)
        inflow_plane_size(gen) == (ns, m.Nz) ||
            throw(DimensionMismatch("InflowTurbulence: the generator plane is not $ns × $(m.Nz)"))
        push!(fs, _InflowFace(side, ax, side in (:east, :north) ? 1.0 : -1.0, gen, src, ns, m.Nz,
                              Float64(Δs), Float64(Δs * m.dz),
                              inflow_lund_coefficients(src, z, t), Ls, Lz,
                              inflow_normal_length(src, z, t), copy(Tm),
                              ntuple(_ -> zeros(ns, m.Nz), 3), falses(ns, m.Nz),
                              zeros(m.Nz), zeros(m.Nz), t, t,
                              ntuple(_ -> zeros(ns, m.Nz), 3), ntuple(_ -> zeros(ns, m.Nz), 3),
                              0.0, 0.0, 0.0, 0.0, 0.0))
    end
    it = InflowTurbulence(inner, sim, parent, fs, z, Float64(u_min), Float64(ρ_ref),
                          parent === nothing ? t : Float64(parent.t),
                          Float64(length_interval), t, Float64(statistics_interval))
    for st in sts
        st.data = it
        st.initialized = false
    end
    return it
end

"""
    free_stream_generator(; U∞, nmodes = 200) -> generator

The `generator` argument of [`InflowTurbulence`](@ref) for the
random-mode provider (the FSD of N-inflow-d, #266): per face, a
[`FreeStreamInflow`](@ref) of a [`FreeStreamDisturbance`](@ref) with
`nmodes` modes whose single integral length is the face mean of the
statistics' along-face streamwise length (``L_s`` of the normal
component), swept through the face at `U∞` (m s⁻¹, Taylor), periodic
along the face (its length ``n_sΔ_s``), seeded per face. The
generator whitens its planes, so the intensity of the mode set (10 %)
does not enter; the Lund amplitudes of the statistics give the
variances. Unlike [`DigitalFilterGenerator`](@ref) it is isotropic with
one length and a frozen field (its time scale is the sweep, not ``T``).
"""
free_stream_generator(; U∞::Real, nmodes::Integer = 200) =
    (side; ns, nz, Δs, Δz, L_s, L_z, T, seed) -> FreeStreamInflow(
        FreeStreamDisturbance(0.1, sum(@view L_s[1, :]) / nz, nmodes; seed = seed, U∞ = U∞,
                              period_y = ns * Δs), ns, nz, Δs, Δz)

Base.show(io::IO, it::InflowTurbulence) =
    print(io, "InflowTurbulence(", join((f.side for f in it.faces), ", "), "; u_min = ",
          it.u_min, " m/s, data = ", nameof(typeof(it.data)), ")")

"""
    remove_inflow_turbulence!(it::InflowTurbulence) -> data

Give the open faces their wrapped boundary data back (no perturbations
from the next refresh on).
"""
function remove_inflow_turbulence!(it::InflowTurbulence)
    for st in open_boundary_states(it.sim)
        st.data === it || continue
        st.data = it.data
        st.initialized = false
    end
    return it.data
end

# =========================================================================
# Lengths, statistics, generator advance
# =========================================================================

"""
    refresh_inflow_lengths!(it::InflowTurbulence, t = it.sim.t) -> it

Rebuild the digital-filter kernels of every face from the current
integral lengths of the statistics source (each [`DigitalFilterGenerator`](@ref)
is recalibrated, see [`digital_filter_kernel`](@ref)). The generators'
current planes and random-number streams carry over, so the AR(1) time
correlation continues across the change. Called every `length_interval`
seconds by the boundary refresh; a generator of another type keeps its
own scales.
"""
function refresh_inflow_lengths!(it::InflowTurbulence, t::Real = it.sim.t)
    m = mesh(it.sim.u)
    for f in it.faces
        f.gen isa DigitalFilterGenerator || continue
        Ls, Lz, _ = _gen_lengths(f.stats, it.z, t, f.Δs, m.dz)
        (Ls == f.L_s && Lz == f.L_z) && continue
        old = f.gen
        new = DigitalFilterGenerator(f.ns, f.nz, f.Δs, m.dz; L_s = Ls, L_z = Lz, T = old.T)
        for c in 1:3
            copyto!(new.ψ[c], old.ψ[c])
        end
        copy!(new.rng, old.rng)
        f.gen = new
        f.L_s = Ls; f.L_z = Lz
    end
    it.t_lengths = Float64(t)
    return it
end

"""
    fold_inflow_statistics!(it::InflowTurbulence; interval = it.statistics_interval) -> Bool

Fold the parent's time elapsed since the last fold into every distinct
statistics source of `it` (`update_inflow_statistics!`), once the parent
clock has advanced by at least `interval` seconds; returns whether it
folded. The boundary refresh calls it with the constructor's
`statistics_interval`. A nest that steps its parent on another task
while the child steps (the parent step started ahead, SC-c #403) must
not let the child's refresh read the parent mid-step: it builds the
[`InflowTurbulence`](@ref) with `statistics_interval = Inf` and calls
this, with its own interval, where the parent is at rest (the Berlin
nest: when a parent window opens, before the next parent step starts).
"""
function fold_inflow_statistics!(it::InflowTurbulence; interval::Real = it.statistics_interval)
    it.parent === nothing && return false
    tp = Float64(it.parent.t)
    tp - it.t_stats >= interval || return false
    Δ = tp - it.t_stats
    seen = IdDict{Any,Bool}()
    for f in it.faces
        haskey(seen, f.stats) && continue
        seen[f.stats] = true
        update_inflow_statistics!(f.stats, Δ)
    end
    it.t_stats = tp
    return true
end

# The face-mean outward normal velocity of the data, per level.
function _face_mean_normal!(f::_InflowFace, np::OpenPlane, m)
    fl = _face_local(np)
    Ht = f.axis == 1 ? m.Hy : m.Hx
    @inbounds for b in 1:f.nz
        s = 0.0; n = 0
        for a in 1:f.ns
            np.fluid[a, b] || continue
            v = f.axis == 1 ? np.data[fl, Ht + a, m.Hz + b] : np.data[Ht + a, fl, m.Hz + b]
            s += v; n += 1
        end
        f.U_out[b] = n == 0 ? 0.0 : f.s_out * s / n
    end
    return f
end

_ramp(U_out, u_min) = clamp(-U_out / u_min, 0.0, 1.0)

# The time window of the wrapped data, if it has one: two-snapshot data (a
# NestBoundaryData, or any WindowedBoundaryData, #329)
# blends linearly over the current window [t0, t0 + Δt].
_data_window(d) = nothing
_data_window(d::WindowedBoundaryData) = d.Δt > 0 ? (d.t0, d.Δt) : nothing

# The applied perturbation is linear in time between two knots,
# (t_prev, Lp) and (t_gen, Lc); τ is not clamped inside a window (a stage
# time beyond its end extrapolates, as the data does).
# `comps` picks the face-frame components (1 normal, 2 along, 3 vertical).
function _knot_value!(out, f::_InflowFace, s::Float64; clamp_τ::Bool, comps = 1:3)
    if f.t_gen > f.t_prev
        τ = (s - f.t_prev) / (f.t_gen - f.t_prev)
        clamp_τ && (τ = clamp(τ, 0.0, 1.0))
        for c in comps
            out[c] .= (1 - τ) .* f.Lp[c] .+ τ .* f.Lc[c]
        end
    else
        for c in comps
            copyto!(out[c], f.Lc[c])
        end
    end
    return out
end

# A new knot at t_new: the curve's value at t_from becomes the first knot;
# the second is the generator advanced from its clock to t_new, Lund
# transformed, ramped with the current face-mean U_n, restricted to the
# receiving faces and stripped of its per-level face mean. Every knot is
# a complete perturbation (zero flux per level), so is every blend of two.
function _new_knot!(it::InflowTurbulence, f::_InflowFace, np::OpenPlane, t_from::Float64,
                    t_new::Float64, owner_lo, owner_hi)
    _knot_value!(f.Lp, f, t_from; clamp_τ = false)
    f.a = inflow_lund_coefficients(f.stats, it.z, t_new)
    f.L_n = inflow_normal_length(f.stats, it.z, t_new)
    @inbounds for k in 1:f.nz, c in 1:3
        f.T[c, k] = f.L_n[c, k] / max(abs(f.U_out[k]), it.u_min)
    end
    set_integral_time!(f.gen, f.T)
    P1, P2, P3 = f.Lc
    unit_fluctuations!(P1, P2, P3, f.gen, t_new - f.t_gen)
    apply_lund!(P1, P2, P3, f.a)
    @inbounds for b in 1:f.nz
        r = _ramp(f.U_out[b], it.u_min)
        f.ramp[b] = r
        # receiving faces of this level: fluid, and not a corner owned by an x-face
        nact = 0; s = 0.0
        for a in 1:f.ns
            act = r > 0 && np.fluid[a, b]
            if act && a == 1 && owner_lo !== nothing && owner_lo.ramp[b] > 0
                act = false
            end
            if act && a == f.ns && owner_hi !== nothing && owner_hi.ramp[b] > 0
                act = false
            end
            f.active[a, b] = act
            if act
                nact += 1; s += P1[a, b]
            end
        end
        mean1 = nact == 0 ? 0.0 : s / nact
        for a in 1:f.ns
            if f.active[a, b]
                P1[a, b] = r * (P1[a, b] - mean1)
                P2[a, b] *= r; P3[a, b] *= r
            else
                P1[a, b] = 0.0; P2[a, b] = 0.0; P3[a, b] = 0.0
            end
        end
    end
    f.t_prev = t_from
    f.t_gen = t_new
    return f
end

# Form the face's perturbations at time t (a new knot first, when t — or
# the data's window — has moved past the last one) and their ledgers.
function _perturbation!(it::InflowTurbulence, f::_InflowFace, np::OpenPlane, t::Float64,
                        owner_lo::Union{Nothing,_InflowFace}, owner_hi::Union{Nothing,_InflowFace})
    tol = 1e-9 * max(1.0, abs(t))                   # a clock round-off is not a new time
    win = _data_window(it.data)
    if win === nothing
        t > f.t_gen + tol && _new_knot!(it, f, np, f.t_gen, t, owner_lo, owner_hi)
        _knot_value!(f.P, f, t; clamp_τ = true)
    else
        t0, Δt = Float64(win[1]), Float64(win[2])
        t0 + Δt > f.t_gen + tol && _new_knot!(it, f, np, t0, t0 + Δt, owner_lo, owner_hi)
        _knot_value!(f.P, f, t; clamp_τ = false)
    end
    P1, P2, P3 = f.P
    Q = 0.0; Qlev = 0.0; Qs = 0.0; ke = 0.0; area = 0.0
    @inbounds for b in 1:f.nz
        Uin = max(-f.U_out[b], 0.0)
        ql = 0.0
        for a in 1:f.ns
            np.fluid[a, b] || continue
            Qs += abs(f.U_out[b]) * f.dA
            area += f.dA
            ql += P1[a, b] * f.dA
            ke += 0.5 * (P1[a, b]^2 + P2[a, b]^2 + P3[a, b]^2) * Uin * f.dA
        end
        Q += f.s_out * ql
        Qlev = max(Qlev, abs(ql))
    end
    f.Q = Q; f.Q_level = Qlev; f.Q_scale = Qs
    f.area = area
    f.ke = area > 0 ? it.ρ_ref * ke / area : 0.0
    return f
end

# Add the perturbations to the face's buffers: every layer of the normal
# slab (the boundary face and the layers beyond it), the tangential and
# vertical halos. Grid components of the face frame: (u, v, w) on an
# x-face, (v, u, w) on a y-face.
function _add_perturbation!(planes, f::_InflowFace, m, lo_open::Bool)
    P1, P2, P3 = f.P
    nn, an = f.axis == 1 ? (:u, :v) : (:v, :u)
    Ht = f.axis == 1 ? m.Hy : m.Hx
    for p in planes
        p.side === f.side || continue
        p.mode === :mirror && continue
        P, skip_a1, kmin = p.name === nn ? (P1, false, 1) :
                           p.name === an ? (P2, lo_open, 1) :
                           p.name === :w ? (P3, false, 2) : (nothing, false, 1)
        P === nothing && continue
        d = p.data
        nl = size(d, f.axis)
        @inbounds for b in kmin:f.nz, a in (skip_a1 ? 2 : 1):f.ns
            v = P[a, b]                     # zero off the receiving faces of both knots
            for l in 1:nl
                if f.axis == 1
                    d[l, Ht + a, m.Hz + b] += v
                else
                    d[Ht + a, l, m.Hz + b] += v
                end
            end
        end
    end
    return planes
end

function _normal_plane(planes, f::_InflowFace)
    nn = f.axis == 1 ? :u : :v
    for p in planes
        p.side === f.side && p.name === nn && p.normal && return p
    end
    return nothing
end

function boundary_values!(planes::AbstractVector{<:OpenPlane}, it::InflowTurbulence, t)
    boundary_values!(planes, it.data, t)
    tt = Float64(t)
    m = mesh(it.sim.u)
    fold_inflow_statistics!(it)
    tt - it.t_lengths >= it.length_interval && refresh_inflow_lengths!(it, tt)
    nps = [_normal_plane(planes, f) for f in it.faces]
    for (f, np) in zip(it.faces, nps)
        np === nothing || _face_mean_normal!(f, np, m)
    end
    # x-faces first: their ramps decide the ownership of the corners
    byside = Dict(f.side => f for f in it.faces)
    for ax in (1, 2), (f, np) in zip(it.faces, nps)
        (f.axis == ax && np !== nothing) || continue
        if ax == 1
            # x-faces own their corners; their ramp must be current
            _perturbation!(it, f, np, tt, nothing, nothing)
        else
            _perturbation!(it, f, np, tt, get(byside, :west, nothing), get(byside, :east, nothing))
        end
        lo = f.axis == 1 ? _open_side(it.sim.bcs, :south) : _open_side(it.sim.bcs, :west)
        _add_perturbation!(planes, f, m, lo)
    end
    return planes
end

# =========================================================================
# Diagnostics, restart, two-way
# =========================================================================

"""
    added_mass_flux(it::InflowTurbulence) -> NamedTuple

The mass check of the last boundary refresh, per face (by side): `Q`, the
net outward volume flux the perturbations added (m³ s⁻¹); `Q_level`, the
largest added flux of a single level; and `relative`, ``|Q|/\\sum|U_n|ΔA``
(the face's mean-flow flux magnitude). The per-level mean removal makes
`Q` round-off; the N-open-b mass-flux correction then sees exactly the
data's imbalance.
"""
added_mass_flux(it::InflowTurbulence) =
    NamedTuple{Tuple(f.side for f in it.faces)}(Tuple(
        (Q = f.Q, Q_level = f.Q_level, relative = abs(f.Q) / max(f.Q_scale, floatmin(Float64)))
        for f in it.faces))

"""
    injected_ke_flux(it::InflowTurbulence) -> NamedTuple

The flux of perturbation kinetic energy carried into the domain by the
mean inflow at the last refresh, ``ρ_{\\rm ref}⟨\\tfrac12|u'|^2\\max(-U_n, 0)⟩``
(W m⁻²), as a face mean per side and as `total`, the area-weighted mean
over the perturbed faces. The energy-ledger term of the synthetic inflow.
"""
function injected_ke_flux(it::InflowTurbulence)
    A = sum(f.area for f in it.faces)
    tot = A > 0 ? sum(f.ke * f.area for f in it.faces) / A : 0.0
    return NamedTuple{(Tuple(f.side for f in it.faces)..., :total)}(
        (Tuple(f.ke for f in it.faces)..., tot))
end

"""
    inflow_perturbation(it::InflowTurbulence, side) -> (normal, along, vertical)

The perturbation planes (`ns × nz`, face frame) the face `side` received at
the last refresh, after the ramp, the corner rule and the mean removal.
"""
function inflow_perturbation(it::InflowTurbulence, side::Symbol)
    for f in it.faces
        f.side === side && return f.P
    end
    throw(ArgumentError("inflow_perturbation: no perturbed face :$side"))
end

"""
    synthetic_normal_velocity(it::InflowTurbulence, side, t) -> plane or nothing

The normal perturbation (`ns × nz`, grid sign: the ``u`` of a west/east
face, the ``v`` of a south/north face) the face `side` carries at time `t`,
on the knot curve the boundary refresh follows (linear in the data's
window). The contract of
[`synthetic_normal_velocity`](@ref Aeolus.BoundaryConditions.synthetic_normal_velocity):
a two-way [`NestedRun`](@ref)'s interface flux correction gives the parent
the child's flux minus this plane, so the parent's interface faces never
receive the synthetic eddies (#418). A face without perturbations
delegates to the wrapped data.
"""
function synthetic_normal_velocity(it::InflowTurbulence, side::Symbol, t::Real)
    for f in it.faces
        f.side === side || continue
        P = zeros(f.ns, f.nz)
        _knot_value!((P,), f, Float64(t); clamp_τ = _data_window(it.data) === nothing, comps = 1:1)
        inner = synthetic_normal_velocity(it.data, side, t)
        return inner === nothing ? P : P .+ inner
    end
    return synthetic_normal_velocity(it.data, side, t)
end

"""
    feedback_exclude_width(it::InflowTurbulence) -> width (m)

The default frame next to the open faces that a two-way feedback should
skip (N-nest-e, #263, `exclude_width`): twice the largest along-face
integral length of the generators, ``2\\max L_s``, so the parent is not
fed the synthetic eddies before they have adjusted to the child's
dynamics.
"""
feedback_exclude_width(it::InflowTurbulence) = 2 * maximum(maximum(f.L_s) for f in it.faces)

"""
    exclude_inflow_feedback!(run::NestedRun, it::InflowTurbulence) -> exclude_width

Give the two-way feedback of `run` the inflow band of `it` (N-inflow-e,
#268): its [`TwoWayFeedback`](@ref) is replaced by a copy whose
`exclude_width` is [`feedback_exclude_width`](@ref)`(it)` in parent
cells, rounded up (``⌈2\\max L_s/Δx_p⌉``, `Δx_p` the smaller horizontal
parent spacing), and never narrower than the one it had. The faces it
excludes stay the feedback's `exclude_sides` (`:inflow` by default: the
faces whose mean normal velocity points into the child at each
feedback). A one-way run is returned unchanged (0).

The band was introduced against a feedback loop measured on the refine-3
test nest (child kinetic energy ×5.5 and parent ×3.2 in 200 s). The path
of that loop was the interface flux correction, which since #418 leaves
the synthetic part out
([`synthetic_normal_velocity`](@ref Aeolus.BoundaryConditions.synthetic_normal_velocity)).
With that, the band no longer measurably matters on the test nests: on
the 100 s nest the child's kinetic energy is ×0.988 of one-way without
it and ×0.995 with it (`docs/src/theory/transition.md`).
"""
function exclude_inflow_feedback!(run::NestedRun, it::InflowTurbulence)
    fb = run.twoway
    fb === nothing && return 0
    pm = run.nest.parent_mesh
    w = max(fb.exclude_width, ceil(Int, feedback_exclude_width(it) / min(pm.dx, pm.dy) - 1e-9))
    run.twoway = TwoWayFeedback(fb.buffer, fb.z_anterp, fb.τ, w, fb.exclude_sides, fb.reflux,
                                fb.reproject, fb.tracer_fluxes, fb.theta_flux)
    return w
end

# a snapshot of the numeric fields of a mutable statistics source (running sums)
function _stats_state(src)
    ismutable(src) || return nothing
    return Dict{Symbol,Any}(n => (v = getfield(src, n); v isa Array{<:Number} ? copy(v) : v)
                            for n in fieldnames(typeof(src))
                            if (v = getfield(src, n); v isa Array{<:Number} || v isa Number))
end
function _restore_stats!(src, s)
    s === nothing && return src
    for (n, v) in s
        v isa Array ? copyto!(getfield(src, n), v) : setfield!(src, n, v)
    end
    return src
end

function generator_state(it::InflowTurbulence)
    seen = IdDict{Any,Any}()
    for f in it.faces
        haskey(seen, f.stats) || (seen[f.stats] = _stats_state(f.stats))
    end
    return (gens = [deepcopy(generator_state(f.gen)) for f in it.faces],
            t_gen = [f.t_gen for f in it.faces], t_prev = [f.t_prev for f in it.faces],
            Lpc = [(deepcopy(f.Lp), deepcopy(f.Lc)) for f in it.faces], gen_objs = [f.gen for f in it.faces],
            L = [(copy(f.L_s), copy(f.L_z), copy(f.T)) for f in it.faces],
            a = [deepcopy(f.a) for f in it.faces],
            stats = seen, t_stats = it.t_stats, t_lengths = it.t_lengths)
end
function restore_generator_state!(it::InflowTurbulence, s)
    for (q, f) in enumerate(it.faces)
        f.gen = s.gen_objs[q]
        restore_generator_state!(f.gen, s.gens[q])
        f.t_gen = s.t_gen[q]; f.t_prev = s.t_prev[q]
        for c in 1:3
            copyto!(f.Lp[c], s.Lpc[q][1][c]); copyto!(f.Lc[c], s.Lpc[q][2][c])
        end
        f.L_s, f.L_z, Tm = copy.(s.L[q])
        copyto!(f.T, Tm)
        f.a = deepcopy(s.a[q])
    end
    for (src, st) in s.stats
        _restore_stats!(src, st)
    end
    it.t_stats = s.t_stats
    it.t_lengths = s.t_lengths
    return it
end
