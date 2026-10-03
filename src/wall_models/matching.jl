# Log-layer mismatch (Phase 57, P57-b, #183): the input of an equilibrium
# wall model. The velocity the law is evaluated with is sampled at a
# matching height h_wm off the first cell (Kawai & Larsson 2012), and
# filtered in time (Yang, Park & Moin 2017) and/or along the wall
# (Bou-Zeid, Meneveau & Parlange 2005) before the law sees it. The stress
# is still applied at the wall face by `apply_wall_stress!` (its `wind`
# and `height` keywords); this file only decides what speed and height
# go in.

export MatchingInput, matching_distance, matching_input!, update_input_filter!,
       exponential_filter_weight, wall_parallel_filter!

"""
    MatchingInput(; h_wm = nothing, h_wm_cells = 1, T_f = 0, spatial_filter = false)

The input of a wall model at one wall: where the velocity is sampled and
how it is filtered before the wall law is evaluated with it. Pass it to
[`matching_input!`](@ref), which returns the `wind` and `height`
keywords of [`apply_wall_stress!`](@ref).

**Matching height** (Kawai & Larsson 2012). The LES velocity in the first
one or two cells next to a wall is under-resolved and dominated by the
numerical errors of the SGS model there, and an equilibrium law fed with
it gives a wall stress that is wrong by the same amount: the *log-layer
mismatch*, a shift of the mean profile against the log law. Kawai &
Larsson sample the velocity a few cells away, at ``h_{wm}``, where the
LES resolves the turbulence, and apply the stress of the law at
``h_{wm}`` at the wall face.

  * `h_wm` (m) — the matching distance from the wall; the velocity is
    linearly interpolated between the two cell centres around it (so
    ``h_{wm} ≥ Δz/2``);
  * `h_wm_cells` — otherwise the centre of the `h_wm_cells`-th cell from
    the wall, ``h = (n - ½)Δz``; an integer, or an `(Nx, Ny)` integer
    matrix (one per column; counted from the column's wall level with a
    level mask). `h_wm_cells = 1` with no filter is the classical
    first-cell matching.

**Temporal filter** (Yang, Park & Moin 2017). With `T_f > 0` (s) the
cell-centre velocity components ``(u, v)`` at the matching point are
passed through the exponential (single-pole) filter

```math
\\tilde u^{n+1} = \\tilde u^n + ε\\,(u^{n+1} - \\tilde u^n), \\qquad
ε = 1 - e^{-Δt/T_f},
```

once per accepted step ([`update_input_filter!`](@ref)), and the law is
evaluated at ``|\\tilde{\\mathbf u}|``. The weight is the exact one of
the continuous filter ``T_f\\,\\dot{\\tilde u} = u - \\tilde u`` for an
input held over the step, so a unit step is followed by
``1 - e^{-t/T_f}`` at every step whatever the Δt
([`exponential_filter_weight`](@ref)). The filter removes the
instantaneous fluctuation that the equilibrium law (a relation between
*mean* quantities) does not hold for; Yang et al. show that the input
filtering alone removes most of the log-layer mismatch.

**Wall-parallel filter** (Bou-Zeid, Meneveau & Parlange 2005). With
`spatial_filter = true` the horizontal velocity is first filtered along
the wall with the 3 × 3 box of `Operators.box_filter!` (`n = (3, 3, 1)`;
the test filter of the dynamic model, so the two cannot drift), the
discrete form of the test filter of twice the grid width that Bou-Zeid
et al. apply to the wall-model input. Its second moment is
``2Δ^2/3`` per direction; it preserves the plane mean of a periodic
field.

Filtered state: `uf`, `vf` (`(Nx, Ny)`, the filtered components at the
matching point) and `primed` (false until the first sample).

# References

  * Kawai, S. & Larsson, J. (2012). Wall-modeling in large eddy
    simulation: length scales, grid resolution, and accuracy.
    *Phys. Fluids*, 24, 015105.
  * Yang, X. I. A., Park, G. I. & Moin, P. (2017). Log-layer mismatch
    and modeling of the fluctuating wall stress in wall-modeled
    large-eddy simulations. *Phys. Rev. Fluids*, 2, 104601.
  * Bou-Zeid, E., Meneveau, C. & Parlange, M. B. (2005). A
    scale-dependent Lagrangian dynamic model for large eddy simulation of
    complex turbulent flows. *Phys. Fluids*, 17, 025105.
"""
mutable struct MatchingInput{T<:AbstractFloat, C}
    h_wm::T                 # m; NaN when `cells` is used
    cells::C                # Int or (Nx, Ny) Int matrix
    T_f::T                  # s; 0 = no temporal filter
    spatial::Bool
    uf::Matrix{T}
    vf::Matrix{T}
    primed::Bool
    work::Any               # (uF, vF) Fields of the wall-parallel filter, allocated on first use
end

function MatchingInput(; h_wm = nothing, h_wm_cells = 1, T_f::Real = 0.0,
                       spatial_filter::Bool = false)
    T_f >= 0 || throw(ArgumentError("MatchingInput: T_f must be ≥ 0, got $T_f"))
    if h_wm !== nothing
        h_wm > 0 || throw(ArgumentError("MatchingInput: h_wm must be > 0, got $h_wm"))
        h_wm_cells == 1 || throw(ArgumentError("MatchingInput: give h_wm or h_wm_cells, not both"))
    end
    h_wm_cells isa Integer || h_wm_cells isa AbstractMatrix{<:Integer} ||
        throw(ArgumentError("MatchingInput: h_wm_cells must be an integer or an integer matrix"))
    all(>=(1), h_wm_cells) || throw(ArgumentError("MatchingInput: h_wm_cells must be ≥ 1"))
    cells = h_wm_cells isa Integer ? Int(h_wm_cells) : Matrix{Int}(h_wm_cells)
    return MatchingInput{Float64, typeof(cells)}(h_wm === nothing ? NaN : float(h_wm), cells,
                                                 float(T_f), spatial_filter,
                                                 zeros(0, 0), zeros(0, 0), false, nothing)
end

@inline _ncells(c::Int, i, j) = c
@inline _ncells(c::AbstractMatrix, i, j) = @inbounds c[i, j]

"""
    matching_distance(mi::MatchingInput, dz, i = 1, j = 1) -> h

The distance from the wall (m) at which `mi` samples column `(i, j)`:
`h_wm`, or ``(n - ½)Δz`` for `h_wm_cells = n`.
"""
matching_distance(mi::MatchingInput, dz::Real, i::Integer = 1, j::Integer = 1) =
    isnan(mi.h_wm) ? (_ncells(mi.cells, i, j) - 0.5) * dz : mi.h_wm

"""
    exponential_filter_weight(Δt, T_f) -> ε

The weight ``ε = 1 - e^{-Δt/T_f}`` of the exponential filter
``\\tilde u ← \\tilde u + ε(u - \\tilde u)`` of [`MatchingInput`](@ref):
the exact solution of ``T_f\\,\\dot{\\tilde u} = u - \\tilde u`` over Δt
for a held input (Yang, Park & Moin 2017). `T_f = 0` gives 1 (no
filter). Computed with `expm1`, so it is accurate for ``Δt ≪ T_f``.
"""
exponential_filter_weight(Δt::Real, T_f::Real) = T_f > 0 ? -expm1(-Δt / T_f) : one(float(Δt))

"""
    wall_parallel_filter!(uF, vF, sim) -> (uF, vF)

The wall-parallel input filter of [`MatchingInput`](@ref): `uF`, `vF`
receive the 3 × 3 box average in x and y (`Operators.box_filter!` with
`n = (3, 3, 1)`, nothing in z) of `sim.u` and `sim.v`, halos filled with
the simulation's boundary conditions. The plane mean of a horizontally
periodic field is preserved exactly (to round-off).
"""
function wall_parallel_filter!(uF, vF, sim)
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    box_filter!(uF, sim.u, (3, 3, 1))
    box_filter!(vF, sim.v, (3, 3, 1))
    fill_halos!(uF, sim.bcs); fill_halos!(vF, sim.bcs)
    return uF, vF
end

# The wall level (1-based) of column (i, j) on `side`, or 0 for no wall.
@inline _wall_level(side, mask, i, j, Nz) =
    side === :bottom ? _level(mask, i, j, 1, Nz) : _level(mask, i, j, Nz, Nz)

# Cell-centre (u, v) of every walled column at the matching point, the
# matching distance. Instantaneous, or
# from the wall-parallel filtered field.
function _sample_matching(mi::MatchingInput, sim, side::Symbol, mask)
    side in (:bottom, :top) ||
        throw(ArgumentError("matching input: side must be :bottom or :top, got :$side"))
    g = sim.mesh
    Nx, Ny, Nz, Hx, Hy, Hz, dz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz, g.dz
    T = typeof(float(dz))
    if mi.spatial
        mask === nothing ||
            throw(ArgumentError("matching input: the wall-parallel filter is for a plane wall (mask = nothing)"))
        if mi.work === nothing
            mi.work = (Field(FaceX(), g), Field(FaceY(), g))
        end
        uF, vF = wall_parallel_filter!(mi.work[1], mi.work[2], sim)
        u, v = uF.data, vF.data
    else
        # Read-only: a periodic neighbour is taken from the interior, so the
        # simulation's halos are not touched (a non-periodic side reads
        # its halo, which is filled here).
        (sim.bcs.west isa Periodic && sim.bcs.south isa Periodic) ||
            (fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs))
        u, v = sim.u.data, sim.v.data
    end
    xper = sim.bcs.west isa Periodic; yper = sim.bcs.south isa Periodic
    uc = zeros(T, Nx, Ny); vc = zeros(T, Nx, Ny); hh = zeros(T, Nx, Ny)
    _sample_loop!(uc, vc, hh, u, v, mi, g, side, mask, xper, yper)
    return uc, vc, hh
end

# function barrier: `u`, `v` have a concrete type here
function _sample_loop!(uc, vc, hh, u::AbstractArray{T,3}, v::AbstractArray{T,3}, mi, g, side, mask,
                       xper, yper) where {T}
    Nx, Ny, Nz, Hx, Hy, Hz, dz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz, g.dz
    s = side === :bottom ? 1 : -1
    # the cell-centre (u, v) of storage cell (I, J, K)
    @inline function cen(K, I, J)
        Ie = (xper && I == Hx + Nx) ? Hx + 1 : I + 1
        Jn = (yper && J == Hy + Ny) ? Hy + 1 : J + 1
        return (@inbounds((u[I, J, K] + u[Ie, J, K]) / 2), @inbounds((v[I, J, K] + v[I, Jn, K]) / 2))
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        kw = _wall_level(side, mask, i, j, Nz)
        kw == 0 && continue
        h = T(matching_distance(mi, dz, i, j))
        x = h / dz + T(0.5)                 # 1 = the first cell centre
        x >= 1 - 1e-12 ||
            throw(ArgumentError("matching input: h_wm = $h is below the first cell centre Δz/2 = $(dz/2)"))
        n = max(1, floor(Int, x + 1e-12)); f = max(zero(T), x - n)
        f < 1e-12 && (f = zero(T))
        k1 = kw + s * (n - 1); k2 = k1 + s
        (1 <= k1 <= Nz && (f == 0 || 1 <= k2 <= Nz)) ||
            throw(ArgumentError("matching input: h_wm = $h from the wall at level $kw of column ($i, $j) leaves the domain"))
        I, J = Hx + i, Hy + j
        a1, b1 = cen(Hz + k1, I, J)
        if f == 0
            uc[i, j], vc[i, j] = a1, b1
        else
            a2, b2 = cen(Hz + k2, I, J)
            uc[i, j] = (1 - f) * a1 + f * a2; vc[i, j] = (1 - f) * b1 + f * b2
        end
        hh[i, j] = h
    end
    return nothing
end

function _prime!(mi::MatchingInput, uc, vc)
    mi.uf = copy(uc); mi.vf = copy(vc); mi.primed = true
    return mi
end

"""
    update_input_filter!(mi::MatchingInput, sim, Δt; side = :bottom, mask = nothing) -> mi

Advance the temporal filter of `mi` over an accepted step of length `Δt`
with the velocity at the end of the step (call it once per step, after
the projection). The first call (or any call while `mi` is not yet
`primed`) sets the filtered state to the current sample. A no-op without
a temporal filter (`T_f = 0`).
"""
function update_input_filter!(mi::MatchingInput, sim, Δt::Real; side::Symbol = :bottom,
                              mask = nothing)
    mi.T_f > 0 || return mi
    uc, vc, _ = _sample_matching(mi, sim, side, mask)
    mi.primed || return _prime!(mi, uc, vc)
    ε = exponential_filter_weight(Δt, mi.T_f)
    @. mi.uf += ε * (uc - mi.uf)
    @. mi.vf += ε * (vc - mi.vf)
    return mi
end

"""
    matching_input!(mi::MatchingInput, sim; side = :bottom, mask = nothing) -> (wind, height)

The `wind` and `height` keywords of [`apply_wall_stress!`](@ref) for the
input `mi` at the wall `side` (`:bottom` or `:top`; `mask` as in
`apply_wall_stress!`): per column, the speed of the (filtered) velocity
at the matching point and the matching distance. With a temporal filter
the speed is that of the filtered state, which changes only in
[`update_input_filter!`](@ref) (so every substage of a Runge-Kutta step
sees the same input); a filter that is not primed yet is primed with the
current sample.
"""
function matching_input!(mi::MatchingInput, sim; side::Symbol = :bottom, mask = nothing)
    if mi.T_f > 0
        if mi.primed
            _, _, hh = _sample_heights(mi, sim, side, mask)
            return (hypot.(mi.uf, mi.vf), hh)
        end
        uc, vc, hh = _sample_matching(mi, sim, side, mask)
        _prime!(mi, uc, vc)
        return (hypot.(uc, vc), hh)
    end
    uc, vc, hh = _sample_matching(mi, sim, side, mask)
    return (hypot.(uc, vc), hh)
end

# The matching distances alone (no sampling, no filtering).
function _sample_heights(mi::MatchingInput, sim, side, mask)
    g = sim.mesh
    hh = zeros(typeof(float(g.dz)), g.Nx, g.Ny)
    @inbounds for j in 1:g.Ny, i in 1:g.Nx
        _wall_level(side, mask, i, j, g.Nz) == 0 && continue
        hh[i, j] = matching_distance(mi, g.dz, i, j)
    end
    return nothing, nothing, hh
end
