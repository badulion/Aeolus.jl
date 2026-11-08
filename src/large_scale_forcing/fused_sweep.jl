# The large-scale forcings inside the fused sweep of the step core (#533
# plan step 3). Included inside `module LargeScaleForcing`, after the
# forcings.
#
# Every forcing of this module is, at each level, an affine function of
# the forced field's own column:
#
#   ∂φ/∂t(i, j, k) = c0[k] + cm[k] (φ[k−1] − φ[k]) + cp[k] (φ[k+1] − φ[k])
#                          + λ[k] (r[k] − φ[k]),
#
# with per-level coefficients that depend on the stage-start state only
# through its plane means (nudging, mean-gradient subsidence, a damping
# toward the slab mean). The geostrophic pressure gradient and the
# advective tendencies are `c0`; the nudging and the mean-gradient
# subsidence `c0` from the plane means; the local (upwind) subsidence
# `cm`, `cp`; a Rayleigh damping toward a reference `λ`, `r`.
#
# `column_forcing!` computes the coefficients before the sweep (in a
# sweep term's `prepare!`), from the stage-start state exactly as
# `apply_forcing!` reads it, and keeps the same records
# (`tendency_integrals`, `damping_tendency`) from the plane sums;
# `add_column_rows!` adds the tendency of one row inside the sweep. The
# whole-field `apply_forcing!` stays the reference the tests compare it
# against. The one place the forcings' coefficients are computed for the
# sweep: the large-scale forcing component (`Components.LargeScale`) and
# the urban site's open-face θ relaxation use it.

using ..Fields: Field
using ..Operators.FiniteVolume: in_damping, damping_rate, _check_damping_target
import ..IO: checkpoint_opaque

"""
    ColumnForcing()

The per-level coefficients of the large-scale forcings of one substage
for the fused sweep (#533): for each field `f` (u, v, w, θ, qv, the
sweep's `SWEEP_U` … `SWEEP_QV`) and interior level `k`,

```math
\\left.\\frac{\\partial φ}{\\partial t}\\right|_{ls} = c_0 + c_m (φ_{k-1} - φ_k)
+ c_p (φ_{k+1} - φ_k) + λ (r - φ_k),
```

filled by [`column_forcing!`](@ref) before the sweep and added row by
row by [`add_column_rows!`](@ref) inside it. Scratch (resized to the
mesh on first use, not checkpoint state).
"""
mutable struct ColumnForcing
    coef::Array{Float64,3}          # (5, Nz, 5): c0, cm, cp, λ, r of level k, field f
    kind::Matrix{Int8}              # (Nz, 5): 0 none, 1 c0 only, 2 c0 + damping, 3 the full stencil
    sums::Matrix{Float64}           # (Nz, 5): plane sums of the stage-start state
    have::Vector{Bool}              # (5): sums[:, f] computed in this substage
    z::Vector{Vector{Float64}}      # (5): the damping coordinates of field f's location (cached)
    Nz::Int
    Hz::Int
    active::Bool
end
ColumnForcing() = ColumnForcing(zeros(5, 0, 5), zeros(Int8, 0, 5), zeros(0, 5), fill(false, 5),
                                [Float64[] for _ in 1:5], 0, 0, false)
checkpoint_opaque(::ColumnForcing) = true     # scratch, recomputed every substage

"""
    reset_column_forcing!(col::ColumnForcing, sim) -> col

Zero the coefficients of `col` for a new substage of `sim` (sized to its
mesh) and forget the plane sums.
"""
function reset_column_forcing!(col::ColumnForcing, sim)
    m = sim.mesh
    Nz = m.Nz
    if col.Nz != Nz
        col.coef = zeros(5, Nz, 5); col.kind = zeros(Int8, Nz, 5); col.sums = zeros(Nz, 5)
        col.z = [Float64[] for _ in 1:5]
        col.Nz = Nz
    else
        fill!(col.coef, 0.0); fill!(col.kind, Int8(0))
    end
    col.Hz = m.Hz
    fill!(col.have, false)
    col.active = false
    return col
end

"""
    finish_column_forcing!(col::ColumnForcing) -> col

After the last [`column_forcing!`](@ref) of a substage: which levels of
which fields the sweep has to touch, and with which form.
"""
function finish_column_forcing!(col::ColumnForcing)
    C = col.coef
    active = false
    @inbounds for f in 1:5, k in 1:col.Nz
        kd = (C[2, k, f] != 0 || C[3, k, f] != 0) ? 3 : C[4, k, f] != 0 ? 2 : C[1, k, f] != 0 ? 1 : 0
        col.kind[k, f] = kd
        active |= kd > 0
    end
    col.active = active
    return col
end

# --- the fields ------------------------------------------------------------------

const _SWEEP_SLOT = (u = 1, v = 2, w = 3, θ = 4, qv = 5)

# the sweep slot of forcing target `name` (the errors of `_state_tendency`)
function _column_slot(sim, name::Symbol, who)
    name === :qv && !_has_moisture(sim) &&
        throw(ArgumentError("$who: :qv requested but the simulation carries no moisture"))
    haskey(_SWEEP_SLOT, name) || throw(ArgumentError("$who: unknown field :$name"))
    return _SWEEP_SLOT[name]
end
_column_field(sim, f::Int) = f == 1 ? sim.u : f == 2 ? sim.v : f == 3 ? sim.w : f == 4 ? sim.θ : sim.qv

# the plane sums of field f of the stage-start state, once per substage
# (the running sum of each level in the order of `_plane_mean_level!`, so
# sum / (Nx Ny) is that kernel's mean, bit for bit)
function _column_sums!(col::ColumnForcing, sim, f::Int)
    col.have[f] && return col.sums
    m = sim.mesh
    d = _column_field(sim, f).data
    launch_weighted!(backend(d), m.Nx * m.Ny, _plane_sum_level!, (1:m.Nz,), col.sums, f, d, m)
    col.have[f] = true
    return col.sums
end
@inline function _plane_sum_level!(k, sums, f, d, m)
    Nx, Ny, Hx, Hy, Hz = m.Nx, m.Ny, m.Hx, m.Hy, m.Hz
    s = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        s += d[Hx+i, Hy+j, Hz+k]
    end
    @inbounds sums[k, f] = s
    return nothing
end
@inline _column_mean(col, m, k, f) = @inbounds col.sums[k, f] / (m.Nx * m.Ny)

# --- which forcings have a column form --------------------------------------------

"""
    column_forcing_supported(F) -> Bool

Whether forcing `F` (an [`AbstractLargeScaleForcing`](@ref), a `Tuple` of
them, or `nothing`) has a column form for the fused sweep
([`column_forcing!`](@ref)); decided by the types, so a sweep term's
choice infers. Every forcing of this module has one, except a
[`ReferenceDamping`](@ref) toward a 3-D target (a `Field` or a
halo-padded array), which is not a level profile. A forcing type of
another module opts in by adding methods of this and of
`column_forcing!`.
"""
column_forcing_supported(::Any) = false
column_forcing_supported(::Nothing) = true
column_forcing_supported(::Tuple{}) = true
column_forcing_supported(F::Tuple) = column_forcing_supported(first(F)) && column_forcing_supported(Base.tail(F))
column_forcing_supported(::Union{GeostrophicPressureGradient,ProfileNudging,Subsidence,AdvectiveTendencies}) = true
column_forcing_supported(::ReferenceDamping{P,X}) where {P,X} = !(X <: Union{Field,AbstractArray{<:Real,3}})

# --- the coefficients of each forcing ---------------------------------------------

"""
    column_forcing!(col::ColumnForcing, sim, F, t, Δt; f = sim.f_coriolis) -> col

Add the column coefficients of forcing `F` (or a `Tuple` of forcings, in
order) over the step `[t, t + Δt]` to `col`, from the state of `sim` as
[`apply_forcing!`](@ref) reads it, and keep `F`'s records
([`tendency_integrals`](@ref), [`damping_tendency`](@ref)) as
`apply_forcing!` does; nothing is written to the tendency fields. Within
round-off, the tendency `add_column_rows!` then adds is `apply_forcing!`'s
(the contributions of several forcings are summed per level before they
are added; the records are computed from the plane sums of the state
rather than point by point).
"""
column_forcing!(col::ColumnForcing, sim, ::Nothing, t, Δt; kw...) = col
column_forcing!(col::ColumnForcing, sim, ::Tuple{}, t, Δt; kw...) = col
function column_forcing!(col::ColumnForcing, sim, F::Tuple, t, Δt; kw...)
    column_forcing!(col, sim, first(F), t, Δt; kw...)
    return column_forcing!(col, sim, Base.tail(F), t, Δt; kw...)
end

function column_forcing!(col::ColumnForcing, sim, F::GeostrophicPressureGradient, t, Δt;
                         f = sim.f_coriolis, kw...)
    m = sim.mesh
    T = eltype(sim.u.data)
    t1 = t + Δt
    C = col.coef
    @inbounds for k in 1:m.Nz
        z = _zc(m, k)
        C[1, k, 1] += T(-(f * forcing_mean(F.V_g, z, t, t1)))
        C[1, k, 2] += T(f * forcing_mean(F.U_g, z, t, t1))
    end
    return col
end

function column_forcing!(col::ColumnForcing, sim, F::AdvectiveTendencies, t, Δt; kw...)
    m = sim.mesh
    F.q !== nothing && !_has_moisture(sim) && throw(ArgumentError(
        "AdvectiveTendencies: a humidity tendency needs a field to go to (sim has no moisture; pass `tq`)"))
    T = eltype(sim.u.data)
    _column_level_tendency!(col, m, T, F.θ, 4, t, t + Δt)
    _column_level_tendency!(col, m, T, F.q, 5, t, t + Δt)
    _column_level_tendency!(col, m, T, F.u, 1, t, t + Δt)
    _column_level_tendency!(col, m, T, F.v, 2, t, t + Δt)
    return col
end
_column_level_tendency!(col, m, ::Type, ::Nothing, f, t, t1) = col
function _column_level_tendency!(col, m, ::Type{T}, x::X, f, t, t1) where {T,X}
    @inbounds for k in 1:m.Nz
        col.coef[1, k, f] += T(forcing_mean(x, _zc(m, k), t, t1))
    end
    return col
end

function column_forcing!(col::ColumnForcing, sim, F::ProfileNudging, t, Δt; kw...)
    m = sim.mesh
    invτ = 1 / F.τ
    wfn = _weight_fn(F.weight, sim)
    t1 = t + Δt
    names = keys(F.targets)
    dA = m.dx * m.dy * (m.Nx * m.Ny)
    resize!(F.buf, m.Nz)
    vals = ntuple(Val(length(names))) do n
        fs = _column_slot(sim, names[n], "ProfileNudging")
        _column_sums!(col, sim, fs)
        _column_nudge!(col, m, F.targets[n], fs, invτ, wfn, t, t1, F.buf, dA)
    end
    F.last[] = NamedTuple{names}(vals)
    return col
end

function column_forcing!(col::ColumnForcing, sim, F::Subsidence, t, Δt; kw...)
    m = sim.mesh
    t1 = t + Δt
    names = F.explicit ? F.fields : Tuple(f for f in F.fields if f !== :qv || _has_moisture(sim))
    vals = ntuple(length(names)) do n
        fs = _column_slot(sim, names[n], "Subsidence")
        (F.form === :mean_gradient ? _column_subside_mean!(col, sim, m, F.w_s, fs, t, t1) :
                                     _column_subside_local!(col, sim, m, F.w_s, fs, t, t1))::Float64
    end
    F.last[] = NamedTuple{names}(vals)
    return col
end

# −w(z)(⟨φ⟩ − φ_ref)/τ per level (the arithmetic of `_nudge_level!`); ∫ dV
function _column_nudge!(col, m, ref::R, fs, invτ, wfn::W, t, t1, buf, dA) where {R,W}
    I = 0.0
    @inbounds for k in 1:m.Nz
        z = _zc(m, k)
        c = invτ * (_column_mean(col, m, k, fs) - forcing_mean(ref, z, t, t1))
        wfn === nothing || (c = wfn(z) * c)
        col.coef[1, k, fs] -= c
        buf[k] = c
        I -= c * dA * _dzc(m, k)
    end
    return I
end

# −w_s ∂⟨φ⟩/∂z, uniform per level (the arithmetic of `_subside_mean_level!`)
function _column_subside_mean!(col, sim, m, w_s::W, fs, t, t1) where {W}   # (W: specialise on a function)
    Nz = m.Nz
    Nz >= 2 || return 0.0
    _column_sums!(col, sim, fs)
    dA = m.dx * m.dy * (m.Nx * m.Ny)
    I = 0.0
    @inbounds for k in 1:Nz
        kl, ku = _grad_levels(k, Nz)
        G = (_column_mean(col, m, ku, fs) - _column_mean(col, m, kl, fs)) / (_zc(m, ku) - _zc(m, kl))
        s = -forcing_mean(w_s, _zc(m, k), t, t1) * G
        col.coef[1, k, fs] += s
        I += s * dA * _dzc(m, k)
    end
    return I
end

# −w_s ∂φ/∂z of each column, first-order upwind (the stencil of
# `_subside_local_level!`): the one-sided difference toward the upwind
# level, as cm (φ[k−1] − φ[k]) or cp (φ[k+1] − φ[k]); its integral from
# the plane sums
function _column_subside_local!(col, sim, m, w_s::W, fs, t, t1) where {W}
    Nz = m.Nz
    Nz >= 2 || return 0.0
    S = _column_sums!(col, sim, fs)
    dAc = m.dx * m.dy
    I = 0.0
    @inbounds for k in 1:Nz
        w = forcing_mean(w_s, _zc(m, k), t, t1)
        kl, ku = w < 0 ? (k == Nz ? (Nz - 1, Nz) : (k, k + 1)) :
                         (k == 1 ? (1, 2) : (k - 1, k))
        idz = 1 / (_zc(m, ku) - _zc(m, kl))
        if ku == k + 1                      # −w (φ[k+1] − φ[k]) / Δz
            cp = -w * idz
            col.coef[3, k, fs] += cp
            Ik = cp * (S[k+1, fs] - S[k, fs])
        else                                # −w (φ[k] − φ[k−1]) / Δz
            cm = w * idz
            col.coef[2, k, fs] += cm
            Ik = cm * (S[k-1, fs] - S[k, fs])
        end
        I += Ik * dAc * _dzc(m, k)
    end
    return I
end

# −λ(z) (φ − φ_ref): λ and the target per level (several dampings of one
# field combine into one rate and its weighted target); the levels and the
# integral of `damping_tendency` from the plane sums
function column_forcing!(col::ColumnForcing, sim, F::ReferenceDamping, t, Δt; kw...)
    fs = _column_slot(sim, F.field, "ReferenceDamping")
    m = sim.mesh
    Nz = m.Nz
    length(F.buf) == Nz || resize!(F.buf, Nz)
    length(F.levels) == Nz || resize!(F.levels, Nz)
    fill!(F.levels, 0.0)
    z = col.z[fs]
    if length(z) != Nz
        zc = damping_coordinates(m, location(_column_field(sim, fs)))
        zc isa AbstractVector ||
            throw(ArgumentError("ReferenceDamping in the fused sweep: the grid's heights are not a level " *
                                "profile ($(typeof(m)))"))
        z = col.z[fs] = Vector{Float64}(zc)
    end
    _check_damping_target(F.target, m)
    S = _column_sums!(col, sim, fs)        # the levels' record (and a :mean target)
    F.integral[] = _column_damping!(col, F, F.profile, F.target, z, S, fs, m, t)
    return col
end
function _column_damping!(col, F, profile::P, target::X, z, S, fs, m, t) where {P,X}
    N = m.Nx * m.Ny
    C = col.coef
    s = 0.0
    @inbounds for k in 1:m.Nz
        r = _column_target(F, target, col, m, z, k, fs, t)
        in_damping(profile, z[k]) || continue
        λ = damping_rate(profile, z[k])
        λ0 = C[4, k, fs]
        if λ0 == 0
            C[4, k, fs] = λ; C[5, k, fs] = r
        else
            C[5, k, fs] = (λ0 * C[5, k, fs] + λ * r) / (λ0 + λ); C[4, k, fs] = λ0 + λ
        end
        F.levels[k] = -λ * (S[k, fs] - N * r)
        s += F.levels[k] * _level_dz(m, k)
    end
    return s * m.dx * m.dy
end
# the level target, as `_resolve_target` and `_damping_target` give it
_column_target(F, target::Real, col, m, z, k, fs, t) = Float64(target)
_column_target(F, target::AbstractVector, col, m, z, k, fs, t) = @inbounds Float64(target[k])
_column_target(F, target::Union{ForcingTable,Function}, col, m, z, k, fs, t) =
    @inbounds (F.buf[k] = forcing_value(target, z[k], t))
_column_target(F, ::Symbol, col, m, z, k, fs, t) = @inbounds (F.buf[k] = _column_mean(col, m, k, fs))

# --- inside the sweep -------------------------------------------------------------------

"""
    add_column_rows!(col::ColumnForcing, buf, a, b, s, j, k, ii, st)

Add the column forcing `col` of every field on row `(ii, j, k)` into the
fused sweep's tendency buffer `buf[:, a, b, :, s]` (the `add_row!` of a
sweep term that holds a [`ColumnForcing`](@ref)): per level a constant,
a damping toward a level target, or the full column stencil, each a
vectorised loop; levels and fields without forcing are skipped.
"""
@inline function add_column_rows!(col::ColumnForcing, buf, a, b, s, j, k, ii, st)
    col.active || return nothing
    kl = k - col.Hz
    (1 <= kl <= col.Nz) || return nothing
    _column_field_row!(col, buf, a, b, s, j, k, kl, ii, st.u, 1)
    _column_field_row!(col, buf, a, b, s, j, k, kl, ii, st.v, 2)
    _column_field_row!(col, buf, a, b, s, j, k, kl, ii, st.w, 3)
    _column_field_row!(col, buf, a, b, s, j, k, kl, ii, st.θ, 4)
    _column_field_row!(col, buf, a, b, s, j, k, kl, ii, st.qv, 5)
    return nothing
end
@inline _column_field_row!(col, buf, a, b, s, j, k, kl, ii, ::Nothing, f) = nothing
@inline function _column_field_row!(col, buf, a, b, s, j, k, kl, ii, φ::AbstractArray, f)
    kd = @inbounds col.kind[kl, f]
    kd == 0 && return nothing
    T = eltype(buf)
    C = col.coef
    @inbounds c0 = T(C[1, kl, f])
    if kd == 1
        _crow_const!(buf, a, b, f, s, ii, c0)
    elseif kd == 2
        @inbounds _crow_damp!(buf, a, b, f, s, ii, j, k, φ, c0, T(C[4, kl, f]), T(C[5, kl, f]))
    else
        @inbounds _crow_full!(buf, a, b, f, s, ii, j, k, φ, c0, T(C[2, kl, f]), T(C[3, kl, f]),
                              T(C[4, kl, f]), T(C[5, kl, f]))
    end
    return nothing
end
@noinline function _crow_const!(buf, a, b, f, s, ii, c0)
    @inbounds @simd ivdep for i in ii
        buf[i, a, b, f, s] += c0
    end
    return nothing
end
@noinline function _crow_damp!(buf, a, b, f, s, ii, j, k, φ, c0, λ, r)
    @inbounds @simd ivdep for i in ii
        buf[i, a, b, f, s] += c0 + λ * (r - φ[i, j, k])
    end
    return nothing
end
@noinline function _crow_full!(buf, a, b, f, s, ii, j, k, φ, c0, cm, cp, λ, r)
    @inbounds @simd ivdep for i in ii
        φk = φ[i, j, k]
        buf[i, a, b, f, s] += c0 + cm * (φ[i, j, k-1] - φk) + cp * (φ[i, j, k+1] - φk) + λ * (r - φk)
    end
    return nothing
end
