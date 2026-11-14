# =========================================================================
# The SGS closures' tendencies as parts of a term of the fused sweep
# (#533 plan step 4). Included by FiniteVolume after fused_sweep.jl.
# =========================================================================
#
# The closure's coefficients (νt, κt, the anisotropic νv, κv or a_v of a
# stability-corrected closure, a diffusivity tensor, a stored stress) are
# computed before the sweep, as the separate loops compute them
# (`Operators.compute_sgs!`, in the term's `prepare!`); inside the sweep
# they are read-only inputs. The parts below run, per row, the point
# bodies of the whole-field operators themselves (`_b_eddy_u!`,
# `_b_sgs_stress_div!`, `_b_stress_div!`, `_b_eddy_scalar!`, the tensor
# face fluxes `_tensor_f*`), writing into the sweep's row buffer through
# `_RowField` instead of into a tendency field: one source of truth, and a
# fused value is the separate loop's within round-off (bit-identical in
# the tests). The term that holds them, `Operators.SGSTerm`, is built from
# the closure contract in src/operators/sgs_sweep.jl.
#
# Footprint: with the coefficients precomputed, every part reads the
# state (u, v, w, θ, qv) at most one point away in each direction (the
# compact flux differences; the tensor's cross terms at the diagonal
# neighbours), so each part's radius over the state is (1, 1, 1), the
# base term's: the ring keeps 2 rows and each slab 1 edge plane per end.
# The coefficients are read up to one point away as well, but they are
# inputs the predictor never overwrites, so they do not enter the lag.
# A stored stress is read only through its inputs: radius 0.

# a row (j, k) of field f of the sweep's tendency buffer, seen as a 3-D
# array, so a whole-field body's `t[i, j, k] += x` adds into the row
struct _RowField{B<:AbstractArray}
    buf::B
    a::Int
    b::Int
    f::Int
    s::Int
end
@inline Base.getindex(r::_RowField, i::Integer, j::Integer, k::Integer) = @inbounds r.buf[i, r.a, r.b, r.f, r.s]
@inline Base.setindex!(r::_RowField, x, i::Integer, j::Integer, k::Integer) =
    (@inbounds r.buf[i, r.a, r.b, r.f, r.s] = x; r)

# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------

"""
    EddyMomentum(νh, νv, symmetric, idx, idy, idz)

The eddy-viscosity SGS momentum tendency as a part of a sweep term (#533):
the Laplacian form `∂ⱼ(νh ∂ⱼuᵢ)` of `add_eddy_viscosity_u!` (`_v!`, `_w!`)
when `symmetric` is `false`, the stress form `∂ⱼ(2νᵢⱼSᵢⱼ)` of
`add_sgs_stress_divergence!` (`νh`, `νv`, a `RegularGrid`'s `1/Δ`
`idx, idy, idz`) when `true`; `νh`, `νv` the halo-padded coefficient
arrays. The form is chosen once per row, so one term type serves both
values of the `sgs_stress` keyword.
"""
struct EddyMomentum{A<:AbstractArray,T}
    νh::A
    νv::A
    symmetric::Bool
    idx::T
    idy::T
    idz::T
end

"""
    StressMomentum(τ11, τ22, τ33, τ12, τ13, τ23, idx, idy, idz)

The divergence `−∂ⱼτᵢⱼ` of a stored stress (`add_stress_divergence!`, the
prognostic Reynolds stress) as a part of a sweep term (#533). Reads no
state, only the six stress arrays.
"""
struct StressMomentum{A<:AbstractArray,T}
    τ11::A
    τ22::A
    τ33::A
    τ12::A
    τ13::A
    τ23::A
    idx::T
    idy::T
    idz::T
end

part_footprint(::Nothing) = (0, 0, 0)
part_footprint(::EddyMomentum) = (1, 1, 1)
part_footprint(::StressMomentum) = (0, 0, 0)
part_inputs(::Nothing) = ()
part_inputs(p::EddyMomentum) = (p.νh, p.νv)
part_inputs(p::StressMomentum) = (p.τ11, p.τ22, p.τ33, p.τ12, p.τ13, p.τ23)

@inline add_part_row!(::Nothing, buf, a, b, s, j, k, ii, st) = nothing
@inline function add_part_row!(p::EddyMomentum, buf, a, b, s, j, k, ii, st)
    if p.symmetric
        T = eltype(buf)
        _sgs_stress_row!(buf, a, b, s, j, k, ii, st.u, st.v, st.w, p.νh, p.νv, p.idx, p.idy, p.idz, one(T))
    else
        _sgs_lap_row!(buf, a, b, s, j, k, ii, st.u, st.v, st.w, p.νh, st.m)
    end
    return nothing
end
@inline add_part_row!(p::StressMomentum, buf, a, b, s, j, k, ii, st) =
    _stress_div_row!(buf, a, b, s, j, k, ii, p.τ11, p.τ22, p.τ33, p.τ12, p.τ13, p.τ23, p.idx, p.idy, p.idz)

# The row loops: @noinline, the arrays as separate arguments (the rule of
# `Backends._row!`, docs/src/design/ka_port.md "SC-e"); `ivdep`: the bodies
# read the state and the coefficients, and write the row buffer only at i.
@noinline function _sgs_lap_row!(buf, a, b, s, j, k, ii, u, v, w, n, m)
    ru = _RowField(buf, a, b, SWEEP_U, s)
    rv = _RowField(buf, a, b, SWEEP_V, s)
    rw = _RowField(buf, a, b, SWEEP_W, s)
    @inbounds @simd ivdep for i in ii
        _b_eddy_u!(i, j, k, ru, u, n, m, nothing, nothing)
        _b_eddy_v!(i, j, k, rv, v, n, m, nothing, nothing)
        _b_eddy_w!(i, j, k, rw, w, n, m, nothing, nothing)
    end
    return nothing
end
@noinline function _sgs_stress_row!(buf, a, b, s, j, k, ii, u, v, w, nh, nv, idx, idy, idz, o)
    ru = _RowField(buf, a, b, SWEEP_U, s)
    rv = _RowField(buf, a, b, SWEEP_V, s)
    rw = _RowField(buf, a, b, SWEEP_W, s)
    @inbounds @simd ivdep for i in ii
        _b_sgs_stress_div!(i, j, k, ru, rv, rw, u, v, w, nh, nv, idx, idy, idz, o, o, nothing, nothing)
    end
    return nothing
end
@noinline function _stress_div_row!(buf, a, b, s, j, k, ii, τ11, τ22, τ33, τ12, τ13, τ23, idx, idy, idz)
    ru = _RowField(buf, a, b, SWEEP_U, s)
    rv = _RowField(buf, a, b, SWEEP_V, s)
    rw = _RowField(buf, a, b, SWEEP_W, s)
    @inbounds @simd ivdep for i in ii
        _b_stress_div!(i, j, k, ru, rv, rw, τ11, τ22, τ33, τ12, τ13, τ23, idx, idy, idz)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# scalars
# ---------------------------------------------------------------------------

"""
    EddyScalar(Kh, Kv, s)

The diagonal SGS scalar flux divergence `s·[∂x(Kh ∂xϕ) + ∂y(Kh ∂yϕ) +
∂z(Kv ∂zϕ)]` of `_add_eddy_diffusion_scalar!` (every public form of
`add_eddy_diffusion_scalar!` without a mask or weights) as a part of a
sweep term (#533); `Kh`, `Kv` the halo-padded cell-centred coefficient
arrays (the `(Kh, Kv, s)` of `Operators.sgs_scalar_face_diffusivity`).
"""
struct EddyScalar{A<:AbstractArray,S}
    Kh::A
    Kv::A
    s::S
end

"""
    TensorScalar(K11, K22, K33, K12, K13, K23, idx, idy, idz, bounded, ranges)

The full-tensor SGS scalar flux divergence of
`add_tensor_diffusion_scalar!` (the isentropic diffusion, the Reynolds-stress
closure's Daly–Harlow flux; `bounded` its walls, `ranges` the interior
`(ii, jj, kk)` of a `RegularGrid`) as a part of a sweep term (#533). The
separate loops store the face fluxes, then difference them; here each
face flux of a point is evaluated where it is needed (`_tensor_f*`, the
same function), so every interior face is computed twice, with the same
result, and nothing is stored.
"""
struct TensorScalar{A<:AbstractArray,T}
    K11::A
    K22::A
    K33::A
    K12::A
    K13::A
    K23::A
    idx::T
    idy::T
    idz::T
    bx::Bool
    by::Bool
    bz::Bool
    i1::Int
    i2::Int
    j1::Int
    j2::Int
    k1::Int
    k2::Int
end
function TensorScalar(K11, K22, K33, K12, K13, K23, idx, idy, idz, bounded::NTuple{3,Bool}, (ii, jj, kk))
    return TensorScalar(K11, K22, K33, K12, K13, K23, idx, idy, idz, bounded...,
                        first(ii), last(ii), first(jj), last(jj), first(kk), last(kk))
end

part_footprint(::EddyScalar) = (1, 1, 1)
part_footprint(::TensorScalar) = (1, 1, 1)
part_inputs(p::EddyScalar) = (p.Kh, p.Kv)
part_inputs(p::TensorScalar) = (p.K11, p.K22, p.K33, p.K12, p.K13, p.K23)

# the scalar part of field f (SWEEP_Θ or SWEEP_QV) of state array ϕ
@inline add_scalar_row!(p, f, buf, a, b, s, j, k, ii, ::Nothing, m) = nothing   # dry: no qv
@inline add_scalar_row!(p::EddyScalar, f, buf, a, b, s, j, k, ii, ϕ::AbstractArray, m) =
    _sgs_scalar_row!(buf, a, b, f, s, j, k, ii, ϕ, p.Kh, p.Kv, p.s, m)
@inline add_scalar_row!(p::TensorScalar, f, buf, a, b, s, j, k, ii, ϕ::AbstractArray, m) =
    _sgs_tensor_row!(buf, a, b, f, s, j, k, ii, ϕ, p.K11, p.K22, p.K33, p.K12, p.K13, p.K23,
                     p.idx, p.idy, p.idz, p.bx, p.by, p.bz, p.i1, p.i2, p.j1, p.j2, p.k1, p.k2)

@noinline function _sgs_scalar_row!(buf, a, b, f, s, j, k, ii, ϕ, Kh, Kv, sc, m)
    r = _RowField(buf, a, b, f, s)
    @inbounds @simd ivdep for i in ii
        _b_eddy_scalar!(i, j, k, r, ϕ, Kh, Kv, sc, m)
    end
    return nothing
end

# t += (Fx[i] − Fx[i−1])·idx + … with each face flux evaluated in place:
# the expression of `_b_flux_divergence!` over the fluxes of `_b_tensor_f*!`
# (the halo face Hx = i1 − 1 of the x faces is the flux `Fx[i1 − 1]`).
@noinline function _sgs_tensor_row!(buf, a, b, f, s, j, k, ii, d, K11, K22, K33, K12, K13, K23,
                                    idx, idy, idz, bx, by, bz, i1, i2, j1, j2, k1, k2)
    Hx, Hy, Hz = i1 - 1, j1 - 1, k1 - 1
    @inbounds for i in ii
        fxp = _tensor_fx(i,     j, k, d, K11, K12, K13, idx, idy, idz, bx, by, bz, j1, j2, k1, k2, Hx, i2, true, true)
        fxm = _tensor_fx(i - 1, j, k, d, K11, K12, K13, idx, idy, idz, bx, by, bz, j1, j2, k1, k2, Hx, i2, true, true)
        fyp = _tensor_fy(i, j,     k, d, K22, K12, K23, idx, idy, idz, bx, by, bz, i1, i2, k1, k2, Hy, j2, true, true)
        fym = _tensor_fy(i, j - 1, k, d, K22, K12, K23, idx, idy, idz, bx, by, bz, i1, i2, k1, k2, Hy, j2, true, true)
        fzp = _tensor_fz(i, j, k,     d, K33, K13, K23, idx, idy, idz, bx, by, bz, i1, i2, j1, j2, Hz, k2, true, true)
        fzm = _tensor_fz(i, j, k - 1, d, K33, K13, K23, idx, idy, idz, bx, by, bz, i1, i2, j1, j2, Hz, k2, true, true)
        buf[i, a, b, f, s] += (fxp - fxm) * idx + (fyp - fym) * idy + (fzp - fzm) * idz
    end
    return nothing
end

# ---------------------------------------------------------------------------
# the point kernels of the per-point (GPU) form (#533 step 8; integration of
# steps 4 and 8): the same point bodies as the row loops above, at one point
# ---------------------------------------------------------------------------

# field f of the tendency row a point kernel adds into, seen as a 3-D array
@inline _row_field(tb::SweepRow, f) = _RowField(tb.buf, tb.a, tb.b, f, tb.s)

@inline add_part_point!(::Nothing, tb, i, j, k, st) = nothing
@inline function add_part_point!(p::EddyMomentum, tb, i, j, k, st)
    ru, rv, rw = _row_field(tb, SWEEP_U), _row_field(tb, SWEEP_V), _row_field(tb, SWEEP_W)
    if p.symmetric
        o = one(eltype(tb.buf))
        _b_sgs_stress_div!(i, j, k, ru, rv, rw, st.u, st.v, st.w, p.νh, p.νv, p.idx, p.idy, p.idz, o, o,
                           nothing, nothing)
    else
        _b_eddy_u!(i, j, k, ru, st.u, p.νh, st.m, nothing, nothing)
        _b_eddy_v!(i, j, k, rv, st.v, p.νh, st.m, nothing, nothing)
        _b_eddy_w!(i, j, k, rw, st.w, p.νh, st.m, nothing, nothing)
    end
    return nothing
end
@inline add_part_point!(p::StressMomentum, tb, i, j, k, st) =
    _b_stress_div!(i, j, k, _row_field(tb, SWEEP_U), _row_field(tb, SWEEP_V), _row_field(tb, SWEEP_W),
                   p.τ11, p.τ22, p.τ33, p.τ12, p.τ13, p.τ23, p.idx, p.idy, p.idz)

@inline add_scalar_point!(p, f, tb, i, j, k, ::Nothing, m) = nothing   # dry: no qv
@inline add_scalar_point!(p::EddyScalar, f, tb, i, j, k, ϕ::AbstractArray, m) =
    _b_eddy_scalar!(i, j, k, _row_field(tb, f), ϕ, p.Kh, p.Kv, p.s, m)
@inline function add_scalar_point!(p::TensorScalar, f, tb, i, j, k, d::AbstractArray, m)
    i1, i2, j1, j2, k1, k2 = p.i1, p.i2, p.j1, p.j2, p.k1, p.k2
    Hx, Hy, Hz = i1 - 1, j1 - 1, k1 - 1
    K11, K22, K33, K12, K13, K23 = p.K11, p.K22, p.K33, p.K12, p.K13, p.K23
    idx, idy, idz, bx, by, bz = p.idx, p.idy, p.idz, p.bx, p.by, p.bz
    fxp = _tensor_fx(i,     j, k, d, K11, K12, K13, idx, idy, idz, bx, by, bz, j1, j2, k1, k2, Hx, i2, true, true)
    fxm = _tensor_fx(i - 1, j, k, d, K11, K12, K13, idx, idy, idz, bx, by, bz, j1, j2, k1, k2, Hx, i2, true, true)
    fyp = _tensor_fy(i, j,     k, d, K22, K12, K23, idx, idy, idz, bx, by, bz, i1, i2, k1, k2, Hy, j2, true, true)
    fym = _tensor_fy(i, j - 1, k, d, K22, K12, K23, idx, idy, idz, bx, by, bz, i1, i2, k1, k2, Hy, j2, true, true)
    fzp = _tensor_fz(i, j, k,     d, K33, K13, K23, idx, idy, idz, bx, by, bz, i1, i2, j1, j2, Hz, k2, true, true)
    fzm = _tensor_fz(i, j, k - 1, d, K33, K13, K23, idx, idy, idz, bx, by, bz, i1, i2, j1, j2, Hz, k2, true, true)
    tb[i, f] += (fxp - fxm) * idx + (fyp - fym) * idy + (fzp - fzm) * idz
    return nothing
end
