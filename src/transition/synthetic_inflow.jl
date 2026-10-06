# N-inflow-a (#252) — synthetic inflow turbulence on an inflow plane.
#
# Included inside `module Transition`, after free_stream_disturbance.jl.
#
# The contract: an `AbstractInflowGenerator` produces, step by step, three
# planes (ũ, ṽ, w̃) of zero-mean, unit-variance, mutually uncorrelated
# fluctuations with prescribed space and time correlation. The Lund et al.
# (1998) transform `u_i = a_ij ũ_j` (a = Cholesky factor of R_ij) turns
# them into fluctuations with a prescribed Reynolds-stress tensor. Two
# providers implement the contract:
#
#   DigitalFilterGenerator   Xie & Castro (2008) digital filter: separable
#                            exponential spatial filter + an AR(1) blend in
#                            time; L and T per component and height
#                            (PALM: Kadasch et al. 2021).
#   FreeStreamInflow         the isotropic random-mode field of
#                            `FreeStreamDisturbance` (P59-d) swept through
#                            the plane at U∞ (Taylor), whitened.
#
# `lund_coefficients` is the Cholesky factorisation of a Reynolds-stress
# tensor and its realizability check; nothing else in the inflow path
# factorises a stress.

# =========================================================================
# The contract
# =========================================================================

"""
    AbstractInflowGenerator

Supertype of the synthetic inflow-turbulence generators. An implementer
works on an inflow plane of `ns × nz` points (``s`` the horizontal
coordinate along the face, ``z`` the vertical) and provides

| method | meaning |
| :-- | :-- |
| [`unit_fluctuations!`](@ref)`(ũ, ṽ, w̃, gen, Δt)` | advance by `Δt` and write the current planes |
| [`generator_state`](@ref)`(gen)` | snapshot of everything the next planes depend on |
| [`restore_generator_state!`](@ref)`(gen, s)` | return to a snapshot (bit-identical continuation) |
| `inflow_plane_size(gen)` | `(ns, nz)` |

The planes are zero-mean, unit-variance and mutually uncorrelated
(``⟨ũ_iũ_j⟩ = δ_{ij}``); the Reynolds stresses are imposed afterwards by
[`apply_lund!`](@ref) with the factor of [`lund_coefficients`](@ref).
Providers: [`DigitalFilterGenerator`](@ref) and [`FreeStreamInflow`](@ref).
"""
abstract type AbstractInflowGenerator end

"""
    unit_fluctuations!(ũ, ṽ, w̃, gen::AbstractInflowGenerator, Δt) -> (ũ, ṽ, w̃)

Advance the generator `gen` by the time step `Δt` (s) and write its
current zero-mean, unit-variance, mutually uncorrelated fluctuation planes
into the `ns × nz` arrays `ũ`, `ṽ`, `w̃` (the components along the face
normal, along the face, and vertical, in that order). `Δt = 0` returns the
current planes without advancing.
"""
function unit_fluctuations! end

"""
    generator_state(gen) -> state
    restore_generator_state!(gen, state) -> gen

Snapshot of the generator's memory (planes, random-number generator, clock)
into buffers `gen` owns, and its restoration, in the pattern of
`Operators.sgs_state` / `restore_sgs_state!`: after
`s = generator_state(gen)`, any number of [`unit_fluctuations!`](@ref)
calls and `restore_generator_state!(gen, s)`, the next planes are bit for
bit those that would have followed the snapshot. The returned `state` is
overwritten by the next `generator_state` call; `deepcopy` it to keep
several.
"""
function generator_state end
@doc (@doc generator_state) function restore_generator_state! end

"""
    inflow_plane_size(gen::AbstractInflowGenerator) -> (ns, nz)
"""
inflow_plane_size(gen::AbstractInflowGenerator) = (gen.ns, gen.nz)

function _check_planes(gen, planes...)
    sz = inflow_plane_size(gen)
    for p in planes
        size(p) == sz || throw(DimensionMismatch("inflow plane is $(size(p)), generator is $sz"))
    end
end

# =========================================================================
# Lund transform
# =========================================================================

_stress6(R::NTuple{6,Real}) = map(Float64, R)
function _stress6(R::AbstractMatrix)
    size(R) == (3, 3) || throw(DimensionMismatch("lund_coefficients: R must be 3 × 3"))
    s = abs(R[1, 1]) + abs(R[2, 2]) + abs(R[3, 3])
    for (i, j) in ((1, 2), (1, 3), (2, 3))
        abs(R[i, j] - R[j, i]) <= 1e-12 * s ||
            throw(ArgumentError("lund_coefficients: R is not symmetric (R[$i,$j] ≠ R[$j,$i])"))
    end
    return (Float64(R[1, 1]), Float64(R[2, 2]), Float64(R[3, 3]),
            Float64(R[1, 2]), Float64(R[1, 3]), Float64(R[2, 3]))
end

"""
    lund_coefficients(R; rtol = 1e-12) -> a::Matrix{Float64}

The lower-triangular amplitude matrix of Lund, Wu & Squires (1998, Eq. 13),
the Cholesky factor of the Reynolds-stress tensor ``R = a a^{\\rm T}``:

```math
a = \\begin{pmatrix} \\sqrt{R_{11}} & 0 & 0\\\\
     R_{21}/a_{11} & \\sqrt{R_{22} - a_{21}^2} & 0\\\\
     R_{31}/a_{11} & (R_{32} - a_{21}a_{31})/a_{22} & \\sqrt{R_{33} - a_{31}^2 - a_{32}^2}
     \\end{pmatrix},
```

so that ``u_i = a_{ij}ũ_j`` has ``⟨u_iu_j⟩ = R_{ij}`` when
``⟨ũ_iũ_j⟩ = δ_{ij}`` ([`apply_lund!`](@ref)). `R` is a symmetric 3 × 3
matrix or the 6-tuple ``(R_{11}, R_{22}, R_{33}, R_{12}, R_{13}, R_{23})``
(the ordering of `PrognosticSGS`), components (normal, along-face,
vertical) — for a west face ``(u, v, w)``, so ``R_{13} = \\overline{u'w'}``.

The radicands are the leading principal minors divided by the previous
ones, so they are all non-negative exactly when `R` is positive
semi-definite — Schumann's (1977) realizability conditions. A radicand
below ``-\\,\\text{rtol}\\cdot\\operatorname{tr}R`` (or a negative trace)
throws an `ArgumentError` naming the violated condition. A radicand
within round-off of zero (a perfectly correlated or vanishing component,
e.g. ``w' = 0`` at a wall) gives a zero pivot; the column below it must
then vanish too, otherwise `R` is not realizable.

References: Lund, T. S., Wu, X. & Squires, K. D. (1998). Generation of
turbulent inflow data for spatially-developing boundary layer simulations.
*J. Comput. Phys.*, 140, 233-258. Schumann, U. (1977). Realizability of
Reynolds-stress turbulence models. *Phys. Fluids*, 20, 721-725.
"""
function lund_coefficients(R; rtol::Real = 1e-12)
    R11, R22, R33, R12, R13, R23 = _stress6(R)
    tr = R11 + R22 + R33
    tr >= 0 || throw(ArgumentError("lund_coefficients: R is not realizable (negative trace $tr)"))
    tol = rtol * max(tr, floatmin(Float64))
    pivot(x, what) = x < -tol ?
        throw(ArgumentError("lund_coefficients: R is not realizable (Schumann 1977): $what = $x < 0")) :
        sqrt(max(x, 0.0))
    # a zero pivot: the entries below it must vanish (to round-off)
    divide(num, p, what) = p > sqrt(tol) ? num / p :
        (abs(num) <= sqrt(tol) * sqrt(tr) ? 0.0 :
         throw(ArgumentError("lund_coefficients: R is not realizable (Schumann 1977): " *
                             "$what ≠ 0 with a vanishing variance")))
    a11 = pivot(R11, "R11")
    a21 = divide(R12, a11, "R12")
    a31 = divide(R13, a11, "R13")
    a22 = pivot(R22 - a21^2, "R11 R22 - R12² (over R11)")
    a32 = divide(R23 - a21 * a31, a22, "R23 - R12 R13/R11")
    a33 = pivot(R33 - a31^2 - a32^2, "det R (over the 2 × 2 minor)")
    return [a11 0.0 0.0; a21 a22 0.0; a31 a32 a33]
end

"""
    apply_lund!(u, v, w, a) -> (u, v, w)

Impose a Reynolds-stress tensor on unit fluctuations in place: at every
point of the `ns × nz` planes, ``u_i ← a_{ij}u_j`` with the lower-triangular
`a` of [`lund_coefficients`](@ref) (Lund et al. 1998). `a` is one 3 × 3
matrix (a uniform tensor) or a vector of `nz` matrices (one per height,
for ``R_{ij}(z)``). The mean flow is not added.
"""
function apply_lund!(u::AbstractMatrix, v::AbstractMatrix, w::AbstractMatrix, a)
    size(u) == size(v) == size(w) || throw(DimensionMismatch("apply_lund!: planes differ in size"))
    ns, nz = size(u)
    a isa AbstractMatrix || length(a) == nz ||
        throw(DimensionMismatch("apply_lund!: $(length(a)) coefficient matrices for $nz levels"))
    @inbounds for k in 1:nz
        A = a isa AbstractMatrix ? a : a[k]
        a11, a21, a22, a31, a32, a33 = A[1, 1], A[2, 1], A[2, 2], A[3, 1], A[3, 2], A[3, 3]
        for j in 1:ns
            ũ, ṽ, w̃ = u[j, k], v[j, k], w[j, k]
            u[j, k] = a11 * ũ
            v[j, k] = a21 * ũ + a22 * ṽ
            w[j, k] = a31 * ũ + a32 * ṽ + a33 * w̃
        end
    end
    return u, v, w
end

# =========================================================================
# Digital filter (Xie & Castro 2008)
# =========================================================================

# Normalised exponential filter b_k ∝ exp(-π|k|/(2n)), k = -N..N, Σb² = 1.
function _df_kernel(n::Real, N::Integer)
    n <= 0 && return [1.0]
    b = [exp(-π * abs(k) / (2n)) for k in -N:N]
    return b ./ sqrt(sum(abs2, b))
end

# Autocorrelation R_r = Σ_k b_k b_{k+r} (r = 0..2N) of a normalised kernel.
function _df_autocorrelation(b)
    m = length(b)
    return [sum(b[i] * b[i+r] for i in 1:(m-r)) for r in 0:(m-1)]
end

# Integral length of the discrete correlation (trapezoid, R_0 = 1):
# Δ (½ R_0 + Σ_{r≥1} R_r) — the measured ∫_0^∞ R dr of the filtered field.
_df_integral(b, Δ) = (R = _df_autocorrelation(b); Δ * (sum(R) - 0.5 * R[1]))

"""
    digital_filter_kernel(L, Δ) -> (b, n, N)

The normalised filter coefficients ``b_k ∝ \\exp(-π|k|/(2n))``,
``k = -N … N``, ``\\sum b_k^2 = 1`` (Xie & Castro 2008, Eq. 11) whose
filtered white noise has the integral length ``∫_0^∞ R\\,dr = L`` on a grid
of spacing `Δ` (trapezoid on the discrete correlation
``R_r = \\sum_k b_kb_{k+r}``).

**The length convention.** Xie & Castro take ``n = L/Δ`` and state
``R(r) = \\exp(-πr/(2nΔ))``. Neither holds for the integral length: the
autocorrelation of the exponential filter is (continuous limit)
``R(r) = (1 + πr/(2nΔ))\\exp(-πr/(2nΔ))``, with ``∫_0^∞ R\\,dr = 4nΔ/π``,
so the nominal ``n = L/Δ`` gives ``4L/π ≈ 1.27L`` untruncated, and
1.11 L to 1.17 L (for ``L/Δ`` = 2 to 16) with X&C's truncation
``N = 2n``. The width `n` is therefore calibrated (bisection to 1e-13)
so the discrete integral is `L` to 1e-10. The support is
``N = ⌈2L/Δ⌉`` (≈ 2.4 n); it is fixed by `L`, not `n`, so the integral is
continuous in `n` during the calibration. The calibrated widths are
``n/(πL/(4Δ))`` = 1.14, 1.07, 1.05, 1.04 for ``L/Δ`` = 2, 4, 8, 16 (the
truncation shortens the correlation slightly), and the correlation shape
is ``R(r) ≈ (1 + 2r/L)\\,e^{-2r/L}`` (RMS difference 0.012 to 0.019 over
``r ≤ 2L`` for ``L/Δ ≥ 1``).

An `ArgumentError` is thrown for ``L ≤ Δ/2``: the white-noise limit
(``n → 0``) already has the trapezoid integral ``Δ/2``.
"""
function digital_filter_kernel(L::Real, Δ::Real)
    Δ > 0 || throw(ArgumentError("digital_filter_kernel: Δ must be > 0"))
    L > Δ / 2 || throw(ArgumentError("digital_filter_kernel: integral length L = $L ≤ Δ/2 = $(Δ / 2) " *
                                     "is not representable on spacing Δ = $Δ"))
    N = ceil(Int, 2L / Δ)
    f(n) = _df_integral(_df_kernel(n, N), Δ) - L
    lo, hi = 1e-3, 4.0 * L / Δ + 1
    while f(hi) < 0
        hi *= 2
        hi > 1e6 && error("digital_filter_kernel: calibration failed for L = $L, Δ = $Δ")
    end
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f(mid) < 0 ? (lo = mid) : (hi = mid)
        hi - lo < 1e-13 * hi && break
    end
    n = 0.5 * (lo + hi)
    b = _df_kernel(n, N)
    abs(_df_integral(b, Δ) / L - 1) < 1e-10 ||
        error("digital_filter_kernel: calibration did not converge for L = $L, Δ = $Δ")
    return b, n, N
end

# Expand a scale specification to a 3 × nz matrix: scalar; 3-tuple
# (per component); vector of length nz (per height); 3-tuple of
# scalars/vectors; or a 3 × nz matrix.
function _scale_matrix(x, nz, name)
    M = zeros(3, nz)
    if x isa Real
        M .= x
    elseif x isa AbstractMatrix
        size(x) == (3, nz) || throw(DimensionMismatch("$name: matrix must be 3 × $nz"))
        M .= x
    elseif x isa Tuple
        length(x) == 3 || throw(ArgumentError("$name: a tuple must have 3 entries (u, v, w)"))
        for c in 1:3
            M[c, :] .= _scale_matrix(x[c], nz, name)[1, :]
        end
    elseif x isa AbstractVector
        length(x) == nz || throw(DimensionMismatch("$name: vector must have nz = $nz entries"))
        for c in 1:3
            M[c, :] .= x
        end
    else
        throw(ArgumentError("$name: give a number, a vector over height, a 3-tuple or a 3 × nz matrix"))
    end
    return M
end

"""
    DigitalFilterGenerator(ns, nz, Δs, Δz; L_s, L_z, T, seed = 1)

The digital-filter inflow generator of Xie & Castro (2008), as in PALM
(Kadasch et al. 2021), on an `ns × nz` inflow plane with spacings `Δs`
(along the face) and `Δz` (vertical). Implements
[`AbstractInflowGenerator`](@ref).

* `L_s`, `L_z` (m) are the integral lengths along the face and in the
  vertical, `T` (s) the integral time; each may be a number, a vector over
  the `nz` levels, a 3-tuple (one entry — number or vector — per component
  ``ũ, ṽ, w̃``) or a 3 × `nz` matrix.
* **Space.** A fresh plane of Gaussian white noise ``r`` is filtered with
  the separable kernel of [`digital_filter_kernel`](@ref),
  ``ψ_{\\rm new}(j,k) = \\sum_{j'}\\sum_{k'} b^s_{j'}(k)\\,b^z_{k'}(k)\\,r(j+j', k+k')``,
  the coefficients those of the output level ``k`` (so the scales may
  vary with height). The kernels are calibrated so the measured
  ``∫_0^∞R\\,ds = L_s`` and ``∫_0^∞R\\,dz = L_z`` (the X&C nominal
  ``n = L/Δ`` gives 1.11-1.17 L; see [`digital_filter_kernel`](@ref)).
* **Time.** ``ψ(t) = ψ(t-Δt)\\,ρ + ψ_{\\rm new}\\sqrt{1-ρ^2}``, ``ρ = e^{-Δt/T}``
  — the blend of X&C (2008, Eq. 13), ``e^{-πΔt/(2T_{XC})}`` and
  ``(1 - e^{-πΔt/T_{XC}})^{1/2}``, with ``T_{XC} = πT/2``: their
  ``T_{XC}`` is not the integral time (``∫_0^∞e^{-πτ/(2T_{XC})}dτ = 2T_{XC}/π``),
  so ``T`` here is, and the correlation is ``R(τ) = e^{-τ/T}`` for any
  sequence of (adaptive) steps. ``ρ^2 + (1-ρ^2) = 1`` keeps unit variance.
* The first planes are drawn spatially filtered (stationary start).
  Components are independent, so ``⟨ũ_iũ_j⟩ = δ_{ij}``; impose ``R_{ij}``
  with [`apply_lund!`](@ref).

`seed` seeds a `Random.Xoshiro`. [`generator_state`](@ref) /
[`restore_generator_state!`](@ref) snapshot the three planes and the
random-number generator. Cost: one step on a 64 × 40 face with
``L = 4Δ`` is quoted in `docs/src/theory/transition.md`.

References: Xie, Z.-T. & Castro, I. P. (2008). Efficient generation of
inflow conditions for large eddy simulation of street-scale flows.
*Flow Turbul. Combust.*, 81, 449-470. Kadasch, E. et al. (2021). Mesoscale
nesting interface of the PALM model system 6.0. *Geosci. Model Dev.*, 14,
5435-5465. Klein, M., Sadiki, A. & Janicka, J. (2003). A digital filter
based generation of inflow data for spatially developing direct numerical
or large eddy simulations. *J. Comput. Phys.*, 186, 652-665.
"""
mutable struct DigitalFilterGenerator <: AbstractInflowGenerator
    ns::Int
    nz::Int
    Δs::Float64
    Δz::Float64
    L_s::Matrix{Float64}             # requested, 3 × nz
    L_z::Matrix{Float64}
    T::Matrix{Float64}
    n_s::Matrix{Float64}             # calibrated filter widths
    n_z::Matrix{Float64}
    bs::Matrix{Vector{Float64}}      # kernels per (component, level)
    bz::Matrix{Vector{Float64}}
    Ms::Int                          # max support along s / z
    Mz::Int
    ψ::NTuple{3,Matrix{Float64}}     # current planes
    r::Matrix{Float64}               # white noise, (ns + 2Ms) × (nz + 2Mz)
    tmp::Matrix{Float64}             # z-filtered noise, (ns + 2Ms) × nz
    new::Matrix{Float64}             # freshly filtered plane
    rng::Random.Xoshiro
    snapψ::NTuple{3,Matrix{Float64}}
    snaprng::Random.Xoshiro
end

function DigitalFilterGenerator(ns::Integer, nz::Integer, Δs::Real, Δz::Real;
                                L_s, L_z, T, seed::Integer = 1)
    ns >= 1 && nz >= 1 || throw(ArgumentError("DigitalFilterGenerator: need ns, nz ≥ 1"))
    Ls = _scale_matrix(L_s, nz, "L_s"); Lz = _scale_matrix(L_z, nz, "L_z")
    Tm = _scale_matrix(T, nz, "T")
    all(>(0), Tm) || throw(ArgumentError("DigitalFilterGenerator: T must be > 0"))
    cache = Dict{Tuple{Float64,Float64},Tuple{Vector{Float64},Float64}}()
    kern(L, Δ) = get!(cache, (L, Δ)) do
        b, n, _ = digital_filter_kernel(L, Δ); (b, n)
    end
    bs = Matrix{Vector{Float64}}(undef, 3, nz); bz = similar(bs)
    n_s = zeros(3, nz); n_z = zeros(3, nz)
    for k in 1:nz, c in 1:3
        bs[c, k], n_s[c, k] = kern(Ls[c, k], Float64(Δs))
        bz[c, k], n_z[c, k] = kern(Lz[c, k], Float64(Δz))
    end
    Ms = maximum(b -> length(b) ÷ 2, bs); Mz = maximum(b -> length(b) ÷ 2, bz)
    rng = Random.Xoshiro(seed)
    ψ = ntuple(_ -> zeros(ns, nz), 3)
    gen = DigitalFilterGenerator(ns, nz, Float64(Δs), Float64(Δz), Ls, Lz, Tm, n_s, n_z,
                                 bs, bz, Ms, Mz, ψ, zeros(ns + 2Ms, nz + 2Mz),
                                 zeros(ns + 2Ms, nz), zeros(ns, nz), rng,
                                 ntuple(_ -> zeros(ns, nz), 3), copy(rng))
    for c in 1:3
        _df_filtered_noise!(gen, c)
        copyto!(gen.ψ[c], gen.new)
    end
    return gen
end

# gen.new ← spatially filtered fresh white noise for component c.
function _df_filtered_noise!(gen::DigitalFilterGenerator, c::Int)
    r, tmp, new = gen.r, gen.tmp, gen.new
    Random.randn!(gen.rng, r)
    ns, nz, Ms, Mz = gen.ns, gen.nz, gen.Ms, gen.Mz
    nj = ns + 2Ms
    @inbounds for k in 1:nz
        b = gen.bz[c, k]; N = length(b) ÷ 2
        for jj in 1:nj
            s = 0.0
            for q in -N:N
                s += b[q+N+1] * r[jj, k+Mz+q]
            end
            tmp[jj, k] = s
        end
        b = gen.bs[c, k]; N = length(b) ÷ 2
        for j in 1:ns
            s = 0.0
            for q in -N:N
                s += b[q+N+1] * tmp[j+Ms+q, k]
            end
            new[j, k] = s
        end
    end
    return new
end

function unit_fluctuations!(ũ::AbstractMatrix, ṽ::AbstractMatrix, w̃::AbstractMatrix,
                            gen::DigitalFilterGenerator, Δt::Real)
    _check_planes(gen, ũ, ṽ, w̃)
    Δt >= 0 || throw(ArgumentError("unit_fluctuations!: Δt must be ≥ 0"))
    if Δt > 0
        for c in 1:3
            _df_filtered_noise!(gen, c)
            ψ = gen.ψ[c]
            @inbounds for k in 1:gen.nz
                ρ = exp(-Δt / gen.T[c, k]); σ = sqrt(1 - ρ^2)
                for j in 1:gen.ns
                    ψ[j, k] = ρ * ψ[j, k] + σ * gen.new[j, k]
                end
            end
        end
    end
    copyto!(ũ, gen.ψ[1]); copyto!(ṽ, gen.ψ[2]); copyto!(w̃, gen.ψ[3])
    return ũ, ṽ, w̃
end

function generator_state(gen::DigitalFilterGenerator)
    for c in 1:3
        copyto!(gen.snapψ[c], gen.ψ[c])
    end
    copy!(gen.snaprng, gen.rng)
    return (ψ = gen.snapψ, rng = gen.snaprng)
end
function restore_generator_state!(gen::DigitalFilterGenerator, s)
    for c in 1:3
        copyto!(gen.ψ[c], s.ψ[c])
    end
    copy!(gen.rng, s.rng)
    return gen
end

# =========================================================================
# FreeStreamDisturbance as a second provider
# =========================================================================

"""
    disturbance_velocity(fst::FreeStreamDisturbance, x, y, z) -> (u, v, w)

The continuous random-mode velocity of `fst` at a point,
``u'(x) = \\sum_n A_nσ_n\\cos(k_n·x + φ_n)`` — the field whose staggered,
divergence-free grid version [`disturbance_velocity!`](@ref) samples
(the two agree to ``O((κΔ)^2)``). Used on inflow planes by
[`FreeStreamInflow`](@ref), where the discrete curl's ``\\mathrm{sinc}``
attenuation would bias the variance.
"""
function disturbance_velocity(fst::FreeStreamDisturbance, x::Real, y::Real, z::Real)
    u = v = w = 0.0
    @inbounds for m in axes(fst.k, 2)
        c = fst.A[m] * cos(fst.k[1, m] * x + fst.k[2, m] * y + fst.k[3, m] * z + fst.φ[m])
        u += c * fst.σ[1, m]; v += c * fst.σ[2, m]; w += c * fst.σ[3, m]
    end
    return u, v, w
end

"""
    disturbance_covariance(fst::FreeStreamDisturbance) -> C::Matrix{Float64}

The exact one-point covariance of the mode set,
``C_{ij} = \\sum_n \\tfrac12 A_n^2 σ_{n,i}σ_{n,j}`` (the space average of
``u'_iu'_j`` for distinct wavevectors); ``\\tfrac12\\operatorname{tr}C = q``.
A finite mode set is only approximately isotropic: the diagonal scatters
around ``(Tu\\,U_∞)^2`` and the off-diagonals around 0 by
``O(n_{\\rm modes}^{-1/2})``.
"""
function disturbance_covariance(fst::FreeStreamDisturbance)
    C = zeros(3, 3)
    for m in eachindex(fst.A), j in 1:3, i in 1:3
        C[i, j] += 0.5 * fst.A[m]^2 * fst.σ[i, m] * fst.σ[j, m]
    end
    return C
end

"""
    FreeStreamInflow(fst::FreeStreamDisturbance, ns, nz, Δs, Δz;
                     origin = (0, 0), x = 0, t = 0)

[`FreeStreamDisturbance`](@ref) as an [`AbstractInflowGenerator`](@ref):
the frozen random-mode field advected at `fst.U∞` (Taylor) through an
inflow plane at `x`, with plane points ``s_j = s_0 + (j - ½)Δs``,
``z_k = z_0 + (k - ½)Δz`` (`origin = (s_0, z_0)`; ``s`` is the ``y`` of the
mode set). Each step evaluates the continuous mode sum
([`disturbance_velocity`](@ref)) at ``x - U_∞t`` and whitens it with the
Cholesky factor of the exact mode-set covariance
([`disturbance_covariance`](@ref), [`lund_coefficients`](@ref)):
``ũ = a^{-1}u'``. The planes are then exactly zero-mean, unit-variance and
uncorrelated over a long sweep, whatever the anisotropy of the finite
mode set; since ``a`` is lower triangular, ``ũ`` is ``u'`` rescaled, so its
longitudinal integral scale is that of `fst` (``L_{11}``). Requires
`Tu > 0` and `U∞ > 0`. The state is the clock `t`. Unlike
[`DigitalFilterGenerator`](@ref) it is isotropic with a single scale.
"""
mutable struct FreeStreamInflow <: AbstractInflowGenerator
    fst::FreeStreamDisturbance
    ns::Int
    nz::Int
    s::Vector{Float64}
    z::Vector{Float64}
    x::Float64
    t::Float64
    ainv::Matrix{Float64}            # whitening a⁻¹ (lower triangular)
    ey::Matrix{ComplexF64}           # cis(k_y s_j) per mode, ns × nmodes
    ez::Matrix{ComplexF64}           # cis(k_z z_k) per mode, nz × nmodes
    snapt::Float64
end

# Inverse of a non-singular lower-triangular 3 × 3 matrix (forward substitution).
function _lower_inverse(a)
    all(>(0), (a[1, 1], a[2, 2], a[3, 3])) ||
        throw(ArgumentError("FreeStreamInflow: the mode-set covariance is singular"))
    b = zeros(3, 3)
    for j in 1:3
        for i in j:3
            s = i == j ? 1.0 : 0.0
            for m in j:(i-1)
                s -= a[i, m] * b[m, j]
            end
            b[i, j] = s / a[i, i]
        end
    end
    return b
end

function FreeStreamInflow(fst::FreeStreamDisturbance, ns::Integer, nz::Integer,
                          Δs::Real, Δz::Real; origin = (0.0, 0.0), x::Real = 0.0,
                          t::Real = 0.0)
    fst.Tu > 0 && fst.U∞ > 0 ||
        throw(ArgumentError("FreeStreamInflow: need Tu > 0 and U∞ > 0 (unit fluctuations of a zero field)"))
    s = [origin[1] + (j - 0.5) * Δs for j in 1:ns]
    z = [origin[2] + (k - 0.5) * Δz for k in 1:nz]
    a = lund_coefficients(disturbance_covariance(fst))
    nm = length(fst.A)
    ey = [cis(fst.k[2, m] * s[j]) for j in 1:ns, m in 1:nm]
    ez = [cis(fst.k[3, m] * z[k]) for k in 1:nz, m in 1:nm]
    return FreeStreamInflow(fst, ns, nz, s, z, Float64(x), Float64(t), _lower_inverse(a), ey, ez,
                            Float64(t))
end

function unit_fluctuations!(ũ::AbstractMatrix, ṽ::AbstractMatrix, w̃::AbstractMatrix,
                            gen::FreeStreamInflow, Δt::Real)
    _check_planes(gen, ũ, ṽ, w̃)
    Δt >= 0 || throw(ArgumentError("unit_fluctuations!: Δt must be ≥ 0"))
    gen.t += Δt
    fst = gen.fst
    fill!(ũ, 0); fill!(ṽ, 0); fill!(w̃, 0)
    xs = gen.x - fst.U∞ * gen.t
    # Σ_m A_m σ_m Re[e^{i(k_x x + φ)} e^{i k_y s} e^{i k_z z}], separable in s and z
    @inbounds for m in eachindex(fst.A)
        ex = fst.A[m] * cis(fst.k[1, m] * xs + fst.φ[m])
        σ1, σ2, σ3 = fst.σ[1, m], fst.σ[2, m], fst.σ[3, m]
        for k in 1:gen.nz
            exz = ex * gen.ez[k, m]
            for j in 1:gen.ns
                c = real(exz * gen.ey[j, m])
                ũ[j, k] += c * σ1; ṽ[j, k] += c * σ2; w̃[j, k] += c * σ3
            end
        end
    end
    # whiten: ũ ← a⁻¹ u′ (lower triangular, in place)
    A = gen.ainv
    @inbounds for k in 1:gen.nz, j in 1:gen.ns
        u, v, w = ũ[j, k], ṽ[j, k], w̃[j, k]
        ũ[j, k] = A[1, 1] * u
        ṽ[j, k] = A[2, 1] * u + A[2, 2] * v
        w̃[j, k] = A[3, 1] * u + A[3, 2] * v + A[3, 3] * w
    end
    return ũ, ṽ, w̃
end

generator_state(gen::FreeStreamInflow) = (gen.snapt = gen.t; (t = gen.snapt,))
restore_generator_state!(gen::FreeStreamInflow, s) = (gen.t = s.t; gen)
