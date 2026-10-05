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
