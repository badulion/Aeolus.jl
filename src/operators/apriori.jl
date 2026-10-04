# A-priori SGS-stress analysis (Phase 56, P56-c).
#
# Included into `Operators`. The exact subfilter stress of a resolved
# (DNS) field under an explicit filter, the SGS dissipation, model stresses
# for comparison (eddy viscosity / Smagorinsky, Bardina scale similarity)
# and the standard a-priori metrics. The filters are those of
# `filters.jl` (or any callable with the same signature); nothing here
# filters on its own.
#
# Everything is evaluated at cell centres on the interior of a
# `RegularGrid`, and derivatives treat the interior as triply periodic,
# as the Fourier filters and the spectrum of P54-c do.

# -------------------------------------------------------------------------
# Symmetric tensor at cell centres
# -------------------------------------------------------------------------

"""
    SymmetricTensorField(mesh::RegularGrid)
    SymmetricTensorField{T}(Nx, Ny, Nz)

The six independent components `xx, yy, zz, xy, xz, yz` of a symmetric
3 × 3 tensor at the interior cell centres (plain `Nx × Ny × Nz` arrays, no
halos). Used for the subfilter stress ``τ_{ij}``, model stresses and the
filtered strain rate ``\\bar S_{ij}``. `τ[i, j]` returns the component
array for `i, j ∈ 1:3`; `components(τ)` the six arrays in the order
above.
"""
struct SymmetricTensorField{T}
    xx::Array{T,3}; yy::Array{T,3}; zz::Array{T,3}
    xy::Array{T,3}; xz::Array{T,3}; yz::Array{T,3}
end

SymmetricTensorField{T}(Nx::Integer, Ny::Integer, Nz::Integer) where {T} =
    SymmetricTensorField{T}((zeros(T, Nx, Ny, Nz) for _ in 1:6)...)
SymmetricTensorField(g::RegularGrid{T}) where {T} = SymmetricTensorField{T}(g.Nx, g.Ny, g.Nz)
Base.similar(τ::SymmetricTensorField{T}) where {T} = SymmetricTensorField{T}(size(τ.xx)...)
Base.size(τ::SymmetricTensorField) = size(τ.xx)

"""
    components(τ::SymmetricTensorField) -> (xx, yy, zz, xy, xz, yz)
"""
components(τ::SymmetricTensorField) = (τ.xx, τ.yy, τ.zz, τ.xy, τ.xz, τ.yz)

const _SYM_INDEX = ((1, 4, 5), (4, 2, 6), (5, 6, 3))
Base.getindex(τ::SymmetricTensorField, i::Integer, j::Integer) = components(τ)[_SYM_INDEX[i][j]]

# -------------------------------------------------------------------------
# Exact subfilter stress
# -------------------------------------------------------------------------

# Cell-centred copy of a staggered component: the two-point average with
# periodic wrap-around (halos not needed). It lived in spectra.jl until the
# energy spectrum stopped centring (#205, P54-c review); the a-priori
# analysis is collocated by construction and is now its only user.
function _to_center_periodic(f::Field{L}) where {L}
    a = interior_view(f)
    Nx, Ny, Nz = size(a)
    s = L === FaceX ? (1, 0, 0) : L === FaceY ? (0, 1, 0) : L === FaceZ ? (0, 0, 1) : (0, 0, 0)
    s == (0, 0, 0) && return Array(a)
    c = similar(a, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        c[i,j,k] = 0.5 * (a[i,j,k] + a[mod1(i + s[1], Nx), mod1(j + s[2], Ny), mod1(k + s[3], Nz)])
    end
    return c
end

# Cell-centred copy of a velocity component into `out` (a CellCenter field):
# the two-point average with periodic wrap for a staggered component.
function _center_into!(out::Field{CellCenter}, f::Field)
    interior_view(out) .= _to_center_periodic(f)
    return out
end

const _PAIRS = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))

"""
    subfilter_stress!(τ, u, v, w, filter; ubar = nothing, bcs = periodic) -> τ

The exact subfilter (SGS) stress of a resolved velocity field under an
explicit filter ``\\overline{\\,·\\,}``,

```math
τ_{ij} = \\overline{u_i u_j} - \\bar u_i \\bar u_j ,
```

at the interior cell centres, written into `τ::SymmetricTensorField`.
`u, v, w` are staggered (`FaceX/FaceY/FaceZ`; interpolated to centres by
the periodic two-point average of [`isotropic_energy_spectrum`](@ref)) or
all `CellCenter`. `filter` is any callable `filter(out, f)` acting on
`CellCenter` fields, e.g.

```julia
(o, f) -> gaussian_filter!(o, f, 4h)       # Gaussian, Δ = 4h
(o, f) -> box_filter!(o, f, 5)             # top hat over 5 cells
(o, f) -> spectral_cutoff_filter!(o, f, kc)
```

The halos of every field handed to `filter` are filled with `bcs`
(triply periodic by default), so stencil filters work too. With
`ubar = (ū, v̄, w̄)` (three `CellCenter` fields) the filtered velocity is
returned as well.

For a filter with a positive kernel ``τ_{ij}`` is a covariance of the
velocity under the kernel weights and hence positive semi-definite at
every point (Vreman, Geurts & Kuerten 1994); it is Galilean invariant for
any filter that preserves constants (Speziale 1985).

References: Leonard, A. (1974), *Adv. Geophys.* 18A, 237; Vreman, B.,
Geurts, B. & Kuerten, H. (1994), *J. Fluid Mech.* 278, 351; Pope, S. B.
(2000), *Turbulent Flows*, §13.3.
"""
function subfilter_stress!(τ::SymmetricTensorField, u::Field, v::Field, w::Field, filter;
                           ubar = nothing,
                           bcs::FieldBoundaryConditions = FieldBoundaryConditions(:periodic))
    g = mesh(u)
    g isa RegularGrid || throw(ArgumentError("subfilter_stress!: RegularGrid only"))
    size(τ) == (g.Nx, g.Ny, g.Nz) ||
        throw(DimensionMismatch("subfilter_stress!: τ is $(size(τ)), interior $((g.Nx, g.Ny, g.Nz))"))
    uc = ntuple(_ -> Field(CellCenter(), g), 3)
    for (c, f) in zip(uc, (u, v, w))
        _center_into!(c, f)
    end
    ub = ubar === nothing ? ntuple(_ -> Field(CellCenter(), g), 3) : ubar
    _apply_filter!(ub[1], uc[1], filter, bcs)
    _apply_filter!(ub[2], uc[2], filter, bcs)
    _apply_filter!(ub[3], uc[3], filter, bcs)
    prod_ = Field(CellCenter(), g); fprod = Field(CellCenter(), g)
    ubi = map(interior_view, ub); uci = map(interior_view, uc)
    for (n, (i, j)) in enumerate(_PAIRS)
        interior_view(prod_) .= uci[i] .* uci[j]
        _apply_filter!(fprod, prod_, filter, bcs)
        components(τ)[n] .= interior_view(fprod) .- ubi[i] .* ubi[j]
    end
    return τ
end

function _apply_filter!(out, f, filter, bcs)
    fill_halos!(f, bcs)
    filter(out, f)
    return out
end

"""
    bardina_stress!(τ, ū, v̄, w̄, filter; C_B = 1, bcs = periodic) -> τ

Scale-similarity model of Bardina, Ferziger & Reynolds (1980): the
subfilter stress of the *filtered* field under the same filter,

```math
τ^{B}_{ij} = C_B\\left(\\overline{\\bar u_i \\bar u_j} - \\bar{\\bar u}_i \\bar{\\bar u}_j\\right),
```

i.e. [`subfilter_stress!`](@ref) applied to `ū, v̄, w̄` (`CellCenter`).
A **reference** model for a-priori comparisons only: it correlates well
with the exact stress (Liu, Meneveau & Katz 1994 measure ``ρ ≈ 0.6–0.8``
against ``≈ 0.2–0.4`` for Smagorinsky) but dissipates too little to run on
its own, so it is not offered as a closure.

References: Bardina, J., Ferziger, J. H. & Reynolds, W. C. (1980), AIAA
paper 80-1357; Liu, S., Meneveau, C. & Katz, J. (1994), *J. Fluid Mech.*
275, 83.
"""
function bardina_stress!(τ::SymmetricTensorField, ub::Field, vb::Field, wb::Field, filter;
                         C_B::Real = 1, bcs::FieldBoundaryConditions = FieldBoundaryConditions(:periodic))
    subfilter_stress!(τ, ub, vb, wb, filter; bcs)
    if C_B != 1
        for c in components(τ)
            c .*= C_B
        end
    end
    return τ
end

# -------------------------------------------------------------------------
# Filtered strain rate, eddy-viscosity model stresses
# -------------------------------------------------------------------------

function filtered_strain!(S::SymmetricTensorField, ub::Field{CellCenter}, vb::Field{CellCenter}, wb::Field{CellCenter})
    g = mesh(ub)
    a = (interior_view(ub), interior_view(vb), interior_view(wb))
    Nx, Ny, Nz = size(a[1])
    h = (g.dx, g.dy, g.dz)
    D(f, d, i, j, k) = d == 1 ? (f[mod1(i+1,Nx),j,k] - f[mod1(i-1,Nx),j,k]) / (2h[1]) :
                       d == 2 ? (f[i,mod1(j+1,Ny),k] - f[i,mod1(j-1,Ny),k]) / (2h[2]) :
                                (f[i,j,mod1(k+1,Nz)] - f[i,j,mod1(k-1,Nz)]) / (2h[3])
    C = components(S)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        for (n, (p, q)) in enumerate(_PAIRS)
            C[n][i,j,k] = p == q ? D(a[p], p, i, j, k) : 0.5 * (D(a[p], q, i, j, k) + D(a[q], p, i, j, k))
        end
    end
    return S
end
filtered_strain(ub::Field{CellCenter}, vb::Field{CellCenter}, wb::Field{CellCenter}) =
    filtered_strain!(SymmetricTensorField(mesh(ub)), ub, vb, wb)
# Both names are exported and share one docstring.
const _FILTERED_STRAIN_DOC = """
    filtered_strain(ū, v̄, w̄) -> S̄::SymmetricTensorField
    filtered_strain!(S̄, ū, v̄, w̄) -> S̄

Strain rate ``\\bar S_{ij} = \\tfrac12(∂_j\\bar u_i + ∂_i\\bar u_j)`` of a
cell-centred (filtered) velocity, by second-order centred differences
with periodic wrap-around on the interior (halos are not used).
"""
@doc _FILTERED_STRAIN_DOC filtered_strain!
@doc _FILTERED_STRAIN_DOC filtered_strain

# Pointwise contraction A_ij B_ij of two symmetric tensors.
_contract(A, B, I) = A.xx[I]*B.xx[I] + A.yy[I]*B.yy[I] + A.zz[I]*B.zz[I] +
                     2 * (A.xy[I]*B.xy[I] + A.xz[I]*B.xz[I] + A.yz[I]*B.yz[I])

_strain_mag(S, I) = sqrt(2 * _contract(S, S, I))

"""
    eddy_viscosity_stress!(τ, S̄, νt) -> τ

The deviatoric model stress of an eddy-viscosity closure,
``τ_{ij} = -2 ν_t \\bar S_{ij}`` (Boussinesq hypothesis), with `νt` a
number or an array of the interior size (e.g. the interior of the `νt`
field a closure filled, interpolated to the analysis grid).
"""
function eddy_viscosity_stress!(τ::SymmetricTensorField, S::SymmetricTensorField, νt)
    for (t, s) in zip(components(τ), components(S))
        t .= -2 .* νt .* s
    end
    return τ
end

"""
    smagorinsky_stress!(τ, S̄, C_s, Δ) -> τ

Smagorinsky (1963) model stress ``τ_{ij} = -2 (C_sΔ)^2 |\\bar S| \\bar S_{ij}``,
``|\\bar S| = \\sqrt{2\\bar S_{ij}\\bar S_{ij}}``, evaluated on a filtered
field for a-priori tests (`Δ` the explicit filter width).
"""
function smagorinsky_stress!(τ::SymmetricTensorField, S::SymmetricTensorField, C_s::Real, Δ::Real)
    νt = similar(S.xx)
    @inbounds for I in eachindex(νt)
        νt[I] = (C_s * Δ)^2 * _strain_mag(S, I)
    end
    return eddy_viscosity_stress!(τ, S, νt)
end

# -------------------------------------------------------------------------
# Dissipation and a-priori metrics
# -------------------------------------------------------------------------

# Deviatoric copy of a stress.
function _deviatoric(τ::SymmetricTensorField)
    d = similar(τ)
    tr = (τ.xx .+ τ.yy .+ τ.zz) ./ 3
    d.xx .= τ.xx .- tr; d.yy .= τ.yy .- tr; d.zz .= τ.zz .- tr
    d.xy .= τ.xy; d.xz .= τ.xz; d.yz .= τ.yz
    return d
end

"""
    sgs_dissipation(τ, S̄::SymmetricTensorField) -> Π
    sgs_dissipation(τ, (ū, v̄, w̄)) -> Π

Pointwise SGS dissipation (the resolved-to-subfilter energy transfer)

```math
Π = -τ_{ij}\\bar S_{ij},
```

positive for forward scatter. Only the deviatoric part of ``τ`` is used
(the isotropic part does no work against a solenoidal strain; using it
would add the round-off ``τ_{kk}\\bar S_{ii}/3`` of the discrete
divergence). The second form computes ``\\bar S`` with
[`filtered_strain`](@ref). For the Smagorinsky model
``Π = (C_sΔ)^2|\\bar S|^3 ≥ 0`` everywhere.

Reference: Pope, S. B. (2000), *Turbulent Flows*, eq. 13.123.
"""
function sgs_dissipation(τ::SymmetricTensorField, S::SymmetricTensorField)
    d = _deviatoric(τ)
    Π = similar(S.xx)
    @inbounds for I in eachindex(Π)
        Π[I] = -_contract(d, S, I)
    end
    return Π
end
sgs_dissipation(τ::SymmetricTensorField, ubar::NTuple{3,Field{CellCenter}}) =
    sgs_dissipation(τ, filtered_strain(ubar...))

_mean(a) = sum(a) / length(a)
function _corr(a, b)
    ma = _mean(a); mb = _mean(b)
    sab = 0.0; saa = 0.0; sbb = 0.0
    @inbounds for I in eachindex(a, b)
        x = a[I] - ma; y = b[I] - mb
        sab += x * y; saa += x * x; sbb += y * y
    end
    return (saa > 0 && sbb > 0) ? sab / sqrt(saa * sbb) : NaN
end

"""
    stress_eigvals(τ::SymmetricTensorField) -> (λ1, λ2, λ3)

Pointwise eigenvalues (ascending, three arrays) of a symmetric tensor
field, by the closed form of Smith (1961) with a deflation step for
near-degenerate pairs: the package's one 3 × 3 symmetric eigensolver,
[`_sym3_eigvals`](@ref) (`operators/sigma_closure.jl`, shared with the
σ-model). For the subfilter stress of a positive filter ``λ_1 ≥ 0``
(Vreman et al. 1994).
"""
function stress_eigvals(τ::SymmetricTensorField{T}) where {T}
    λ = ntuple(_ -> similar(τ.xx), 3)
    @inbounds for I in eachindex(τ.xx)
        e = _sym3_eigvals(τ.xx[I], τ.yy[I], τ.zz[I], τ.xy[I], τ.xz[I], τ.yz[I])   # descending
        λ[1][I] = e[3]; λ[2][I] = e[2]; λ[3][I] = e[1]
    end
    return λ
end

"""
    lumley_invariants(τ::SymmetricTensorField) -> (II, III)

Pointwise invariants of the anisotropy tensor
``b_{ij} = τ_{ij}/τ_{kk} - δ_{ij}/3`` (Lumley 1978):
``II = b_{ij}b_{ji}``, ``III = b_{ij}b_{jk}b_{ki}``. A realizable
(positive semi-definite) stress lies in the Lumley triangle,
``0 ≤ II ≤ 2/3``; in Pope's variables ``η^2 = II/6``, ``ξ^3 = III/6`` it
is bounded by the axisymmetric lines ``|ξ| = η`` and the two-component
line ``η^2 = 1/27 + 2ξ^3``. Points with
``τ_{kk} ≤ 0`` give `NaN`.

Reference: Lumley, J. L. (1978), *Adv. Appl. Mech.* 18, 123; Pope, S. B.
(2000), *Turbulent Flows*, §11.3.
"""
function lumley_invariants(τ::SymmetricTensorField)
    II = similar(τ.xx); III = similar(τ.xx)
    @inbounds for I in eachindex(II)
        tr = τ.xx[I] + τ.yy[I] + τ.zz[I]
        if !(tr > 0)
            II[I] = NaN; III[I] = NaN; continue
        end
        b11 = τ.xx[I] / tr - 1/3; b22 = τ.yy[I] / tr - 1/3; b33 = τ.zz[I] / tr - 1/3
        b12 = τ.xy[I] / tr; b13 = τ.xz[I] / tr; b23 = τ.yz[I] / tr
        II[I] = b11^2 + b22^2 + b33^2 + 2 * (b12^2 + b13^2 + b23^2)
        III[I] = b11^3 + b22^3 + b33^3 + 3 * (b12^2 * (b11 + b22) + b13^2 * (b11 + b33) + b23^2 * (b22 + b33)) +
                 6 * b12 * b13 * b23
    end
    return II, III
end

# Inside the Lumley triangle (Pope's η–ξ form), with a relative slack.
function _in_lumley_triangle(II, III; tol = 1e-10)
    isnan(II) && return false
    η = sqrt(max(II, 0) / 6); ξ = cbrt(III / 6)
    return abs(ξ) <= η + tol && η^2 <= 1/27 + 2ξ^3 + tol
end

"""
    apriori_metrics(τ_model, τ_exact, S̄) -> NamedTuple

Standard a-priori comparison of a model stress with the exact subfilter
stress on the same filtered field (Clark, Ferziger & Reynolds 1979;
Liu, Meneveau & Katz 1994). All averages are volume averages over the
interior.

* `ρ` — per-component correlation coefficients of the **deviatoric**
  stresses, a NamedTuple `(xx, yy, zz, xy, xz, yz)`; an eddy-viscosity
  model is deviatoric by construction, so the trace, which it does not
  predict, is removed from both.
* `ρ_offdiag` — the mean of the three off-diagonal correlations.
* `ρ_Π` — correlation of the pointwise dissipations
  ``Π = -τ_{ij}\\bar S_{ij}`` (see [`sgs_dissipation`](@ref)).
* `Π_ratio` — ``⟨Π_{model}⟩/⟨Π_{exact}⟩``, the net energy-transfer ratio.
* `backscatter_exact`, `backscatter_model` — the fraction of points with
  ``Π < 0``.
* `Π_exact`, `Π_model` — the mean dissipations.
* `II`, `III` — volume means of the Lumley invariants of the exact stress
  ([`lumley_invariants`](@ref)); `realizable` — the fraction of points of
  the exact stress inside the Lumley triangle.
"""
function apriori_metrics(τm::SymmetricTensorField, τe::SymmetricTensorField, S::SymmetricTensorField)
    dm = _deviatoric(τm); de = _deviatoric(τe)
    names = (:xx, :yy, :zz, :xy, :xz, :yz)
    ρ = NamedTuple{names}(Tuple(_corr(a, b) for (a, b) in zip(components(dm), components(de))))
    Πm = sgs_dissipation(τm, S); Πe = sgs_dissipation(τe, S)
    II, III = lumley_invariants(τe)
    ok = [_in_lumley_triangle(II[I], III[I]) for I in eachindex(II)]
    fin = .!isnan.(II)
    return (; ρ, ρ_offdiag = (ρ.xy + ρ.xz + ρ.yz) / 3,
              ρ_Π = _corr(Πm, Πe), Π_ratio = _mean(Πm) / _mean(Πe),
              Π_model = _mean(Πm), Π_exact = _mean(Πe),
              backscatter_model = count(<(0), Πm) / length(Πm),
              backscatter_exact = count(<(0), Πe) / length(Πe),
              II = _mean(II[fin]), III = _mean(III[fin]),
              realizable = count(ok) / length(ok))
end

"""
    dissipation_matched_cs(τ_exact, S̄, Δ) -> C_s

The Smagorinsky coefficient whose mean dissipation equals the exact one,

```math
C_s^2 = \\frac{⟨Π_{exact}⟩}{Δ^2 ⟨|\\bar S|^3⟩},
```

the a-priori estimate of Clark, Ferziger & Reynolds (1979) (the value
behind Lilly's 1967 ``C_s ≈ 0.17`` for a sharp inertial-range cut-off).
Returns `NaN` if the exact mean transfer is not positive.
"""
function dissipation_matched_cs(τe::SymmetricTensorField, S::SymmetricTensorField, Δ::Real)
    Πe = _mean(sgs_dissipation(τe, S))
    S3 = _mean([_strain_mag(S, I)^3 for I in eachindex(S.xx)])
    return Πe > 0 ? sqrt(Πe / (Δ^2 * S3)) : NaN
end
