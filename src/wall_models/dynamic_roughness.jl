# The dynamic wall model for partially resolved roughness (P57-e, #190):
# the roughness length of the wall law is z0 = α σ_Δ, σ_Δ the RMS of the
# sub-filter surface height, with α found by requiring the total surface
# drag to be the same at the grid scale Δ and at the test-filter scale
# (Anderson & Meneveau 2011). The test filter is Operators' box /
# spectral cut-off filter (P54-c), the root solver SurfaceExchange's
# Illinois iteration; nothing is restated here.

export SubfilterHeight, subfilter_height, DynamicRoughness,
       surface_gradient_drag, roughness_surface_drag, roughness_drag_residual, dynamic_alpha,
       update_dynamic_roughness!, with_roughness, dynamic_wall_laws, dynamic_wall_laws!

# =========================================================================
# Sub-filter statistics of a surface height map
# =========================================================================

"""
    SubfilterHeight

The statistics of a surface height field ``h(x, y)`` the dynamic
roughness model of [`DynamicRoughness`](@ref) needs, on an `(Nx, Ny)`
grid of spacing `dx`, `dy`, at the grid scale Δ and at the test-filter
scale ``\\hat Δ`` (built by [`subfilter_height`](@ref)):

  * `h`, `ht` — the resolved height ``h_Δ`` and ``h_{\\hat Δ}`` (m);
  * `σ2`, `σ2t` — the sub-filter height variance
    ``σ_Δ^2 = \\overline{(h - h_Δ)^2}`` and
    ``σ_{\\hat Δ}^2 = \\widehat{\\overline{(h - h_{\\hat Δ})^2}}`` (m²);
  * `dhx`, `dhy`, `dhxt`, `dhyt` — the gradients of ``h_Δ`` and
    ``h_{\\hat Δ}`` (second-order central differences);
  * `filter` — `:box` or `:spectral`; `periodic` — whether x and y wrap.
"""
struct SubfilterHeight{T<:AbstractFloat}
    h::Matrix{T}
    σ2::Matrix{T}
    ht::Matrix{T}
    σ2t::Matrix{T}
    dhx::Matrix{T}
    dhy::Matrix{T}
    dhxt::Matrix{T}
    dhyt::Matrix{T}
    dx::T
    dy::T
    filter::Symbol
    periodic::Bool
end

Base.size(s::SubfilterHeight) = size(s.h)

# The test filter of a 2-D map: Operators' box filter (3 × 3 cells, the
# test filter of the dynamic Smagorinsky model) or its sharp spectral
# cut-off at k_c = π/(2Δ), on a one-level grid. Never reimplemented here.
function _test_filter(A::AbstractMatrix, dx::Real, dy::Real, filter::Symbol, periodic::Bool)
    Nx, Ny = size(A)
    T = float(eltype(A))
    g = RegularGrid(T; size = (Nx, Ny, 1), extent = (Nx * dx, Ny * dy, one(T)), halo = (1, 1, 1))
    f = Field(CellCenter(), g); o = Field(CellCenter(), g)
    view(interior_view(f), :, :, 1) .= A
    if filter === :box
        bcs = periodic ? FieldBoundaryConditions(:periodic, :periodic, :wall) :
                         FieldBoundaryConditions(:wall, :wall, :wall)
        fill_halos!(f, bcs)
        box_filter!(o, f, (3, 3, 1))
    elseif filter === :spectral
        periodic || throw(ArgumentError("the spectral test filter needs a periodic surface"))
        spectral_cutoff_filter!(o, f, π / (2 * max(dx, dy)))
    else
        throw(ArgumentError("test filter must be :box or :spectral, got :$filter"))
    end
    return Matrix{T}(view(interior_view(o), :, :, 1))
end

# Second-order central differences of a cell-centred map (one-sided at a
# non-periodic edge).
function _gradients(h::AbstractMatrix{T}, dx, dy, periodic::Bool) where {T}
    Nx, Ny = size(h)
    gx = zeros(T, Nx, Ny); gy = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if periodic
            gx[i, j] = (h[mod1(i + 1, Nx), j] - h[mod1(i - 1, Nx), j]) / (2dx)
            gy[i, j] = (h[i, mod1(j + 1, Ny)] - h[i, mod1(j - 1, Ny)]) / (2dy)
        else
            ie, iw = min(i + 1, Nx), max(i - 1, 1)
            jn, js = min(j + 1, Ny), max(j - 1, 1)
            gx[i, j] = Nx > 1 ? (h[ie, j] - h[iw, j]) / ((ie - iw) * dx) : zero(T)
            gy[i, j] = Ny > 1 ? (h[i, jn] - h[i, js]) / ((jn - js) * dy) : zero(T)
        end
    end
    return gx, gy
end

"""
    subfilter_height(m1, m2; dx, dy, resolved = m1, periodic = true,
                     filter = periodic ? :spectral : :box) -> SubfilterHeight
    subfilter_height(h_fine, r; dx, dy, resolved = nothing, kwargs...)

The sub-filter statistics of a surface height field, from its first two
moments over each grid cell, `m1` ``= \\overline{h}`` and `m2`
``= \\overline{h^2}`` (`(Nx, Ny)` matrices, m and m²), or from a height
map `h_fine` given on a grid `r` times finer in each direction
(`(r Nx, r Ny)`; the moments are its `r × r` block means). `dx`, `dy`
are the spacing of the coarse (LES) grid.

`resolved` is the height the LES represents, ``h_Δ`` (default the cell
mean ``\\overline h``; the Berlin case passes the immersed-boundary
staircase, quantised to Δz). The sub-filter variance about it is

```math
σ_Δ^2 = \\overline{(h - h_Δ)^2} = m_2 - 2 h_Δ m_1 + h_Δ^2 ,
```

and at the test-filter scale, with ``\\hat{\\cdot}`` the test filter
`filter` — `:spectral`, `Operators.spectral_cutoff_filter!` at
``k_c = π/(2Δ)`` (the test-to-grid ratio 2 of Anderson & Meneveau 2011;
the default on a periodic surface), or `:box`, the 3 × 3 box of
`Operators.box_filter!` (ratio 3 for the block-mean height; local, so
it also works on a non-periodic surface) —
``h_{\\hat Δ} = \\hat h_Δ`` and

```math
σ_{\\hat Δ}^2 = \\hat m_2 - 2 h_{\\hat Δ} \\hat m_1 + h_{\\hat Δ}^2 ,
```

the variance of the fine height about ``h_{\\hat Δ}`` over the composite
(cell mean, then test filter) kernel — exact for the box, whose weights
are positive; for the spectral cut-off it is clamped at 0. A flat
surface has ``σ_Δ = σ_{\\hat Δ} = 0``.

Reference: Anderson, W. & Meneveau, C. (2011). Dynamic roughness model
for large-eddy simulation of turbulent flow over multiscale, fractal-like
rough surfaces. *J. Fluid Mech.*, 679, 288-314.
"""
function subfilter_height(m1::AbstractMatrix, m2::AbstractMatrix; dx::Real, dy::Real,
                          resolved = m1, periodic::Bool = true,
                          filter::Symbol = periodic ? :spectral : :box)
    size(m1) == size(m2) == size(resolved) ||
        throw(DimensionMismatch("subfilter_height: m1, m2 and resolved must have the same size"))
    T = float(promote_type(eltype(m1), eltype(m2), eltype(resolved)))
    M1 = Matrix{T}(m1); M2 = Matrix{T}(m2); h = Matrix{T}(resolved)
    σ2 = max.(M2 .- 2 .* h .* M1 .+ h .^ 2, zero(T))
    ht = _test_filter(h, dx, dy, filter, periodic)
    M1t = _test_filter(M1, dx, dy, filter, periodic)
    M2t = _test_filter(M2, dx, dy, filter, periodic)
    σ2t = max.(M2t .- 2 .* ht .* M1t .+ ht .^ 2, zero(T))
    gx, gy = _gradients(h, T(dx), T(dy), periodic)
    gxt, gyt = _gradients(ht, T(dx), T(dy), periodic)
    return SubfilterHeight{T}(h, σ2, ht, σ2t, gx, gy, gxt, gyt, T(dx), T(dy), filter, periodic)
end

function subfilter_height(hf::AbstractMatrix, r::Integer; dx::Real, dy::Real,
                          resolved = nothing, kwargs...)
    r >= 1 || throw(ArgumentError("subfilter_height: refinement r must be ≥ 1"))
    Nf, Mf = size(hf)
    (Nf % r == 0 && Mf % r == 0) ||
        throw(DimensionMismatch("subfilter_height: fine map $(size(hf)) is not a multiple of r = $r"))
    Nx, Ny = Nf ÷ r, Mf ÷ r
    T = float(eltype(hf))
    m1 = zeros(T, Nx, Ny); m2 = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx, b in 1:r, a in 1:r
        x = T(hf[(i - 1) * r + a, (j - 1) * r + b])
        m1[i, j] += x; m2[i, j] += x^2
    end
    m1 ./= r^2; m2 ./= r^2
    return subfilter_height(m1, m2; dx = dx, dy = dy,
                            resolved = resolved === nothing ? m1 : resolved, kwargs...)
end

# =========================================================================
# The resolved form drag: the surface-gradient drag model
# =========================================================================

"""
    surface_gradient_drag(u, v, dhdx, dhdy; C_d = 1.0) -> (fx, fy)

The kinematic force per unit planform area (m² s⁻²) a resolved surface of
slope ``∇h = (∂_x h, ∂_y h)`` exerts on the flow of velocity ``(u, v)``
next to it, in the surface-gradient-based drag model of Anderson &
Meneveau (2011, §2):

```math
f_i = -C_d\\,(\\tilde{\\mathbf u}\\cdot∇h)^2\\,H(\\tilde{\\mathbf u}\\cdot∇h)\\,\\frac{∂h}{∂x_i},
```

the pressure drag on the windward (``\\tilde{\\mathbf u}\\cdot∇h > 0``)
faces of the resolved relief; leeward faces exert none. It is zero on a
flat surface and quadratic in the wind. `C_d` is a model constant
(default 1; the drag on resolved relief is the part of the total drag
the dynamic procedure of [`DynamicRoughness`](@ref) balances against the
modelled stress, so its value changes α, not the procedure).

Reference: Anderson, W. & Meneveau, C. (2011). *J. Fluid Mech.*, 679,
288-314.
"""
@inline function surface_gradient_drag(u::Real, v::Real, gx::Real, gy::Real; C_d::Real = 1.0)
    p = u * gx + v * gy
    p > 0 || return (zero(float(p)), zero(float(p)))
    c = -C_d * p^2
    return (c * gx, c * gy)
end

# =========================================================================
# The dynamic procedure
# =========================================================================

"""
    DynamicRoughness(stats::SubfilterHeight; z0_floor, C_d = 1.0,
                     α = nothing, α_fallback = 0.0, mask = nothing,
                     α_max = nothing, c = DEFAULT_SURFACE_CONSTANTS)

The dynamic surface roughness model of Anderson & Meneveau (2011) for a
surface that is **partially resolved**: the relief larger than the grid
(``h_Δ``) is resolved, the rest enters the wall law as a roughness
length proportional to the RMS sub-filter height,

```math
z_{0,Δ}(x, y) = \\sqrt{z_{0,f}^2 + α^2 σ_Δ^2(x, y)} ,
```

where ``z_{0,f}`` (`z0_floor`, m) is the roughness of the scales below
the height map itself (the per-class table of the Berlin case; for the
pure Anderson & Meneveau model a negligible value). A flat surface
(``σ_Δ ≡ 0``) returns ``z_{0,f}`` exactly.

**The dynamic α.** The total surface drag — modelled wall stress plus the
form drag on the resolved relief ([`surface_gradient_drag`](@ref)) — must
not depend on how much of the surface is resolved. Evaluated at the grid
scale Δ and at the test-filter scale ``\\hat Δ`` (the filter of `stats`),
the plane-mean drag along the mean wind ``\\hat{\\mathbf e}``,

```math
D_Δ(α) = \\Big\\langle \\Big[\\frac{κ}{\\ln(z/z_{0,Δ}(α))}\\Big]^2 |\\tilde{\\mathbf u}|\\,\\tilde{\\mathbf u}\\cdot\\hat{\\mathbf e}
          + \\mathbf f(\\tilde{\\mathbf u}, ∇h_Δ)\\cdot(-\\hat{\\mathbf e}) \\Big\\rangle ,
```

and ``D_{\\hat Δ}`` the same with the test-filtered velocity, ``h_{\\hat Δ}``
and ``σ_{\\hat Δ}``, gives α as the root of

```math
D_{\\hat Δ}(α) - D_Δ(α) = 0
```

(Anderson & Meneveau 2011, their eq. 3.3, with the neutral log law of
[`RoughLogLaw`](@ref)), solved per plane with SurfaceExchange's Illinois
iteration on the bracket ``(0, α_{max})``, ``α_{max}`` the largest α for
which every ``z_0 < z``. A plane-mean residual with no sign change in the
bracket keeps `α_fallback` (default 0, i.e. ``z_0 = z_{0,f}``) and marks
the solve unconverged.

`α = <number>` fixes α instead (the non-dynamic model, e.g. a value
tuned at one resolution). `mask` (`(Nx, Ny)` `Bool`, `true` = included)
restricts the plane means to the columns with a wall. κ is
`SurfaceExchange.von_karman(c)`.

Fields after [`update_dynamic_roughness!`](@ref): `α`, `converged`, `z0`
(the `(Nx, Ny)` roughness lengths, m), `nsolves`, `nfail`.
[`dynamic_wall_laws`](@ref) / [`with_roughness`](@ref) turn them into a
per-column [`RoughLogLaw`](@ref) or [`MOSTWallLaw`](@ref).

# References

  * Anderson, W. & Meneveau, C. (2011). Dynamic roughness model for
    large-eddy simulation of turbulent flow over multiscale, fractal-like
    rough surfaces. *J. Fluid Mech.*, 679, 288-314.
  * Macdonald, R. W., Griffiths, R. F. & Hall, D. J. (1998). An improved
    method for the estimation of surface roughness of obstacle arrays.
    *Atmos. Environ.*, 32, 1857-1864.
  * Cheng, H. & Castro, I. P. (2002). Near wall flow over urban-like
    roughness. *Boundary-Layer Meteorol.*, 104, 229-259.
"""
mutable struct DynamicRoughness{T<:AbstractFloat, C<:SurfaceConstants}
    stats::SubfilterHeight{T}
    z0_floor::T
    C_d::T
    fixed_α::Union{Nothing, T}
    α_fallback::T
    α_max::Union{Nothing, T}
    mask::Union{Nothing, Matrix{Bool}}
    c::C
    α::T
    converged::Bool
    z0::Matrix{T}
    nsolves::Int
    nfail::Int
end

function DynamicRoughness(stats::SubfilterHeight{T}; z0_floor::Real, C_d::Real = 1.0,
                          α = nothing, α_fallback::Real = 0.0, mask = nothing,
                          α_max = nothing,
                          c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) where {T}
    z0_floor > 0 || throw(ArgumentError("DynamicRoughness: z0_floor must be > 0"))
    C_d >= 0 || throw(ArgumentError("DynamicRoughness: C_d must be ≥ 0"))
    mask === nothing || size(mask) == size(stats) ||
        throw(DimensionMismatch("DynamicRoughness: mask size $(size(mask)) ≠ $(size(stats))"))
    fa = α === nothing ? nothing : T(α)
    a0 = fa === nothing ? T(α_fallback) : fa
    dr = DynamicRoughness{T, typeof(c)}(stats, T(z0_floor), T(C_d), fa, T(α_fallback),
                                        α_max === nothing ? nothing : T(α_max),
                                        mask === nothing ? nothing : Matrix{Bool}(mask), c,
                                        a0, fa !== nothing, zeros(T, size(stats)), 0, 0)
    _set_z0!(dr, a0)
    return dr
end

# z0 = √(z0f² + α²σ²); exactly z0f where σ = 0.
@inline _z0(z0f, α, σ2) = σ2 > 0 ? sqrt(z0f^2 + α^2 * σ2) : z0f

function _set_z0!(dr::DynamicRoughness, α)
    s = dr.stats
    @inbounds for I in eachindex(dr.z0)
        dr.z0[I] = _z0(dr.z0_floor, α, s.σ2[I])
    end
    dr.α = α
    return dr
end

@inline _in(::Nothing, I) = true
@inline _in(m::AbstractMatrix{Bool}, I) = @inbounds m[I]
@inline _hval(h::Real, I) = h
@inline _hval(h::AbstractMatrix, I) = @inbounds h[I]

# The parts of the plane-mean drag at both scales that do not depend on α:
# per column A = κ² |u| u·ê (the modelled stress is A/ln²(z/z0)), and the
# plane-mean resolved drag along ê, R_Δ and R_Δ̂.
function _drag_parts(dr::DynamicRoughness{T}, uc, vc, ut, vt) where {T}
    s = dr.stats
    κ = T(von_karman(dr.c))
    n = 0; ub = zero(T); vb = zero(T)
    @inbounds for I in eachindex(uc)
        _in(dr.mask, I) || continue
        ub += uc[I]; vb += vc[I]; n += 1
    end
    n > 0 || throw(ArgumentError("DynamicRoughness: no column in the mask"))
    M = hypot(ub, vb)
    ex, ey = M > 0 ? (ub / M, vb / M) : (one(T), zero(T))
    A1 = zeros(T, size(uc)); A2 = zeros(T, size(uc))
    R1 = zero(T); R2 = zero(T)
    @inbounds for I in eachindex(uc)
        _in(dr.mask, I) || continue
        u1, v1, u2, v2 = T(uc[I]), T(vc[I]), T(ut[I]), T(vt[I])
        A1[I] = κ^2 * hypot(u1, v1) * (u1 * ex + v1 * ey)
        A2[I] = κ^2 * hypot(u2, v2) * (u2 * ex + v2 * ey)
        f1 = surface_gradient_drag(u1, v1, s.dhx[I], s.dhy[I]; C_d = dr.C_d)
        f2 = surface_gradient_drag(u2, v2, s.dhxt[I], s.dhyt[I]; C_d = dr.C_d)
        R1 -= f1[1] * ex + f1[2] * ey
        R2 -= f2[1] * ex + f2[2] * ey
    end
    return A1, A2, R1 / n, R2 / n, n
end

# The residual F(α) = D_Δ̂(α) − D_Δ(α), from the parts.
function _residual(dr::DynamicRoughness{T}, α, A1, A2, R1, R2, n, h) where {T}
    s = dr.stats; z0f = dr.z0_floor
    W = zero(T)
    @inbounds for I in eachindex(A1)
        _in(dr.mask, I) || continue
        z = T(_hval(h, I))
        W += A2[I] / log(z / _z0(z0f, α, s.σ2t[I]))^2 - A1[I] / log(z / _z0(z0f, α, s.σ2[I]))^2
    end
    return W / n + (R2 - R1)
end

# Test-filtered column velocities (the filter of the statistics).
_filtered_velocity(dr::DynamicRoughness, uc, vc) =
    (_test_filter(uc, dr.stats.dx, dr.stats.dy, dr.stats.filter, dr.stats.periodic),
     _test_filter(vc, dr.stats.dx, dr.stats.dy, dr.stats.filter, dr.stats.periodic))

"""
    roughness_surface_drag(dr::DynamicRoughness, uc, vc, h; α = dr.α,
                           scale = :grid) -> (wall, form, total)

The plane-mean total surface drag along the mean wind (kinematic,
m² s⁻²) that the model of [`DynamicRoughness`](@ref) predicts for the
cell-centre velocities `uc`, `vc` at distance `h` (m) from the surface,
split into the modelled wall stress
``⟨[κ/\\ln(h/z_0(α))]^2 |\\tilde{\\mathbf u}|\\,\\tilde{\\mathbf u}\\cdot\\hat{\\mathbf e}⟩``
and the resolved form drag ([`surface_gradient_drag`](@ref)). `scale =
:grid` evaluates it with ``h_Δ``, ``σ_Δ`` and the velocities as given;
`:test` with ``h_{\\hat Δ}``, ``σ_{\\hat Δ}`` and the test-filtered
velocities. ``\\sqrt{\\text{total}}`` is the plane-mean friction velocity
of the surface. The dynamic α is the one for which the two scales agree.
"""
function roughness_surface_drag(dr::DynamicRoughness{T}, uc::AbstractMatrix, vc::AbstractMatrix,
                                h; α::Real = dr.α, scale::Symbol = :grid) where {T}
    scale in (:grid, :test) || throw(ArgumentError("scale must be :grid or :test, got :$scale"))
    ut, vt = _filtered_velocity(dr, uc, vc)
    A1, A2, R1, R2, n = _drag_parts(dr, uc, vc, ut, vt)
    s = dr.stats
    A, R, σ2 = scale === :grid ? (A1, R1, s.σ2) : (A2, R2, s.σ2t)
    W = zero(T)
    @inbounds for I in eachindex(A)
        _in(dr.mask, I) || continue
        W += A[I] / log(T(_hval(h, I)) / _z0(dr.z0_floor, T(α), σ2[I]))^2
    end
    return (wall = W / n, form = R, total = W / n + R)
end

"""
    roughness_drag_residual(dr::DynamicRoughness, α, uc, vc, h) -> F

The residual ``F(α) = D_{\\hat Δ}(α) - D_Δ(α)`` (m² s⁻²) of the dynamic
procedure of [`DynamicRoughness`](@ref) for the cell-centre velocities
`uc`, `vc` (`(Nx, Ny)`, m s⁻¹) at the distance `h` from the surface (a
number or an `(Nx, Ny)` matrix, m): the total-drag mismatch between the
test-filter and the grid scale. [`dynamic_alpha`](@ref) finds its root.
"""
function roughness_drag_residual(dr::DynamicRoughness, α::Real, uc::AbstractMatrix,
                                 vc::AbstractMatrix, h)
    ut, vt = _filtered_velocity(dr, uc, vc)
    A1, A2, R1, R2, n = _drag_parts(dr, uc, vc, ut, vt)
    return _residual(dr, float(α), A1, A2, R1, R2, n, h)
end

"""
    dynamic_alpha(dr::DynamicRoughness, uc, vc, h; maxiter = 200) -> (α, converged)

The dynamic α of Anderson & Meneveau (2011) for the cell-centre
velocities `uc`, `vc` (`(Nx, Ny)`, m s⁻¹) at distance `h` (m; number or
matrix) from the surface: the root of [`roughness_drag_residual`](@ref)
on ``(0, α_{max})`` by the Illinois iteration of `SurfaceExchange` (the
shared root solver; a log-spaced scan of 48 points finds the first
bracket with ``F < 0 < F``). A flat surface (``σ ≡ 0``), or a residual
with no sign change (including one that vanishes identically, e.g. a
periodic relief the grid cannot resolve: then α is undetermined),
returns `(dr.α_fallback, false)`; a fixed-α model
returns `(α, true)`.
"""
function dynamic_alpha(dr::DynamicRoughness{T}, uc::AbstractMatrix, vc::AbstractMatrix, h;
                       maxiter::Integer = 200) where {T}
    dr.fixed_α === nothing || return (dr.fixed_α, true)
    s = dr.stats; z0f = dr.z0_floor
    # the admissible range: every z0 below its matching height
    amax = T(Inf); rough = false
    @inbounds for I in eachindex(s.σ2)
        _in(dr.mask, I) || continue
        z = T(_hval(h, I))
        z > z0f || throw(ArgumentError("DynamicRoughness: z0_floor = $z0f ≥ matching height $z"))
        smax = max(s.σ2[I], s.σ2t[I])
        smax > 0 || continue
        rough = true
        amax = min(amax, sqrt((z^2 - z0f^2) / smax))
    end
    rough || return (dr.α_fallback, false)               # flat: no sub-filter relief
    amax *= 1 - T(1e-6)
    dr.α_max === nothing || (amax = min(amax, dr.α_max))
    ut, vt = _filtered_velocity(dr, uc, vc)
    A1, A2, R1, R2, n = _drag_parts(dr, uc, vc, ut, vt)
    F(α) = _residual(dr, α, A1, A2, R1, R2, n, h)
    # first bracket F(a) < 0 < F(b) on a log-spaced scan of (1e-8 α_max, α_max)
    nscan = 48
    a = amax * T(1e-8); fa = F(a)
    for m in 1:nscan
        b = amax * T(10)^(T(-8) * (1 - T(m) / nscan))
        fb = F(b)
        if fa < 0 && fb > 0
            α, _, conv = SurfaceExchange._illinois(F, a, b, fa, fb, maxiter)
            return (α, conv)
        elseif fa < 0 && fb == 0
            return (b, true)
        end
        a, fa = b, fb
    end
    return (dr.α_fallback, false)
end

"""
    update_dynamic_roughness!(dr::DynamicRoughness, uc, vc, h) -> dr

Solve [`dynamic_alpha`](@ref) for the current velocities and set
`dr.α`, `dr.converged` and the per-column roughness lengths `dr.z0`
(``\\sqrt{z_{0,f}^2 + α^2σ_Δ^2}``).
"""
function update_dynamic_roughness!(dr::DynamicRoughness, uc::AbstractMatrix,
                                   vc::AbstractMatrix, h)
    α, conv = dynamic_alpha(dr, uc, vc, h)
    dr.nsolves += 1
    conv || (dr.nfail += 1)
    dr.converged = conv
    _set_z0!(dr, α)
    return dr
end

"""
    with_roughness(law, z0) -> law′

The wall law `law` with its momentum roughness length replaced by `z0`
(m), everything else kept: a [`RoughLogLaw`](@ref) (its constants `c`)
or a [`MOSTWallLaw`](@ref) (stability functions, Obukhov length, `U_min`,
constants). The parameter-provider interface of
[`DynamicRoughness`](@ref).
"""
with_roughness(law::RoughLogLaw, z0::Real) = RoughLogLaw(z0; c = law.c)
with_roughness(law::MOSTWallLaw, z0::Real) =
    MOSTWallLaw(law.sf, z0, law.L; U_min = law.U_min, c = law.c)

"""
    dynamic_wall_laws(dr::DynamicRoughness, law) -> Matrix
    dynamic_wall_laws!(laws, dr::DynamicRoughness)

Per-column wall laws with the roughness lengths `dr.z0`:
`with_roughness(law, dr.z0[i, j])` for a template `law`
([`RoughLogLaw`](@ref) or [`MOSTWallLaw`](@ref)), ready for a
[`WallModel`](@ref). The in-place form updates an existing matrix of
laws (e.g. one `MOSTWallLaw` per column, each with its own `L`).
"""
dynamic_wall_laws(dr::DynamicRoughness, law::AbstractWallLaw) =
    [with_roughness(law, z) for z in dr.z0]

function dynamic_wall_laws!(laws::AbstractMatrix{<:AbstractWallLaw}, dr::DynamicRoughness)
    size(laws) == size(dr.z0) || throw(DimensionMismatch("dynamic_wall_laws!: size mismatch"))
    @inbounds for I in eachindex(laws)
        laws[I] = with_roughness(laws[I], dr.z0[I])
    end
    return laws
end
