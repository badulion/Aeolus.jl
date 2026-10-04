# Explicit LES filters (Phase 54, P54-c).
#
# Included into `Operators`. The box filter is the test filter of the
# dynamic Smagorinsky model (`_dynamic_smagorinsky!` sums its stencil
# through `_box_stencil_sum`, so the two cannot drift); the Gaussian and
# spectral cut-off filters work through FFTW on triply periodic fields.

# -------------------------------------------------------------------------
# Box (top-hat) filter
# -------------------------------------------------------------------------

# Sum `g(a, b, c)` over the (2rx+1)(2ry+1)(2rz+1) box centred on (i, j, k),
# with `c` (z) the outer and `a` (x) the inner loop. `g` returns a tuple;
# every component is accumulated from zero in the same order, so a caller
# that sums several quantities at once gets, component by component, the
# same floating-point result as summing each one alone.
@inline function _box_stencil_sum(g::G, i, j, k, rx, ry, rz) where {G}
    acc = map(zero, @inline(g(i, j, k)))
    @inbounds for dk in -rz:rz, dj in -ry:ry, di in -rx:rx
        acc = map(+, acc, @inline(g(i + di, j + dj, k + dk)))
    end
    return acc
end

_box_halfwidths(n::Integer) = _box_halfwidths((n, n, n))
function _box_halfwidths(n::NTuple{3,Integer})
    all(m -> m >= 1 && isodd(m), n) ||
        throw(ArgumentError("box filter width must be odd and ≥ 1 in every direction, got $n"))
    return n[1] ÷ 2, n[2] ÷ 2, n[3] ÷ 2
end

"""
    box_filter!(out, f, n = 3; mask = nothing) -> out

Top-hat (box) filter of `f` over `n × n × n` cells (or `n = (nx, ny, nz)`,
each odd), written to the interior of `out`. Works at any
[`Location`](@ref): the stencil is taken on the storage indices of `f`, so
a `FaceX` field is averaged over neighbouring x-faces, and so on. `out`
and `f` must be distinct fields at the same location; the halos of `f`
must be filled and at least `n ÷ 2` wide. The halos of `out` are not
touched.

Without a mask the result is the plain average
``\\bar f_{ijk} = n^{-3} \\sum f``, which preserves the domain mean of a
periodic field. Its second moment is ``(n^2-1)Δ^2/12`` per direction, the
quantity behind the default test-to-grid width ratio of
[`compute_dynamic_smagorinsky_νt!`](@ref) (Lund 1997).

`mask` (a `Bool` array the size of `f.data`, `true` = solid, at the
location of `f`; see [`staggered_solid_mask`](@ref)) restricts the
average to fluid neighbours and renormalises by their number,

```math
\\bar f_{ijk} = \\frac{\\sum_{\\text{fluid}} f}{N_{\\text{fluid}}},
```

so values inside obstacles never enter the result. Solid cells (and
cells with no fluid neighbour) get `0`.

This is the test filter of the dynamic Smagorinsky model (Germano et al.
1991); [`compute_dynamic_smagorinsky_νt!`](@ref) uses the same stencil sum.

Reference: Lund, T. S. (1997), *On the use of discrete filters for large
eddy simulation*, CTR Annual Research Briefs, 83.
"""
function box_filter!(out::Field{L}, f::Field{L}, n = 3; mask = nothing) where {L}
    out === f && throw(ArgumentError("box_filter!: `out` and `f` must be different fields"))
    g = mesh(f)
    g isa RegularGrid || throw(ArgumentError("box_filter!: RegularGrid only"))
    rx, ry, rz = _box_halfwidths(n)
    (g.Hx >= rx && g.Hy >= ry && g.Hz >= rz) ||
        throw(ArgumentError("box_filter!: halo $((g.Hx, g.Hy, g.Hz)) narrower than the filter half-width $((rx, ry, rz))"))
    mask === nothing || size(mask) == size(f.data) ||
        throw(DimensionMismatch("box_filter!: mask has size $(size(mask)), field data $(size(f.data))"))
    _box_filter!(out.data, f.data, g, rx, ry, rz, mask)
    return out
end

function _box_filter!(od, fd::AbstractArray{T}, g, rx, ry, rz, ::Nothing) where {T}
    inv_n = one(T) / ((2rx + 1) * (2ry + 1) * (2rz + 1))
    @inbounds for k in g.Hz+1:g.Hz+g.Nz, j in g.Hy+1:g.Hy+g.Ny, i in g.Hx+1:g.Hx+g.Nx
        (s,) = _box_stencil_sum((a, b, q) -> (@inbounds(fd[a,b,q]),), i, j, k, rx, ry, rz)
        od[i,j,k] = s * inv_n
    end
    return nothing
end

function _box_filter!(od, fd::AbstractArray{T}, g, rx, ry, rz, mask) where {T}
    @inbounds for k in g.Hz+1:g.Hz+g.Nz, j in g.Hy+1:g.Hy+g.Ny, i in g.Hx+1:g.Hx+g.Nx
        if mask[i,j,k]
            od[i,j,k] = zero(T)
            continue
        end
        s, c = _box_stencil_sum((a, b, q) -> @inbounds(mask[a,b,q] ? (zero(T), 0) : (fd[a,b,q], 1)),
                                i, j, k, rx, ry, rz)
        od[i,j,k] = c > 0 ? s / c : zero(T)
    end
    return nothing
end

"""
    staggered_solid_mask(mask, loc::Location) -> BitArray

The solid mask at a staggered location from a cell-centre mask (`true` =
solid, the storage-sized array of an `IBM.SolidMask` or
`CutCell.solid_mask`): a face is solid when either cell it separates is
solid, the rule of `IBM.apply_solid_mask!`. For `CellCenter()` a copy of
`mask` is returned. The outermost storage layer, whose upwind neighbour
does not exist, copies the cell mask.
"""
staggered_solid_mask(mask::AbstractArray{Bool,3}, ::CellCenter) = BitArray(mask)
staggered_solid_mask(mask::AbstractArray{Bool,3}, ::FaceX) = _face_mask(mask, (1, 0, 0))
staggered_solid_mask(mask::AbstractArray{Bool,3}, ::FaceY) = _face_mask(mask, (0, 1, 0))
staggered_solid_mask(mask::AbstractArray{Bool,3}, ::FaceZ) = _face_mask(mask, (0, 0, 1))

function _face_mask(mask, s)
    out = BitArray(mask)
    @inbounds for I in CartesianIndices(mask)
        J = CartesianIndex(I[1] - s[1], I[2] - s[2], I[3] - s[3])
        checkbounds(Bool, mask, J) && (out[I] = mask[I] | mask[J])
    end
    return out
end

# -------------------------------------------------------------------------
# Fourier filters (triply periodic)
# -------------------------------------------------------------------------

# Angular wavenumbers of the rfft / fft axes of an Nx × Ny × Nz interior.
_rfft_k(N, L) = (2π / L) .* (0:N÷2)
_fft_k(N, L) = (2π / L) .* fftfreq(N, N)

function _fourier_filter!(out::Field{L}, f::Field{L}, transfer) where {L}
    g = mesh(f)
    g isa RegularGrid || throw(ArgumentError("Fourier filters: RegularGrid only"))
    fi = interior_view(f)
    F = rfft(Array(fi))
    kx = _rfft_k(g.Nx, g.Lx); ky = _fft_k(g.Ny, g.Ly); kz = _fft_k(g.Nz, g.Lz)
    @inbounds for c in 1:size(F, 3), b in 1:size(F, 2), a in 1:size(F, 1)
        F[a,b,c] *= transfer(kx[a], ky[b], kz[c])
    end
    interior_view(out) .= irfft(F, g.Nx)
    return out
end

"""
    gaussian_filter!(out, f, Δ) -> out

Gaussian filter of width `Δ` (a length, or `(Δx, Δy, Δz)`) applied in
Fourier space, with the transfer function (Pope 2000, table 13.2)

```math
\\hat G(\\mathbf k) = \\exp\\!\\left(-\\frac{k_x^2 Δ_x^2 + k_y^2 Δ_y^2 + k_z^2 Δ_z^2}{24}\\right),
```

i.e. ``\\exp(-k^2Δ^2/24)`` for an isotropic width, the kernel with the
same second moment ``Δ^2/12`` as a top hat of width ``Δ``. The interior of
`f` is treated as **triply periodic** (FFTW); halos are ignored and those
of `out` are not written. Any [`Location`](@ref) works, the filter acting
on the stored samples. The mean (``k = 0``) is preserved exactly.

Reference: Pope, S. B. (2000), *Turbulent Flows*, §13.2.
"""
function gaussian_filter!(out::Field{L}, f::Field{L}, Δ) where {L}
    Δx, Δy, Δz = Δ isa Real ? (Δ, Δ, Δ) : Δ
    cx = Δx^2 / 24; cy = Δy^2 / 24; cz = Δz^2 / 24
    return _fourier_filter!(out, f, (kx, ky, kz) -> exp(-(cx * kx^2 + cy * ky^2 + cz * kz^2)))
end

"""
    spectral_cutoff_filter!(out, f, k_c) -> out

Sharp spectral filter: removes every Fourier mode of the (triply
periodic) interior of `f` with ``|\\mathbf k| > k_c`` (angular wavenumber,
rad m⁻¹) and keeps the rest unchanged. The filter is a projection, so it
is idempotent; the mean is preserved. The natural cut-off of an
``N``-point grid is ``k_c = π/Δ`` (Pope 2000, §13.2); the LES resolution
of a grid twice as coarse is ``k_c = π/(2Δ)``.
"""
function spectral_cutoff_filter!(out::Field{L}, f::Field{L}, k_c::Real) where {L}
    kc2 = float(k_c)^2
    return _fourier_filter!(out, f, (kx, ky, kz) -> kx^2 + ky^2 + kz^2 <= kc2 ? 1.0 : 0.0)
end
