# =========================================================================
# Phase 55 (P55-b, #175) — the stress-form SGS momentum divergence.
# =========================================================================
#
# Included into `module Operators`, after the FV kernels. One momentum-
# stress operator, `add_stress_divergence!(tu, tv, tw, τ)`, for any
# symmetric stress τ placed on the C-grid (diagonals at cell centres,
# off-diagonals on the edges); `add_sgs_stress_divergence!` is the eddy-
# viscosity special case τᵢⱼ = −2νᵢⱼSᵢⱼ with anisotropic (νh, νv), fused
# (the stresses formed on the fly) and equal to the generic operator
# applied to `sgs_stress_tensor!` to round-off (bit for bit, in fact).
# The kernels are in FiniteVolume.jl.

"""
    StressTensor(mesh)
    StressTensor(τ11, τ22, τ33, τ12, τ13, τ23)

The six independent components of a symmetric stress tensor `τᵢⱼ`
(m² s⁻², kinematic) on the C-grid, each stored in an array shaped like a
`Field{CellCenter}`'s `data` (halos included), with the placement of the
fluxes of [`add_eddy_viscosity_u!`](@ref):

| component | location of `τ[i, j, k]` |
| :-- | :-- |
| `τ11`, `τ22`, `τ33` | the centre of cell `(i, j, k)` |
| `τ12` | the xy-edge `(x_{i-½}, y_{j-½}, z_k)` (where u-face `i` meets v-face `j`) |
| `τ13` | the xz-edge `(x_{i-½}, y_j, z_{k-½})` |
| `τ23` | the yz-edge `(x_i, y_{j-½}, z_{k-½})` |

[`add_stress_divergence!`](@ref) reads the diagonals on the interior plus
the shell of cells below it in their own direction (`τ11` on
`i ∈ first-1:last`) and the off-diagonals on the interior plus the shell of
edges above it in both of their directions (`τ12` on `i, j ∈ first:last+1`);
[`sgs_stress_tensor!`](@ref) fills exactly those ranges. The `mesh` form
allocates zeros.
"""
struct StressTensor{A<:AbstractArray{<:Real,3}}
    τ11::A
    τ22::A
    τ33::A
    τ12::A
    τ13::A
    τ23::A
end
function StressTensor(g::RegularGrid{T}) where {T}
    z() = zeros(T, size(Field(CellCenter(), g).data))
    return StressTensor(z(), z(), z(), z(), z(), z())
end

"""
    add_stress_divergence!(tu, tv, tw, τ::StressTensor)

Add the divergence of a symmetric kinematic stress, `−∂τᵢⱼ/∂xⱼ`, to the
face-centred momentum tendencies (`tu @ FaceX`, `tv @ FaceY`,
`tw @ FaceZ`). The generic momentum-stress operator of the SGS closures: the
diagonal stresses at cell centres are differenced to the faces of their own
component, the off-diagonal stresses on the edges to the two faces they
border (placement in [`StressTensor`](@ref)). It is conservative by
construction — every stress value enters two neighbouring faces with
opposite signs — so on a periodic domain `Σ tu = Σ tv = Σ tw = 0` to
round-off for any τ.

With `τ =` [`sgs_stress_tensor!`](@ref)`(τ, u, v, w, νh, νv)` it equals
[`add_sgs_stress_divergence!`](@ref)`(tu, tv, tw, u, v, w, νh, νv)` bit for
bit. A prognostic or explicit Reynolds stress (P56-d) is passed here
directly.
"""
add_stress_divergence!(tu, tv, tw, τ::StressTensor) =
    (_add_stress_divergence!(tu, tv, tw, τ.τ11, τ.τ22, τ.τ33, τ.τ12, τ.τ13, τ.τ23, mesh(tu));
     nothing)

_coefdata(ν::Field) = ν.data
_coefdata(ν::AbstractArray{<:Real,3}) = ν

"""
    sgs_stress_tensor!(τ::StressTensor, u, v, w, νh, νv = νh) -> τ

The eddy-viscosity stress `τᵢⱼ = −2νᵢⱼSᵢⱼ` on the C-grid (placement and
index ranges in [`StressTensor`](@ref)), from the staggered velocity and
the cell-centred horizontal and vertical eddy viscosities (`Field`s or
halo-padded arrays; halos of all five filled). The strain is the compact
difference at each stress location: `Sᵢᵢ` is the diagonal of
[`_velocity_gradient_at`](@ref) (face to face through the cell), and
`S12 = ½(∂u/∂y + ∂v/∂x)` on the xy-edge from the two `u` and two `v` values
adjacent to it (likewise `S13`, `S23`). `ν` is the cell value for the
diagonals and the 4-cell average onto the edge for the off-diagonals, as in
[`add_eddy_viscosity_u!`](@ref). Anisotropy follows the CM1 convention
(Bryan 2002; also Kosović 1997's split of horizontal and vertical
mixing): `ν11 = ν22 = ν12 = νh`, `ν33 = ν13 = ν23 = νv`.
"""
function sgs_stress_tensor!(τ::StressTensor, u, v, w, νh, νv = νh)
    _sgs_stress_tensor!(τ.τ11, τ.τ22, τ.τ33, τ.τ12, τ.τ13, τ.τ23, u, v, w,
                        _coefdata(νh), _coefdata(νv), mesh(u))
    return τ
end

"""
    add_sgs_stress_divergence!(tu, tv, tw, u, v, w, νh, νv = νh; part = :full)

Add the stress-form SGS momentum divergence

```math
\\frac{\\partial}{\\partial x_j}\\left(2 ν_{ij} S_{ij}\\right)
= \\frac{\\partial}{\\partial x_j}\\left[ν_{ij}\\left(\\frac{\\partial u_i}{\\partial x_j}
  + \\frac{\\partial u_j}{\\partial x_i}\\right)\\right]
```

to the momentum tendencies, with `τᵢⱼ = −2νᵢⱼSᵢⱼ` formed on the fly
(the stencil, placement and the CM1 anisotropy convention of
[`sgs_stress_tensor!`](@ref); `νh`, `νv` cell-centred `Field`s or halo-padded
arrays, halos of everything filled). The isotropic closure is
`νv = νh`. Unlike the Laplacian form of [`add_eddy_viscosity_u!`](@ref),
`∂ⱼ(ν∂ⱼuᵢ)`, it keeps the transposed term `∂ⱼ(ν∂ᵢuⱼ)`, which is
`ν∂ᵢ(∇·u) = 0` for constant ν but not where ν varies; it gives no tendency
for a solid-body rotation, and its kinetic-energy tendency is exactly
`−Σ 2νᵢⱼSᵢⱼSᵢⱼ ΔV ≤ 0` ([`sgs_stress_dissipation`](@ref)) on a periodic
domain.

`part` selects a piece of the split `τᵢⱼ = −νᵢⱼ(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)`:
`:full` (default), `:diagonal` (the Laplacian part `∂ⱼ(νᵢⱼ∂ⱼuᵢ)`, with the
edge interpolation of the Laplacian kernels; this is what the IMEX steppers
treat implicitly) or `:transpose` (the remainder `∂ⱼ(νᵢⱼ∂ᵢuⱼ)`, explicit).
`:diagonal` + `:transpose` = `:full`.

`weights = (Wc, Wf)` (default `nothing`) is the sound-proof form
`(1/W) ∂ⱼ(W τᵢⱼ)` of a core with the per-level constraint weights `W`
(`Anelastic.constraint_weights`, ONE-j #460): the vertical fluxes `τ₁₃`,
`τ₂₃`, `τ₃₃` are weighted, so `Σ W uᵢ ΔV` is conserved. RegularGrid only.

References: Bryan & Fritsch (2002), *Mon. Wea. Rev.* 130, 2917 (CM1);
Kosović (1997), *J. Fluid Mech.* 336, 151; Deardorff (1970),
*J. Fluid Mech.* 41, 453 (the stress form on a staggered grid).
"""
function add_sgs_stress_divergence!(tu, tv, tw, u, v, w, νh, νv = νh; part::Symbol = :full,
                                    weights = nothing)
    a, b = part === :full ? (1, 1) : part === :diagonal ? (1, 0) :
           part === :transpose ? (0, 1) :
           throw(ArgumentError("part must be :full, :diagonal or :transpose, got $(repr(part))"))
    # checked: the kernel reads the weights without bounds checks (#559 review)
    wc, wf = weights === nothing ? (nothing, nothing) : _level_weights(u, weights)
    _add_sgs_stress_divergence!(tu, tv, tw, u, v, w, _coefdata(νh), _coefdata(νv),
                                a, b, mesh(u), wc, wf)
    return nothing
end

"""
    sgs_stress_dissipation(u, v, w, νh, νv = νh) -> D

The resolved-to-subgrid kinetic-energy transfer of the stress-form closure,
`D = Σ 2νᵢⱼSᵢⱼSᵢⱼ ΔV` (m⁵ s⁻³; divide by the domain volume for the mean
dissipation rate in W kg⁻¹), summed over the interior cells for the
diagonal terms and over the edges with interior indices for the
off-diagonal ones, each of which counts for `ij` and `ji`. On a periodic
domain the kinetic-energy tendency of
[`add_sgs_stress_divergence!`](@ref), `Σ (u tu + v tv + w tw) ΔV`, is
exactly `−D`; with walls it differs by the boundary edges.
"""
sgs_stress_dissipation(u, v, w, νh, νv = νh) =
    _sgs_stress_dissipation(u, v, w, _coefdata(νh), _coefdata(νv), mesh(u))

# The `sgs_stress` keyword of the cores (`:laplacian` | `:symmetric`).
function _check_sgs_stress(s::Symbol)
    s === :laplacian || s === :symmetric ||
        throw(ArgumentError("sgs_stress must be :laplacian or :symmetric, got $(repr(s))"))
    return s
end
