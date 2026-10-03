# =========================================================================
# Era VI, Phase 55 (P55-c, #181) — the tensorial scalar eddy diffusivity.
# =========================================================================
#
# Included into `module Operators`, after the FV kernels. The single
# scalar-flux operator of Era VI: `add_tensor_diffusion_scalar!(tϕ, ϕ, K)`
# for any symmetric diffusivity tensor K at cell centres, in the triad
# (functional) discretisation of Griffies et al. (1998). The diagonal
# special case is `add_eddy_diffusion_scalar!(tϕ, ϕ, κh, κv)`; the rotated
# isentropic tensor of Redi (1982) is built by `isentropic_tensor!`.
# P55-e (κh ≠ κv under stratification) and P56-d (the Daly–Harlow heat
# flux Kᵢⱼ = C (e/ε) τᵢⱼ) reuse this operator. The kernels are in
# FiniteVolume.jl.

"""
    DiffusivityTensor(mesh)
    DiffusivityTensor(K11, K22, K33, K12, K13, K23)

The six independent components of a symmetric scalar eddy-diffusivity
tensor `Kᵢⱼ` (m² s⁻¹), each a `Field{CellCenter}` (all six at the centre of
the cell, halos included), plus the face-flux scratch
[`add_tensor_diffusion_scalar!`](@ref) uses. The `mesh` form allocates
zeros. Halos are filled with `fill_halos!(K, bcs)` (each component with the
same boundary conditions, as for a cell-centred eddy coefficient).

For the down-gradient (variance-dissipating) property of the operator every
cell's `K` must be positive semi-definite; [`isentropic_tensor!`](@ref)
builds one that is.
"""
struct DiffusivityTensor{F<:Field{CellCenter}, A<:AbstractArray}
    K11::F
    K22::F
    K33::F
    K12::F
    K13::F
    K23::F
    flux::NTuple{3,A}             # face-flux scratch (x, y, z)
end
function DiffusivityTensor(K11::F, K22::F, K33::F, K12::F, K13::F, K23::F) where {F<:Field{CellCenter}}
    z() = zero(K11.data)
    return DiffusivityTensor(K11, K22, K33, K12, K13, K23, (z(), z(), z()))
end
function DiffusivityTensor(g::RegularGrid)
    c() = Field(CellCenter(), g)
    return DiffusivityTensor(c(), c(), c(), c(), c(), c())
end

_components(K::DiffusivityTensor) = (K.K11, K.K22, K.K33, K.K12, K.K13, K.K23)

"""
    fill_halos!(K::DiffusivityTensor, bcs) -> K

Fill the halos of the six components of `K` with `bcs` (the boundary
conditions of the cell-centred eddy coefficients).
"""
function BoundaryConditions.fill_halos!(K::DiffusivityTensor, bcs)
    foreach(c -> fill_halos!(c, bcs), _components(K))
    return K
end

"""
    diagonal_tensor!(K::DiffusivityTensor, κh, κv = κh) -> K

Set `K = diag(κh, κh, κv)` (fields or numbers; halos included, so a
field's halos must be filled): the tensor of the diagonal operator
[`add_eddy_diffusion_scalar!`](@ref)`(tϕ, ϕ, κh, κv)`, which
[`add_tensor_diffusion_scalar!`](@ref) reproduces to round-off.
"""
function diagonal_tensor!(K::DiffusivityTensor, κh, κv = κh)
    _setcomp!(K.K11.data, κh); _setcomp!(K.K22.data, κh); _setcomp!(K.K33.data, κv)
    fill!(K.K12.data, 0); fill!(K.K13.data, 0); fill!(K.K23.data, 0)
    return K
end
_setcomp!(a, x::Real) = fill!(a, x)
_setcomp!(a, x::Field) = copyto!(a, x.data)
_setcomp!(a, x::AbstractArray) = copyto!(a, x)

"""
    bounded_directions(bcs) -> NTuple{3,Bool}

`true` for each direction (x, y, z) whose boundaries are not periodic in
the `FieldBoundaryConditions` `bcs` (walls, no-flux or prescribed-value
faces): the `bounded` keyword of [`add_tensor_diffusion_scalar!`](@ref).
"""
bounded_directions(bcs::FieldBoundaryConditions) =
    (!(bcs.west isa Periodic), !(bcs.south isa Periodic), !(bcs.bottom isa Periodic))

"""
    add_tensor_diffusion_scalar!(tϕ, ϕ, K::DiffusivityTensor;
                                 bounded = (false, false, false), part = :full)

Add the divergence of the down-gradient flux of a symmetric diffusivity
tensor, `∂/∂xᵢ(Kᵢⱼ ∂ϕ/∂xⱼ)`, to the cell-centred scalar tendency `tϕ`, with
the six components of `K` at cell centres.

**Discretisation (triads, Griffies et al. 1998).** Each cell is split into
eight octants; octant `(a, b, c)` pairs one x-face, one y-face and one
z-face gradient of the cell, `g = (gx, gy, gz)`. The operator is the
gradient of the discrete functional

    F[ϕ] = −½ Σ_cells (ΔV/8) Σ_octants gᵀ K_cell g,   ΔV ∂ϕₙ/∂t = ∂F/∂ϕₙ,

which in flux form puts on each face the mean of `(K g)ₙ` over the eight
octants that touch it. Three properties follow from the functional and hold
to round-off for any K, independently of resolution:

* **conservation**: `Σ ΔV ∂ϕ/∂t = 0` (periodic or no-flux boundaries);
* **down-gradient**: `Σ ΔV ϕ ∂ϕ/∂t = 2F ≤ 0` when every cell's `K` is
  positive semi-definite (Griffies et al. 1998, Sec. 3) — the scalar
  variance cannot grow, however strongly K is rotated;
* the operator is **symmetric**.

The diagonal part is exactly the face-averaged operator of
[`add_eddy_diffusion_scalar!`](@ref) with `K11` on the x faces, `K22` on
the y faces and `K33` on the z faces, so `K = diag(κh, κh, κv)` reproduces
`add_eddy_diffusion_scalar!(tϕ, ϕ, κh, κv)` to round-off. For a constant
`K` a plane wave `cos(k·x)` decays at
`K11 k̂ₓ² + K22 k̂_y² + K33 k̂_z² + 2K12 sₓs_y + 2K13 sₓs_z + 2K23 s_y s_z`,
`k̂ = 2 sin(kΔ/2)/Δ`, `s = sin(kΔ)/Δ`.

`bounded[d]` marks direction `d` (x, y, z) as walled
([`bounded_directions`](@ref)`(bcs)` derives it from the boundary
conditions of `ϕ`): the wall faces' gradients are left out of the cross
terms, and the wall flux keeps only its diagonal part `K_nn ∂ϕ/∂n` from the
halo, as in the diagonal operator (zero for a no-flux halo, with which the
three properties hold with walls). Prescribed surface fluxes are added
separately ([`add_surface_flux!`](@ref)).

`part` selects `:full` (default), `:diagonal` (the `K11, K22, K33` terms
only) or `:offdiagonal` (the cross terms only); `:diagonal + :offdiagonal =
:full`. The IMEX cores treat the diagonal implicitly and the off-diagonal
part explicitly (see the theory page *Smagorinsky–Lilly SGS*, section
"Tensorial scalar diffusivity").

Halos of `ϕ` and of the components of `K` must be filled (a periodic
direction needs `K` in the first halo cell; corners included, which
`fill_halos!` provides).

References: Redi (1982), *J. Phys. Oceanogr.* 12, 1154; Griffies, Gnanadesikan,
Pacanowski, Larichev, Dukowicz & Smith (1998), *J. Phys. Oceanogr.* 28, 805.
"""
function add_tensor_diffusion_scalar!(tϕ, ϕ, K::DiffusivityTensor;
                                      bounded::NTuple{3,Bool} = (false, false, false),
                                      part::Symbol = :full)
    part in (:full, :diagonal, :offdiagonal) ||
        throw(ArgumentError("part must be :full, :diagonal or :offdiagonal, got $(repr(part))"))
    _add_tensor_diffusion_scalar!(tϕ, ϕ, K.K11.data, K.K22.data, K.K33.data,
                                  K.K12.data, K.K13.data, K.K23.data,
                                  K.flux[1], K.flux[2], K.flux[3], bounded,
                                  part !== :offdiagonal, part !== :diagonal, mesh(ϕ))
    return nothing
end
