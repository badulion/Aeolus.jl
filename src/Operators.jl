module Operators

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields
using ..Fields: Field, mesh

export ∂x, ∂y, ∂z, laplacian, divergence,
       advect_centered!, advect_upwind!,
       interpolate_to_face!, interpolate_to_center!,
       momentum_advect!,
       add_buoyancy!, add_coriolis!,
       compute_smagorinsky_νt!,
       add_eddy_viscosity_u!, add_eddy_viscosity_v!, add_eddy_viscosity_w!,
       add_eddy_diffusion_scalar!,
       add_surface_flux!, add_sponge!,
       advect_weno5!

# -------------------------------------------------------------------------
# Public API — dispatches on mesh type AND location combination.
# -------------------------------------------------------------------------
# Each concrete `AbstractMesh` subtype provides its own kernel. FV kernels
# live in `src/FiniteVolume.jl`; a future SEM backend will add matching
# `_∂x!(out, f, ::SpectralElementMesh)` methods and slot in without
# touching call sites. The location parameters (CellCenter / FaceX / …) on
# the fields select the correct discrete stencil via multiple dispatch.

"""
    ∂x(out, f); ∂y(out, f); ∂z(out, f)

Partial derivative of `f` into `out`. The *combination* of input and
output [`Location`](@ref)s selects the discrete stencil:

  * `(CellCenter, CellCenter)` → 2nd-order centered, cell → cell.
  * `(FaceX, CellCenter)`      → 2-point gradient, cell → face
    (pressure-gradient use).
  * `(CellCenter, FaceX)`      → 2-point difference, face → cell
    (divergence contribution).

Halos of `f` must be filled before calling. Returns `out`.
"""
∂x(out::Field, f::Field) = _∂x!(out, f, mesh(f))
@doc (@doc ∂x) ∂y(out::Field, f::Field) = _∂y!(out, f, mesh(f))
@doc (@doc ∂x) ∂z(out::Field, f::Field) = _∂z!(out, f, mesh(f))

"""
    laplacian(out, f)

Apply the 7-point 2nd-order centered discrete Laplacian to `f`, writing
`∇²f` into `out`. `out` must live at the same [`Location`](@ref) as `f`.
Halos of `f` must be filled.
"""
laplacian(out::Field, f::Field)        = _laplacian!(out, f, mesh(f))

"""
    divergence(out, u, v, w)

Discrete divergence `∇·u` written to `out` at `CellCenter`.

Two methods:
  * `u@FaceX, v@FaceY, w@FaceZ` → staggered Arakawa-C divergence
    (the physically correct one; used in the pressure projection).
  * `u@CellCenter, v@CellCenter, w@CellCenter` → collocated 2nd-order
    centered divergence (diagnostic only).

Halos of `u, v, w` must be filled.
"""
divergence(out::Field, u, v, w)        = _divergence!(out, u, v, w, mesh(u))

"""
    advect_centered!(tend, u, v, w, ϕ)

Add `-(u·∇)ϕ` (or equivalently `-∂(uⱼϕ)/∂xⱼ` for `∇·u = 0`) to `tend`
using 2nd-order centered stencils. Two methods:

  * `u,v,w @ Face…, ϕ @ CellCenter` — staggered flux form (used for θ in
    the Boussinesq core).
  * `u,v,w,ϕ @ CellCenter` — collocated advection-form (diagnostic /
    passive-tracer runs).

Non-monotone: overshoots near sharp gradients. Use [`advect_upwind!`](@ref)
for stronger monotonicity at the cost of diffusion.
"""
advect_centered!(tend, u, v, w, ϕ)     = _advect_centered!(tend, u, v, w, ϕ, mesh(ϕ))

"""
    advect_upwind!(tend, u, v, w, ϕ)

Add `-(u·∇)ϕ` to `tend` using 1st-order upwind differencing. Monotone
but highly diffusive; a placeholder for the WENO-5 kernel planned in
Phase 2. Collocated (`CellCenter`) only.
"""
advect_upwind!(tend, u, v, w, ϕ)       = _advect_upwind!(tend, u, v, w, ϕ, mesh(ϕ))

"""
    momentum_advect!(tu, tv, tw, u, v, w)

Add `-∂(uⱼuᵢ)/∂xⱼ` (momentum flux divergence) to the three face-centered
tendencies `(tu, tv, tw)`. 2nd-order centered flux form on Arakawa-C.
Halos of `u, v, w` must be filled.
"""
momentum_advect!(tu, tv, tw, u, v, w)  = _momentum_advect!(tu, tv, tw, u, v, w, mesh(u))

"""
    add_buoyancy!(tw, θ, θ₀, g_accel)
    add_buoyancy!(tw, θ, qv, θ₀, g_accel)

Add the Boussinesq buoyancy tendency to the w-momentum tendency `tw`.

  * 4-argument form: dry buoyancy `g·(θ - θ₀)/θ₀`.
  * 5-argument form: moist buoyancy `g·(θᵥ - θ₀)/θ₀` with virtual
    potential temperature `θᵥ = θ (1 + 0.61 qv)` (Phase 11).

`θ` (and `qv`) are interpolated in z to the z-face location of `w`.
"""
add_buoyancy!(tw, θ, θ₀, g_accel) = _add_buoyancy!(tw, θ, θ₀, g_accel, mesh(θ))
add_buoyancy!(tw, θ, qv, θ₀, g_accel) = _add_buoyancy_moist!(tw, θ, qv, θ₀, g_accel, mesh(θ))

"""
    add_coriolis!(tu, tv, u, v, f)

f-plane Coriolis: add `+f v` to `tu` (with `v` interpolated to `u`'s face)
and `-f u` to `tv` (with `u` interpolated to `v`'s face). 4-point
horizontal averaging preserves symmetry.
"""
add_coriolis!(tu, tv, u, v, f)    = _add_coriolis!(tu, tv, u, v, f, mesh(u))

"""
    compute_smagorinsky_νt!(νt, u, v, w, C_s)

Compute Smagorinsky-Lilly eddy viscosity at cell centers:

```math
\\nu_t = (C_s \\Delta)^2 \\, |S|, \\quad |S| = \\sqrt{2\\, S_{ij} S_{ij}}
```

with `Δ = (dx·dy·dz)^{1/3}` and strain-rate tensor
`Sᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)`. Diagonal components live natively at cell
centers on a C-grid; off-diagonals are formed at edges and averaged to
the center. Halos of `u, v, w` must be filled.
"""
compute_smagorinsky_νt!(νt, u, v, w, C_s) = _compute_smagorinsky_νt!(νt, u, v, w, C_s, mesh(u))

"""
    add_eddy_viscosity_u!(tu, u, νt)
    add_eddy_viscosity_v!(tv, v, νt)
    add_eddy_viscosity_w!(tw, w, νt)

Add the conservative eddy-viscosity tendency `∂/∂xⱼ(νₜ ∂uᵢ/∂xⱼ)` to the
face-centered momentum tendencies. `νt` is interpolated from cell centers
to the flux face (edges for off-diagonal fluxes).
"""
add_eddy_viscosity_u!(tu, u, νt) = _add_eddy_viscosity_u!(tu, u, νt, mesh(u))

@doc (@doc add_eddy_viscosity_u!) add_eddy_viscosity_v!(tv, v, νt) = _add_eddy_viscosity_v!(tv, v, νt, mesh(v))
@doc (@doc add_eddy_viscosity_u!) add_eddy_viscosity_w!(tw, w, νt) = _add_eddy_viscosity_w!(tw, w, νt, mesh(w))

"""
    add_eddy_diffusion_scalar!(tϕ, ϕ, νt, Pr_t)

Add `∂/∂xⱼ((νₜ/Prₜ) ∂ϕ/∂xⱼ)` to the cell-centered scalar tendency.
"""
add_eddy_diffusion_scalar!(tϕ, ϕ, νt, Pr_t) = _add_eddy_diffusion_scalar!(tϕ, ϕ, νt, Pr_t, mesh(ϕ))

"""
    add_surface_flux!(tϕ, F_bot, F_top)

Inject prescribed diffusive fluxes at the bottom (`F_bot`) and top
(`F_top`) walls of the domain into the cell-centered scalar tendency.
Sign convention: `F_bot > 0` means flux into the domain from below.
"""
add_surface_flux!(tϕ, F_bot, F_top) = _add_surface_flux!(tϕ, F_bot, F_top, mesh(tϕ))

"""
    add_sponge!(tf, f, α_max, z_start, z_top; f_ref = 0)

Rayleigh damping toward `f_ref` in the region `z ≥ z_start`. Damping rate
grows smoothly `α(z) = α_max sin²(π/2 (z - z_start)/(z_top - z_start))`.
Adds `-α(z)·(f - f_ref)` to `tf` for cells inside the sponge region.
"""
add_sponge!(tf, f, α_max, z_start, z_top; f_ref = 0) =
    _add_sponge!(tf, f, α_max, z_start, z_top, f_ref, mesh(f))

"""
    advect_weno5!(tend, u, v, w, ϕ)

Add `-∂(uⱼϕ)/∂xⱼ` to `tend` using WENO-5 upwind reconstruction. Non-
oscillatory near sharp gradients — the recommended scheme for scalars
(θ, tracers, moisture) once the domain has walls or fronts. Requires
`halo ≥ 3` in every direction.

Staggered case only: `u, v, w @ Face…`, `ϕ, tend @ CellCenter`.
"""
advect_weno5!(tend, u, v, w, ϕ) = _advect_weno5!(tend, u, v, w, ϕ, mesh(ϕ))

"""
    interpolate_to_face!(out, f)
    interpolate_to_center!(out, f)

Linear (2-point) averaging between staggered [`Location`](@ref)s. The
target location is inferred from `out`. Halos of `f` must be filled.
"""
interpolate_to_face!(out::Field, f::Field)   = _interp_to_face!(out, f, mesh(f))

@doc (@doc interpolate_to_face!) interpolate_to_center!(out::Field, f::Field) = _interp_to_center!(out, f, mesh(f))

# -------------------------------------------------------------------------
# Load backend kernels.
# -------------------------------------------------------------------------
include("FiniteVolume.jl")
using .FiniteVolume: _∂x!, _∂y!, _∂z!, _laplacian!, _divergence!,
                     _advect_centered!, _advect_upwind!,
                     _interp_to_face!, _interp_to_center!,
                     _momentum_advect!,
                     _add_buoyancy!, _add_buoyancy_moist!, _add_coriolis!,
                     _compute_smagorinsky_νt!,
                     _add_eddy_viscosity_u!, _add_eddy_viscosity_v!, _add_eddy_viscosity_w!,
                     _add_eddy_diffusion_scalar!,
                     _add_surface_flux!, _add_sponge!,
                     _advect_weno5!

end # module
