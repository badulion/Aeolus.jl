module Operators

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields
using ..Fields: Field, mesh, interior_view
using ..Grids: interior
using ..BoundaryConditions: BoundaryConditions, fill_halos!, Periodic, FieldBoundaryConditions
using ..TurbulenceStability: buoyancy_frequency_squared, gradient_richardson, RI_SHEAR2_FLOOR,
                             AbstractRiStability, BrownEtAl1994,
                             stability_factor_m, stability_factor_h
using ..MoistThermodynamics: saturated_buoyancy_frequency_squared, DEFAULT_CONSTANTS
using FFTW: rfft, irfft, fft, ifft, fftfreq, plan_rfft
using Random: AbstractRNG, default_rng, randn

export ∂x, ∂y, ∂z, laplacian, divergence,
       advect_centered!, advect_upwind!,
       interpolate_to_face!, interpolate_to_center!,
       momentum_advect!,
       add_buoyancy!, add_coriolis!,
       compute_smagorinsky_νt!,
       compute_dynamic_smagorinsky_νt!, DynamicSmagorinskyWorkspace,
       add_eddy_viscosity_u!, add_eddy_viscosity_v!, add_eddy_viscosity_w!,
       add_eddy_diffusion_scalar!,
       add_surface_flux!, add_sponge!,
       advect_weno5!,
       box_filter!, gaussian_filter!, spectral_cutoff_filter!, staggered_solid_mask,
       isotropic_energy_spectrum, isotropic_random_field!, rescale_to_spectrum!
# #245: the FFTW planning policy (operators/fftw_planning.jl)
export set_fftw_policy!, fftw_policy, fftw_planner_flags, save_fftw_wisdom, load_fftw_wisdom!
# Era VI (P54-a, #167): the SGS closure contract (operators/sgs_closures.jl)
export AbstractSGSClosure, SGSFields, sgs_fields, sgs_coefficients!, advance_sgs!,
       sgs_state, restore_sgs_state!, add_sgs_momentum!, add_sgs_scalar!,
       sgs_tracer_diffusivity!,
       compute_sgs!, sgs_closure,
       AbstractEddyViscosityClosure, sgs_viscosity!, turbulent_prandtl,
       StaticSmagorinsky, DynamicSmagorinsky, FunctionSGS,
       sgs_tke!                 # N-inflow-b (#258)
# Era VI (P54-b, #173): Lagrangian-averaged dynamic Smagorinsky (operators/lagrangian_sgs.jl)
export LagrangianDynamicSmagorinsky
# Era VI (P55-b, #175): the stress-form momentum divergence (operators/sgs_stress.jl)
export StressTensor, add_stress_divergence!, sgs_stress_tensor!,
       add_sgs_stress_divergence!, sgs_stress_dissipation
# Era VI (P55-c, #181): the tensorial scalar eddy diffusivity (operators/scalar_tensor.jl)
export DiffusivityTensor, diagonal_tensor!, bounded_directions,
       add_tensor_diffusion_scalar!, isentropic_tensor!,
       sgs_scalar_tensor, IsentropicDiffusion, sgs_scalar_face_diffusivity
export buoyancy_frequency_squared!, gradient_richardson!, strain_rate_squared!   # P58-a (#170)
# The one damping-profile implementation behind add_sponge! and the fringe (P59-d, #185).
export AbstractRamp, SinSquaredRamp, NordstromRamp, smooth_step,
       DampingProfile, sponge_profile, damping_rate, damping_coordinates
# Era VI (P55-d, #176): AMD and Vreman closures (operators/sgs_amd_vreman.jl)
export AMDClosure, VremanClosure, compute_amd_coefficients!, compute_amd_scalar_diffusivity!,
       compute_vreman_νt!,
       sgs_buoyancy_parameter
# Era VI, Phase 56 (P56-c, #178): a-priori SGS-stress analysis (operators/apriori.jl)
export SymmetricTensorField, subfilter_stress!, bardina_stress!, filtered_strain, filtered_strain!,
       eddy_viscosity_stress!, smagorinsky_stress!, sgs_dissipation, stress_eigvals,
       lumley_invariants, apriori_metrics, dissipation_matched_cs
# Era VI (P59-b, #184): the σ-model closure (operators/sigma_closure.jl)
export SigmaModel, compute_sigma_νt!
# Era VI (P55-e, #187): the stability-corrected anisotropic closure (operators/sgs_stability.jl)
export StabilityCorrected, dynamic_sgs_coefficient, anisotropic_length_factors,
       sgs_stratification!

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
    add_eddy_diffusion_scalar!(tϕ, ϕ, κt::Field; mask = nothing, weights = nothing)
    add_eddy_diffusion_scalar!(tϕ, ϕ, νt::Field, Pr_t)
    add_eddy_diffusion_scalar!(tϕ, ϕ, κh::Field, κv::Field; mask = nothing, weights = nothing)

Add the conservative eddy-diffusion tendency `∂/∂xⱼ(κₜ ∂ϕ/∂xⱼ)` to the
cell-centred scalar tendency `tϕ`, with the cell-centred diffusivity `κt`
averaged linearly to each face. The four-argument form with a number is
`(1/Prₜ) ∂/∂xⱼ(νₜ ∂ϕ/∂xⱼ)`: the same kernel, with the turbulent Prandtl
number applied as a factor after the flux divergence.

The form with two fields is the **diagonal** operator (P55-c, #181):
`∂/∂x(κh ∂ϕ/∂x) + ∂/∂y(κh ∂ϕ/∂y) + ∂/∂z(κv ∂ϕ/∂z)`, `κh` on the x and y
faces and `κv` on the z faces. It is the diffusivity tensor
`diag(κh, κh, κv)`; the three-argument isotropic form is its case
`κv = κh`, bit for bit (one kernel). The general symmetric tensor is
[`add_tensor_diffusion_scalar!`](@ref). Halos of `ϕ` and of the
coefficient fields must be filled.

Keywords of the three-argument form (P56-a, #177), also taken by the
diagonal form (κh on the x, y faces and κv on the z faces, e.g. the
coefficients of a stability-corrected closure, P55-e):

  * `mask` — a storage-sized `AbstractArray{Bool,3}` of solid cells (e.g.
    `IBM.SolidMask(...).mask`). A face with a solid cell on either side
    carries **no flux**, so a scalar does not diffuse into or out of a
    building, and the tendency of a solid cell is left untouched. Every
    open face carries the unmasked flux, so the fluid integral `Σ ϕ dV` is
    conserved exactly (up to round-off). The halo entries of `mask` must
    be consistent with the boundary conditions of `ϕ` (periodic copies
    across a periodic seam; a `false` halo next to a no-flux wall is
    harmless, since that face carries no gradient).
  * `weights = (Wc, Wf)` — the sound-proof (anelastic / pseudo-incompressible)
    form `(1/W) ∇·(W κₜ ∇ϕ)`, with a cell weight `Wc[k]` and a z-face weight
    `Wf[k]` (bottom face of cell `k`) per storage level (`Anelastic.constraint_weights`).
    It conserves `Σ W ϕ dV` instead of `Σ ϕ dV`.

With both `nothing` (default) this is bit for bit the unmasked kernel.
"""
function add_eddy_diffusion_scalar!(tϕ, ϕ, κt::Field; mask = nothing, weights = nothing)
    mask === nothing && weights === nothing &&
        return _add_eddy_diffusion_scalar!(tϕ, ϕ, κt, one(eltype(κt.data)), mesh(ϕ))
    wc, wf = _check_mask_weights(ϕ, mask, weights)
    return _add_eddy_diffusion_scalar_masked!(tϕ, ϕ, κt, mesh(ϕ), mask, wc, wf)
end
add_eddy_diffusion_scalar!(tϕ, ϕ, νt::Field, Pr_t::Real) =
    _add_eddy_diffusion_scalar!(tϕ, ϕ, νt, 1 / Pr_t, mesh(ϕ))
function add_eddy_diffusion_scalar!(tϕ, ϕ, κh::Field, κv::Field; mask = nothing, weights = nothing)
    mask === nothing && weights === nothing &&
        return _add_eddy_diffusion_scalar!(tϕ, ϕ, κh, κv, one(eltype(κh.data)), mesh(ϕ))
    wc, wf = _check_mask_weights(ϕ, mask, weights)
    return _add_eddy_diffusion_scalar_masked!(tϕ, ϕ, κh, κv, mesh(ϕ), mask, wc, wf)
end

# the storage-size mask and per-level weights of the masked / weighted form
function _check_mask_weights(ϕ, mask, weights)
    if mask !== nothing
        size(mask) == size(ϕ.data) ||
            throw(DimensionMismatch("mask has size $(size(mask)); expected the storage size $(size(ϕ.data))"))
    end
    wc, wf = weights === nothing ? (nothing, nothing) : weights
    if weights !== nothing
        n = size(ϕ.data, 3)
        (length(wc) >= n && length(wf) >= n) ||
            throw(DimensionMismatch("weights: need Wc and Wf per storage level ($n), got $(length(wc)), $(length(wf))"))
    end
    return wc, wf
end

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

`f_ref` (#254) is

* a number — the constant target of the Phase-7 velocity sponge;
* a **level vector** of length `Nz` — one target per interior level of
  `f`'s location, e.g. the initial θ̄(z) of a free-atmosphere top;
* a `Field` or a halo-padded 3-D array of `f`'s shape — a full 3-D
  target, e.g. the parent state a nest is damped toward.

A `ForcingTable` or a time-dependent profile is evaluated into a level
vector by `LargeScaleForcing.ReferenceDamping`, which also books the
tendency integral. A constant level vector gives the same tendency as the
equal number, bit for bit.

The rate is [`damping_rate`](@ref) of [`sponge_profile`](@ref)`(α_max,
z_start, z_top)`, the damping-profile implementation the fringe region of
`Transition` (P59-d) shares; the arithmetic is the one kernel
`FiniteVolume._add_damping!` on a `RegularGrid` and a `StretchedGrid`.

References: Klemp, J. B. & Lilly, D. K. (1978), *J. Atmos. Sci.*, 35,
78-107; Durran, D. R. (2010), *Numerical Methods for Fluid Dynamics*,
2nd ed., §8.4.
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
# Dynamic Smagorinsky (Germano et al. 1991, Lilly 1992).
# -------------------------------------------------------------------------
# Moved here from the urban example (examples/urban/turbulence_addons.jl,
# "F3", at dfaae75), where it was never called. Changes against that copy:
# the sign of M_ij is corrected (the copy's C_s² was negative wherever the
# flow cascades energy to small scales, so the clip set it to 0); the strain stencil is evaluated on one extra shell of cells from the
# filled halos, so every interior cell gets a dynamic coefficient (the
# example fell back to C_s = 0.16 on the outermost interior shell); the
# averaging of the Germano identity is selectable (plane average, the
# standard stabilisation, by default; the example's pointwise ratio as
# `averaging = :local`); the test-filter width ratio is a keyword with a
# default from the filter's second moment (the example used 2); and the
# scratch arrays live in a reusable workspace.

"""
    DynamicSmagorinskyWorkspace(mesh)

Scratch storage for [`compute_dynamic_smagorinsky_νt!`](@ref): the
cell-centred velocity and strain-rate components on the interior plus one
shell, and `cs2`, the interior-sized coefficient ``C_s^2`` of the last
call (a diagnostic).
"""
struct DynamicSmagorinskyWorkspace{T}
    Uc::Array{T,3}; Vc::Array{T,3}; Wc::Array{T,3}
    S11::Array{T,3}; S22::Array{T,3}; S33::Array{T,3}
    S12::Array{T,3}; S13::Array{T,3}; S23::Array{T,3}
    magS::Array{T,3}
    LM::Array{T,3}; MM::Array{T,3}
    cs2::Array{T,3}
end

function DynamicSmagorinskyWorkspace(g::RegularGrid{T}) where {T}
    e() = zeros(T, g.Nx + 2, g.Ny + 2, g.Nz + 2)
    i() = zeros(T, g.Nx, g.Ny, g.Nz)
    return DynamicSmagorinskyWorkspace{T}(e(), e(), e(), e(), e(), e(), e(), e(), e(), e(),
                                          i(), i(), i())
end

"""
    _velocity_gradient_at(u, v, w, i, j, k, idx, idy, idz) -> g

The full resolved velocity-gradient tensor at the centre of cell `(i, j, k)`
(indices into the halo-padded `data` arrays of the staggered `u @ FaceX`,
`v @ FaceY`, `w @ FaceZ`), with `idx = 1/Δx` etc. Returns the 3 × 3 tensor
as a tuple of rows, `g[a][b] = ∂u_a/∂x_b`:

* diagonal: the face-to-face difference through the cell (`∂u/∂x =
  (u[i+1] − u[i])/Δx`), exact for a linear field and the stencil of the
  discrete divergence;
* off-diagonal: the centred difference over two cells of the component
  averaged to the cell centre in its own direction (`∂u/∂y = ((u[i,j+1] +
  u[i+1,j+1]) − (u[i,j−1] + u[i+1,j−1]))/(4Δy)`), also exact for a linear
  field.

The single gradient stencil of the SGS closures: [`_strain_at`](@ref) (and so
the static and dynamic Smagorinsky ν_t) is built from it, and the AMD /
Vreman, σ, gradient-Richardson and TKE-production closures of Era VI read the
same tensor. Halos of `u, v, w` must be filled.

`idz_off` (default `idz`) is the inverse of the vertical distance used by
the off-diagonal `∂/∂z` terms: on a vertically stretched grid the
diagonal `∂w/∂z` spans the cell (`idz = 1/Δz_c[k]`) while `∂u/∂z`, `∂v/∂z`
span the two neighbouring centres, `idz_off = 2/(Δz_f[k] + Δz_f[k+1])`
(the `StretchedGrid` Smagorinsky kernel, #151). With the default the
uniform stencil is unchanged, bit for bit.
"""
@inline function _velocity_gradient_at(u, v, w, i, j, k, idx, idy, idz, idz_off = idz)
    @inbounds begin
        du_dx = (u[i+1,j,k] - u[i,j,k]) * idx
        dv_dy = (v[i,j+1,k] - v[i,j,k]) * idy
        dw_dz = (w[i,j,k+1] - w[i,j,k]) * idz
        du_dy = ((u[i,j+1,k] + u[i+1,j+1,k]) - (u[i,j-1,k] + u[i+1,j-1,k])) * (0.25 * idy)
        dv_dx = ((v[i+1,j,k] + v[i+1,j+1,k]) - (v[i-1,j,k] + v[i-1,j+1,k])) * (0.25 * idx)
        du_dz = ((u[i,j,k+1] + u[i+1,j,k+1]) - (u[i,j,k-1] + u[i+1,j,k-1])) * (0.25 * idz_off)
        dw_dx = ((w[i+1,j,k] + w[i+1,j,k+1]) - (w[i-1,j,k] + w[i-1,j,k+1])) * (0.25 * idx)
        dv_dz = ((v[i,j,k+1] + v[i,j+1,k+1]) - (v[i,j,k-1] + v[i,j+1,k-1])) * (0.25 * idz_off)
        dw_dy = ((w[i,j+1,k] + w[i,j+1,k+1]) - (w[i,j-1,k] + w[i,j-1,k+1])) * (0.25 * idy)
    end
    return ((du_dx, du_dy, du_dz),
            (dv_dx, dv_dy, dv_dz),
            (dw_dx, dw_dy, dw_dz))
end

"""
    _vorticity_at(u, v, w, i, j, k, idx, idy, idz) -> (ω_x, ω_y, ω_z)

Vorticity `ω = ∇×u` at the centre of cell `(i, j, k)`, from the same
[`_velocity_gradient_at`](@ref) tensor as the strain rate (`ω_x = ∂w/∂y −
∂v/∂z`, …). Used by `Transition` (turbulent fraction γ(z)).
"""
@inline function _vorticity_at(u, v, w, i, j, k, idx, idy, idz)
    g = _velocity_gradient_at(u, v, w, i, j, k, idx, idy, idz)
    return g[3][2] - g[2][3], g[1][3] - g[3][1], g[2][1] - g[1][2]
end

"""
    _strain_at(u, v, w, i, j, k, idx, idy, idz[, idz_off]) -> (S11, S22, S33, S12, S13, S23, |S|)

Strain-rate components `Sᵢⱼ = ½(gᵢⱼ + gⱼᵢ)` and `|S| = √(2 SᵢⱼSᵢⱼ)` at the
centre of cell `(i, j, k)`, from [`_velocity_gradient_at`](@ref). The stencil
of [`compute_smagorinsky_νt!`](@ref) and [`compute_dynamic_smagorinsky_νt!`](@ref).
"""
@inline function _strain_at(u, v, w, i, j, k, idx, idy, idz, idz_off = idz)
    g = _velocity_gradient_at(u, v, w, i, j, k, idx, idy, idz, idz_off)
    S11 = g[1][1]; S22 = g[2][2]; S33 = g[3][3]
    S12 = 0.5 * (g[1][2] + g[2][1])
    S13 = 0.5 * (g[1][3] + g[3][1])
    S23 = 0.5 * (g[2][3] + g[3][2])
    S2 = 2 * (S11^2 + S22^2 + S33^2) + 4 * (S12^2 + S13^2 + S23^2)
    return S11, S22, S33, S12, S13, S23, sqrt(S2)
end

"""
    compute_dynamic_smagorinsky_νt!(νt, u, v, w;
                                    workspace = DynamicSmagorinskyWorkspace(mesh(u)),
                                    averaging = :plane, filter_ratio = 3, cs_max = 0.5, mask = nothing) -> νt

Eddy viscosity ``ν_t = C_s^2 Δ^2 |S|`` with the coefficient computed from
the resolved field by the dynamic procedure of Germano et al. (1991) in
the least-squares form of Lilly (1992). With a test filter ``\\widetilde{·}``
(here the 3 × 3 × 3 box average of cell-centred quantities),

```math
L_{ij} = \\widetilde{u_i u_j} - \\tilde u_i \\tilde u_j, \\qquad
M_{ij} = 2Δ^2\\left(\\widetilde{|S| S_{ij}} - α^2 |\\tilde S| \\tilde S_{ij}\\right), \\qquad
C_s^2 = \\frac{⟨L^d_{ij} M_{ij}⟩}{⟨M_{ij} M_{ij}⟩},
```

with ``L^d`` the deviatoric part of ``L``, ``Δ = (Δx Δy Δz)^{1/3}`` and
``α = \\hat Δ / Δ`` the test-to-grid filter width ratio (`filter_ratio`).
``C_s^2`` is clipped to ``[0, c_{s,\\max}^2]`` (no backscatter).

* `averaging = :plane` — ``⟨·⟩`` is the average over each horizontal
  level (Lilly 1992; the standard choice with horizontally homogeneous
  statistics), so ``C_s`` depends on height only;
* `averaging = :local` — the pointwise ratio (no averaging), which is
  noisy and relies on the clip.

`filter_ratio`: a discrete box of three cells has second moment
``2Δ^2/3``; adding it to the implicit grid filter (``Δ^2/12``) gives an
effective width ``\\sqrt{Δ^2 + 12·2Δ^2/3} = 3Δ``, hence the default 3.
The value 2, often quoted for the 3-point trapezoidal filter, is the
other common choice (see Lund 1997, *Annual Research Briefs*, CTR).

`mask` (optional; a `Bool` array the size of the cell-centred storage,
`true` = solid, e.g. `IBM.SolidMask.mask`, whose halos are used as given)
makes the procedure aware of immersed obstacles (P54-b, #173): the
velocity is read with the no-slip value 0 on every face touching a solid
(so whatever is stored inside an obstacle never reaches a fluid cell), the
test filter is the masked box of [`box_filter!`](@ref) (fluid neighbours
only, renormalised by their number), the plane average runs over the fluid
cells of each level, and ``C_s^2 = ν_t = 0`` in solid cells.

The strain rate uses the stencil of [`compute_smagorinsky_νt!`](@ref), so
for a fixed ``C_s^2`` the two agree exactly. Halos of `u, v, w` must be
filled and be at least 2 wide (the strain rate is evaluated on one shell
of halo cells so the test filter reaches every interior cell). The
coefficient is left in `workspace.cs2`.

References: Germano, Piomelli, Moin & Cabot (1991), *Phys. Fluids A* 3,
1760; Lilly (1992), *Phys. Fluids A* 4, 633.
"""
function compute_dynamic_smagorinsky_νt!(νt::Field, u::Field, v::Field, w::Field;
                                         workspace::DynamicSmagorinskyWorkspace = DynamicSmagorinskyWorkspace(mesh(u)),
                                         averaging::Symbol = :plane,
                                         filter_ratio::Real = 3,
                                         cs_max::Real = 0.5,
                                         mask = nothing)
    g = mesh(u)
    g isa RegularGrid || throw(ArgumentError("compute_dynamic_smagorinsky_νt!: RegularGrid only"))
    min(g.Hx, g.Hy, g.Hz) >= 2 ||
        throw(ArgumentError("compute_dynamic_smagorinsky_νt!: needs halo ≥ 2, mesh has $((g.Hx, g.Hy, g.Hz))"))
    averaging in (:plane, :local) ||
        throw(ArgumentError("averaging must be :plane or :local, got $(repr(averaging))"))
    m = mask === nothing ? nothing : _SGSMask(mask, g, nothing)
    _dynamic_smagorinsky!(νt.data, u.data, v.data, w.data, workspace, g,
                          averaging === :plane, float(filter_ratio)^2, float(cs_max)^2, m)
    return νt
end

# -------------------------------------------------------------------------
# Solid cells (immersed boundaries) in the dynamic procedure (P54-b, #173)
# -------------------------------------------------------------------------

"""
    _SGSMask(mask, g, periodic)

The solid mask of the dynamic closures, prepared once: the cell mask with
its halos filled (`periodic[d]`: wrapped; otherwise the nearest interior
layer is copied; `periodic === nothing`: the halos of `mask` are used as
given), the face masks of [`staggered_solid_mask`](@ref), and three
storage-sized velocity buffers. The dynamic procedure reads the velocity
through [`_masked_velocities!`](@ref), which puts the no-slip value 0 on
every face that touches a solid cell, so nothing stored inside an obstacle
can reach a fluid coefficient.
"""
struct _SGSMask{B<:AbstractArray{Bool,3}, T}
    cell::B; fx::B; fy::B; fz::B
    ub::Array{T,3}; vb::Array{T,3}; wb::Array{T,3}
end

function _SGSMask(mask::AbstractArray{Bool,3}, g::RegularGrid{T}, periodic) where {T}
    sz = (g.Nx + 2g.Hx, g.Ny + 2g.Hy, g.Nz + 2g.Hz)
    size(mask) == sz ||
        throw(DimensionMismatch("solid mask has size $(size(mask)), the mesh storage is $sz"))
    cell = BitArray(mask)
    periodic === nothing || _fill_mask_halos!(cell, g, periodic)
    return _SGSMask(cell, staggered_solid_mask(cell, FaceX()), staggered_solid_mask(cell, FaceY()),
                    staggered_solid_mask(cell, FaceZ()), zeros(T, sz), zeros(T, sz), zeros(T, sz))
end
_SGSMask(m::_SGSMask, g, periodic) = m

# Halos of a storage-sized cell mask: periodic wrap, or the nearest interior layer.
function _fill_mask_halos!(m, g, periodic::NTuple{3,Bool})
    N = (g.Nx, g.Ny, g.Nz); H = (g.Hx, g.Hy, g.Hz)
    @inbounds for d in 1:3
        n, h = N[d], H[d]
        for I in CartesianIndices(m)
            l = I[d]
            (l <= h || l > h + n) || continue
            s = periodic[d] ? h + mod(l - h - 1, n) + 1 : clamp(l, h + 1, h + n)
            J = Base.setindex(Tuple(I), s, d)
            m[I] = m[J...]
        end
    end
    return m
end

"""
    _masked_velocities!(m::_SGSMask, ud, vd, wd) -> (ub, vb, wb)

Copies of the velocity storage with every face that touches a solid set to
the no-slip value 0 (the rule of `IBM.apply_solid_mask!`), halos included.
"""
function _masked_velocities!(m::_SGSMask, ud, vd, wd)
    @inbounds for I in eachindex(ud)
        m.ub[I] = m.fx[I] ? zero(eltype(m.ub)) : ud[I]
        m.vb[I] = m.fy[I] ? zero(eltype(m.vb)) : vd[I]
        m.wb[I] = m.fz[I] ? zero(eltype(m.wb)) : wd[I]
    end
    return m.ub, m.vb, m.wb
end
_masked_velocities!(::Nothing, ud, vd, wd) = (ud, vd, wd)

_cellmask(::Nothing) = nothing
_cellmask(m::_SGSMask) = m.cell

# -------------------------------------------------------------------------
# The Germano identity: one kernel for every averaging (P54-b, #173)
# -------------------------------------------------------------------------

# Pass 1: cell-centred velocity and strain on the interior + one shell.
# Local index l ∈ 1:N+2 is data index H - 1 + l.
function _germano_pass1!(ws::DynamicSmagorinskyWorkspace, ud, vd, wd, g)
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    Uc, Vc, Wc, magS = ws.Uc, ws.Vc, ws.Wc, ws.magS
    S11, S22, S33, S12, S13, S23 = ws.S11, ws.S22, ws.S33, ws.S12, ws.S13, ws.S23
    @inbounds for lk in 1:Nz+2, lj in 1:Ny+2, li in 1:Nx+2
        i = Hx - 1 + li; j = Hy - 1 + lj; k = Hz - 1 + lk
        Uc[li,lj,lk] = 0.5 * (ud[i,j,k] + ud[i+1,j,k])
        Vc[li,lj,lk] = 0.5 * (vd[i,j,k] + vd[i,j+1,k])
        Wc[li,lj,lk] = 0.5 * (wd[i,j,k] + wd[i,j,k+1])
        s11, s22, s33, s12, s13, s23, ms = _strain_at(ud, vd, wd, i, j, k, idx, idy, idz)
        S11[li,lj,lk] = s11; S22[li,lj,lk] = s22; S33[li,lj,lk] = s33
        S12[li,lj,lk] = s12; S13[li,lj,lk] = s13; S23[li,lj,lk] = s23
        magS[li,lj,lk] = ms
    end
    return nothing
end

# |S| alone on the interior (into the shell-indexed ws.magS): what ν_t needs
# when the coefficient comes from memory (the Lagrangian closure).
function _strain_magnitude!(ws::DynamicSmagorinskyWorkspace, ud, vd, wd, g)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        ws.magS[i+1,j+1,k+1] = _strain_at(ud, vd, wd, Hx + i, Hy + j, Hz + k, idx, idy, idz)[7]
    end
    return nothing
end

# The 21 test-filtered quantities of the box around shell cell (a, b, c).
@inline function _germano_terms(ws, a, b, c)
    @inbounds begin
        uu = ws.Uc[a,b,c]; vv = ws.Vc[a,b,c]; ww = ws.Wc[a,b,c]
        s11 = ws.S11[a,b,c]; s22 = ws.S22[a,b,c]; s33 = ws.S33[a,b,c]
        s12 = ws.S12[a,b,c]; s13 = ws.S13[a,b,c]; s23 = ws.S23[a,b,c]
        ms = ws.magS[a,b,c]
    end
    return (uu, vv, ww, uu*uu, vv*vv, ww*ww, uu*vv, uu*ww, vv*ww,
            s11, s22, s33, s12, s13, s23,
            ms*s11, ms*s22, ms*s33, ms*s12, ms*s13, ms*s23)
end

# L^d_ij M_ij and M_ij M_ij from the box sums `t` and the normalisation
# `inv` (1/27, or 1/N_fluid with a mask).
@inline function _germano_LM_MM(t, inv, Δ2, α2)
    Ut, Vt, Wt, UU, VV, WW, UV, UW, VW,
    A11, A22, A33, A12, A13, A23,
    B11, B22, B33, B12, B13, B23 = t
    Ut *= inv; Vt *= inv; Wt *= inv
    L11 = UU*inv - Ut*Ut; L22 = VV*inv - Vt*Vt; L33 = WW*inv - Wt*Wt
    L12 = UV*inv - Ut*Vt; L13 = UW*inv - Ut*Wt; L23 = VW*inv - Vt*Wt
    tr = (L11 + L22 + L33) / 3
    L11 -= tr; L22 -= tr; L33 -= tr
    A11 *= inv; A22 *= inv; A33 *= inv; A12 *= inv; A13 *= inv; A23 *= inv
    magSt = sqrt(2 * (A11^2 + A22^2 + A33^2) + 4 * (A12^2 + A13^2 + A23^2))
    # L = T − τ̃ with τ = −2C_s²Δ²|S|S (deviatoric) gives L = C_s² M with
    # M = 2Δ²(~(|S|S) − α²|S̃|S̃). (The example's copy had the opposite
    # sign, so C_s² came out negative in forward scatter and was clipped.)
    c1 = 2 * Δ2 * α2 * magSt; c2 = 2 * Δ2 * inv
    M11 = c2*B11 - c1*A11; M22 = c2*B22 - c1*A22; M33 = c2*B33 - c1*A33
    M12 = c2*B12 - c1*A12; M13 = c2*B13 - c1*A13; M23 = c2*B23 - c1*A23
    LM = L11*M11 + L22*M22 + L33*M33 + 2 * (L12*M12 + L13*M13 + L23*M23)
    MM = M11*M11 + M22*M22 + M33*M33 + 2 * (M12*M12 + M13*M13 + M23*M23)
    return LM, MM
end

# Pass 2: L^d_ij M_ij and M_ij M_ij at every interior cell. The test
# filter is the 3 × 3 × 3 box of `box_filter!`, summed for all 21
# quantities in one sweep of the shared stencil; with a mask it is the
# masked box (fluid neighbours only, renormalised by their number), and
# LM = MM = 0 in solid cells.
function _germano_pass2!(ws::DynamicSmagorinskyWorkspace{T}, g, α2, ::Nothing) where {T}
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    inv27 = one(T) / 27
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        t = _box_stencil_sum((a, b, c) -> _germano_terms(ws, a, b, c), i + 1, j + 1, k + 1, 1, 1, 1)
        ws.LM[i,j,k], ws.MM[i,j,k] = _germano_LM_MM(t, inv27, Δ2, α2)
    end
    return nothing
end

function _germano_pass2!(ws::DynamicSmagorinskyWorkspace{T}, g, α2, cell) where {T}
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    oH = (g.Hx - 1, g.Hy - 1, g.Hz - 1)                 # shell → data index
    z21 = ntuple(_ -> zero(T), Val(21))
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        if cell[g.Hx + i, g.Hy + j, g.Hz + k]
            ws.LM[i,j,k] = zero(T); ws.MM[i,j,k] = zero(T)
            continue
        end
        s = _box_stencil_sum(i + 1, j + 1, k + 1, 1, 1, 1) do a, b, c
            t = _germano_terms(ws, a, b, c)
            @inbounds cell[oH[1] + a, oH[2] + b, oH[3] + c] ? z21 : t
        end
        (n,) = _box_stencil_sum(i + 1, j + 1, k + 1, 1, 1, 1) do a, b, c
            @inbounds (cell[oH[1] + a, oH[2] + b, oH[3] + c] ? 0 : 1,)
        end
        ws.LM[i,j,k], ws.MM[i,j,k] = _germano_LM_MM(s, one(T) / n, Δ2, α2)
    end
    return nothing
end

"""
    _germano_identity!(ws, ud, vd, wd, g, α2, mask = nothing)

The Germano identity of the dynamic Smagorinsky model at every interior
cell: `ws.LM = L^d_ij M_ij`, `ws.MM = M_ij M_ij`, and the strain magnitude
`|S|` in `ws.magS` (shell-indexed), from the halo-filled velocity storage
`ud, vd, wd`. The one kernel shared by the plane / local averaging of
[`compute_dynamic_smagorinsky_νt!`](@ref) and the Lagrangian averaging of
[`LagrangianDynamicSmagorinsky`](@ref). With a `_SGSMask` the velocity is
read with zeros on solid faces and the test filter runs over fluid cells.
"""
function _germano_identity!(ws::DynamicSmagorinskyWorkspace, ud, vd, wd, g, α2, mask = nothing)
    ub, vb, wb = _masked_velocities!(mask, ud, vd, wd)
    _germano_pass1!(ws, ub, vb, wb, g)
    _germano_pass2!(ws, g, α2, _cellmask(mask))
    return nothing
end

# ν_t = C_s² Δ² |S| on the interior, 0 in solid cells.
function _dynamic_νt!(nd, ws::DynamicSmagorinskyWorkspace{T}, g, cell) where {T}
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    cs2, magS = ws.cs2, ws.magS
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        solid = cell !== nothing && cell[Hx+i, Hy+j, Hz+k]
        nd[Hx+i, Hy+j, Hz+k] = solid ? zero(T) : cs2[i,j,k] * Δ2 * magS[i+1, j+1, k+1]
    end
    return nothing
end

function _dynamic_smagorinsky!(nd, ud, vd, wd, ws::DynamicSmagorinskyWorkspace{T}, g,
                               plane::Bool, α2, cs2max, mask = nothing) where {T}
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    _germano_identity!(ws, ud, vd, wd, g, α2, mask)
    cell = _cellmask(mask)

    # Averaging of the identity and the clip to [0, cs_max²]. With a mask
    # the plane average runs over the fluid cells of the level only.
    cs2 = ws.cs2
    if plane
        @inbounds for k in 1:Nz
            lm = zero(T); mm = zero(T)
            for j in 1:Ny, i in 1:Nx
                cell !== nothing && cell[g.Hx+i, g.Hy+j, g.Hz+k] && continue
                lm += ws.LM[i,j,k]; mm += ws.MM[i,j,k]
            end
            c = mm > 0 ? clamp(lm / mm, zero(T), T(cs2max)) : zero(T)
            for j in 1:Ny, i in 1:Nx
                cs2[i,j,k] = cell !== nothing && cell[g.Hx+i, g.Hy+j, g.Hz+k] ? zero(T) : c
            end
        end
    else
        @inbounds for I in eachindex(cs2)
            mm = ws.MM[I]
            cs2[I] = mm > 0 ? clamp(ws.LM[I] / mm, zero(T), T(cs2max)) : zero(T)
        end
    end
    _dynamic_νt!(nd, ws, g, cell)
    return nothing
end

# The FFTW planning policy every plan_* call uses (#245).
include("operators/fftw_planning.jl")
# Explicit filters, energy spectrum and isotropic initialiser (Phase 54, P54-c).
include("operators/filters.jl")
include("operators/spectra.jl")
# A-priori SGS-stress analysis (Phase 56, P56-c).
include("operators/apriori.jl")

# -------------------------------------------------------------------------
# Era VI: the SGS closure contract (wraps the kernels above).
# -------------------------------------------------------------------------
include("operators/sgs_closures.jl")
include("operators/lagrangian_sgs.jl")
# Era VI (P54-d, #174): wall-bounded flow diagnostics (Φ_m, stress budget, E11)
include("operators/abl_diagnostics.jl")
# Era VI, P55-d (#176): AMD and Vreman closures on the contract.
include("operators/sgs_amd_vreman.jl")
# The σ-model closure (P59-b, #184).
include("operators/sigma_closure.jl")

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
                     _add_eddy_diffusion_scalar!, _add_eddy_diffusion_scalar_masked!,
                     _add_surface_flux!, _add_sponge!,
                     _advect_weno5!,
                     _add_sgs_stress_divergence!, _add_stress_divergence!,
                     _sgs_stress_tensor!, _sgs_stress_dissipation,
                     _add_tensor_diffusion_scalar!, _isentropic_tensor!
# The shared damping profile (sponge + fringe, P59-d #185).
using .FiniteVolume: AbstractRamp, SinSquaredRamp, NordstromRamp, smooth_step,
                     DampingProfile, sponge_profile, in_damping, damping_rate,
                     damping_coordinates, _add_damping!

# -------------------------------------------------------------------------
# Era VI: the stress-form SGS momentum divergence (wraps the kernels above).
# -------------------------------------------------------------------------
include("operators/sgs_stress.jl")

# -------------------------------------------------------------------------
# Era VI: the tensorial scalar eddy diffusivity (wraps the kernels above).
# -------------------------------------------------------------------------
include("operators/scalar_tensor.jl")
# Pointwise N² and Ri_g (Era VI, P58-a).
include("operators/stratification.jl")

# Era VI: the stability-corrected anisotropic eddy viscosity / diffusivity
# (P55-e; uses the contract, the scalar operators and the N² / Ri_g stencils).
include("operators/sgs_stability.jl")

end # module
