module Operators

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields
using ..Fields: Field, mesh
using ..BoundaryConditions: fill_halos!

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
       advect_weno5!
# Era VI (P54-a, #167): the SGS closure contract (operators/sgs_closures.jl)
export AbstractSGSClosure, SGSFields, sgs_fields, sgs_coefficients!, advance_sgs!,
       sgs_state, restore_sgs_state!, add_sgs_momentum!, add_sgs_scalar!,
       compute_sgs!, sgs_closure,
       AbstractEddyViscosityClosure, sgs_viscosity!, turbulent_prandtl,
       StaticSmagorinsky, DynamicSmagorinsky, FunctionSGS
# Era VI (P55-b, #175): the stress-form momentum divergence (operators/sgs_stress.jl)
export StressTensor, add_stress_divergence!, sgs_stress_tensor!,
       add_sgs_stress_divergence!, sgs_stress_dissipation

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
    add_eddy_diffusion_scalar!(tϕ, ϕ, κt::Field)
    add_eddy_diffusion_scalar!(tϕ, ϕ, νt::Field, Pr_t)

Add the conservative eddy-diffusion tendency `∂/∂xⱼ(κₜ ∂ϕ/∂xⱼ)` to the
cell-centred scalar tendency `tϕ`, with the cell-centred diffusivity `κt`
averaged linearly to each face. The four-argument form is
`(1/Prₜ) ∂/∂xⱼ(νₜ ∂ϕ/∂xⱼ)`: the same kernel, with the turbulent Prandtl
number applied as a factor after the flux divergence. Halos of `ϕ` and of
the coefficient field must be filled.
"""
add_eddy_diffusion_scalar!(tϕ, ϕ, κt::Field) =
    _add_eddy_diffusion_scalar!(tϕ, ϕ, κt, one(eltype(κt.data)), mesh(ϕ))
add_eddy_diffusion_scalar!(tϕ, ϕ, νt::Field, Pr_t::Real) =
    _add_eddy_diffusion_scalar!(tϕ, ϕ, νt, 1 / Pr_t, mesh(ϕ))

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
"""
@inline function _velocity_gradient_at(u, v, w, i, j, k, idx, idy, idz)
    @inbounds begin
        du_dx = (u[i+1,j,k] - u[i,j,k]) * idx
        dv_dy = (v[i,j+1,k] - v[i,j,k]) * idy
        dw_dz = (w[i,j,k+1] - w[i,j,k]) * idz
        du_dy = ((u[i,j+1,k] + u[i+1,j+1,k]) - (u[i,j-1,k] + u[i+1,j-1,k])) * (0.25 * idy)
        dv_dx = ((v[i+1,j,k] + v[i+1,j+1,k]) - (v[i-1,j,k] + v[i-1,j+1,k])) * (0.25 * idx)
        du_dz = ((u[i,j,k+1] + u[i+1,j,k+1]) - (u[i,j,k-1] + u[i+1,j,k-1])) * (0.25 * idz)
        dw_dx = ((w[i+1,j,k] + w[i+1,j,k+1]) - (w[i-1,j,k] + w[i-1,j,k+1])) * (0.25 * idx)
        dv_dz = ((v[i,j,k+1] + v[i,j+1,k+1]) - (v[i,j,k-1] + v[i,j+1,k-1])) * (0.25 * idz)
        dw_dy = ((w[i,j+1,k] + w[i,j+1,k+1]) - (w[i,j-1,k] + w[i,j-1,k+1])) * (0.25 * idy)
    end
    return ((du_dx, du_dy, du_dz),
            (dv_dx, dv_dy, dv_dz),
            (dw_dx, dw_dy, dw_dz))
end

"""
    _strain_at(u, v, w, i, j, k, idx, idy, idz) -> (S11, S22, S33, S12, S13, S23, |S|)

Strain-rate components `Sᵢⱼ = ½(gᵢⱼ + gⱼᵢ)` and `|S| = √(2 SᵢⱼSᵢⱼ)` at the
centre of cell `(i, j, k)`, from [`_velocity_gradient_at`](@ref). The stencil
of [`compute_smagorinsky_νt!`](@ref) and [`compute_dynamic_smagorinsky_νt!`](@ref).
"""
@inline function _strain_at(u, v, w, i, j, k, idx, idy, idz)
    g = _velocity_gradient_at(u, v, w, i, j, k, idx, idy, idz)
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
                                    averaging = :plane, filter_ratio = 3, cs_max = 0.5) -> νt

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
                                         cs_max::Real = 0.5)
    g = mesh(u)
    g isa RegularGrid || throw(ArgumentError("compute_dynamic_smagorinsky_νt!: RegularGrid only"))
    min(g.Hx, g.Hy, g.Hz) >= 2 ||
        throw(ArgumentError("compute_dynamic_smagorinsky_νt!: needs halo ≥ 2, mesh has $((g.Hx, g.Hy, g.Hz))"))
    averaging in (:plane, :local) ||
        throw(ArgumentError("averaging must be :plane or :local, got $(repr(averaging))"))
    _dynamic_smagorinsky!(νt.data, u.data, v.data, w.data, workspace, g,
                          averaging === :plane, float(filter_ratio)^2, float(cs_max)^2)
    return νt
end

function _dynamic_smagorinsky!(nd, ud, vd, wd, ws::DynamicSmagorinskyWorkspace{T}, g,
                               plane::Bool, α2, cs2max) where {T}
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    Uc, Vc, Wc, magS = ws.Uc, ws.Vc, ws.Wc, ws.magS
    S11, S22, S33, S12, S13, S23 = ws.S11, ws.S22, ws.S33, ws.S12, ws.S13, ws.S23

    # Pass 1: cell-centred velocity and strain on the interior + one shell.
    # Local index l ∈ 1:N+2 is data index H - 1 + l.
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

    # Pass 2: L_ij M_ij and M_ij M_ij at every interior cell.
    inv27 = one(T) / 27
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        li = i + 1; lj = j + 1; lk = k + 1
        Ut = zero(T); Vt = zero(T); Wt = zero(T)
        UU = zero(T); VV = zero(T); WW = zero(T); UV = zero(T); UW = zero(T); VW = zero(T)
        A11 = zero(T); A22 = zero(T); A33 = zero(T); A12 = zero(T); A13 = zero(T); A23 = zero(T)
        B11 = zero(T); B22 = zero(T); B33 = zero(T); B12 = zero(T); B13 = zero(T); B23 = zero(T)
        for dk in -1:1, dj in -1:1, di in -1:1
            a = li + di; b = lj + dj; c = lk + dk
            uu = Uc[a,b,c]; vv = Vc[a,b,c]; ww = Wc[a,b,c]
            Ut += uu; Vt += vv; Wt += ww
            UU += uu*uu; VV += vv*vv; WW += ww*ww
            UV += uu*vv; UW += uu*ww; VW += vv*ww
            s11 = S11[a,b,c]; s22 = S22[a,b,c]; s33 = S33[a,b,c]
            s12 = S12[a,b,c]; s13 = S13[a,b,c]; s23 = S23[a,b,c]
            ms = magS[a,b,c]
            A11 += s11; A22 += s22; A33 += s33; A12 += s12; A13 += s13; A23 += s23
            B11 += ms*s11; B22 += ms*s22; B33 += ms*s33
            B12 += ms*s12; B13 += ms*s13; B23 += ms*s23
        end
        Ut *= inv27; Vt *= inv27; Wt *= inv27
        L11 = UU*inv27 - Ut*Ut; L22 = VV*inv27 - Vt*Vt; L33 = WW*inv27 - Wt*Wt
        L12 = UV*inv27 - Ut*Vt; L13 = UW*inv27 - Ut*Wt; L23 = VW*inv27 - Vt*Wt
        tr = (L11 + L22 + L33) / 3
        L11 -= tr; L22 -= tr; L33 -= tr
        A11 *= inv27; A22 *= inv27; A33 *= inv27; A12 *= inv27; A13 *= inv27; A23 *= inv27
        magSt = sqrt(2 * (A11^2 + A22^2 + A33^2) + 4 * (A12^2 + A13^2 + A23^2))
        # L = T − τ̃ with τ = −2C_s²Δ²|S|S (deviatoric) gives L = C_s² M with
        # M = 2Δ²(~(|S|S) − α²|S̃|S̃). (The example's copy had the opposite
        # sign, so C_s² came out negative in forward scatter and was clipped.)
        c1 = 2 * Δ2 * α2 * magSt; c2 = 2 * Δ2 * inv27
        M11 = c2*B11 - c1*A11; M22 = c2*B22 - c1*A22; M33 = c2*B33 - c1*A33
        M12 = c2*B12 - c1*A12; M13 = c2*B13 - c1*A13; M23 = c2*B23 - c1*A23
        ws.LM[i,j,k] = L11*M11 + L22*M22 + L33*M33 + 2 * (L12*M12 + L13*M13 + L23*M23)
        ws.MM[i,j,k] = M11*M11 + M22*M22 + M33*M33 + 2 * (M12*M12 + M13*M13 + M23*M23)
    end

    # Averaging of the identity and the clip to [0, cs_max²].
    cs2 = ws.cs2
    if plane
        @inbounds for k in 1:Nz
            lm = zero(T); mm = zero(T)
            for j in 1:Ny, i in 1:Nx
                lm += ws.LM[i,j,k]; mm += ws.MM[i,j,k]
            end
            c = mm > 0 ? clamp(lm / mm, zero(T), T(cs2max)) : zero(T)
            for j in 1:Ny, i in 1:Nx
                cs2[i,j,k] = c
            end
        end
    else
        @inbounds for I in eachindex(cs2)
            mm = ws.MM[I]
            cs2[I] = mm > 0 ? clamp(ws.LM[I] / mm, zero(T), T(cs2max)) : zero(T)
        end
    end

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        nd[Hx+i, Hy+j, Hz+k] = cs2[i,j,k] * Δ2 * magS[i+1, j+1, k+1]
    end
    return nothing
end

# -------------------------------------------------------------------------
# Era VI: the SGS closure contract (wraps the kernels above).
# -------------------------------------------------------------------------
include("operators/sgs_closures.jl")

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
                     _advect_weno5!,
                     _add_sgs_stress_divergence!, _add_stress_divergence!,
                     _sgs_stress_tensor!, _sgs_stress_dissipation

# -------------------------------------------------------------------------
# Era VI: the stress-form SGS momentum divergence (wraps the kernels above).
# -------------------------------------------------------------------------
include("operators/sgs_stress.jl")

end # module
