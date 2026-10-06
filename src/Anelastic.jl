"""
    Anelastic

Era II / Phase 27 — **anelastic** and **pseudo-incompressible** dynamics as
selectable alternatives to the Boussinesq core in
[`Simulations`](@ref Aeolus.Simulations).

The three sound-proof systems shipped by Aeolus differ *only* in the
mass-like constraint that the pressure projection enforces:

| System                 | Constraint          | Projection operator      |
| :--------------------- | :------------------ | :----------------------- |
| Boussinesq             | `∇·u = 0`           | `∇²φ`                    |
| Anelastic (Lipps–Hemler 1982) | `∇·(ρ̄ u) = 0` | `∇·(ρ̄ ∇φ)`             |
| Pseudo-incompressible (Durran 1989) | `∇·(ρ̄θ̄ u) = 0` | `∇·(ρ̄θ̄ ∇φ)` |

Everything else — advection, buoyancy, SGS closure, boundary conditions —
follows from that one choice, because the flux-form operators must be
weighted by the *same* profile that appears in the constraint or the
projection stops being exact.

Contents
--------

  * [`ReferenceState`](@ref) and its four constructors
    ([`isothermal_reference`](@ref), [`isentropic_reference`](@ref),
    [`constant_stability_reference`](@ref), [`reference_from_profile`](@ref)).
  * [`DensityWeightedPoissonSolver`](@ref) — matrix-free Jacobi-preconditioned
    CG for the variable-coefficient Poisson problem `∇·(w(z) ∇φ) = f`.
  * [`AnelasticSimulation`](@ref) / [`PseudoIncompressibleSimulation`](@ref)
    with [`step_anelastic!`](@ref) / [`step_pseudo_incompressible!`](@ref).
  * [`system_comparison_diagnostics`](@ref) — the instrument that shows the
    three systems diverging from one another.

### Formulation note (which pressure variable?)

Lipps & Hemler write the momentum equation with `-cₚθ̄∇π′`; Ogura & Phillips
and most LES codes instead project the **mass flux** `ρ̄u` onto its
divergence-free part, which puts a plain scalar potential `φ` in the
momentum equation,

```
u = u* − ∇φ,     ∇·(ρ̄ u) = 0   ⟹   ∇·(ρ̄ ∇φ) = ∇·(ρ̄ u*).
```

Aeolus implements the mass-flux projection, so `AnelasticSimulation` uses
the `∇·(ρ̄ ∇·)` operator. The `-cₚθ̄∇π′` form would instead give the weight
`ρ̄θ̄` — numerically the *same* operator family, and exactly the weight used
by [`PseudoIncompressibleSimulation`](@ref). The two weightings therefore
bracket the usual sound-proof formulations, and they coincide (up to an
irrelevant constant factor) whenever the base state is neutral, `θ̄ = const`.
"""
module Anelastic

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ,
               interior, znodes
using ..Fields
using ..Fields: Field, interior_view, set!
using ..BoundaryConditions
using ..BoundaryConditions: FieldBoundaryConditions, fill_halos!,
                            Periodic, NoFlux, ImpermeableWall,
                            update_open_boundaries!, prepare_open_boundaries!,
                            assert_open_face_correction, mark_open_boundary_stage!
import ..BoundaryConditions: boundary_flux_weights
using ..Operators
using ..Operators: laplacian, divergence, ∂x, ∂y, ∂z,
                   add_coriolis!, compute_smagorinsky_νt!,
                   add_eddy_viscosity_u!, add_eddy_viscosity_v!,
                   add_eddy_viscosity_w!, add_eddy_diffusion_scalar!,
                   add_surface_flux!, add_sponge!,
                   compute_sgs!, add_sgs_momentum!, add_sgs_scalar!, advance_sgs!,
                   _active_sgs, _check_sgs_stress
using ..PressureSolvers: AbstractPressureSolver
using ..RunLog: current_domain   # structured events (LOG-a, #333)

import ..PressureSolvers: solve!, poisson_axis_bcs
import ..TimeSteppers: step!
import ..Simulations: run!, kinetic_energy, max_divergence, mean_θ_profile

using ..Simulations: horizontal_mean, horizontal_variance, add_velocity_sponge!
import ..Simulations
using ..TurbulenceStability: buoyancy_frequency_squared   # the one N² (P58-a)
using ..Operators: sgs_fields
using ..TimeIntegration: CFLConstraints, _max_advective_rate, _SNAPSHOTS
import ..TimeIntegration: adaptive_dt, StateSnapshot, save_state!, restore_state!,
                          state_snapshot, _all_finite

export ReferenceState,
       isothermal_reference, isentropic_reference,
       constant_stability_reference, reference_from_profile,
       hydrostatic_residual, max_hydrostatic_residual,
       reference_scale_height,
       DensityWeightedPoissonSolver, apply_weighted_laplacian!,
       solver_iterations, solver_residual,
       MassWeighting, AnelasticWeighting, PseudoIncompressibleWeighting,
       MassWeightedSimulation, AnelasticSimulation, PseudoIncompressibleSimulation,
       step_anelastic!, step_pseudo_incompressible!,
       anelastic_divergence!, pseudo_incompressible_divergence!,
       mass_flux_divergence!, weighted_divergence!,
       max_anelastic_divergence, max_mass_flux_divergence,
       max_pseudo_incompressible_divergence, constraint_residual,
       add_anelastic_buoyancy!,
       weighted_momentum_advect!, weighted_scalar_advect!,
       set_reference_state!, reference_state, constraint_weights,
       θ_variance, max_w, bubble_top_height, unweighted_kinetic_energy,
       system_comparison_diagnostics

# =========================================================================
# Thermodynamic constants (dry air)
# =========================================================================

"Gas constant of dry air, J kg⁻¹ K⁻¹."
const R_DRY   = 287.0
"Isobaric specific heat of dry air, J kg⁻¹ K⁻¹."
const CP_DRY  = 1004.0
"Reference pressure for the Exner function / potential temperature, Pa."
const P_REF   = 1.0e5
"Standard gravity, m s⁻²."
const G_STD   = 9.81

# =========================================================================
# ReferenceState
# =========================================================================

"""
    ReferenceState{T, M}

Prescribed, horizontally homogeneous base state `(ρ̄, θ̄, p̄, T̄)(z)` for the
sound-proof systems, **in discrete hydrostatic balance on the mesh it was
built for**.

# Stored profiles

Cell-centred, length `Nz` (interior levels only):

| Field | Meaning |
| :---- | :------ |
| `ρ̄`  | base-state density, kg m⁻³ |
| `θ̄`  | base-state potential temperature, K |
| `p̄`  | base-state pressure, Pa |
| `T̄`  | base-state temperature, K |
| `π̄`  | Exner function `(p̄/p₀)^{R/cₚ}` |
| `N²` | Brunt–Väisälä frequency squared, s⁻² |
| `P`  | pseudo-density `ρ̄θ̄` (the Durran mass variable) |

Face-located (`FaceZ`), length `Nz+1`; entry `k` is the **bottom face of
cell `k`**, so entry `Nz+1` is the domain top:

| Field | Meaning |
| :---- | :------ |
| `ρ̄_f` | face density — the coefficient of the anelastic mass flux |
| `θ̄_f` | face potential temperature |
| `P_f`  | `ρ̄_f θ̄_f` |

Each profile also has a halo-padded twin (`ρ̄c_pad`, `ρ̄f_pad`, `θ̄c_pad`,
`θ̄f_pad`, `Pc_pad`, `Pf_pad`) whose index is the *storage-array* `k` index,
so kernels can write `ρ̄c_pad[k]` next to `u.data[i,j,k]` with no offset
arithmetic. Halo entries are filled by log-linear extrapolation, which keeps
them strictly positive.

# Discrete hydrostatic balance

The balance that is exact (to round-off) is the one the staggered
C-grid actually evaluates in the `w`-momentum equation — a cell-to-cell
pressure difference against the *face* density:

```
(p̄[k] − p̄[k−1]) / Δz  =  −g ρ̄_f[k],        k = 2 … Nz
```

and `ρ̄_f[k]` is the **logarithmic mean** of `ρ̄[k−1]` and `ρ̄[k]`,

```
ρ̄_f = (ρ̄[k−1] − ρ̄[k]) / log(ρ̄[k−1]/ρ̄[k]).
```

The logarithmic mean is symmetric (hence the projection operator stays
self-adjoint, see [`DensityWeightedPoissonSolver`](@ref)), lies between the
geometric and arithmetic means (hence strictly positive), and — the reason
it is chosen here — it makes the discrete balance above *identically exact*
for an exponential density profile. That is what lets
[`isothermal_reference`](@ref) be the exact analytic `ρ₀e^{−z/H}` and still
satisfy the discrete balance to `~1e-13`. Check it with
[`max_hydrostatic_residual`](@ref).

`θ̄_f` uses the **arithmetic** mean instead, because the buoyancy kernel
interpolates the prognostic `θ` to the `w`-face arithmetically; matching the
two guarantees that a base-state-only atmosphere produces exactly zero
buoyancy. At the two wall faces both face profiles mirror the adjacent cell
value, matching the `NoFlux`/`ImpermeableWall` halo fill.

See [`isothermal_reference`](@ref), [`isentropic_reference`](@ref),
[`constant_stability_reference`](@ref), [`reference_from_profile`](@ref).
"""
struct ReferenceState{T, M<:RegularGrid{T}}
    mesh::M
    # thermodynamic constants actually used (stored so diagnostics agree)
    R_d::T
    cp::T
    cv::T
    p₀::T
    g::T
    # cell-centred profiles, length Nz
    ρ̄::Vector{T}
    θ̄::Vector{T}
    p̄::Vector{T}
    T̄::Vector{T}
    π̄::Vector{T}
    N²::Vector{T}
    P::Vector{T}
    # face profiles, length Nz+1 (entry k = bottom face of cell k)
    ρ̄_f::Vector{T}
    θ̄_f::Vector{T}
    P_f::Vector{T}
    # halo-padded, storage-index-aligned twins, length Nz + 2Hz
    ρ̄c_pad::Vector{T}
    ρ̄f_pad::Vector{T}
    θ̄c_pad::Vector{T}
    θ̄f_pad::Vector{T}
    Pc_pad::Vector{T}
    Pf_pad::Vector{T}
end

function Base.show(io::IO, r::ReferenceState{T}) where {T}
    print(io, "ReferenceState{$T}(Nz=$(r.mesh.Nz), ")
    print(io, "ρ̄ ∈ [$(round(minimum(r.ρ̄); digits=4)), $(round(maximum(r.ρ̄); digits=4))], ")
    print(io, "θ̄ ∈ [$(round(minimum(r.θ̄); digits=3)), $(round(maximum(r.θ̄); digits=3))], ")
    print(io, "max|hydrostatic residual| = $(max_hydrostatic_residual(r)))")
end

# ------------------------------------------------------------------ means

"""
    _logmean(a, b)

Logarithmic (Lorenz) mean `(a-b)/log(a/b)`, symmetric in its arguments and
continuous at `a == b` where it equals `a`. Evaluated by a series in
`x = (a-b)/(a+b)` when the arguments are close, so the `0/0` is never
formed and full precision is retained.
"""
@inline function _logmean(a::T, b::T) where {T<:AbstractFloat}
    (a <= 0 || b <= 0) && return (a + b) / 2
    s = a + b
    d = a - b
    d == 0 && return a
    x = d / s
    if abs(x) < T(1e-3)
        # 1/(1 + x²/3 + x⁴/5 + …) = 1 − x²/3 − (4/45)x⁴ + O(x⁶)
        x² = x * x
        return (s / 2) * (1 - x² / 3 - (4 * x² * x²) / 45)
    end
    return d / log(a / b)
end

@inline _arithmean(a::T, b::T) where {T} = (a + b) / 2

# ---------------------------------------------------------------- padding

function _pad_center(vals::Vector{T}, Hz::Int) where {T}
    Nz = length(vals)
    out = Vector{T}(undef, Nz + 2Hz)
    @inbounds out[Hz+1:Hz+Nz] .= vals
    rb = Nz >= 2 ? vals[1] / vals[2]      : one(T)
    rt = Nz >= 2 ? vals[Nz] / vals[Nz-1]  : one(T)
    @inbounds for h in 1:Hz
        out[Hz + 1 - h]  = vals[1]  * rb^h
        out[Hz + Nz + h] = vals[Nz] * rt^h
    end
    return out
end

function _pad_face(vals::Vector{T}, Hz::Int, Nz::Int) where {T}
    n = length(vals)                  # == Nz + 1
    @assert n == Nz + 1
    @assert Hz >= 1 "face profiles need at least one z halo cell"
    out = Vector{T}(undef, Nz + 2Hz)
    @inbounds out[Hz+1:Hz+Nz+1] .= vals
    rb = n >= 2 ? vals[1] / vals[2]    : one(T)
    rt = n >= 2 ? vals[n] / vals[n-1]  : one(T)
    @inbounds for h in 1:Hz
        out[Hz + 1 - h] = vals[1] * rb^h
        idx = Hz + Nz + 1 + h
        idx <= length(out) && (out[idx] = vals[n] * rt^h)
    end
    return out
end

# --------------------------------------------------------- thermodynamics

"""
    _density(p, θ, R, cp, p₀) -> ρ

Density from pressure and potential temperature via the ideal-gas law and
the Exner relation: `ρ = (p₀/(Rθ)) (p/p₀)^{cᵥ/cₚ}`.
"""
@inline _density(p::T, θ::T, R::T, cp::T, p₀::T) where {T} =
    (p₀ / (R * θ)) * (p / p₀)^((cp - R) / cp)

"""
    _brunt_vaisala(θ̄, dz, g) -> N²

`N² = g ∂(log θ̄)/∂z` on cell centres: a centred difference of `log θ̄` in
the interior and one-sided differences at the two end levels. Using the
*logarithmic* derivative (rather than `g/θ̄ · Δθ̄/Δz`) makes the result exact
to round-off for the analytic constant-`N²` profile `θ̄ = θ_s e^{N²z/g}`,
which is what [`constant_stability_reference`](@ref) prescribes. The
difference is the logarithmic form of
`TurbulenceStability.buoyancy_frequency_squared`.
"""
function _brunt_vaisala(θ̄::Vector{T}, dz::T, g::T) where {T}
    Nz = length(θ̄)
    N² = zeros(T, Nz)
    Nz == 1 && return N²
    @inbounds begin
        N²[1] = buoyancy_frequency_squared(θ̄[1], θ̄[2], dz, g)
        for k in 2:Nz-1
            N²[k] = buoyancy_frequency_squared(θ̄[k-1], θ̄[k+1], 2dz, g)
        end
        N²[Nz] = buoyancy_frequency_squared(θ̄[Nz-1], θ̄[Nz], dz, g)
    end
    return N²
end

"""
    _assemble_reference(mesh, θ̄, p̄, ρ̄; R_d, cp, p₀, g) -> ReferenceState

Build every derived profile (Exner, `T̄`, `N²`, the face profiles and their
halo-padded twins) from cell-centred `θ̄`, `p̄`, `ρ̄`.
"""
function _assemble_reference(mesh::RegularGrid{T}, θ̄::Vector{T},
                             p̄::Vector{T}, ρ̄::Vector{T};
                             R_d::T, cp::T, p₀::T, g::T) where {T}
    Nz, Hz, dz = mesh.Nz, mesh.Hz, mesh.dz
    @assert length(θ̄) == Nz && length(p̄) == Nz && length(ρ̄) == Nz
    all(>(0), ρ̄) || error("reference density must be strictly positive")
    all(>(0), θ̄) || error("reference potential temperature must be strictly positive")

    π̄ = @. (p̄ / p₀)^(R_d / cp)
    T̄ = @. θ̄ * π̄
    N² = _brunt_vaisala(θ̄, dz, g)
    P  = ρ̄ .* θ̄

    # Face profiles. Interior faces: log-mean for ρ̄ (exact hydrostatic
    # balance for exponential profiles), arithmetic for θ̄ (matches the
    # arithmetic θ interpolation in the buoyancy kernel). Wall faces mirror
    # the adjacent cell, matching the NoFlux / ImpermeableWall halo fill.
    ρ̄_f = Vector{T}(undef, Nz + 1)
    θ̄_f = Vector{T}(undef, Nz + 1)
    @inbounds begin
        ρ̄_f[1] = ρ̄[1];  θ̄_f[1] = θ̄[1]
        for k in 2:Nz
            ρ̄_f[k] = _logmean(ρ̄[k-1], ρ̄[k])
            θ̄_f[k] = _arithmean(θ̄[k-1], θ̄[k])
        end
        ρ̄_f[Nz+1] = ρ̄[Nz];  θ̄_f[Nz+1] = θ̄[Nz]
    end
    P_f = ρ̄_f .* θ̄_f

    ρ̄c_pad = _pad_center(ρ̄, Hz)
    θ̄c_pad = _pad_center(θ̄, Hz)
    ρ̄f_pad = _pad_face(ρ̄_f, Hz, Nz)
    θ̄f_pad = _pad_face(θ̄_f, Hz, Nz)
    Pc_pad = ρ̄c_pad .* θ̄c_pad
    Pf_pad = ρ̄f_pad .* θ̄f_pad

    return ReferenceState{T, typeof(mesh)}(mesh, R_d, cp, cp - R_d, p₀, g,
                                           ρ̄, θ̄, p̄, T̄, π̄, N², P,
                                           ρ̄_f, θ̄_f, P_f,
                                           ρ̄c_pad, ρ̄f_pad,
                                           θ̄c_pad, θ̄f_pad,
                                           Pc_pad, Pf_pad)
end

"""
    _integrate_hydrostatic(mesh, θ̄, p_surf; R_d, cp, p₀, g) -> (p̄, ρ̄)

Integrate the **discrete** hydrostatic equation upward from a surface
pressure `p_surf` prescribed at `z = z₀` (the bottom wall face), given a
prescribed `θ̄(z)` at cell centres.

Level 1 sits half a cell above the wall, so it is closed with the half-cell
balance `p̄[1] = p_surf − g(Δz/2)ρ̄[1]`; this only sets the absolute pressure
level. Every subsequent level solves the *nonlinear* two-point balance

```
p̄[k] = p̄[k−1] − g Δz · logmean(ρ̄[k−1], ρ(p̄[k], θ̄[k]))
```

by fixed-point iteration (contraction factor `≈ gΔz·cᵥρ̄/(2cₚp̄) ~ 10⁻³`, so
a handful of sweeps reaches round-off). The result satisfies the discrete
balance to machine precision *by construction* — no `O(Δz²)` slop.
"""
function _integrate_hydrostatic(mesh::RegularGrid{T}, θ̄::Vector{T},
                                p_surf::T; R_d::T, cp::T, p₀::T, g::T) where {T}
    Nz, dz = mesh.Nz, mesh.dz
    p̄ = Vector{T}(undef, Nz)
    ρ̄ = Vector{T}(undef, Nz)

    # --- level 1: half-cell balance from the surface ---------------------
    p = p_surf
    for _ in 1:200
        ρ = _density(p, θ̄[1], R_d, cp, p₀)
        pn = p_surf - g * (dz / 2) * ρ
        Δ = abs(pn - p); p = pn
        Δ <= 8 * eps(T) * abs(p) && break
    end
    p̄[1] = p
    ρ̄[1] = _density(p, θ̄[1], R_d, cp, p₀)

    # --- levels 2…Nz: discrete two-point balance ------------------------
    @inbounds for k in 2:Nz
        p = p̄[k-1] - g * dz * ρ̄[k-1]          # explicit first guess
        for _ in 1:200
            ρ  = _density(p, θ̄[k], R_d, cp, p₀)
            pn = p̄[k-1] - g * dz * _logmean(ρ̄[k-1], ρ)
            pn > 0 || error("hydrostatic integration produced non-positive pressure at k=$k; " *
                            "domain is too deep for the prescribed θ̄ / p_surf")
            Δ = abs(pn - p); p = pn
            Δ <= 8 * eps(T) * abs(p) && break
        end
        p̄[k] = p
        ρ̄[k] = _density(p, θ̄[k], R_d, cp, p₀)
    end
    return p̄, ρ̄
end

# ------------------------------------------------------------ constructors

"""
    isothermal_reference(mesh; T_surf=288.0, p_surf=1e5,
                         R_d=287.0, cp=1004.0, p₀=1e5, g_accel=9.81)

Isothermal base state — the one case with a closed-form answer, so it is
evaluated **analytically** rather than integrated:

```
p̄(z) = p_surf e^{−ζ/H},   ρ̄(z) = p̄/(R T),   H = R T / g,   ζ = z − z₀
```

with `θ̄ = T (p₀/p̄)^{R/cₚ}` following from the definition of potential
temperature. Because the face density uses the logarithmic mean (see
[`ReferenceState`](@ref)), this exact exponential *also* satisfies the
discrete hydrostatic balance to round-off — the two requirements are not in
tension. `reference_scale_height(ref)` is not stored; use `R_d*T_surf/g`.
"""
function isothermal_reference(mesh::RegularGrid{T};
                              T_surf::Real = 288.0,
                              p_surf::Real = P_REF,
                              R_d::Real    = R_DRY,
                              cp::Real     = CP_DRY,
                              p₀::Real     = P_REF,
                              g_accel::Real = G_STD) where {T}
    R = T(R_d); c = T(cp); p0 = T(p₀); gg = T(g_accel)
    Ts = T(T_surf); ps = T(p_surf)
    H  = R * Ts / gg
    ζ  = collect(znodes(mesh, CellCenter())) .- mesh.z0
    p̄  = @. ps * exp(-ζ / H)
    ρ̄  = @. p̄ / (R * Ts)
    θ̄  = @. Ts * (p0 / p̄)^(R / c)
    return _assemble_reference(mesh, θ̄, p̄, ρ̄; R_d = R, cp = c, p₀ = p0, g = gg)
end

"""
    isentropic_reference(mesh; θ_const=300.0, p_surf=1e5, …)

Neutral (constant-`θ`) base state — `N² ≡ 0` exactly. The classic dry
warm-bubble background. Pressure and density come from the discrete
hydrostatic integration, so the balance is exact on the model grid.

With `θ̄ = const` the anelastic weight `ρ̄` and the pseudo-incompressible
weight `ρ̄θ̄` differ only by a constant factor, so
[`AnelasticSimulation`](@ref) and [`PseudoIncompressibleSimulation`](@ref)
produce *identical* trajectories on this base state — a useful exactness
check, and the reason the two systems must be compared on a stratified base
state to see them separate.
"""
function isentropic_reference(mesh::RegularGrid{T};
                              θ_const::Real = 300.0,
                              p_surf::Real  = P_REF,
                              R_d::Real     = R_DRY,
                              cp::Real      = CP_DRY,
                              p₀::Real      = P_REF,
                              g_accel::Real = G_STD) where {T}
    R = T(R_d); c = T(cp); p0 = T(p₀); gg = T(g_accel)
    θ̄ = fill(T(θ_const), mesh.Nz)
    p̄, ρ̄ = _integrate_hydrostatic(mesh, θ̄, T(p_surf); R_d = R, cp = c, p₀ = p0, g = gg)
    return _assemble_reference(mesh, θ̄, p̄, ρ̄; R_d = R, cp = c, p₀ = p0, g = gg)
end

"""
    constant_stability_reference(mesh; N=0.01, θ_surf=300.0, p_surf=1e5, …)

Constant-stability base state, the standard stratified test atmosphere:

```
θ̄(z) = θ_surf exp(N² ζ / g),      ζ = z − z₀
```

which is the exact solution of `N² = g ∂ log θ̄/∂z` for constant `N²`. The
`N²` diagnostic stored in the returned [`ReferenceState`](@ref) reproduces
`N² = N^2` to round-off because it is evaluated as a difference of `log θ̄`
(see `_brunt_vaisala`). `N` is the frequency in s⁻¹; pass `N = 0.01` for the
usual mid-tropospheric value `N² = 10⁻⁴ s⁻²`.
"""
function constant_stability_reference(mesh::RegularGrid{T};
                                      N::Real       = 0.01,
                                      θ_surf::Real  = 300.0,
                                      p_surf::Real  = P_REF,
                                      R_d::Real     = R_DRY,
                                      cp::Real      = CP_DRY,
                                      p₀::Real      = P_REF,
                                      g_accel::Real = G_STD) where {T}
    R = T(R_d); c = T(cp); p0 = T(p₀); gg = T(g_accel)
    N² = T(N)^2
    ζ  = collect(znodes(mesh, CellCenter())) .- mesh.z0
    θ̄  = @. T(θ_surf) * exp(N² * ζ / gg)
    p̄, ρ̄ = _integrate_hydrostatic(mesh, θ̄, T(p_surf); R_d = R, cp = c, p₀ = p0, g = gg)
    return _assemble_reference(mesh, θ̄, p̄, ρ̄; R_d = R, cp = c, p₀ = p0, g = gg)
end

"""
    reference_from_profile(mesh, θ_profile, p_surf=1e5; …)

Hydrostatically integrate an arbitrary potential-temperature profile.
`θ_profile` may be

  * a `Function` — sampled as `θ_profile(z)` at the `Nz` interior cell
    centres, or
  * an `AbstractVector` of length `Nz` — used directly.

Use this for sounding-driven base states (e.g. a mixed layer capped by an
inversion). The discrete hydrostatic balance is exact regardless of how
rough the profile is.
"""
function reference_from_profile(mesh::RegularGrid{T}, θ_profile,
                                p_surf::Real = P_REF;
                                R_d::Real     = R_DRY,
                                cp::Real      = CP_DRY,
                                p₀::Real      = P_REF,
                                g_accel::Real = G_STD) where {T}
    R = T(R_d); c = T(cp); p0 = T(p₀); gg = T(g_accel)
    θ̄ = if θ_profile isa Function
        T[T(θ_profile(z)) for z in znodes(mesh, CellCenter())]
    else
        length(θ_profile) == mesh.Nz ||
            throw(DimensionMismatch("θ_profile has length $(length(θ_profile)), expected Nz=$(mesh.Nz)"))
        T.(collect(θ_profile))
    end
    p̄, ρ̄ = _integrate_hydrostatic(mesh, θ̄, T(p_surf); R_d = R, cp = c, p₀ = p0, g = gg)
    return _assemble_reference(mesh, θ̄, p̄, ρ̄; R_d = R, cp = c, p₀ = p0, g = gg)
end

# ---------------------------------------------------------- balance check

"""
    hydrostatic_residual(ref) -> Vector

Residual of the **discrete** hydrostatic equation on the interior faces
`k = 2 … Nz`:

```
r[k−1] = (p̄[k] − p̄[k−1])/Δz + g ρ̄_f[k]        [Pa m⁻¹]
```

This is exactly the stencil the `w`-momentum equation uses (a cell-to-cell
pressure difference against the face density), so a zero residual means the
base state exerts no spurious vertical acceleration on the model grid. Every
constructor in this module drives it to round-off; see
[`max_hydrostatic_residual`](@ref).
"""
function hydrostatic_residual(ref::ReferenceState{T}) where {T}
    Nz, dz, g = ref.mesh.Nz, ref.mesh.dz, ref.g
    Nz < 2 && return T[]
    r = Vector{T}(undef, Nz - 1)
    @inbounds for k in 2:Nz
        r[k-1] = (ref.p̄[k] - ref.p̄[k-1]) / dz + g * ref.ρ̄_f[k]
    end
    return r
end

"""
    max_hydrostatic_residual(ref) -> Real

`maximum(abs, hydrostatic_residual(ref))`, or `0` for a single-level mesh.
"""
function max_hydrostatic_residual(ref::ReferenceState{T}) where {T}
    r = hydrostatic_residual(ref)
    isempty(r) && return zero(T)
    return maximum(abs, r)
end

"""
    reference_scale_height(ref) -> Real

Density e-folding depth diagnosed from the stored profile,
`H = −(z_top − z_bot)/log(ρ̄[Nz]/ρ̄[1])`. Handy for quantifying "how
non-Boussinesq" a domain is: the Boussinesq error scales with `Lz/H`.
"""
function reference_scale_height(ref::ReferenceState{T}) where {T}
    Nz = ref.mesh.Nz
    Nz < 2 && return T(Inf)
    depth = (Nz - 1) * ref.mesh.dz
    return -depth / log(ref.ρ̄[Nz] / ref.ρ̄[1])
end

# =========================================================================
# DensityWeightedPoissonSolver
# =========================================================================

"""
    DensityWeightedPoissonSolver(mesh, w_c, w_f; tol=1e-10, maxit=500,
                                 bc_x=:periodic, bc_y=:periodic, bc_z=:neumann)
    DensityWeightedPoissonSolver(mesh, ref::ReferenceState; weighting=:ρ, kwargs...)
    DensityWeightedPoissonSolver(mesh, c::Real; kwargs...)

Matrix-free, Jacobi-preconditioned conjugate-gradient solver for the
variable-coefficient Poisson problem

```math
\\nabla\\cdot\\left(w(z)\\,\\nabla\\varphi\\right) = f
```

with `w > 0` a function of height only. Drop-in replacement for the other
[`AbstractPressureSolver`](@ref Aeolus.PressureSolvers.AbstractPressureSolver)s:
the interface is the same `solve!(p, source, solver)`. The constant-coefficient
FFT/DCT solvers cannot do this — the vertical operator is no longer
diagonalised by a cosine transform once `w` varies with `k`.

# Discrete operator

`w` depends only on `z`, so the horizontal faces of cell `k` all carry the
cell value `w_c[k]`, while the two vertical faces carry `w_f[k]` (bottom)
and `w_f[k+1]` (top):

```
(Lφ)ᵢⱼₖ = w_c[k] (δ²ₓ + δ²_y)φ
        + [ w_f[k+1](φᵏ⁺¹ − φᵏ) − w_f[k](φᵏ − φᵏ⁻¹) ] / Δz²
```

**Averaging `w` to faces is what makes `L` self-adjoint.** The coefficient
coupling cells `k` and `k+1` is `w_f[k+1]/Δz²` seen from *either* side, so
`L = Lᵀ` exactly; had we instead used the cell value `w_c[k]` in front of a
centred second difference, the coupling would be `w_c[k]` from below and
`w_c[k+1]` from above and CG would be solving a non-symmetric problem.
`w_f` is supplied by the caller — [`ReferenceState`](@ref) builds it with the
*logarithmic* mean, which is symmetric (so self-adjointness is preserved),
strictly positive, and makes the discrete hydrostatic balance exact for
exponential density; a plain arithmetic mean is equally admissible for the
solver and differs only at `O(Δz²)`.

# Boundary conditions

Each axis is `:periodic` or `:neumann`. In `x` and `y` the default is
periodic; `bc_x = :neumann` / `bc_y = :neumann` sets the flux `w ∂φ/∂n`
through the two lateral faces to zero (mirror ghost) — the closure of a
nested or open-boundary child, whose normal velocity is prescribed on the
side faces (#251; before that the solver was periodic in `x, y` by
construction). In `z`, `bc_z = :neumann` (default) sets the
vertical flux through the two wall faces to zero — the condition consistent
with `ImpermeableWall`/`NoFlux` halos and with `w = 0` at the walls — or
`bc_z = :periodic`, in which case `w_f[1]` and `w_f[Nz+1]` describe the same
face and are symmetrised for you. The Jacobi preconditioner uses the
interior diagonal in every cell; on a Neumann side the boundary cells'
true diagonal is smaller by `w_c/Δ²`, which leaves the preconditioner
symmetric positive definite (CG still converges, to the same solution).

`L` is singular with the constants as its null space, so the source is
projected to zero mean and the returned solution is the unique zero-mean
solution. The preconditioned direction is projected too, which keeps every
iterate inside the zero-mean subspace (a deflated Jacobi preconditioner) —
without that, `M⁻¹ = diag(L)⁻¹` leaks a growing constant into `φ` because the
diagonal varies with `k`.

Convergence is `‖r‖₂ < tol · ‖b‖₂`; inspect with [`solver_iterations`](@ref)
and [`solver_residual`](@ref). Apply the operator directly (for manufactured
solutions) with [`apply_weighted_laplacian!`](@ref).
"""
mutable struct DensityWeightedPoissonSolver{T, RA<:Array{T,3}} <: AbstractPressureSolver
    mesh::RegularGrid{T}
    w_c::Vector{T}          # cell weight, length Nz
    w_f::Vector{T}          # face weight, length Nz+1
    diag::Vector{T}         # positive diagonal of (−L), length Nz
    tol::T
    maxit::Int
    bc_z::Symbol
    bc_x::Symbol
    bc_y::Symbol
    # scratch, all interior-sized
    b::RA
    x::RA
    r::RA
    p_dir::RA
    Ap::RA
    z::RA
    last_iter::Base.RefValue{Int}
    last_residual::Base.RefValue{T}
end

function DensityWeightedPoissonSolver(mesh::RegularGrid{T},
                                      w_c::AbstractVector,
                                      w_f::AbstractVector;
                                      tol::Real  = 1e-10,
                                      maxit::Int = 500,
                                      bc_z::Symbol = :neumann,
                                      bc_x::Symbol = :periodic,
                                      bc_y::Symbol = :periodic) where {T}
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    bc_x in (:neumann, :periodic) || error("bc_x must be :neumann or :periodic, got $bc_x")
    bc_y in (:neumann, :periodic) || error("bc_y must be :neumann or :periodic, got $bc_y")
    length(w_c) == Nz    || throw(DimensionMismatch("w_c must have length Nz=$Nz"))
    length(w_f) == Nz + 1 || throw(DimensionMismatch("w_f must have length Nz+1=$(Nz+1)"))
    bc_z in (:neumann, :periodic) || error("bc_z must be :neumann or :periodic, got $bc_z")
    wc = T.(collect(w_c)); wf = T.(collect(w_f))
    all(>(0), wc) && all(>(0), wf) || error("Poisson weights must be strictly positive")
    if bc_z === :periodic
        # w_f[1] and w_f[Nz+1] are the same physical face under periodicity;
        # symmetrise so the operator stays self-adjoint.
        wsym = _logmean(wf[1], wf[Nz+1])
        wf[1] = wsym; wf[Nz+1] = wsym
    end

    idx2 = 1 / mesh.dx^2; idy2 = 1 / mesh.dy^2; idz2 = 1 / mesh.dz^2
    diag = Vector{T}(undef, Nz)
    @inbounds for k in 1:Nz
        ct = (k < Nz || bc_z === :periodic) ? wf[k == Nz ? Nz + 1 : k + 1] : zero(T)
        cb = (k > 1  || bc_z === :periodic) ? wf[k]                        : zero(T)
        diag[k] = wc[k] * (2idx2 + 2idy2) + (ct + cb) * idz2
        diag[k] > 0 || error("degenerate Jacobi diagonal at k=$k (single-cell domain?)")
    end

    mk() = zeros(T, Nx, Ny, Nz)
    return DensityWeightedPoissonSolver{T, Array{T,3}}(
        mesh, wc, wf, diag, T(tol), maxit, bc_z, bc_x, bc_y,
        mk(), mk(), mk(), mk(), mk(), mk(), Ref(0), Ref(zero(T)))
end

"Anelastic (`:ρ`) or pseudo-incompressible (`:ρθ`) weighting taken from `ref`."
function DensityWeightedPoissonSolver(mesh::RegularGrid{T}, ref::ReferenceState{T};
                                      weighting::Symbol = :ρ, kwargs...) where {T}
    ref.mesh === mesh || ref.mesh == mesh ||
        error("ReferenceState was built for a different mesh")
    if weighting === :ρ
        return DensityWeightedPoissonSolver(mesh, ref.ρ̄, ref.ρ̄_f; kwargs...)
    elseif weighting === :ρθ
        return DensityWeightedPoissonSolver(mesh, ref.P, ref.P_f; kwargs...)
    else
        error("weighting must be :ρ (anelastic) or :ρθ (pseudo-incompressible), got $weighting")
    end
end

"Constant weight `c` — reduces to `c ∇²φ = f` and reproduces `CGPoissonSolver`."
DensityWeightedPoissonSolver(mesh::RegularGrid{T}, c::Real; kwargs...) where {T} =
    DensityWeightedPoissonSolver(mesh, fill(T(c), mesh.Nz), fill(T(c), mesh.Nz + 1); kwargs...)

poisson_axis_bcs(s::DensityWeightedPoissonSolver) = (bc_x = s.bc_x, bc_y = s.bc_y, bc_z = s.bc_z)

"Iterations used by the last [`solve!`](@ref)."
solver_iterations(s::DensityWeightedPoissonSolver) = s.last_iter[]
"Final residual 2-norm of the last [`solve!`](@ref)."
solver_residual(s::DensityWeightedPoissonSolver) = s.last_residual[]

# --- kernels -------------------------------------------------------------

# out = −∇·(w ∇u) on an interior-sized array (positive semi-definite form).
function _apply_neg_wlap!(out::Array{T,3}, u::Array{T,3},
                          w_c::Vector{T}, w_f::Vector{T},
                          dx::T, dy::T, dz::T, bc_z::Symbol,
                          bc_x::Symbol = :periodic, bc_y::Symbol = :periodic) where {T}
    Nx, Ny, Nz = size(u)
    idx2 = 1 / dx^2; idy2 = 1 / dy^2; idz2 = 1 / dz^2
    periodic_z = bc_z === :periodic
    # Neumann side: the ghost mirrors the boundary cell, so the face
    # difference (and the flux through the face) is zero.
    px = bc_x === :periodic; py = bc_y === :periodic
    @inbounds for k in 1:Nz
        wk = w_c[k]
        wt = w_f[k == Nz ? Nz + 1 : k + 1]
        wb = w_f[k]
        kp = k < Nz ? k + 1 : (periodic_z ? 1  : 0)
        km = k > 1  ? k - 1 : (periodic_z ? Nz : 0)
        for j in 1:Ny
            jp = j < Ny ? j + 1 : (py ? 1  : Ny)
            jm = j > 1  ? j - 1 : (py ? Ny : 1)
            for i in 1:Nx
                ip = i < Nx ? i + 1 : (px ? 1  : Nx)
                im = i > 1  ? i - 1 : (px ? Nx : 1)
                c = u[i, j, k]
                lap = wk * ((u[ip, j, k] - 2c + u[im, j, k]) * idx2 +
                            (u[i, jp, k] - 2c + u[i, jm, k]) * idy2)
                ft = kp == 0 ? zero(T) : wt * (u[i, j, kp] - c)
                fb = km == 0 ? zero(T) : wb * (c - u[i, j, km])
                out[i, j, k] = -(lap + (ft - fb) * idz2)
            end
        end
    end
    return out
end

"""
    apply_weighted_laplacian!(out::Field, φ::Field, solver) -> out

Evaluate the *discrete* operator `∇·(w ∇φ)` of `solver` on the interior of
`φ`, writing the result to the interior of `out`. Halos of `φ` are ignored —
the operator applies its own periodic / Neumann closure, exactly as during
[`solve!`](@ref) — so this is the right tool for a manufactured-solution
test: build `f = apply_weighted_laplacian!(f, φ, solver)`, solve for `φ`, and
the recovery error is purely the CG tolerance.
"""
function apply_weighted_laplacian!(out::Field{CellCenter,T}, φ::Field{CellCenter,T},
                                   s::DensityWeightedPoissonSolver{T}) where {T}
    ii, jj, kk = interior(φ.mesh)
    xin  = Array{T,3}(view(φ.data, ii, jj, kk))
    xout = similar(xin)
    _apply_neg_wlap!(xout, xin, s.w_c, s.w_f, s.mesh.dx, s.mesh.dy, s.mesh.dz, s.bc_z,
                     s.bc_x, s.bc_y)
    @inbounds @views out.data[ii, jj, kk] .= .-xout
    return out
end

@inline function _dot(a::Array{T,3}, b::Array{T,3}) where {T}
    s = zero(T)
    @inbounds @simd for i in eachindex(a)
        s += a[i] * b[i]
    end
    return s
end

@inline function _remove_mean!(a::Array{T,3}) where {T}
    m = sum(a) / length(a)
    @inbounds @simd for i in eachindex(a)
        a[i] -= m
    end
    return a
end

# z .= r / diag(k), then project onto the zero-mean subspace.
function _jacobi!(z::Array{T,3}, r::Array{T,3}, diag::Vector{T}) where {T}
    Nx, Ny, Nz = size(r)
    @inbounds for k in 1:Nz
        d = 1 / diag[k]
        for j in 1:Ny, i in 1:Nx
            z[i, j, k] = r[i, j, k] * d
        end
    end
    return _remove_mean!(z)
end

"""
    solve!(p, source, solver::DensityWeightedPoissonSolver)

Solve `∇·(w ∇p) = source` on the interior. `source` is projected to zero
mean (the compatibility condition of the singular operator) and `p` is
returned with zero mean; halos of `p` are untouched.
"""
function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                s::DensityWeightedPoissonSolver{T}) where {T}
    m = source.mesh
    ii, jj, kk = interior(m)

    # Work with the positive semi-definite form (−L)x = −source.
    b = s.b
    @inbounds @views b .= .-source.data[ii, jj, kk]
    _remove_mean!(b)
    bnorm = sqrt(_dot(b, b))

    x = fill!(s.x, zero(T))
    s.last_iter[] = 0
    s.last_residual[] = zero(T)
    if bnorm == 0
        @inbounds @views p.data[ii, jj, kk] .= 0
        return p
    end

    r = s.r; pdir = s.p_dir; Ap = s.Ap; z = s.z
    copyto!(r, b)
    _jacobi!(z, r, s.diag)
    copyto!(pdir, z)
    rz = _dot(r, z)

    dx, dy, dz = m.dx, m.dy, m.dz
    converged = false
    for it in 1:s.maxit
        _apply_neg_wlap!(Ap, pdir, s.w_c, s.w_f, dx, dy, dz, s.bc_z, s.bc_x, s.bc_y)
        pap = _dot(pdir, Ap)
        if pap <= 0
            @warn("DensityWeightedPoissonSolver: non-positive curvature ($pap); stopping at iteration $it",
                  subsystem = :pressure, domain = current_domain(),
                  solver = :density_weighted_cg, reason = :breakdown, iterations = it,
                  curvature = pap)
            break
        end
        α = rz / pap
        @inbounds @simd for i in eachindex(x)
            x[i] += α * pdir[i]
            r[i] -= α * Ap[i]
        end
        res = sqrt(_dot(r, r))
        s.last_iter[] = it
        s.last_residual[] = res
        if res < s.tol * bnorm
            converged = true
            break
        end
        _jacobi!(z, r, s.diag)
        rz_new = _dot(r, z)
        β = rz_new / rz
        @inbounds @simd for i in eachindex(pdir)
            pdir[i] = z[i] + β * pdir[i]
        end
        rz = rz_new
    end
    converged || @warn("DensityWeightedPoissonSolver did not converge: " *
                       "‖r‖=$(s.last_residual[]) after $(s.last_iter[]) iterations " *
                       "(tol·‖b‖=$(s.tol * bnorm))",
                       subsystem = :pressure, domain = current_domain(),
                       solver = :density_weighted_cg, reason = :maxit,
                       iterations = s.last_iter[], residual = s.last_residual[],
                       relative_residual = s.last_residual[] / bnorm, tol = s.tol)

    _remove_mean!(x)
    @inbounds @views p.data[ii, jj, kk] .= x
    return p
end

# =========================================================================
# Weighted flux-form operators
# =========================================================================
#
# Every kernel below takes halo-padded weight vectors indexed by the
# storage-array k index: `Wc[k]` is the weight of cell k, `Wf[k]` the weight
# of that cell's *bottom* face. Because the weight depends only on z, all
# horizontal flux faces of a cell carry Wc[k] and the factor cancels against
# the 1/Wc[k] out front — only the vertical terms actually change relative to
# the Boussinesq kernels, which is exactly the physics of the anelastic
# approximation.

"""
    weighted_divergence!(out, u, v, w, Wc, Wf, mesh; normalize=false)

Flux-form divergence of the weighted velocity, at cell centres:

```
out = ∇·(W u) = Wc[k] (δₓu + δ_y v) + ( Wf[k+1]w[k+1] − Wf[k]w[k] ) / Δz
```

With `normalize = true` the result is divided by `Wc[k]`, giving
`(1/W)∇·(W u)`. `Wc`/`Wf` are the halo-padded weight vectors of a
[`ReferenceState`](@ref).

This is the *exact* discrete adjoint-partner of the projection operator in
[`DensityWeightedPoissonSolver`](@ref): substituting `u ← u − ∂ₓφ` etc.
reproduces `∇·(W∇φ)` term for term, which is why the projection drives the
constraint to the solver tolerance rather than to truncation error.
"""
function weighted_divergence!(out::Field{CellCenter,T},
                              u::Field{FaceX,T}, v::Field{FaceY,T}, w::Field{FaceZ,T},
                              Wc::Vector{T}, Wf::Vector{T}, g::RegularGrid{T};
                              normalize::Bool = false) where {T}
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data; od = out.data
    @inbounds for k in kk
        wc = Wc[k]; wfb = Wf[k]; wft = Wf[k+1]
        scale = normalize ? 1 / wc : one(T)
        for j in jj, i in ii
            d = wc * ((ud[i+1,j,k] - ud[i,j,k]) * idx +
                      (vd[i,j+1,k] - vd[i,j,k]) * idy) +
                (wft * wd[i,j,k+1] - wfb * wd[i,j,k]) * idz
            od[i,j,k] = scale * d
        end
    end
    return out
end

"""
    mass_flux_divergence!(div, u, v, w, ref, mesh=ref.mesh) -> div

`∇·(ρ̄ u)` — the raw anelastic mass-flux divergence (kg m⁻³ s⁻¹), which is
what the pressure projection actually annihilates.
"""
mass_flux_divergence!(div::Field, u::Field, v::Field, w::Field,
                      ref::ReferenceState, g::RegularGrid = ref.mesh) =
    weighted_divergence!(div, u, v, w, ref.ρ̄c_pad, ref.ρ̄f_pad, g; normalize = false)

"""
    anelastic_divergence!(div, u, v, w, refstate, mesh=refstate.mesh) -> div

`(1/ρ̄) ∇·(ρ̄ u)` in flux form — the anelastic continuity residual, in s⁻¹ so
it is directly comparable with the Boussinesq `∇·u`. Note that a correct
anelastic solution has this equal to zero while `∇·u = −w ∂log ρ̄/∂z ≠ 0`;
that difference *is* the anelastic approximation.
"""
anelastic_divergence!(div::Field, u::Field, v::Field, w::Field,
                      refstate::ReferenceState, g::RegularGrid = refstate.mesh) =
    weighted_divergence!(div, u, v, w, refstate.ρ̄c_pad, refstate.ρ̄f_pad, g; normalize = true)

"""
    pseudo_incompressible_divergence!(div, u, v, w, refstate, mesh=refstate.mesh) -> div

`(1/ρ̄θ̄) ∇·(ρ̄θ̄ u)` — the Durran pseudo-incompressible continuity residual.
"""
pseudo_incompressible_divergence!(div::Field, u::Field, v::Field, w::Field,
                                  refstate::ReferenceState, g::RegularGrid = refstate.mesh) =
    weighted_divergence!(div, u, v, w, refstate.Pc_pad, refstate.Pf_pad, g; normalize = true)

"""
    weighted_momentum_advect!(tu, tv, tw, u, v, w, Wc, Wf, mesh)

Add the weighted flux-form momentum advection
`−(1/W) ∂(W uⱼuᵢ)/∂xⱼ` to the three face tendencies.

Under the discrete constraint `∇·(Wu) = 0` this equals the advective form
`−u·∇uᵢ`; the *unweighted* Boussinesq flux form does **not**, because it is
off by `uᵢ ∇·u`, and `∇·u ≠ 0` in a sound-proof system with a varying base
state. Getting this term right is what keeps a deep anelastic run from
spuriously accelerating as the density drops.

Control volumes: `u`/`v` live at cell-centre height, so their own weight is
`Wc[k]` and their vertical flux faces carry `Wf[k]`, `Wf[k+1]`. `w` lives at
a face, so its own weight is `Wf[k]` and its vertical flux faces sit at cell
centres `k−1`, `k` carrying `Wc[k−1]`, `Wc[k]`.
"""
function weighted_momentum_advect!(tu::Field{FaceX,T}, tv::Field{FaceY,T},
                                   tw::Field{FaceZ,T},
                                   u::Field{FaceX,T}, v::Field{FaceY,T},
                                   w::Field{FaceZ,T},
                                   Wc::Vector{T}, Wf::Vector{T},
                                   g::RegularGrid{T}) where {T}
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data

    # --- u-momentum, west face of cell (i,j,k) --------------------------
    @inbounds for k in kk
        iwc = 1 / Wc[k]; wfb = Wf[k]; wft = Wf[k+1]
        for j in jj, i in ii
            uce = 0.5*(ud[i,   j, k] + ud[i+1, j, k])
            ucw = 0.5*(ud[i-1, j, k] + ud[i,   j, k])
            d_uu_dx = (uce*uce - ucw*ucw) * idx

            vun = 0.5*(vd[i-1, j+1, k] + vd[i, j+1, k])
            uvn = 0.5*(ud[i,   j,   k] + ud[i, j+1, k])
            vus = 0.5*(vd[i-1, j,   k] + vd[i, j,   k])
            uvs = 0.5*(ud[i,   j-1, k] + ud[i, j,   k])
            d_vu_dy = (vun*uvn - vus*uvs) * idy

            wut = 0.5*(wd[i-1, j, k+1] + wd[i, j, k+1])
            uwt = 0.5*(ud[i,   j, k  ] + ud[i, j, k+1])
            wub = 0.5*(wd[i-1, j, k  ] + wd[i, j, k  ])
            uwb = 0.5*(ud[i,   j, k-1] + ud[i, j, k  ])
            d_wu_dz = (wft*wut*uwt - wfb*wub*uwb) * idz * iwc

            tu.data[i, j, k] -= d_uu_dx + d_vu_dy + d_wu_dz
        end
    end

    # --- v-momentum, south face of cell (i,j,k) -------------------------
    @inbounds for k in kk
        iwc = 1 / Wc[k]; wfb = Wf[k]; wft = Wf[k+1]
        for j in jj, i in ii
            uve = 0.5*(ud[i+1, j-1, k] + ud[i+1, j, k])
            vue = 0.5*(vd[i,   j,   k] + vd[i+1, j, k])
            uvw = 0.5*(ud[i,   j-1, k] + ud[i,   j, k])
            vuw = 0.5*(vd[i-1, j,   k] + vd[i,   j, k])
            d_uv_dx = (uve*vue - uvw*vuw) * idx

            vcn = 0.5*(vd[i, j,   k] + vd[i, j+1, k])
            vcs = 0.5*(vd[i, j-1, k] + vd[i, j,   k])
            d_vv_dy = (vcn*vcn - vcs*vcs) * idy

            wvt = 0.5*(wd[i, j-1, k+1] + wd[i, j, k+1])
            vwt = 0.5*(vd[i, j,   k  ] + vd[i, j, k+1])
            wvb = 0.5*(wd[i, j-1, k  ] + wd[i, j, k  ])
            vwb = 0.5*(vd[i, j,   k-1] + vd[i, j, k  ])
            d_wv_dz = (wft*wvt*vwt - wfb*wvb*vwb) * idz * iwc

            tv.data[i, j, k] -= d_uv_dx + d_vv_dy + d_wv_dz
        end
    end

    # --- w-momentum, bottom face of cell (i,j,k) ------------------------
    @inbounds for k in kk
        iwf = 1 / Wf[k]; wcb = Wc[k-1]; wct_ = Wc[k]
        for j in jj, i in ii
            uwe = 0.5*(ud[i+1, j, k-1] + ud[i+1, j, k])
            wue = 0.5*(wd[i,   j, k  ] + wd[i+1, j, k])
            uww = 0.5*(ud[i,   j, k-1] + ud[i,   j, k])
            wuw = 0.5*(wd[i-1, j, k  ] + wd[i,   j, k])
            d_uw_dx = (uwe*wue - uww*wuw) * idx

            vwn = 0.5*(vd[i, j+1, k-1] + vd[i, j+1, k])
            wvn = 0.5*(wd[i, j,   k  ] + wd[i, j+1, k])
            vws = 0.5*(vd[i, j,   k-1] + vd[i, j,   k])
            wvs = 0.5*(wd[i, j-1, k  ] + wd[i, j,   k])
            d_vw_dy = (vwn*wvn - vws*wvs) * idy

            wtop = 0.5*(wd[i, j, k  ] + wd[i, j, k+1])
            wbot = 0.5*(wd[i, j, k-1] + wd[i, j, k  ])
            d_ww_dz = (wct_*wtop*wtop - wcb*wbot*wbot) * idz * iwf

            tw.data[i, j, k] -= d_uw_dx + d_vw_dy + d_ww_dz
        end
    end

    return (tu, tv, tw)
end

"""
    weighted_scalar_advect!(tϕ, u, v, w, ϕ, Wc, Wf, mesh)

Add `−(1/W) ∂(W uⱼ ϕ)/∂xⱼ` to a cell-centred scalar tendency, with
2nd-order centred face interpolation of `ϕ`. Discretely conservative for
`W ϕ`, and equal to `−u·∇ϕ` under the discrete `∇·(Wu) = 0` constraint —
i.e. the correct adiabatic `θ` equation for the anelastic
(`W = ρ̄`) and pseudo-incompressible (`W = ρ̄θ̄`) systems.
"""
function weighted_scalar_advect!(tϕ::Field{CellCenter,T},
                                 u::Field{FaceX,T}, v::Field{FaceY,T}, w::Field{FaceZ,T},
                                 ϕ::Field{CellCenter,T},
                                 Wc::Vector{T}, Wf::Vector{T},
                                 g::RegularGrid{T}) where {T}
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data; f = ϕ.data
    @inbounds for k in kk
        iwc = 1 / Wc[k]; wfb = Wf[k]; wft = Wf[k+1]
        for j in jj, i in ii
            uϕe = ud[i+1,j,k] * 0.5 * (f[i,  j,k] + f[i+1,j,k])
            uϕw = ud[i,  j,k] * 0.5 * (f[i-1,j,k] + f[i,  j,k])
            vϕn = vd[i,j+1,k] * 0.5 * (f[i,j,  k] + f[i,j+1,k])
            vϕs = vd[i,j,  k] * 0.5 * (f[i,j-1,k] + f[i,j,  k])
            wϕt = wft * wd[i,j,k+1] * 0.5 * (f[i,j,k  ] + f[i,j,k+1])
            wϕb = wfb * wd[i,j,k  ] * 0.5 * (f[i,j,k-1] + f[i,j,k  ])
            tϕ.data[i,j,k] -= (uϕe - uϕw) * idx + (vϕn - vϕs) * idy +
                              (wϕt - wϕb) * idz * iwc
        end
    end
    return tϕ
end

"""
    add_anelastic_buoyancy!(tw, θ, ref, mesh=ref.mesh)

Add the sound-proof buoyancy tendency to the `w`-momentum tendency using the
*height-dependent* reference potential temperature,

```
b(z) = g · (θ − θ̄(z)) / θ̄(z),
```

evaluated at `FaceZ` with `θ` interpolated arithmetically in `z` and `θ̄`
taken from `ref.θ̄_f` (which uses the matching arithmetic mean, so a
base-state-only atmosphere yields exactly zero buoyancy — including at the
wall faces, where `θ̄_f` mirrors the adjacent cell just as the halo fill
does).

Contrast with the Boussinesq `add_buoyancy!(tw, θ, θ₀, g)`, which divides by
a single constant `θ₀`; over a deep domain the difference in `1/θ̄(z)` is a
leading-order error.
"""
function add_anelastic_buoyancy!(tw::Field{FaceZ,T}, θ::Field{CellCenter,T},
                                 ref::ReferenceState{T},
                                 g::RegularGrid{T} = ref.mesh) where {T}
    ii, jj, kk = interior(g)
    gacc = ref.g
    θ̄f = ref.θ̄f_pad
    @inbounds for k in kk
        θ̄k = θ̄f[k]
        coef = gacc / θ̄k
        for j in jj, i in ii
            θface = 0.5 * (θ.data[i,j,k-1] + θ.data[i,j,k])
            tw.data[i,j,k] += coef * (θface - θ̄k)
        end
    end
    return tw
end

# =========================================================================
# Simulation types
# =========================================================================

"""
    MassWeighting

Tag selecting which profile weights the continuity constraint and therefore
all the flux-form operators and the pressure projection.

  * [`AnelasticWeighting`](@ref) — `W = ρ̄(z)`, Lipps–Hemler anelastic.
  * [`PseudoIncompressibleWeighting`](@ref) — `W = ρ̄(z)θ̄(z)`, Durran
    pseudo-incompressible.
"""
abstract type MassWeighting end

"`W = ρ̄(z)`: the anelastic constraint `∇·(ρ̄u) = 0`. See [`MassWeighting`](@ref)."
struct AnelasticWeighting <: MassWeighting end

"`W = ρ̄(z)θ̄(z)`: the pseudo-incompressible constraint `∇·(ρ̄θ̄u) = 0`. See [`MassWeighting`](@ref)."
struct PseudoIncompressibleWeighting <: MassWeighting end

"""
    MassWeightedSimulation{S<:MassWeighting, …}

Sound-proof LES driver on a staggered Arakawa-C mesh, parameterised by the
[`MassWeighting`](@ref) `S`. Two aliases name the concrete systems:

  * [`AnelasticSimulation`](@ref) `= MassWeightedSimulation{AnelasticWeighting}`
  * [`PseudoIncompressibleSimulation`](@ref) `= MassWeightedSimulation{PseudoIncompressibleWeighting}`

Prognostic state mirrors
[`BoussinesqSimulation`](@ref Aeolus.Simulations.BoussinesqSimulation)
(`u@FaceX`, `v@FaceY`, `w@FaceZ`, `p@CellCenter`, `θ@CellCenter`) plus the
prescribed [`ReferenceState`](@ref) in `sim.ref`. `θ` stores the **full**
potential temperature, initialised to `θ̄(z)` by
[`set_reference_state!`](@ref); buoyancy works with `θ − θ̄(z)` internally.

`sim.p` holds the projection potential rate `φ/Δt`, i.e. the sound-proof
pressure divided by the reference density — the same convention as the
Boussinesq driver.
"""
mutable struct MassWeightedSimulation{S<:MassWeighting,
                                      T,
                                      M<:RegularGrid{T},
                                      BCS<:FieldBoundaryConditions,
                                      TBCS<:FieldBoundaryConditions,
                                      PS<:AbstractPressureSolver,
                                      R<:ReferenceState{T}}
    weighting::S
    mesh::M
    bcs::BCS
    θ_bcs::TBCS
    solver::PS
    ref::R
    # physics parameters
    ν::T
    g_accel::T
    f_coriolis::T
    C_s::T
    Pr_t::T
    θ_flux_bottom::T
    θ_flux_top::T
    sponge_α_max::T
    sponge_z_start::T
    sponge_target::NamedTuple{(:u, :v, :w)}   # N-open-d (#262)
    # enable flags
    has_viscosity::Bool
    has_buoyancy::Bool
    has_coriolis::Bool
    has_sgs::Bool
    has_surface_flux::Bool
    has_sponge::Bool
    # clock
    t::T
    # prognostic
    u::Field{FaceX,      T, Array{T,3}, M}
    v::Field{FaceY,      T, Array{T,3}, M}
    w::Field{FaceZ,      T, Array{T,3}, M}
    p::Field{CellCenter, T, Array{T,3}, M}
    θ::Field{CellCenter, T, Array{T,3}, M}
    # tendencies
    tu::Field{FaceX,      T, Array{T,3}, M}
    tv::Field{FaceY,      T, Array{T,3}, M}
    tw::Field{FaceZ,      T, Array{T,3}, M}
    tθ::Field{CellCenter, T, Array{T,3}, M}
    # scratch
    div_buf::Field{CellCenter, T, Array{T,3}, M}
    φ::Field{CellCenter,       T, Array{T,3}, M}
    dφdx::Field{FaceX,         T, Array{T,3}, M}
    dφdy::Field{FaceY,         T, Array{T,3}, M}
    dφdz::Field{FaceZ,         T, Array{T,3}, M}
    lap_u::Field{FaceX,        T, Array{T,3}, M}
    lap_v::Field{FaceY,        T, Array{T,3}, M}
    lap_w::Field{FaceZ,        T, Array{T,3}, M}
    lap_θ::Field{CellCenter,   T, Array{T,3}, M}
    νt::Field{CellCenter,      T, Array{T,3}, M}
    κt::Field{CellCenter,      T, Array{T,3}, M}   # SGS scalar diffusivity (Era VI, #167)
end

"""
    AnelasticSimulation(mesh; ref = isentropic_reference(mesh), kwargs...)

Lipps–Hemler anelastic LES: `∇·(ρ̄u) = 0`, buoyancy `g(θ−θ̄)/θ̄`, flux-form
advection weighted by `ρ̄`, and a density-weighted pressure projection
(`∇·(ρ̄∇φ) = ∇·(ρ̄u*)`, solved by [`DensityWeightedPoissonSolver`](@ref)).
Advance with [`step_anelastic!`](@ref) (or the generic `step!`).

Keyword arguments (all optional; each physics term is off when its parameter
is `0`):

| Keyword | Default | Effect |
| :------ | :------ | :----- |
| `ref` | `isentropic_reference(mesh)` | the [`ReferenceState`](@ref) |
| `ν` | `0` | molecular viscosity |
| `buoyancy` | `true` | enable `g(θ−θ̄)/θ̄` |
| `f_coriolis` | `0` | f-plane Coriolis |
| `C_s` | `0` | Smagorinsky coefficient (the built-in closure; `step!(…; sgs)` replaces it) |
| `Pr_t` | `1/3` | turbulent Prandtl number |
| `θ_flux_bottom`, `θ_flux_top` | `0` | prescribed surface fluxes |
| `sponge_α_max`, `sponge_z_start` | off | Rayleigh sponge |
| `sponge_target` | `nothing` (zero) | sponge relaxation targets; required with an open top ([`sponge_target`](@ref Aeolus.Simulations.sponge_target), #262) |
| `bcs`, `θ_bcs` | `FieldBoundaryConditions(:periodic, :periodic, :wall)` | BCs |
| `solver` | `DensityWeightedPoissonSolver(mesh, ref; weighting=:ρ)` | projection solver |
| `tol`, `maxit` | `1e-12`, `500` | passed to the default solver |

The vertical BC **must not be periodic**: a `z`-periodic domain with a
non-periodic `ρ̄(z)` has no well-posed anelastic constraint, and the
constructor rejects it rather than silently producing a mass-flux
discontinuity at the seam.

Diagnostics: [`max_anelastic_divergence`](@ref),
[`max_mass_flux_divergence`](@ref), `max_divergence` (the *plain* `∇·u`,
which is deliberately non-zero), `kinetic_energy` (density-weighted),
`mean_θ_profile`, [`bubble_top_height`](@ref),
[`system_comparison_diagnostics`](@ref).
"""
const AnelasticSimulation{T,M,BCS,TBCS,PS,R} =
    MassWeightedSimulation{AnelasticWeighting,T,M,BCS,TBCS,PS,R}

"""
    PseudoIncompressibleSimulation(mesh; ref = constant_stability_reference(mesh), kwargs...)

Durran (1989) pseudo-incompressible LES. Identical structure to
[`AnelasticSimulation`](@ref) but the constraint is on the flux of the
pseudo-density `P = ρ̄θ̄`,

```
∇·(ρ̄θ̄ u) = 0    ⟹    ∇·(ρ̄θ̄ ∇φ) = ∇·(ρ̄θ̄ u*),
```

which retains the thermodynamic coupling that the anelastic filter throws
away: the constraint responds to the base-state stratification, not only to
the mass distribution. Advance with
[`step_pseudo_incompressible!`](@ref).

Because `P` and `ρ̄` differ by a constant when `θ̄` is constant, this system is
*exactly* the anelastic one on a neutral base state; use a stratified base
state (e.g. [`constant_stability_reference`](@ref)) to see them separate,
which is why that is the default `ref` here.

Note: the adiabatic limit is implemented (`Q = 0`). A diabatic
pseudo-incompressible run would put `∇·(ρ̄θ̄u) = (R/cᵥ)(P/π̄)⁻¹·…` on the
right-hand side of the projection; the hook for that is
`post_predictor!`, which runs after the predictor and before the solve.
"""
const PseudoIncompressibleSimulation{T,M,BCS,TBCS,PS,R} =
    MassWeightedSimulation{PseudoIncompressibleWeighting,T,M,BCS,TBCS,PS,R}

_weighting_symbol(::AnelasticWeighting)              = :ρ
_weighting_symbol(::PseudoIncompressibleWeighting)   = :ρθ

_weight_c(::AnelasticWeighting, r::ReferenceState)            = r.ρ̄c_pad
_weight_f(::AnelasticWeighting, r::ReferenceState)            = r.ρ̄f_pad
_weight_c(::PseudoIncompressibleWeighting, r::ReferenceState) = r.Pc_pad
_weight_f(::PseudoIncompressibleWeighting, r::ReferenceState) = r.Pf_pad

"""
    constraint_weights(sim::MassWeightedSimulation) -> (Wc, Wf)

The weight `W` of the constraint `∇·(W u) = 0` a sound-proof simulation
enforces (`ρ̄` anelastic, `ρ̄θ̄` pseudo-incompressible), as the
halo-padded cell and bottom-face vectors its kernels use (`Wc[k]`,
`Wf[k]` at storage index `k`, see [`weighted_divergence!`](@ref)). A
tracer transported in the flux form `∂(W q)/∂t = −∇·(W u q)` with these
weights conserves `Σ W q` and keeps a uniform `q` uniform to the
projection's tolerance.
"""
constraint_weights(sim::MassWeightedSimulation) =
    (_weight_c(sim.weighting, sim.ref), _weight_f(sim.weighting, sim.ref))

# Open boundaries (N-open-b): the boundary mass-flux correction balances the
# flux of the constraint weight W (ρ̄ or ρ̄θ̄), the quantity the projection
# annihilates.
boundary_flux_weights(sim::MassWeightedSimulation) = constraint_weights(sim)

_default_reference(::AnelasticWeighting, mesh)            = isentropic_reference(mesh)
_default_reference(::PseudoIncompressibleWeighting, mesh)  = constant_stability_reference(mesh)

function _build_mass_weighted(S::MassWeighting, mesh::RegularGrid{T};
                              ref::Union{Nothing,ReferenceState} = nothing,
                              ν::Real                = 0,
                              buoyancy::Bool         = true,
                              g_accel::Real          = G_STD,
                              f_coriolis::Real       = 0,
                              C_s::Real              = 0,
                              Pr_t::Real             = 1/3,
                              θ_flux_bottom::Real    = 0,
                              θ_flux_top::Real       = 0,
                              sponge_α_max::Real     = 0,
                              sponge_z_start::Real   = mesh.z0 + mesh.Lz,
                              sponge_target          = nothing,
                              bcs::FieldBoundaryConditions =
                                  FieldBoundaryConditions(:periodic, :periodic, :wall),
                              θ_bcs::FieldBoundaryConditions = bcs,
                              tol::Real   = 1e-12,
                              maxit::Int  = 500,
                              solver::Union{Nothing,AbstractPressureSolver} = nothing,
                              initialize_θ::Bool = true) where {T}
    (bcs.bottom isa Periodic || bcs.top isa Periodic) &&
        error("$(nameof(typeof(S))) requires a non-periodic vertical boundary " *
              "(use :wall or :noflux): a z-periodic domain is incompatible with a " *
              "height-dependent reference profile.")
    mesh.Hz >= 1 || error("sound-proof systems need at least one z halo cell")

    refstate = ref === nothing ? _default_reference(S, mesh) : ref
    refstate isa ReferenceState{T} ||
        error("ReferenceState element type $(eltype(refstate.ρ̄)) does not match mesh eltype $T")

    # The projection's per-axis closure comes from the velocity faces
    # (periodic, or Neumann for walls and prescribed-velocity sides).
    ax = poisson_axis_bcs(bcs)
    ps = solver === nothing ?
        DensityWeightedPoissonSolver(mesh, refstate;
                                     weighting = _weighting_symbol(S),
                                     tol = tol, maxit = maxit, ax...) :
        solver

    u = Field(FaceX(), mesh); v = Field(FaceY(), mesh); w = Field(FaceZ(), mesh)
    p = Field(CellCenter(), mesh); θ = Field(CellCenter(), mesh)
    tu = Field(FaceX(), mesh); tv = Field(FaceY(), mesh); tw = Field(FaceZ(), mesh)
    tθ = Field(CellCenter(), mesh)
    div_buf = Field(CellCenter(), mesh)
    φ    = Field(CellCenter(), mesh)
    dφdx = Field(FaceX(), mesh); dφdy = Field(FaceY(), mesh); dφdz = Field(FaceZ(), mesh)
    lu = Field(FaceX(), mesh); lv = Field(FaceY(), mesh); lw = Field(FaceZ(), mesh)
    lθ = Field(CellCenter(), mesh); νt = Field(CellCenter(), mesh)
    κt = Field(CellCenter(), mesh)

    has_sponge = sponge_α_max != 0 && sponge_z_start < (mesh.z0 + mesh.Lz)

    sim = MassWeightedSimulation{typeof(S), T, typeof(mesh), typeof(bcs),
                                 typeof(θ_bcs), typeof(ps), typeof(refstate)}(
        S, mesh, bcs, θ_bcs, ps, refstate,
        T(ν), T(g_accel), T(f_coriolis), T(C_s), T(Pr_t),
        T(θ_flux_bottom), T(θ_flux_top), T(sponge_α_max), T(sponge_z_start),
        Simulations._sponge_target(sponge_target, mesh, bcs, has_sponge),
        ν != 0, buoyancy, f_coriolis != 0, C_s != 0,
        θ_flux_bottom != 0 || θ_flux_top != 0, has_sponge,
        zero(T),
        u, v, w, p, θ, tu, tv, tw, tθ,
        div_buf, φ, dφdx, dφdy, dφdz, lu, lv, lw, lθ, νt, κt)

    initialize_θ && set_reference_state!(sim)
    return sim
end

AnelasticSimulation(mesh::RegularGrid; kwargs...) =
    _build_mass_weighted(AnelasticWeighting(), mesh; kwargs...)

PseudoIncompressibleSimulation(mesh::RegularGrid; kwargs...) =
    _build_mass_weighted(PseudoIncompressibleWeighting(), mesh; kwargs...)

"""
    reference_state(sim) -> ReferenceState

The base state a sound-proof simulation was built with.
"""
reference_state(sim::MassWeightedSimulation) = sim.ref

"""
    set_reference_state!(sim) -> sim

Reset the prognostic state to the base state at rest: `u = v = w = 0` and
`θ(x,y,z) = θ̄(z)`. Called automatically by the constructors, so a typical
initialisation is

```julia
sim = AnelasticSimulation(mesh; ref = ref)
θ̄ = sim.ref.θ̄
set!(sim.θ, (x,y,z) -> θ̄_at(z) + perturbation(x,y,z))
```

or simply add the perturbation on top of what the constructor left in
`sim.θ`.
"""
function set_reference_state!(sim::MassWeightedSimulation{S,T}) where {S,T}
    fill!(sim.u.data, 0); fill!(sim.v.data, 0); fill!(sim.w.data, 0)
    fill!(sim.p.data, 0)
    ii, jj, kk = interior(sim.mesh)
    θ̄ = sim.ref.θ̄c_pad
    @inbounds for k in kk, j in jj, i in ii
        sim.θ.data[i,j,k] = θ̄[k]
    end
    fill_halos!(sim.θ, sim.θ_bcs)
    sim.t = zero(T)
    return sim
end

# =========================================================================
# The fractional step
# =========================================================================

"In-place y[interior] += a · x[interior]."
function _axpy_interior!(y::Field, a::Number, x::Field, m::AbstractMesh)
    ii, jj, kk = interior(m)
    @inbounds @views y.data[ii, jj, kk] .+= a .* x.data[ii, jj, kk]
    return y
end

"""
    step_anelastic!(sim::AnelasticSimulation, dt; body_force! = nothing, post_predictor! = nothing)

Advance an anelastic simulation by `dt` with the density-weighted
fractional step:

1. fill halos; zero tendencies;
2. `ρ̄`-weighted flux-form momentum and `θ` advection;
3. molecular / SGS diffusion, anelastic buoyancy `g(θ−θ̄)/θ̄`, Coriolis,
   surface flux, sponge, then `body_force!(sim, Δt)`;
4. predictor `u* = uⁿ + Δt F`, then `post_predictor!(sim, Δt)`;
5. refill halos (which re-imposes `w = 0` on the walls) and form
   `D = ∇·(ρ̄u*)`;
6. solve `∇·(ρ̄∇φ) = D` with the [`DensityWeightedPoissonSolver`](@ref);
7. correct `u = u* − ∇φ`, advance `θ`, store `p = φ/Δt`, advance the clock.

Steps 5–7 are *discretely consistent*: the divergence in 5 and the operator
in 6 are built from the same face weights and the same differences, so the
post-step `∇·(ρ̄u)` is bounded by the CG tolerance, not by truncation error.
The callbacks have the same contract as for
[`step!`](@ref Aeolus.Simulations.step!), and so does the SGS closure: the
keyword `sgs` (any [`AbstractSGSClosure`](@ref
Aeolus.Operators.AbstractSGSClosure), or a function `f(νt, u, v, w)`; the
legacy alias `eddy_viscosity!` is accepted) replaces the built-in
Smagorinsky closure (`C_s`), its coefficients go into `sim.νt` / `sim.κt`,
its tendencies are added in step 3 where the built-in ones are, and its
memory is committed once after the projection; `sgs_stress = :laplacian |
:symmetric` selects the form of its momentum divergence as for `step!`. The
SGS stress is applied in the unweighted form, like the built-in closure (the `O(∂ln ρ̄/∂z)` weighting
correction is omitted for the same reason as for the molecular term).
"""
step_anelastic!(sim::AnelasticSimulation, dt::Real; kwargs...) =
    _step_mass_weighted!(sim, dt; kwargs...)

"""
    step_pseudo_incompressible!(sim::PseudoIncompressibleSimulation, dt; kwargs...)

As [`step_anelastic!`](@ref) but with every weight replaced by the
pseudo-density `P = ρ̄θ̄`, so the projection enforces `∇·(ρ̄θ̄u) = 0`.
"""
step_pseudo_incompressible!(sim::PseudoIncompressibleSimulation, dt::Real; kwargs...) =
    _step_mass_weighted!(sim, dt; kwargs...)

"""
    step!(sim::MassWeightedSimulation, dt; kwargs...)

Generic entry point — dispatches to [`step_anelastic!`](@ref) or
[`step_pseudo_incompressible!`](@ref) according to the weighting, so
`step!` / `run!` work uniformly across all three dynamical cores.
"""
step!(sim::MassWeightedSimulation, dt::Real; kwargs...) =
    _step_mass_weighted!(sim, dt; kwargs...)

# Steps 8–10 of the sound-proof fractional step: the weighted divergence of
# the velocity, the variable-coefficient Poisson solve ∇·(W∇φ) = ∇·(Wu), and
# the correction u ← u − ∇φ. `sim.φ` holds the potential afterwards; `sim.p`
# is not touched. Also the re-projection of a two-way nest parent after the
# feedback (N-nest-e, #263).
function _project_mass_weighted!(sim::MassWeightedSimulation{S,T},
                                 Wc = _weight_c(sim.weighting, sim.ref),
                                 Wf = _weight_f(sim.weighting, sim.ref)) where {S,T}
    m = sim.mesh
    # 8. weighted divergence of the predicted velocity
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)
    weighted_divergence!(sim.div_buf, sim.u, sim.v, sim.w, Wc, Wf, m; normalize = false)

    # 9. variable-coefficient Poisson solve: ∇·(W∇φ) = ∇·(W u*)
    solve!(sim.φ, sim.div_buf, sim.solver)

    # 10. correction u = u* − ∇φ
    fill_halos!(sim.φ, sim.bcs)
    ∂x(sim.dφdx, sim.φ); ∂y(sim.dφdy, sim.φ); ∂z(sim.dφdz, sim.φ)
    assert_open_face_correction(sim.bcs, sim.dφdx, sim.dφdy, sim.dφdz)   # ∂φ/∂n = 0 on open faces
    _axpy_interior!(sim.u, -one(T), sim.dφdx, m)
    _axpy_interior!(sim.v, -one(T), sim.dφdy, m)
    _axpy_interior!(sim.w, -one(T), sim.dφdz, m)
    return sim
end

function _step_mass_weighted!(sim::MassWeightedSimulation{S,T}, dt::Real;
                              body_force!::Union{Nothing,Function} = nothing,
                              post_predictor!::Union{Nothing,Function} = nothing,
                              sgs = nothing,
                              eddy_viscosity! = nothing,
                              sgs_stress::Symbol = :laplacian) where {S,T}
    Δt = T(dt)
    closure = _active_sgs(sim, sgs, eddy_viscosity!)
    _check_sgs_stress(sgs_stress)
    m  = sim.mesh
    Wc = _weight_c(sim.weighting, sim.ref)
    Wf = _weight_f(sim.weighting, sim.ref)
    prepare_open_boundaries!(sim)          # no-op unless an open face is unfilled

    # 1. halos
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)
    fill_halos!(sim.θ, sim.θ_bcs)
    mark_open_boundary_stage!(sim)         # radiating outflow: step-start layers (N-open-c)

    # 2. advection, weighted flux form
    fill!(sim.tu.data, 0); fill!(sim.tv.data, 0); fill!(sim.tw.data, 0)
    fill!(sim.tθ.data, 0)
    weighted_momentum_advect!(sim.tu, sim.tv, sim.tw, sim.u, sim.v, sim.w, Wc, Wf, m)
    weighted_scalar_advect!(sim.tθ, sim.u, sim.v, sim.w, sim.θ, Wc, Wf, m)

    # 3. molecular viscosity / diffusion (ν∇²·; the O(∂log ρ̄/∂z) correction
    #    to the weighted viscous flux divergence is negligible next to the
    #    SGS flux in any LES and is deliberately omitted).
    if sim.has_viscosity
        laplacian(sim.lap_u, sim.u); laplacian(sim.lap_v, sim.v); laplacian(sim.lap_w, sim.w)
        laplacian(sim.lap_θ, sim.θ)
        _axpy_interior!(sim.tu, sim.ν, sim.lap_u, m)
        _axpy_interior!(sim.tv, sim.ν, sim.lap_v, m)
        _axpy_interior!(sim.tw, sim.ν, sim.lap_w, m)
        _axpy_interior!(sim.tθ, sim.ν, sim.lap_θ, m)
    end

    # 4. SGS closure (the contract shared with the Boussinesq core: the
    #    built-in Smagorinsky, or the `sgs` keyword).
    if closure !== nothing
        f = compute_sgs!(closure, sim)
        add_sgs_momentum!(sim.tu, sim.tv, sim.tw, closure, f, sim; stress = sgs_stress)
        add_sgs_scalar!(sim.tθ, sim.θ, closure, f, sim)
    end

    # 5. buoyancy with the height-dependent reference θ̄(z)
    sim.has_buoyancy && add_anelastic_buoyancy!(sim.tw, sim.θ, sim.ref, m)

    # 6. Coriolis / surface flux / sponge / user forcing
    sim.has_coriolis && add_coriolis!(sim.tu, sim.tv, sim.u, sim.v, sim.f_coriolis)
    sim.has_surface_flux && add_surface_flux!(sim.tθ, sim.θ_flux_bottom, sim.θ_flux_top)
    sim.has_sponge && add_velocity_sponge!(sim)   # toward sim.sponge_target (#262)
    body_force! === nothing || body_force!(sim, Δt)

    # 7. predictor
    _axpy_interior!(sim.u, Δt, sim.tu, m)
    _axpy_interior!(sim.v, Δt, sim.tv, m)
    _axpy_interior!(sim.w, Δt, sim.tw, m)
    post_predictor! === nothing || post_predictor!(sim, Δt)

    # 7b. open boundaries (N-open-b): data at the end of the step, with the
    #     W-weighted mass-flux correction, written into u* by the fill below
    update_open_boundaries!(sim, sim.t + Δt, Δt)

    # 8.–10. weighted projection
    _project_mass_weighted!(sim, Wc, Wf)

    # 11. scalar update
    _axpy_interior!(sim.θ, Δt, sim.tθ, m)

    # 12. pressure-like diagnostic and clock
    ii, jj, kk = interior(m)
    @inbounds @views sim.p.data[ii, jj, kk] .= sim.φ.data[ii, jj, kk] ./ Δt
    sim.t += Δt
    closure === nothing || advance_sgs!(closure, sim, Δt)   # once per step
    return sim
end

"""
    run!(sim::MassWeightedSimulation, dt, nsteps; callback = nothing)

Advance `nsteps` steps of the appropriate sound-proof fractional step.
`callback(sim, n)` is invoked after each step when supplied.
"""
function run!(sim::MassWeightedSimulation, dt::Real, nsteps::Integer; callback = nothing)
    for n in 1:nsteps
        _step_mass_weighted!(sim, dt)
        callback === nothing || callback(sim, n)
    end
    return sim
end

# =========================================================================
# Diagnostics
# =========================================================================

"""
    max_anelastic_divergence(sim) -> Real

`max |(1/ρ̄)∇·(ρ̄u)|` over the interior — the anelastic continuity residual
in s⁻¹. After [`step_anelastic!`](@ref) this sits at the Poisson-solver
tolerance. Compare with `max_divergence(sim)`, which measures the *plain*
`∇·u` and is deliberately **not** zero: `∇·u = −w ∂log ρ̄/∂z`.

Defined for both weightings (it always measures the `ρ̄` constraint), so a
pseudo-incompressible run can be asked how far it is from the anelastic
constraint and vice versa.
"""
function max_anelastic_divergence(sim::MassWeightedSimulation)
    _fill_velocity_halos!(sim)
    anelastic_divergence!(sim.div_buf, sim.u, sim.v, sim.w, sim.ref, sim.mesh)
    return maximum(abs, interior_view(sim.div_buf))
end

"""
    max_mass_flux_divergence(sim) -> Real

`max |∇·(ρ̄u)|` over the interior (kg m⁻³ s⁻¹) — the un-normalised anelastic
residual, i.e. the quantity the projection literally annihilates.
"""
function max_mass_flux_divergence(sim::MassWeightedSimulation)
    _fill_velocity_halos!(sim)
    mass_flux_divergence!(sim.div_buf, sim.u, sim.v, sim.w, sim.ref, sim.mesh)
    return maximum(abs, interior_view(sim.div_buf))
end

"""
    max_pseudo_incompressible_divergence(sim) -> Real

`max |(1/ρ̄θ̄)∇·(ρ̄θ̄u)|` over the interior — the Durran continuity residual.
"""
function max_pseudo_incompressible_divergence(sim::MassWeightedSimulation)
    _fill_velocity_halos!(sim)
    pseudo_incompressible_divergence!(sim.div_buf, sim.u, sim.v, sim.w, sim.ref, sim.mesh)
    return maximum(abs, interior_view(sim.div_buf))
end

function _fill_velocity_halos!(sim::MassWeightedSimulation)
    fill_halos!(sim.u, sim.bcs)
    fill_halos!(sim.v, sim.bcs)
    fill_halos!(sim.w, sim.bcs)
    return sim
end

"""
    max_divergence(sim::MassWeightedSimulation) -> Real

`max |∇·u|` — the *Boussinesq* constraint evaluated on a sound-proof
solution. Non-zero by design; its size relative to
[`max_anelastic_divergence`](@ref) is a direct measure of how far the run is
from the Boussinesq limit.
"""
function max_divergence(sim::MassWeightedSimulation)
    _fill_velocity_halos!(sim)
    divergence(sim.div_buf, sim.u, sim.v, sim.w)
    return maximum(abs, interior_view(sim.div_buf))
end

"""
    constraint_residual(sim) -> Real

The continuity residual appropriate to `sim`'s own dynamical core: `∇·u` for
Boussinesq, `(1/ρ̄)∇·(ρ̄u)` for anelastic, `(1/ρ̄θ̄)∇·(ρ̄θ̄u)` for
pseudo-incompressible. Lets [`system_comparison_diagnostics`](@ref) hold all
three cores to their own standard.
"""
constraint_residual(sim::AnelasticSimulation)              = max_anelastic_divergence(sim)
constraint_residual(sim::PseudoIncompressibleSimulation)   = max_pseudo_incompressible_divergence(sim)
constraint_residual(sim)                                   = max_divergence(sim)

"""
    kinetic_energy(sim::MassWeightedSimulation) -> Real

**Density-weighted** kinetic energy `½∫ρ̄(u²+v²+w²)dV` over the interior.
`u, v` use the cell density `ρ̄[k]`, `w` the face density `ρ̄_f[k]` — each
component weighted by the density at its own location. This is the energy
that the anelastic system conserves in the absence of forcing; the unweighted
Boussinesq `kinetic_energy` is not comparable in absolute terms across
systems, which is why [`system_comparison_diagnostics`](@ref) also reports
`max_w` and the `θ` variance.
"""
function kinetic_energy(sim::MassWeightedSimulation{S,T}) where {S,T}
    m = sim.mesh
    dV = m.dx * m.dy * m.dz
    ii, jj, kk = interior(m)
    ρc = sim.ref.ρ̄c_pad; ρf = sim.ref.ρ̄f_pad
    ke = zero(T)
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    @inbounds for k in kk
        rc = ρc[k]; rf = ρf[k]
        for j in jj, i in ii
            ke += rc * (ud[i,j,k]^2 + vd[i,j,k]^2) + rf * wd[i,j,k]^2
        end
    end
    return 0.5 * ke * dV
end

"""
    mean_θ_profile(sim::MassWeightedSimulation) -> Vector

Horizontal-mean potential-temperature profile `⟨θ⟩(k)` over the interior,
one entry per interior `z`-level. `sim.θ` stores the full `θ`, so this is
directly comparable with `sim.ref.θ̄`.
"""
mean_θ_profile(sim::MassWeightedSimulation) = horizontal_mean(sim.θ)

"""
    θ_variance(sim) -> Real

Volume mean of the horizontal variance of `θ`, `⟨⟨(θ−⟨θ⟩)²⟩⟩` (K²). A
single scalar measuring how much thermal structure survives — the cleanest
way to see two dynamical cores predicting different mixing.
"""
function θ_variance(sim)
    v = horizontal_variance(sim.θ)
    return sum(v) / length(v)
end

"""
    unweighted_kinetic_energy(sim) -> Real

`½∫(u²+v²+w²)dV` over the interior, with **no** density weighting — the same
definition for every dynamical core, hence the only kinetic-energy number
that may be compared across systems. (`kinetic_energy` of a sound-proof core
carries a `ρ̄` factor and so differs from the Boussinesq value even when the
velocity fields are identical.)
"""
function unweighted_kinetic_energy(sim)
    m = sim.mesh
    dV = m.dx * m.dy * m.dz
    ui = interior_view(sim.u); vi = interior_view(sim.v); wi = interior_view(sim.w)
    ke = zero(eltype(ui))
    @inbounds for I in eachindex(ui)
        ke += ui[I]^2 + vi[I]^2 + wi[I]^2
    end
    return 0.5 * ke * dV
end

"""
    max_w(sim) -> Real

Maximum (signed) vertical velocity over the interior, m s⁻¹. Positive means
somewhere in the domain is rising.
"""
max_w(sim) = maximum(interior_view(sim.w))

"""
    bubble_top_height(sim; frac = 0.5) -> Real

Height of the highest interior cell level whose maximum horizontal
potential-temperature perturbation still exceeds `frac` times the global
maximum perturbation, where the perturbation is measured against the
*horizontal mean* at each level. Returns the bottom of the domain when the
field is perturbation-free.

Defined for any simulation type carrying a `θ` field, so the three
dynamical cores can be compared with one number. `frac = 0.5` is the
conventional half-amplitude contour of the warm-bubble benchmark.
"""
function bubble_top_height(sim; frac::Real = 0.5)
    θi = interior_view(sim.θ)
    Nx, Ny, Nz = size(θi)
    zc = znodes(sim.mesh, CellCenter())
    mean_k = [sum(@view θi[:,:,k]) / (Nx*Ny) for k in 1:Nz]
    peak = zeros(eltype(θi), Nz)
    @inbounds for k in 1:Nz
        mx = -Inf
        for j in 1:Ny, i in 1:Nx
            d = θi[i,j,k] - mean_k[k]
            d > mx && (mx = d)
        end
        peak[k] = mx
    end
    gmax = maximum(peak)
    gmax <= 0 && return zc[1]
    thr = frac * gmax
    ktop = 1
    @inbounds for k in 1:Nz
        peak[k] >= thr && (ktop = k)
    end
    return zc[ktop]
end

"""
    system_comparison_diagnostics(sims::NamedTuple) -> NamedTuple

Reduce a family of simulations — typically the *same* initial condition
integrated by Boussinesq, anelastic and pseudo-incompressible cores — to one
comparable diagnostic record each, and quantify how far they have drifted
apart.

```julia
sims = (boussinesq = b, anelastic = a, pseudo = p)
d = system_comparison_diagnostics(sims)
d.anelastic.max_w            # 4.71
d.anelastic.rel_ke           # KE relative to the first entry
```

Each entry is a `NamedTuple` with

| Field | Meaning |
| :---- | :------ |
| `t` | simulation time |
| `kinetic_energy` | each core's *native* KE — density-weighted for the sound-proof cores |
| `kinetic_energy_unweighted` | `½∫(u²+v²+w²)dV`, the cross-system-comparable KE |
| `max_w` | maximum vertical velocity, m s⁻¹ |
| `θ_variance` | volume-mean horizontal `θ` variance, K² |
| `bubble_top` | [`bubble_top_height`](@ref), m |
| `constraint_residual` | each core's *own* continuity residual |
| `plain_divergence` | `max|∇·u|` for every core — zero only for Boussinesq |
| `rel_ke`, `rel_max_w`, `rel_θ_variance` | signed relative difference from the first entry |

`rel_ke` is built from `kinetic_energy_unweighted`, not from the native KE,
precisely so that it measures a difference in the *flow* rather than a
difference in the definition.

The `rel_*` fields are the validation instrument: they should be `~0` for
the reference entry, `O(Lz/H)` for a sound-proof core in a shallow domain,
and grow as the domain deepens.
"""
function system_comparison_diagnostics(sims::NamedTuple)
    isempty(sims) && return NamedTuple()
    raw = map(_system_record, sims)
    ref = first(raw)
    return map(r -> merge(r, (rel_ke          = _reldiff(r.kinetic_energy_unweighted,
                                                        ref.kinetic_energy_unweighted),
                              rel_max_w       = _reldiff(r.max_w,          ref.max_w),
                              rel_θ_variance  = _reldiff(r.θ_variance,     ref.θ_variance))), raw)
end

_reldiff(a, b) = b == 0 ? (a == 0 ? 0.0 : Inf) : (a - b) / abs(b)

function _system_record(sim)
    return (t                   = sim.t,
            kinetic_energy      = kinetic_energy(sim),
            kinetic_energy_unweighted = unweighted_kinetic_energy(sim),
            max_w               = max_w(sim),
            θ_variance          = θ_variance(sim),
            bubble_top          = bubble_top_height(sim),
            constraint_residual = constraint_residual(sim),
            plain_divergence    = max_divergence(sim))
end

# =========================================================================
# Adaptive dt and step snapshots for the sound-proof cores (N-nest-c, #257)
# =========================================================================
#
# `adaptive_dt`, `StateSnapshot` and the finiteness check are the
# TimeIntegration contract (Phase 23). These methods extend it to the
# mass-weighted cores so a nested run (Nesting.NestedRun) can pick each
# domain's step and roll a rejected step back whatever its dynamical core.

"""
    adaptive_dt(sim::MassWeightedSimulation, constraints; dt_prev = nothing,
                sgs = nothing, kwargs...)

The next time step of a sound-proof simulation: the smallest of the
advective limit `C_advective / max_cell Σ|uᵢ|/Δxᵢ` (the per-cell rate of
the Boussinesq [`cfl_diagnostics`](@ref Aeolus.TimeIntegration.cfl_diagnostics)),
the diffusive limit `C_diffusive Δ²_harm / (ν + max(νt, κt))` with the SGS
coefficients recomputed (pure `compute_sgs!`) from the current state, the
buoyancy limit `C_buoyancy / N` from the horizontal-mean θ profile, and
the Coriolis limit `C_coriolis / |f|`. Then, as for the Boussinesq core,
limited to `growth_max · dt_prev` and clamped to `[dt_min, dt_max]`; a
non-finite state collapses to `dt_min`. The other keywords of the
Boussinesq method (`scheme`, `sgs_stress`, `diagnostics`) are accepted and
ignored, so a driver can pass the same keywords to either core.
"""
function adaptive_dt(sim::MassWeightedSimulation{S,T}, c::CFLConstraints;
                     dt_prev = nothing, sgs = nothing, kwargs...) where {S,T}
    m = sim.mesh
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    rate = _max_advective_rate(sim.u.data, sim.v.data, sim.w.data, m)
    isnan(rate) && return c.dt_min
    dt = rate > 0 ? c.C_advective / rate : T(Inf)

    closure = _active_sgs(sim, sgs, nothing)
    κmax = zero(T)
    if closure !== nothing
        fill_halos!(sim.θ, sim.θ_bcs)
        compute_sgs!(closure, sim)
        f = sgs_fields(closure, sim)
        κmax = max(maximum(interior_view(f.νh)), maximum(interior_view(f.νv)),
                   maximum(interior_view(f.κh)), maximum(interior_view(f.κv)))
    end
    νtot = (sim.has_viscosity ? sim.ν : zero(T)) + κmax
    Δ²harm = one(T) / (1 / m.dx^2 + 1 / m.dy^2 + 1 / m.dz^2)
    νtot > 0 && (dt = min(dt, c.C_diffusive * Δ²harm / νtot))

    if sim.has_buoyancy
        θm = horizontal_mean(sim.θ)
        N² = zero(T)
        for k in 2:length(θm)
            N² = max(N², buoyancy_frequency_squared(θm[k-1], θm[k], m.dz, sim.g_accel,
                                                     (θm[k-1] + θm[k]) / 2))
        end
        N² > 0 && (dt = min(dt, c.C_buoyancy / sqrt(N²)))
    end
    sim.has_coriolis && sim.f_coriolis != 0 &&
        (dt = min(dt, c.C_coriolis / abs(sim.f_coriolis)))
    isnan(dt) && return c.dt_min
    if dt_prev !== nothing
        dtp = T(dt_prev)
        dtp > 0 && (dt = min(dt, c.growth_max * dtp))
    end
    return clamp(dt, c.dt_min, c.dt_max)
end

# A sound-proof core carries no qv: its snapshot holds an empty qv buffer.
StateSnapshot(sim::MassWeightedSimulation{S,T}) where {S,T} =
    StateSnapshot{T}(similar(sim.u.data), similar(sim.v.data), similar(sim.w.data),
                     similar(sim.p.data), similar(sim.θ.data), Array{T,3}(undef, 0, 0, 0),
                     sim.t, false)

function save_state!(snap::StateSnapshot, sim::MassWeightedSimulation)
    copyto!(snap.u, sim.u.data); copyto!(snap.v, sim.v.data)
    copyto!(snap.w, sim.w.data); copyto!(snap.p, sim.p.data)
    copyto!(snap.θ, sim.θ.data)
    snap.t = sim.t; snap.valid = true
    return snap
end

function restore_state!(sim::MassWeightedSimulation, snap::StateSnapshot)
    snap.valid || throw(ArgumentError("StateSnapshot has never been saved"))
    copyto!(sim.u.data, snap.u); copyto!(sim.v.data, snap.v)
    copyto!(sim.w.data, snap.w); copyto!(sim.p.data, snap.p)
    copyto!(sim.θ.data, snap.θ)
    sim.t = snap.t
    return sim
end

state_snapshot(sim::MassWeightedSimulation) =
    get!(() -> StateSnapshot(sim), _SNAPSHOTS, sim)::StateSnapshot

_all_finite(sim::MassWeightedSimulation) =
    all(isfinite, interior_view(sim.u)) && all(isfinite, interior_view(sim.v)) &&
    all(isfinite, interior_view(sim.w)) && all(isfinite, interior_view(sim.θ)) &&
    all(isfinite, interior_view(sim.p))

end # module
