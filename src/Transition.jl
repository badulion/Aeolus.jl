"""
    Transition

The Era VI owner of the laminar/turbulent and intermittency diagnostics
(Phase 59, P59-c, #180). Every later analysis that needs to know whether
a region, or a moment, is turbulent (the stable-boundary-layer collapse,
the CASES-99 "hockey stick", relaminarisation) calls these functions and
does not redefine them.

| function | quantity | reference |
| :-- | :-- | :-- |
| [`turbulent_fraction`](@ref), [`turbulent_indicator!`](@ref), [`laminar_mask!`](@ref) | plane fraction ``γ(z)`` of cells with ``\\lvert ω' \\rvert / ⟨\\lvert ω \\rvert⟩ > `` threshold | Brandt, Schlatter & Henningson (2004) |
| [`intermittency_factor`](@ref) | time fraction of a series above a threshold | Townsend (1976), §1.5 |
| [`flux_richardson`](@ref) | ``Ri_f = -B/P``, resolved + SGS fluxes | Ellison (1957); Ivey & Imberger (1991) |
| [`buoyancy_reynolds`](@ref) | ``Re_b = ε/(ν N^2)`` | Gibson (1980); Smyth & Moum (2000) |
| [`turbulence_velocity_scale`](@ref) | ``V_{TKE} = \\sqrt{e}`` | Sun et al. (2012) |
| [`growth_rate`](@ref) | log-linear least-squares growth rate, ``R^2`` window | — |
| [`supercritical_fraction`](@ref) | plane fraction with ``Ri_g > 1/4`` | Miles (1961); Howard (1961) |
| [`grid_reynolds_number`](@ref) | ``U Δ/(ν + ν_t)`` | — |
| [`friction_reynolds`](@ref) | ``Re_τ = u_τ h/ν`` of a no-slip wall | Pope (2000), §7.1 |
| [`relaminarisation_criteria`](@ref) | ``h/L``, ``L^+ = L u_τ/ν``, bulk ``Ri_b`` of a cooled channel | Nieuwstadt (2005); Flores & Riley (2011) |
| [`relaminarised_fraction`](@ref) | fraction of the ``Ri_g > 1/4`` cells that are γ-laminar with ``ν_t/ν < 10^{-2}`` | — |

The module defines **no Richardson number and no Obukhov length of its
own**: ``L`` is `SurfaceExchange.obukhov_length`, the bulk Richardson
number is `SurfaceExchange.bulk_richardson`, and ``N^2`` and
``Ri_g`` come from the `TurbulenceStability` contract (P58-a, #170) and
its pointwise fields `Operators.buoyancy_frequency_squared!` /
`Operators.gradient_richardson!`; [`supercritical_fraction`](@ref) only
counts cells of such a field. The cell-centred vorticity is
`Operators._vorticity_at`, built from the same velocity-gradient stencil
as the strain rate of the Smagorinsky closure.

# References

  * Brandt, L., Schlatter, P. & Henningson, D. S. (2004). Transition in
    boundary layers subject to free-stream turbulence. *J. Fluid Mech.*,
    517, 167-198.
  * Townsend, A. A. (1976). *The Structure of Turbulent Shear Flow*, 2nd
    ed. Cambridge University Press.
  * Ellison, T. H. (1957). Turbulent transport of heat and momentum from
    an infinite rough plane. *J. Fluid Mech.*, 2, 456-466.
  * Ivey, G. N. & Imberger, J. (1991). On the nature of turbulence in a
    stratified fluid. Part I. *J. Phys. Oceanogr.*, 21, 650-658.
  * Gibson, C. H. (1980). Fossil temperature, salinity, and vorticity
    turbulence in the ocean. In *Marine Turbulence*, Elsevier, 221-257.
  * Smyth, W. D. & Moum, J. N. (2000). Length scales of turbulence in
    stably stratified mixing layers. *Phys. Fluids*, 12, 1327-1342.
  * Sun, J., Mahrt, L., Banta, R. M. & Pichugina, Y. L. (2012).
    Turbulence regimes and turbulence intermittency in the stable
    boundary layer during CASES-99. *J. Atmos. Sci.*, 69, 338-351.
  * Miles, J. W. (1961). On the stability of heterogeneous shear flows.
    *J. Fluid Mech.*, 10, 496-508.
  * Howard, L. N. (1961). Note on a paper of John W. Miles. *J. Fluid
    Mech.*, 10, 509-512.
  * Nieuwstadt, F. T. M. (2005). Direct numerical simulation of stable
    channel flow at large stability. *Boundary-Layer Meteorol.*, 116,
    277-299.
  * Flores, O. & Riley, J. J. (2011). Analysis of turbulence collapse in
    the stably stratified surface layer using direct numerical
    simulation. *Boundary-Layer Meteorol.*, 139, 241-259.
  * Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press.
"""
module Transition

using ..Grids: RegularGrid, CellCenter, FaceX, FaceY, FaceZ, interior, xnodes, ynodes, znodes
using ..Fields: Field, interior_view, mesh
using ..BoundaryConditions: fill_halos!, NoSlipWall, FreeStreamBoundary, FieldBoundaryConditions
using ..Operators: _vorticity_at, _strain_at
using ..Operators: DampingProfile, NordstromRamp, damping_rate, smooth_step, _add_damping!   # P59-d
using ..Simulations: BoussinesqSimulation
import Random
using ..CutCell: wall_shear_stress
using ..MoistThermodynamics: ThermodynamicConstants
using ..SurfaceExchange: obukhov_length, bulk_richardson, SurfaceConstants, von_karman
using ..Operators: sgs_tke!, sgs_fields                                    # N-inflow-b
using LinearAlgebra: Symmetric, Diagonal, eigen, mul!
using FFTW: plan_rfft, irfft

export turbulent_fraction, turbulent_indicator!, laminar_mask!,
       intermittency_factor, flux_richardson, buoyancy_reynolds,
       turbulence_velocity_scale, growth_rate, supercritical_fraction,
       grid_reynolds_number
# P59-d (#185) — spatially developing flat-plate boundary layer.
export BlasiusSolution, blasius_solution, blasius_profile, blasius_velocity,
       blasius_skin_friction, turbulent_skin_friction,
       FringeRegion, fringe_rate, add_fringe!, fringe_forcing,
       blasius_target, set_blasius!, blasius_freestream, fringe_distance, flat_plate_setup,
       streamwise_profile, edge_velocity, boundary_layer_edge, skin_friction, displacement_thickness,
       momentum_thickness, shape_factor, transition_onset,
       onset_reynolds_theta_mayle, onset_reynolds_theta_abu_ghannam_shaw,
       FreeStreamDisturbance, von_karman_spectrum, disturbance_velocity!,
       disturbance_fringe_update
# N-inflow-a (#252): synthetic inflow turbulence (digital filter, Lund transform)
export AbstractInflowGenerator, unit_fluctuations!, generator_state, restore_generator_state!,
       inflow_plane_size, DigitalFilterGenerator, digital_filter_kernel, lund_coefficients, apply_lund!,
       FreeStreamInflow, disturbance_velocity, disturbance_covariance
# N-inflow-b (#258): statistics for the synthetic inflow (parent-resolved, parametrized)
export InflowStatistics, inflow_stresses, inflow_lengths, update_inflow_statistics!,
       inflow_projection_log, realizable_stress, inflow_lund_coefficients,
       ParentResolvedStatistics, inflow_stress_parts
# P59-e (#195): relaminarisation in strongly stable stratification
export friction_reynolds, relaminarisation_criteria, relaminarised_fraction

# =========================================================================
# Turbulent fraction γ(z)
# =========================================================================

const _INDICATORS = (:vorticity_fluctuation,)

_fluid_ok(::Nothing, i, j, k) = true
_fluid_ok(fluid, i, j, k) = @inbounds fluid[i, j, k]

"""
    turbulent_indicator!(ind, u, v, w; indicator = :vorticity_fluctuation,
                         threshold = 0.2, fluid = nothing,
                         normalization = :plane) -> ind

Pointwise laminar/turbulent indicator on the interior cells: `ind[i,j,k]`
(a `Bool` array of the interior size) is `true` where

```math
\\frac{\\lvert ω' \\rvert}{⟨\\lvert ω \\rvert⟩} > \\text{threshold},
\\qquad ω' = ω - ⟨ω⟩,
```

with ``ω = ∇×u`` at the cell centre (the centred stencil of the
Smagorinsky strain rate, `Operators._vorticity_at`) and ``⟨·⟩`` the mean
over the fluid cells of the level (`normalization = :plane`, the
streamwise–spanwise average of Brandt, Schlatter & Henningson 2004) or
over all fluid cells of the domain (`normalization = :volume`, for a
domain whose free stream is quiet). The fluctuation ``ω'`` is always
taken about the plane mean, so a laminar parallel shear flow
(``ω`` uniform in each plane) is laminar everywhere: ``ω' = 0`` up to
round-off, far below any threshold. A plane (or domain) without
vorticity is laminar.

`fluid` (a `Bool` array of the interior size, or `nothing` for all
fluid) marks the cells that count; solid cells are never turbulent.
The halos of `u, v, w` must be filled.
"""
function turbulent_indicator!(ind::AbstractArray{Bool,3}, u::Field, v::Field, w::Field;
                              indicator::Symbol = :vorticity_fluctuation,
                              threshold::Real = 0.2, fluid = nothing,
                              normalization::Symbol = :plane)
    indicator in _INDICATORS ||
        throw(ArgumentError("turbulent_indicator!: unknown indicator :$indicator " *
                            "(known: $(join(string.(':', _INDICATORS), ", ")))"))
    normalization in (:plane, :volume) ||
        throw(ArgumentError("turbulent_indicator!: normalization must be :plane or :volume"))
    m = mesh(u)
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    size(ind) == (Nx, Ny, Nz) || throw(DimensionMismatch("ind must be $((Nx, Ny, Nz))"))
    idx = 1 / m.dx; idy = 1 / m.dy; idz = 1 / m.dz
    ud, vd, wd = u.data, v.data, w.data
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    ωx = zeros(Nx, Ny); ωy = zeros(Nx, Ny); ωz = zeros(Nx, Ny)
    # plane means of the vector ω and of |ω|, and the counts
    mω = zeros(3, Nz); mabs = zeros(Nz); cnt = zeros(Int, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        _fluid_ok(fluid, i, j, k) || continue
        a, b, c = _vorticity_at(ud, vd, wd, Hx + i, Hy + j, Hz + k, idx, idy, idz)
        mω[1, k] += a; mω[2, k] += b; mω[3, k] += c
        mabs[k] += sqrt(a^2 + b^2 + c^2); cnt[k] += 1
    end
    vol_abs = sum(mabs) / max(sum(cnt), 1)
    @inbounds for k in 1:Nz
        n = cnt[k]
        if n > 0
            mω[1, k] /= n; mω[2, k] /= n; mω[3, k] /= n; mabs[k] /= n
        end
        ref = normalization === :plane ? mabs[k] : vol_abs
        thr = threshold * ref
        for j in 1:Ny, i in 1:Nx
            if !_fluid_ok(fluid, i, j, k) || !(ref > 0)
                ind[i, j, k] = false
                continue
            end
            a, b, c = _vorticity_at(ud, vd, wd, Hx + i, Hy + j, Hz + k, idx, idy, idz)
            ind[i, j, k] = sqrt((a - mω[1, k])^2 + (b - mω[2, k])^2 + (c - mω[3, k])^2) > thr
        end
    end
    return ind
end

"""
    laminar_mask!(mask, u, v, w; kwargs...) -> mask

The complement of [`turbulent_indicator!`](@ref) within the fluid:
`mask[i,j,k]` is `true` in a fluid cell that is not turbulent (and
`false` in solid cells). Same keywords.
"""
function laminar_mask!(mask::AbstractArray{Bool,3}, u::Field, v::Field, w::Field;
                       fluid = nothing, kwargs...)
    turbulent_indicator!(mask, u, v, w; fluid = fluid, kwargs...)
    @inbounds for k in axes(mask, 3), j in axes(mask, 2), i in axes(mask, 1)
        mask[i, j, k] = _fluid_ok(fluid, i, j, k) && !mask[i, j, k]
    end
    return mask
end

"""
    turbulent_fraction(ind::AbstractArray{Bool,3}; fluid = nothing) -> γ::Vector
    turbulent_fraction(u, v, w; indicator = :vorticity_fluctuation,
                       threshold = 0.2, fluid = nothing, normalization = :plane) -> γ
    turbulent_fraction(sim::BoussinesqSimulation; kwargs...) -> γ

The turbulent fraction ``γ(z)``: per level, the fraction of the fluid
cells that the indicator marks turbulent (see
[`turbulent_indicator!`](@ref) for the vorticity-fluctuation criterion
and its keywords; Brandt, Schlatter & Henningson 2004). ``0 ≤ γ ≤ 1``;
a level without fluid cells gets `NaN`. The `sim` method fills the
velocity halos with the simulation's boundary conditions first.

The default threshold 0.2 marks as turbulent every cell whose vorticity
fluctuation exceeds a fifth of the plane-mean vorticity magnitude. For a
fully developed isotropic field, where ``ω'`` is close to a Gaussian
vector, the fraction below that is ``\\sim 1\\%`` (the Maxwell
distribution at ``0.2⟨|ω|⟩ = 0.32σ``), so ``γ ≈ 0.99``.
"""
function turbulent_fraction(ind::AbstractArray{Bool,3}; fluid = nothing)
    Nx, Ny, Nz = size(ind)
    γ = fill(NaN, Nz)
    @inbounds for k in 1:Nz
        n = 0; t = 0
        for j in 1:Ny, i in 1:Nx
            _fluid_ok(fluid, i, j, k) || continue
            n += 1; t += ind[i, j, k]
        end
        n > 0 && (γ[k] = t / n)
    end
    return γ
end

function turbulent_fraction(u::Field, v::Field, w::Field; fluid = nothing, kwargs...)
    m = mesh(u)
    ind = Array{Bool,3}(undef, m.Nx, m.Ny, m.Nz)
    turbulent_indicator!(ind, u, v, w; fluid = fluid, kwargs...)
    return turbulent_fraction(ind; fluid = fluid)
end

function turbulent_fraction(sim::BoussinesqSimulation; kwargs...)
    _fill_velocity_halos!(sim)
    return turbulent_fraction(sim.u, sim.v, sim.w; kwargs...)
end

function _fill_velocity_halos!(sim)
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    return sim
end

# =========================================================================
# Time-series intermittency
# =========================================================================

"""
    intermittency_factor(series; threshold) -> Float64
    intermittency_factor(t, series; threshold) -> Float64

The intermittency factor: the fraction of time a signal spends above
`threshold` (Townsend 1976). Without times, every sample counts equally
(a uniformly sampled series): the result is the number of samples with
`series > threshold` divided by the number of samples. With sample times
`t` (increasing), sample `i` stands for the interval ``[t_i, t_{i+1})``
and the result is the time-weighted fraction over ``[t_1, t_N)`` (the
last sample closes the record).

An on/off signal with duty cycle ``d`` gives exactly ``d``.
"""
function intermittency_factor(series::AbstractVector{<:Real}; threshold::Real)
    isempty(series) && throw(ArgumentError("intermittency_factor: empty series"))
    return count(>(threshold), series) / length(series)
end

function intermittency_factor(t::AbstractVector{<:Real}, series::AbstractVector{<:Real};
                              threshold::Real)
    length(t) == length(series) || throw(DimensionMismatch("t and series differ in length"))
    length(t) >= 2 || throw(ArgumentError("intermittency_factor: need two samples"))
    on = 0.0
    @inbounds for i in 1:length(t)-1
        dt = t[i+1] - t[i]
        dt >= 0 || throw(ArgumentError("intermittency_factor: t must be increasing"))
        series[i] > threshold && (on += dt)
    end
    return on / (t[end] - t[1])
end

# =========================================================================
# Ri_f, Re_b, V_TKE, grid Re
# =========================================================================

"""
    flux_richardson(u, v, w, θ; θ₀, g = 9.81, νt = nothing, Pr_t = 1/3)
        -> (Ri_f, B, P, uw, vw, wθ)
    flux_richardson(sim::BoussinesqSimulation)

The flux Richardson number per level,

```math
Ri_f = -\\frac{B}{P}, \\qquad
B = \\frac{g}{θ_0} ⟨w'θ'⟩, \\qquad
P = -⟨u'w'⟩\\frac{∂U}{∂z} - ⟨v'w'⟩\\frac{∂V}{∂z},
```

the ratio of the buoyancy destruction of turbulence kinetic energy to its
shear production (Ellison 1957; Ivey & Imberger 1991, where the
irreversible part is the mixing efficiency). ``⟨·⟩`` is the plane mean
and ``U, V, Θ`` the plane-mean profiles. The fluxes are the **resolved
plus the SGS** fluxes: with an eddy viscosity `νt` (a `CellCenter`
field) the SGS parts

```math
τ_{13} = -2ν_t S_{13}, \\quad τ_{23} = -2ν_t S_{23}, \\quad
⟨w'θ'⟩_{\\rm sgs} = -\\frac{ν_t}{Pr_t}\\frac{∂θ}{∂z}
```

are added (``S_{ij}`` from the Smagorinsky stencil `Operators._strain_at`,
``∂θ/∂z`` centred over ``2Δz``, as in `add_eddy_diffusion_scalar!`).
Resolved fluxes use `u, v, w` interpolated to the cell centre;
``∂U/∂z`` is centred (one-sided at the first and last level).

Returns profiles: `Ri_f` (`NaN` where ``P = 0``), `B`, `P` and the total
kinematic fluxes `uw`, `vw`, `wθ`. Halos of `u, v, w, θ` (and `νt`) must
be filled. The `sim` method uses `sim.θ₀`, `sim.g_accel`, `sim.Pr_t` and,
when the SGS closure is on, `sim.νt`, and fills the halos first.
"""
function flux_richardson(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                         θ::Field{CellCenter}; θ₀::Real, g::Real = 9.81,
                         νt = nothing, Pr_t::Real = 1/3)
    m = mesh(u)
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    idx = 1 / m.dx; idy = 1 / m.dy; idz = 1 / m.dz
    ud, vd, wd, θd = u.data, v.data, w.data, θ.data
    nd = νt === nothing ? nothing : νt.data
    n = Nx * Ny
    U = zeros(Nz); V = zeros(Nz); W = zeros(Nz); Θ = zeros(Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        U[k] += 0.5 * (ud[I, J, K] + ud[I+1, J, K])
        V[k] += 0.5 * (vd[I, J, K] + vd[I, J+1, K])
        W[k] += 0.5 * (wd[I, J, K] + wd[I, J, K+1])
        Θ[k] += θd[I, J, K]
    end
    U ./= n; V ./= n; W ./= n; Θ ./= n
    uw = zeros(Nz); vw = zeros(Nz); wθ = zeros(Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        uc = 0.5 * (ud[I, J, K] + ud[I+1, J, K]) - U[k]
        vc = 0.5 * (vd[I, J, K] + vd[I, J+1, K]) - V[k]
        wc = 0.5 * (wd[I, J, K] + wd[I, J, K+1]) - W[k]
        uw[k] += uc * wc; vw[k] += vc * wc; wθ[k] += wc * (θd[I, J, K] - Θ[k])
        if nd !== nothing
            S = _strain_at(ud, vd, wd, I, J, K, idx, idy, idz)
            ν = nd[I, J, K]
            uw[k] -= 2ν * S[5]
            vw[k] -= 2ν * S[6]
            wθ[k] -= ν / Pr_t * (θd[I, J, K+1] - θd[I, J, K-1]) * (0.5 * idz)
        end
    end
    uw ./= n; vw ./= n; wθ ./= n
    B = (g / θ₀) .* wθ
    P = zeros(Nz)
    @inbounds for k in 1:Nz
        kl = max(k - 1, 1); kh = min(k + 1, Nz)
        kh == kl && continue
        dz = (kh - kl) * m.dz
        P[k] = -uw[k] * (U[kh] - U[kl]) / dz - vw[k] * (V[kh] - V[kl]) / dz
    end
    Ri_f = [P[k] == 0 ? NaN : -B[k] / P[k] for k in 1:Nz]
    return (Ri_f = Ri_f, B = B, P = P, uw = uw, vw = vw, wθ = wθ)
end

function flux_richardson(sim::BoussinesqSimulation)
    sim.has_buoyancy ||
        throw(ArgumentError("flux_richardson: the simulation has no buoyancy (θ₀ = 0)"))
    _fill_velocity_halos!(sim)
    fill_halos!(sim.θ, sim.θ_bcs)
    sgs = sim.has_sgs
    sgs && fill_halos!(sim.νt, sim.bcs)
    return flux_richardson(sim.u, sim.v, sim.w, sim.θ; θ₀ = sim.θ₀, g = sim.g_accel,
                           νt = sgs ? sim.νt : nothing, Pr_t = sim.Pr_t)
end

"""
    buoyancy_reynolds(ε, ν, N²) -> Float64

The buoyancy Reynolds number ``Re_b = ε/(ν N^2)`` (Gibson 1980; Smyth &
Moum 2000): the squared ratio of the Ozmidov scale
``L_O = (ε/N^3)^{1/2}`` to the Kolmogorov scale
``η = (ν^3/ε)^{1/4}`` to the power 4/3, ``Re_b = (L_O/η)^{4/3}``. Below
``Re_b ≈ 20`` stratification suppresses the inertial range (Smyth & Moum
2000). Returns `Inf` for ``N^2 ≤ 0`` (no buoyancy constraint).
"""
buoyancy_reynolds(ε::Real, ν::Real, N2::Real) = N2 > 0 ? ε / (ν * N2) : Inf

"""
    turbulence_velocity_scale(tke) -> Float64
    turbulence_velocity_scale(σu, σv, σw) -> Float64

The turbulence velocity scale of Sun et al. (2012),
``V_{TKE} = \\sqrt{e}`` with ``e = (σ_u^2 + σ_v^2 + σ_w^2)/2`` the
turbulence kinetic energy per unit mass — the variable whose dependence
on the wind speed separates the three turbulence regimes of the stable
boundary layer (the CASES-99 "hockey stick").
"""
turbulence_velocity_scale(tke::Real) =
    tke >= 0 ? sqrt(tke) : throw(DomainError(tke, "TKE must be non-negative"))
turbulence_velocity_scale(σu::Real, σv::Real, σw::Real) =
    sqrt((σu^2 + σv^2 + σw^2) / 2)

"""
    grid_reynolds_number(U, Δ, ν, νt = 0) -> Float64

The grid Reynolds number ``Re_Δ = UΔ/(ν + ν_t)``: the Reynolds number of
the smallest resolved eddies with the total (molecular, or the model's
numerical, plus SGS) viscosity. ``Re_Δ ≲ 1`` means the grid scale is
viscously damped (a DNS-like resolution); a large value means the
closure, not the viscosity, sets the dissipation.
"""
grid_reynolds_number(U::Real, Δ::Real, ν::Real, νt::Real = 0) = U * Δ / (ν + νt)

"""
    supercritical_fraction(Ri; fluid = nothing, Ri_c = 0.25) -> Vector

Per level, the fraction of the fluid cells whose gradient Richardson
number exceeds `Ri_c` (default the Miles–Howard 1/4: a parallel
stratified shear flow with ``Ri_g > 1/4`` everywhere is linearly stable;
Miles 1961, Howard 1961). `Ri` is a `CellCenter` field (from
`Operators.gradient_richardson!`) or an array of the interior size; the
Richardson number itself is not recomputed here. A level without fluid
cells gets `NaN`.
"""
supercritical_fraction(Ri::Field; kwargs...) =
    supercritical_fraction(interior_view(Ri); kwargs...)

function supercritical_fraction(Ri::AbstractArray{<:Real,3}; fluid = nothing,
                                Ri_c::Real = 0.25)
    Nx, Ny, Nz = size(Ri)
    f = fill(NaN, Nz)
    @inbounds for k in 1:Nz
        n = 0; s = 0
        for j in 1:Ny, i in 1:Nx
            _fluid_ok(fluid, i, j, k) || continue
            n += 1; s += Ri[i, j, k] > Ri_c
        end
        n > 0 && (f[k] = s / n)
    end
    return f
end

# =========================================================================
# Relaminarisation in strongly stable stratification (P59-e, #195)
# =========================================================================

# SurfaceConstants whose gravity is the simulation's (the channels are run
# in wall units with g = 1); κ stays the package's von Kármán constant.
_constants_with_gravity(g::Real) =
    SurfaceConstants(thermo = ThermodynamicConstants(grav = Float64(g)))

# u_τ of the bottom wall from the molecular wall stress (P59-a)
_wall_friction_velocity(sim) = sqrt(abs(wall_shear_stress(sim; side = :bottom)))

"""
    friction_reynolds(sim::BoussinesqSimulation; u_τ = nothing, h = sim.mesh.Lz) -> Re_τ

The friction Reynolds number ``Re_τ = u_τ h/ν`` of a channel with a
resolved no-slip bottom wall: `u_τ` defaults to the square root of the
plane-mean molecular wall stress of the bottom wall
(`wall_shear_stress(sim; side = :bottom)`, P59-a), `h` to the domain
height (the depth of an open channel; pass the half-height δ for a closed
one). Pass `u_τ` to use the nominal value that a mean pressure gradient
``u_τ^2/h`` imposes (Nieuwstadt 2005), as the collapse studies do.
"""
function friction_reynolds(sim::BoussinesqSimulation; u_τ = nothing, h::Real = sim.mesh.Lz)
    sim.ν > 0 || throw(ArgumentError("friction_reynolds: the simulation is inviscid (ν = 0)"))
    uτ = u_τ === nothing ? _wall_friction_velocity(sim) : float(u_τ)
    return uτ * h / sim.ν
end

"""
    relaminarisation_criteria(sim::BoussinesqSimulation; heat_flux = sim.θ_flux_bottom,
                              u_τ = nothing, h = sim.mesh.Lz)
        -> (h_over_L, L_plus, Ri_b, L, u_τ, Re_τ)

The three parameters that the stable-channel collapse studies use to
predict relaminarisation of a channel of depth `h` cooled at the bottom
by the kinematic heat flux `heat_flux` ``= \\overline{w'θ'}_0`` (K m s⁻¹,
negative for cooling; default the simulation's prescribed bottom flux):

  * ``h/L`` with the Obukhov length of the surface flux,
    ``L = u_τ^2 θ_0/(κ g θ_*)``, ``θ_* = -\\overline{w'θ'}_0/u_τ``
    (`SurfaceExchange.obukhov_length` with the simulation's ``g`` and
    ``θ_0``; positive for cooling). Nieuwstadt (2005) finds the
    turbulence of an open channel collapses for ``h/L ≳ 1.23`` at
    ``Re_τ = 360``;
  * ``L^+ = L u_τ/ν``, the Obukhov length in wall units; Flores & Riley
    (2011) find collapse for ``L^+ ≲ 100`` (``L^+ = Re_τ/(h/L)``);
  * the bulk Richardson number ``Ri_b = g h (Θ_{top} - Θ_{bot})/(Θ_{top} U_b^2)``
    (`SurfaceExchange.bulk_richardson`) of the plane-mean θ of the top
    and the bottom cell layers and the bulk (volume-mean) streamwise
    velocity ``U_b``.

`u_τ` defaults to the measured friction velocity of the bottom wall (see
[`friction_reynolds`](@ref)); pass the nominal value to evaluate the
criteria of a case set up by its forcing, as the published thresholds
are. Needs buoyancy (``θ_0 > 0``). A neutral flux gives ``h/L = 0`` and
``L^+ = Inf``.
"""
function relaminarisation_criteria(sim::BoussinesqSimulation;
                                   heat_flux::Real = sim.θ_flux_bottom,
                                   u_τ = nothing, h::Real = sim.mesh.Lz)
    sim.has_buoyancy ||
        throw(ArgumentError("relaminarisation_criteria: the simulation has no buoyancy (θ₀ = 0)"))
    sim.ν > 0 || throw(ArgumentError("relaminarisation_criteria: the simulation is inviscid (ν = 0)"))
    uτ = u_τ === nothing ? _wall_friction_velocity(sim) : float(u_τ)
    c = _constants_with_gravity(sim.g_accel)
    θ₀ = float(sim.θ₀)
    L = heat_flux == 0 ? Inf : obukhov_length(uτ, -heat_flux / uτ, θ₀; c = c)
    m = sim.mesh
    Nx, Ny, Nz, Hx, Hy, Hz = m.Nx, m.Ny, m.Nz, m.Hx, m.Hy, m.Hz
    ud, θd = sim.u.data, sim.θ.data
    Ub = 0.0; θb = 0.0; θt = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        for k in 1:Nz
            Ub += 0.5 * (ud[Hx+i, Hy+j, Hz+k] + ud[Hx+i+1, Hy+j, Hz+k])
        end
        θb += θd[Hx+i, Hy+j, Hz+1]; θt += θd[Hx+i, Hy+j, Hz+Nz]
    end
    n = Nx * Ny
    Ub /= n * Nz; θb /= n; θt /= n
    Ri_b = bulk_richardson(h, Ub, θt, θb; c = c)
    return (h_over_L = h / L, L_plus = L * uτ / sim.ν, Ri_b = Ri_b, L = L, u_τ = uτ,
            Re_τ = uτ * h / sim.ν)
end

"""
    relaminarised_fraction(laminar, Ri, νt, ν; fluid = nothing, Ri_c = 0.25,
                           νt_ratio = 1e-2) -> (profile, fraction, nstable)

The relaminarised share of the stable layer: among the fluid cells whose
gradient Richardson number exceeds `Ri_c` (default the Miles–Howard 1/4),
the fraction that are laminar by the vorticity-fluctuation indicator
(`laminar`, a `Bool` array from [`laminar_mask!`](@ref), i.e. ``γ = 0``
there) **and** carry a negligible eddy viscosity, ``ν_t/ν <`` `νt_ratio`.
A laminar-consistent closure (the σ-model) switches itself off in such
cells; Smagorinsky does not, as long as the resolved strain is non-zero.

`Ri` and `νt` are `CellCenter` fields or arrays of the interior size
(the Richardson number comes from `Operators.gradient_richardson!`; none
is recomputed here). Returns the per-level fraction (`NaN` on a level
with no stable cell), the fraction over the whole stable layer (`NaN` if
it is empty) and the number of stable cells.
"""
function relaminarised_fraction(laminar::AbstractArray{Bool,3}, Ri, νt, ν::Real;
                                fluid = nothing, Ri_c::Real = 0.25, νt_ratio::Real = 1e-2)
    R = Ri isa Field ? interior_view(Ri) : Ri
    V = νt isa Field ? interior_view(νt) : νt
    Nx, Ny, Nz = size(laminar)
    size(R) == size(V) == (Nx, Ny, Nz) ||
        throw(DimensionMismatch("laminar, Ri and νt must have the interior size $((Nx, Ny, Nz))"))
    ν > 0 || throw(ArgumentError("relaminarised_fraction: ν must be positive"))
    prof = fill(NaN, Nz)
    nst = 0; nrel = 0
    lim = νt_ratio * ν
    @inbounds for k in 1:Nz
        n = 0; r = 0
        for j in 1:Ny, i in 1:Nx
            _fluid_ok(fluid, i, j, k) || continue
            R[i, j, k] > Ri_c || continue
            n += 1
            r += laminar[i, j, k] && V[i, j, k] < lim
        end
        n > 0 && (prof[k] = r / n)
        nst += n; nrel += r
    end
    return (profile = prof, fraction = nst > 0 ? nrel / nst : NaN, nstable = nst)
end

# =========================================================================
# Growth-rate fit
# =========================================================================

"""
    growth_rate(t, A; window = nothing, r2_min = 0.999, rate_tol = 0.01, min_points = 10)
        -> (rate, intercept, r2, window, npoints)

Exponential growth rate ``σ`` of an amplitude series ``A(t) ∝ e^{σt}``
from the least-squares fit of ``\\ln A = σ t + c``.

* `window = (t₁, t₂)`: fit the samples with ``t₁ ≤ t ≤ t₂``.
* `window = nothing` (default): the fit window is chosen automatically
  as the **longest** contiguous run of at least `min_points` samples
  whose fit has ``R^2 ≥ r_{2,\\min}`` (default 0.999) **and** whose three
  thirds, fitted separately, give rates that agree to `rate_tol`
  (relative, default 1 %). The second condition is what makes the window
  the stretch of clean exponential growth: ``R^2`` alone grows with the
  span of ``\\ln A``, so a long window that includes the adjustment of
  the seed or the onset of saturation still passes 0.999 with a rate
  biased by a few per cent (measured: 0.1859 instead of the 0.1864 of
  the linear stage of the Kelvin–Helmholtz test, and 2.8 % low on a
  synthetic saturating series), while the rates of the thirds of a bent
  ``\\ln A`` differ (thirds rather than halves: a seed transient at the
  start and saturation at the end bias both halves alike, but not the
  middle third). An error is thrown if no window qualifies.

Returns the rate, the intercept, the ``R^2`` of the fit, the time window
used and the number of samples in it. Non-positive amplitudes are not
allowed.
"""
function growth_rate(t::AbstractVector{<:Real}, A::AbstractVector{<:Real};
                     window = nothing, r2_min::Real = 0.999, rate_tol::Real = 0.01,
                     min_points::Integer = 10)
    length(t) == length(A) || throw(DimensionMismatch("t and A differ in length"))
    all(>(0), A) || throw(ArgumentError("growth_rate: amplitudes must be positive"))
    y = log.(Float64.(A)); x = Float64.(t)
    if window !== nothing
        idx = findall(ti -> window[1] <= ti <= window[2], x)
        length(idx) >= 3 || throw(ArgumentError("growth_rate: fewer than 3 samples in the window"))
        σ, c, r2 = _linfit(view(x, idx), view(y, idx))
        return (rate = σ, intercept = c, r2 = r2, window = (x[idx[1]], x[idx[end]]),
                npoints = length(idx))
    end
    N = length(x)
    N >= min_points || throw(ArgumentError("growth_rate: fewer than min_points samples"))
    # prefix sums about the series means (conditioning) → O(1) per window
    x0 = sum(x) / N; y0 = sum(y) / N
    Sx = zeros(N + 1); Sy = zeros(N + 1); Sxx = zeros(N + 1); Syy = zeros(N + 1); Sxy = zeros(N + 1)
    @inbounds for i in 1:N
        a = x[i] - x0; b = y[i] - y0
        Sx[i+1] = Sx[i] + a; Sy[i+1] = Sy[i] + b
        Sxx[i+1] = Sxx[i] + a * a; Syy[i+1] = Syy[i] + b * b; Sxy[i+1] = Sxy[i] + a * b
    end
    # slope and R² of samples a..b from the prefix sums
    @inline function seg(a, b)
        n = b - a + 1
        sx = Sx[b+1] - Sx[a]; sy = Sy[b+1] - Sy[a]
        cxx = (Sxx[b+1] - Sxx[a]) - sx * sx / n
        cyy = (Syy[b+1] - Syy[a]) - sy * sy / n
        cxy = (Sxy[b+1] - Sxy[a]) - sx * sy / n
        cxx > 0 || return (NaN, 0.0)
        return (cxy / cxx, cyy > 0 ? cxy * cxy / (cxx * cyy) : 1.0)
    end
    best = (0, 0); bestspan = -Inf
    @inbounds for i in 1:N-min_points+1
        for j in N:-1:i+min_points-1
            x[j] - x[i] <= bestspan && break      # cannot beat the best span
            σ, r2 = seg(i, j)
            r2 >= r2_min || continue
            n3 = (j - i + 1) ÷ 3
            σ1, _ = seg(i, i + n3 - 1); σ2, _ = seg(i + n3, j - n3); σ3, _ = seg(j - n3 + 1, j)
            if max(σ1, σ2, σ3) - min(σ1, σ2, σ3) <= rate_tol * abs(σ)
                best = (i, j); bestspan = x[j] - x[i]
                break
            end
        end
    end
    best == (0, 0) &&
        error("growth_rate: no window of ≥ $min_points samples has R² ≥ $r2_min and " *
              "rates of its thirds within $rate_tol")
    i, j = best
    σ, c, r2 = _linfit(view(x, i:j), view(y, i:j))
    return (rate = σ, intercept = c, r2 = r2, window = (x[i], x[j]), npoints = j - i + 1)
end

function _linfit(x, y)
    n = length(x)
    mx = sum(x) / n; my = sum(y) / n
    cxx = sum(xi -> (xi - mx)^2, x)
    cxy = sum(i -> (x[i] - mx) * (y[i] - my), eachindex(x))
    cyy = sum(yi -> (yi - my)^2, y)
    σ = cxy / cxx
    r2 = cyy > 0 ? cxy^2 / (cxx * cyy) : 1.0
    return σ, my - σ * mx, r2
end

# =========================================================================
# Spatially developing flat-plate boundary layer (P59-d, #185)
# =========================================================================

include("transition/blasius.jl")
include("transition/flat_plate.jl")
include("transition/free_stream_disturbance.jl")
include("transition/synthetic_inflow.jl")             # N-inflow-a (#252)
include("transition/inflow_statistics.jl")            # N-inflow-b (#258)

end # module
