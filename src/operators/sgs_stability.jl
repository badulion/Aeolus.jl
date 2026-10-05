# =========================================================================
# Era VI, Phase 55 (P55-e, #187) — stratification-dependent anisotropic
# eddy viscosity and diffusivity: the StabilityCorrected wrapper.
# =========================================================================
#
# Included into `module Operators`, after the SGS closure contract
# (sgs_closures.jl), the scalar-tensor closures (scalar_tensor.jl) and the
# pointwise N² / Ri_g stencils (stratification.jl). It is the one
# stability-corrected closure of Era VI (P55 "stratification-dependent
# anisotropic" and the former P58 "buoyancy-aware Smagorinsky" merged):
#
#   νh = ν_base·ℓh            νv = ν_base·ℓv·f_m(Ri)
#   κh = ν_base·ℓh / Prₜ      κv = ν_base·ℓv·f_h(Ri) / Prₜ
#
# with ℓh = (Δh/Δ)², ℓv = (Δv/Δ)² (the anisotropic length scales; 1 for
# `length_scales = :isotropic`), Ri_g from `gradient_richardson!` and f_m,
# f_h from the TurbulenceStability contract (P58-a, #170). No stability
# function is defined here. The four fields reach the P55-b stress-form
# momentum operator and the P55-c diagonal scalar operator through the
# contract defaults of `add_sgs_momentum!` / `add_sgs_scalar!`, so the
# correction works on every core (step!, step_rk3!, step_imex!,
# step_adaptive!, and the anelastic / pseudo-incompressible step!).

"""
    dynamic_sgs_coefficient(c::AbstractEddyViscosityClosure) -> Bool

`true` for a closure whose coefficient is computed dynamically from the
resolved flow (the Germano identity: [`DynamicSmagorinsky`](@ref) and
[`LagrangianDynamicSmagorinsky`](@ref), scale invariant or scale dependent).
[`StabilityCorrected`](@ref) applies no `f_m` multiplier on top of such a
coefficient: the dynamic procedure already sees the damping of the resolved
motions by the stratification, and multiplying by `f_m` would count it
twice. Only the scalar diffusivity is corrected, through the turbulent
Prandtl number `Prₜ(Ri) = Prₜ f_m(Ri)/f_h(Ri)`. Default `false`.
"""
dynamic_sgs_coefficient(::AbstractEddyViscosityClosure) = false
dynamic_sgs_coefficient(::DynamicSmagorinsky) = true
dynamic_sgs_coefficient(::LagrangianDynamicSmagorinsky) = true   # P54-b / LASD P54-e

"""
    anisotropic_length_factors(mesh::RegularGrid) -> (ℓh, ℓv)

The factors `ℓh = (Δh/Δ)²`, `ℓv = (Δv/Δ)²` that turn an eddy viscosity built
on the isotropic filter width `Δ = (dx dy dz)^{1/3}` (the `Δ` of
[`compute_smagorinsky_νt!`](@ref)) into the horizontal and vertical ones
built on `Δh = √(dx dy)` and `Δv = dz` (Bryan & Fritsch 2002; CM1's
anisotropic closure), for `ν ∝ Δ²`. Exactly `(1, 1)` when
`dx == dy == dz`. On the Berlin grid (`dx/dz = 2.5`) they are
`2.5^{2/3} = 1.842` and `2.5^{-4/3} = 0.2947`.

Reference: Bryan, G. H. & Fritsch, J. M. (2002), *Mon. Wea. Rev.* 130,
2917 (the CM1 model; its anisotropic subgrid turbulence option).
"""
function anisotropic_length_factors(g::RegularGrid)
    dx, dy, dz = g.dx, g.dy, g.dz
    dx == dy == dz && return (one(dx), one(dx))
    Δ2 = ((dx * dy * dz)^(1 / 3))^2
    return (dx * dy / Δ2, dz^2 / Δ2)
end

"""
    StabilityCorrected(base::AbstractEddyViscosityClosure, mesh;
                       scheme = BrownEtAl1994(), length_scales = :anisotropic,
                       qv = nothing, S2_min = RI_SHEAR2_FLOOR)

A closure that corrects the eddy viscosity `ν_base` of `base`
([`StaticSmagorinsky`](@ref), [`DynamicSmagorinsky`](@ref),
[`FunctionSGS`](@ref), `ML.MLSGS`, or any other
[`AbstractEddyViscosityClosure`](@ref)) for the local stratification and
splits it into horizontal and vertical coefficients:

```math
ν_h = ν_{base}\\,ℓ_h, \\qquad ν_v = ν_{base}\\,ℓ_v\\,f_m(Ri), \\qquad
κ_h = \\frac{ν_{base}\\,ℓ_h}{Pr_t}, \\qquad κ_v = \\frac{ν_{base}\\,ℓ_v\\,f_h(Ri)}{Pr_t}.
```

* **Ri.** The gradient Richardson number at the cell centre,
  [`gradient_richardson!`](@ref)`(Ri, N², u, v, w; S2_min)`, with
  ``|S|^2 = 2S_{ij}S_{ij}`` (the strain the Smagorinsky closure sees) and
  ``N^2`` from [`buoyancy_frequency_squared!`](@ref): of θ, or of the
  virtual potential temperature when a humidity is known (`qv`, a
  `Field{CellCenter}` such as a tracer, or the core's own `sim.qv` when it
  carries moisture). The reference is `sim.θ₀` on the Boussinesq core and
  the profile `θ̄(z)` (`sim.ref`) on the anelastic / pseudo-incompressible
  cores, the reference their buoyancy uses.
* **f_m, f_h** are the stability factors of the TurbulenceStability
  contract, `stability_factor_m(scheme, Ri)` and
  `stability_factor_h(scheme, Ri)` (P58-a, #170): both 1 at `Ri = 0`, so a
  neutral flow is not changed. `scheme` is any `AbstractRiStability`:
  `BrownEtAl1994()` (default; the LEM / MONC functions, zero above
  ``Ri_c = 1/4``, enhanced mixing for ``Ri < 0``), `LillyCutoff()` (Lilly
  1962; Mason 1989), `LongTail()`, `ShortTail()`,
  `SurfaceExchange.MOSTConsistent(…)`. The neutral turbulent Prandtl
  number is `base`'s ([`turbulent_prandtl`](@ref)), not
  `neutral_prandtl(scheme)`: the factors are normalised to 1 at
  `Ri = 0`, so the base closure keeps the one neutral value of the run.
* **Length scales.** `length_scales = :anisotropic` (default) uses the
  horizontal filter width `Δh = √(dx dy)` for the horizontal and `Δv = dz`
  for the vertical coefficients instead of `Δ = (dx dy dz)^{1/3}`
  (Bryan & Fritsch 2002; CM1), through the factors
  [`anisotropic_length_factors`](@ref) (exact for `ν ∝ Δ²`, true of every
  Smagorinsky-type closure). `:isotropic` keeps `ℓh = ℓv = 1`. On cubic
  cells the two are identical.
* **Dynamic bases** ([`dynamic_sgs_coefficient`](@ref)): no `f_m`
  multiplier on the momentum, `ν_v = ν_{base}ℓ_v` (the dynamic coefficient
  already responds to the stratification; a second factor would double
  count it). Only the scalar is corrected, through
  ``Pr_t(Ri) = Pr_t f_m/f_h``: ``κ_v = ν_{base}ℓ_v (f_h/f_m)/Pr_t``, and
  ``κ_v = 0`` where ``f_m = 0`` (beyond a critical Ri).

**Not for every closure.** A closure with its own buoyancy term (the
anisotropic minimum-dissipation model, P55-d) or its own length scale (a
prognostic-TKE closure, P56-b) already accounts for the stratification and
is not wrapped; the constructor takes `AbstractEddyViscosityClosure`s only.

`νh` and `κh` are stored in `sim.νt`, `sim.κt` (the fields the diagnostics
and the CFL estimate read), `νv` and `κv` in the closure's own fields
`c.νv`, `c.κv`; `c.N2` and `c.Ri` hold the last `N²` and `Ri_g`. The scalar
tendency is the diagonal operator `(1/Prₜ) ∂ⱼ(aⱼ ∂ⱼϕ)` with `a = ν_base ℓh`
on the x, y faces and `ν_base ℓv f_h` on the z faces (the base closure's
order of operations, so that with `N² = 0` and `ℓh = ℓv = 1` the closure is
`base` **bit for bit**: coefficients, momentum and scalar tendencies, and
every core's step). Under `step_imex!` the vertical coefficients are part of
the implicit operator. `base`'s memory and commit cadence are untouched
(`advance_sgs!`, `sgs_state`, `restore_sgs_state!` forward to it).

It composes with [`IsentropicDiffusion`](@ref):
`IsentropicDiffusion(StabilityCorrected(base, mesh), mesh)` diffuses θ with
the rotated tensor, `κh` along the isentropes and this closure's `κv`
across them.

References: Lilly, D. K. (1962), *Tellus* 14, 148; Mason, P. J. (1989),
*J. Atmos. Sci.* 46, 1492; Brown, A. R., Derbyshire, S. H. & Mason, P. J.
(1994), *Q. J. R. Meteorol. Soc.* 120, 1485; Bryan, G. H. & Fritsch, J. M.
(2002), *Mon. Wea. Rev.* 130, 2917.
"""
struct StabilityCorrected{B<:AbstractEddyViscosityClosure, S<:AbstractRiStability,
                          F<:Field{CellCenter}, Q} <: AbstractSGSClosure
    base::B
    scheme::S
    length_scales::Symbol
    ℓh::Float64
    ℓv::Float64
    νv::F
    κv::F
    av::F            # Prandtl-free vertical scalar coefficient ν_base ℓv f_h (= κv Prₜ)
    N2::F
    Ri::F
    qv::Q            # nothing, or the humidity field of θ_v
    qvh::F           # scratch: a copy of the humidity with filled halos
    S2_min::Float64
end

function StabilityCorrected(base::AbstractEddyViscosityClosure, g::RegularGrid;
                            scheme::AbstractRiStability = BrownEtAl1994(),
                            length_scales::Symbol = :anisotropic,
                            qv::Union{Nothing,Field} = nothing,
                            S2_min::Real = RI_SHEAR2_FLOOR)
    length_scales in (:anisotropic, :isotropic) ||
        throw(ArgumentError("length_scales must be :anisotropic or :isotropic, got $(repr(length_scales))"))
    S2_min > 0 || throw(ArgumentError("S2_min must be positive, got $S2_min"))
    ℓh, ℓv = length_scales === :anisotropic ? anisotropic_length_factors(g) : (1.0, 1.0)
    c() = Field(CellCenter(), g)
    return StabilityCorrected(base, scheme, length_scales, Float64(ℓh), Float64(ℓv),
                              c(), c(), c(), c(), c(), qv, c(), Float64(S2_min))
end
StabilityCorrected(base::AbstractSGSClosure, g; kwargs...) =
    throw(ArgumentError("StabilityCorrected wraps an AbstractEddyViscosityClosure " *
                        "(ν_base and a turbulent Prandtl number); $(typeof(base)) is not one. " *
                        "Closures with their own buoyancy term or length scale are not wrapped."))

sgs_fields(c::StabilityCorrected, sim) = SGSFields(sim.νt, c.νv, sim.κt, c.κv)
laminar_consistent(c::StabilityCorrected) = laminar_consistent(c.base)

# The reference θ and gravity of the core's buoyancy: the constant θ₀ of the
# Boussinesq core, or the θ̄(z) profile of a sound-proof core's ReferenceState
# (storage-indexed, as `buoyancy_frequency_squared!` takes it).
function _buoyancy_reference(sim)
    hasproperty(sim, :θ₀) && return (sim.θ₀, sim.g_accel)
    if hasproperty(sim, :ref) && hasproperty(sim.ref, :θ̄c_pad)
        return (sim.ref.θ̄c_pad, sim.ref.g)
    end
    throw(ArgumentError("StabilityCorrected: the simulation has neither θ₀ nor a reference state θ̄(z)"))
end

# The humidity of θ_v with filled halos (scratch), or nothing (dry N²).
function _stability_humidity!(c::StabilityCorrected, sim)
    src = c.qv !== nothing ? c.qv :
          (hasproperty(sim, :has_moisture) && sim.has_moisture) ? sim.qv : nothing
    src === nothing && return nothing
    copyto!(c.qvh.data, src.data)
    fill_halos!(c.qvh, _scalar_bcs(sim))
    return c.qvh
end

"""
    sgs_stratification!(c::StabilityCorrected, sim) -> (N2, Ri)

`N²` and `Ri_g` of the current state of `sim` into `c.N2` and `c.Ri`
(interior cells), as [`sgs_coefficients!`](@ref) forms them. Halos of the
velocities and of `θ` must be filled. On a core without buoyancy
(`sim.has_buoyancy == false`, e.g. a Boussinesq core with `θ₀ = 0`) θ is a
passive scalar and `N² = Ri = 0`: the closure is then `base` (with the
anisotropic length scales).
"""
function sgs_stratification!(c::StabilityCorrected, sim)
    if hasproperty(sim, :has_buoyancy) && !sim.has_buoyancy
        # no buoyancy (a Boussinesq core with θ₀ = 0): θ is a passive
        # scalar, the flow is dynamically neutral, N² = 0 and Ri = 0
        fill!(c.N2.data, 0); fill!(c.Ri.data, 0)
        return (c.N2, c.Ri)
    end
    θ₀, g = _buoyancy_reference(sim)
    q = _stability_humidity!(c, sim)
    if q === nothing
        buoyancy_frequency_squared!(c.N2, sim.θ; θ₀ = θ₀, g = g)
    else
        buoyancy_frequency_squared!(c.N2, sim.θ, q; θ₀ = θ₀, g = g)
    end
    gradient_richardson!(c.Ri, c.N2, sim.u, sim.v, sim.w; S2_min = c.S2_min)
    return (c.N2, c.Ri)
end

function sgs_coefficients!(f::SGSFields, c::StabilityCorrected, sim)
    sgs_viscosity!(f.νh, c.base, sim)                     # ν_base
    sgs_stratification!(c, sim)
    _stability_correct!(f.νh.data, f.νv.data, f.κh.data, f.κv.data, c.av.data, c.Ri.data,
                        c.scheme, c.ℓh, c.ℓv, turbulent_prandtl(c.base, sim),
                        dynamic_sgs_coefficient(c.base), mesh(f.νh))
    return f
end

# νh ← ν_base ℓh, νv ← ν_base ℓv f_m, a_v ← ν_base ℓv f_h, κ = a/Prₜ (by
# division, as the base closure forms κt), interior cells. With f = 1 and
# ℓ = 1 every product is exact, so the neutral case is the base bit for bit.
function _stability_correct!(νh, νv, κh, κv, av, Ri, scheme, ℓh, ℓv, Pr, dynamic::Bool,
                             g::RegularGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        nb = νh[i, j, k]
        r  = Ri[i, j, k]
        fm = stability_factor_m(scheme, r)
        fh = stability_factor_h(scheme, r)
        gm, gh = dynamic ? (one(fm), fm > 0 ? fh / fm : zero(fh)) : (fm, fh)
        h = nb * ℓh
        νh[i, j, k] = h
        νv[i, j, k] = nb * ℓv * gm
        a = nb * ℓv * gh
        av[i, j, k] = a
        κh[i, j, k] = h / Pr
        κv[i, j, k] = a / Pr
    end
    return nothing
end

# Three halo fills (νh, νv, a_v); κh, κv = νh/Prₜ, a_v/Prₜ on the whole array
# afterwards, as the base closure forms κt (the halo copies commute with the
# division, so this is what filling κh, κv would give).
function _fill_sgs_halos!(f::SGSFields, c::StabilityCorrected, sim)
    bcs = sim.bcs
    fill_halos!(f.νh, bcs); fill_halos!(f.νv, bcs); fill_halos!(c.av, bcs)
    Pr = turbulent_prandtl(c.base, sim)
    f.κh.data .= f.νh.data ./ Pr
    f.κv.data .= c.av.data ./ Pr
    return f
end

advance_sgs!(c::StabilityCorrected, sim, dt) = (advance_sgs!(c.base, sim, dt); c)
sgs_state(c::StabilityCorrected) = sgs_state(c.base)
restore_sgs_state!(c::StabilityCorrected, s) = (restore_sgs_state!(c.base, s); c)

# (1/Prₜ) ∂ⱼ(aⱼ ∂ⱼϕ), a = νh on the x, y faces and a_v on the z faces: the
# kernel and order of operations of the base closure's scalar tendency.
add_sgs_scalar!(tϕ, ϕ, c::StabilityCorrected, f::SGSFields, sim) =
    (_add_eddy_diffusion_scalar!(tϕ, ϕ, f.νh, c.av, 1 / turbulent_prandtl(c.base, sim),
                                 mesh(ϕ)); nothing)
