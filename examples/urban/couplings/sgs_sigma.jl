# The σ-model SGS closure (P59-b, #184) as a provider of the `:sgs` slot
# (defined in sgs.jl, which loads before this file).
#
#   :sigma   Operators.SigmaModel(C_σ) — ν_t = (C_σΔ)² σ₃(σ₁−σ₂)(σ₂−σ₃)/σ₁²
#            from the singular values of the resolved velocity gradient
#            (Nicoud et al. 2011), κ_t = ν_t/Pr_t with the core's Pr_t.
#            Its ν_t vanishes in laminar shear, solid-body rotation and
#            any 2-D flow, and falls off as z³ at a no-slip wall, where
#            Smagorinsky stays on.
#
# Like every eddy-viscosity provider of the slot it runs the stress-form
# momentum divergence, `sgs_stress = :symmetric` (P55-b #175, through
# `sgs_step_options`; option `stress = :laplacian` for the Laplacian form).
#
# Off by default: `:tke` is the default provider (#182; flipping it is the
# maintainer's call, #184). The smoke matrix runs this provider
# automatically (one run per non-default provider).

"""
    SGSSigma

Slot `:sgs`, provider `:sigma`: the σ-model closure `SigmaModel(C_σ)`
(option `C_σ`, default 1.35, Nicoud et al. 2011) passed as the `sgs` of
every step, on any core and time stepper, with the stress-form momentum
divergence (option `stress = :symmetric`, as for `SGSSmagorinsky`; #175). `diagnostics` declares the mean ν_t in
the canopy (levels with buildings) and in the free stream above the sponge
start, next to what the Smagorinsky closure (`cfg.C_s`) gives on the same
flow.
"""
struct SGSSigma{C} <: AbstractCoupling
    closure::C
    stress::Symbol      # the form of the momentum divergence (#175)
end
coupling_name(::Type{<:SGSSigma}) = :sgs_sigma
slot(::Type{<:SGSSigma}) = :sgs
provider(::Type{<:SGSSigma}) = :sigma
is_default(::Type{<:SGSSigma}) = false
modules_used(::Type{<:SGSSigma}) = (:Operators,)
default_options(::Type{<:SGSSigma}) = (C_σ = 1.35, SGS_STRESS_DEFAULT...)
build(::Type{<:SGSSigma}, case, o) = SGSSigma(SigmaModel(o.C_σ), o.stress)

step_options(c::SGSSigma, case) = sgs_step_options(c.closure, c.stress)

"""
    sigma_νt_profile(c::SGSSigma, case) -> NamedTuple

Mean ν_t (m² s⁻¹) of the current flow over the canopy levels (those with a
solid cell) and over the levels at and above the sponge start
(`cfg.sponge_frac·Lz`), for the σ-model and for Smagorinsky with
`cfg.C_s`, and the extreme values of the σ-model ν_t above the sponge
start. Both are evaluated with the pure `compute_sgs!` / the Smagorinsky
kernel: nothing is committed and `sim.νt` is left as the step left it.
"""
function sigma_νt_profile(c::SGSSigma, case)
    sim, m = case.sim, case.mesh
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    νσ = Field(CellCenter(), m); νs = Field(CellCenter(), m)
    compute_sigma_νt!(νσ, sim.u, sim.v, sim.w, c.closure.C_σ)
    compute_smagorinsky_νt!(νs, sim.u, sim.v, sim.w, case.cfg.C_s)
    kr = highest_solid_level(case)
    ksp = floor(Int, case.cfg.sponge_frac * m.Nz) + 1      # first level at/above the sponge start
    lvl(f, ks) = isempty(ks) ? NaN : mean(view(interior_view(f), :, :, ks))
    top = view(interior_view(νσ), :, :, ksp:m.Nz)
    return (canopy_σ = lvl(νσ, 1:kr), canopy_smag = lvl(νs, 1:kr),
            free_σ = lvl(νσ, ksp:m.Nz), free_smag = lvl(νs, ksp:m.Nz),
            free_σ_max = maximum(top), min_σ = minimum(interior_view(νσ)), ksp = ksp)
end

"""
    diagnostics(c::SGSSigma, case)

The σ-model's mean ν_t in the canopy and above the sponge start against
Smagorinsky with `cfg.C_s` on the same flow ([`sigma_νt_profile`](@ref)):
`nu_t_canopy`, `nu_t_canopy_smagorinsky`, `nu_t_free`,
`nu_t_free_smagorinsky` (m² s⁻¹).
"""
function diagnostics(c::SGSSigma, case)
    p = per_state(() -> sigma_νt_profile(c, case), case)
    return Diagnostic[
        Diagnostic("nu_t_canopy", "m2 s-1", () -> p().canopy_σ),
        Diagnostic("nu_t_canopy_smagorinsky", "m2 s-1", () -> p().canopy_smag),
        Diagnostic("nu_t_free", "m2 s-1", () -> p().free_σ; description = "above the sponge start"),
        Diagnostic("nu_t_free_smagorinsky", "m2 s-1", () -> p().free_smag;
                   description = "above the sponge start")]
end

function smoke_check(c::SGSSigma, case)
    p = sigma_νt_profile(c, case)
    # laminar consistency on the laminar version of the current flow (level
    # means of u, v; w = 0): ν_t = 0 exactly where Smagorinsky is on, the
    # check AMD and Vreman share (sgs_switchoff_checks, sgs_amd_vreman.jl)
    return ["σ-model ν_t ≥ 0" => p.min_σ >= 0,
            "σ-model ν_t > 0 in the canopy" => p.canopy_σ > 0,
            sgs_switchoff_checks(case, c.closure)...,
            sgs_stress_checks(c.stress, case)...,
            # relaminarisation criterion of P59-e (#195): negligible against
            # the background ν in the free stream above the sponge start
            # (with the geostrophic sponge of #171 that free stream is shear
            # free, so Smagorinsky's ν_t there is no longer the larger one)
            "σ-model ν_t/ν < 1e-2 above the sponge start" => p.free_σ < 1e-2 * case.cfg.ν]
end

register!(SGSSigma)
