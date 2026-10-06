# The subgrid-scale (SGS) closure, in the `:sgs` slot. Every provider
# hands the dynamics an `AbstractSGSClosure` (the Era VI contract,
# src/operators/sgs_closures.jl, #167) through the `sgs` keyword of the
# step (`step_options`); every core and time stepper takes it — step!,
# step_imex!, step_adaptive! (which also rolls the closure's memory back on
# a rejected step), and the anelastic / pseudo-incompressible step!. The
# closure's coefficients fill `sim.νt` / `sim.κt` and its tendencies are
# added where the built-in Smagorinsky adds them; its memory is committed
# once per accepted step.
#
#   :tke         (default, couplings/sgs_tke.jl) PrognosticSGS.TKEClosure —
#                the Deardorff/Moeng prognostic SGS TKE closure (#182).
#   :reynolds_stress (option, couplings/sgs_reynolds_stress.jl)
#                PrognosticSGS.ReynoldsStressClosure — the Deardorff (1973)
#                differential Reynolds-stress closure with LRR-IP (#188);
#                its momentum flux is its own stress (no `stress` option).
#   :smagorinsky Operators.StaticSmagorinsky(cfg.C_s) — the
#                Smagorinsky–Lilly closure, C_s = cfg.C_s (0.1); bit for
#                bit the built-in closure of the cores (the default before
#                #182).
#   :dynamic     the Germano–Lilly dynamic coefficient from the resolved
#                flow, obstacle aware (the buildings' solid mask, #173).
#                Default averaging :lagrangian, scale_dependent = true —
#                Operators.LagrangianDynamicSmagorinsky (Meneveau, Lund &
#                Cabot 1996: averaged along pathlines, no homogeneous
#                planes needed; scale dependent after Bou-Zeid, Meneveau
#                & Parlange 2005, LASD, #186; `scale_dependent = false`
#                is the scale-invariant LASI); `averaging = :plane` / `:local` run
#                Operators.DynamicSmagorinsky (moved to src from the
#                example's dead turbulence_addons.jl by #132).
#   :mlsgs       ML.MLSGS — a user closure `fn(νt, u, v, w, mesh)`. There
#                is no trained model: the default `fn` is the Smagorinsky
#                closure with cfg.C_s, which reproduces the default run
#                bit for bit (the hook is the test).
#   :sigma       Operators.SigmaModel — the σ-model of Nicoud et al. (2011),
#                ν_t from the singular values of the resolved gradient; in
#                couplings/sgs_sigma.jl (#184).
#
# Every eddy-viscosity provider (:smagorinsky, :dynamic, :mlsgs, :amd,
# :vreman) runs its closure with the stress-form momentum divergence,
# `sgs_stress = :symmetric` (∂ⱼ(2νᵢⱼSᵢⱼ), P55-b #175; option `stress`,
# `:laplacian` for the pre-Era-VI ∂ⱼ(νₜ∂ⱼuᵢ)), through the one helper
# `sgs_step_options`. The :tke provider (couplings/sgs_tke.jl) keeps the
# Laplacian form: its shear production is built as the exact discrete dual
# of that operator (PrognosticSGS.shear_production!), which is what makes
# the resolved-KE sink and the TKE source agree to round-off; with the
# stress form the two would no longer balance. The cores' default stays `:laplacian`; the flagship
# opts in.
#
# The eddy-viscosity closures of :smagorinsky, :dynamic and :mlsgs are
# wrapped, by the one builder `sgs_provider_closure`, in
# Operators.StabilityCorrected (P55-e, #187), on by default for these
# providers: νh = ν ℓh, νv = ν ℓv f_m(Ri_g), κh = ν ℓh/Prₜ,
# κv = ν ℓv f_h(Ri_g)/Prₜ with the TurbulenceStability functions
# (`CaseConfig.sgs_stability`: `:brown94` default, Brown et al. 1994;
# `:lilly`, `:long_tail`, `:short_tail`, `:most`; `:none` = the uncorrected
# closure, the pre-P55-e run bit for bit) and the anisotropic length scales
# Δh = √(dx dy), Δv = dz (dx/dz = 2.5: ℓh = 1.842, ℓv = 0.2947). Ri_g uses
# the θ_v of the qv tracer (the moist buoyancy's θ_v) and the core's own
# reference θ (θ₀, or θ̄(z) on the sound-proof cores). The dynamic provider
# gets no f_m on the momentum (only Prₜ(Ri)). The :tke closure (its own
# stability length) and :amd (its own buoyancy term) already account for
# the stratification and are not wrapped; :vreman and :sigma keep their
# validated isotropic form (νh = νv).
#
# The scalar (θ) SGS flux goes through the closure contract's diagonal
# operator, add_eddy_diffusion_scalar!(tϕ, ϕ, κh, κv) (P55-c, #181), which
# for these isotropic closures is the κh = κv case, bit for bit the
# pre-P55-c run. Option `scalar_tensor = :isentropic` (every provider, via
# `sgs_scalar_closure`) wraps the closure in Operators.IsentropicDiffusion:
# the rotated Redi tensor of θ in the triad form of Griffies et al. (1998),
# κh along the isentropes and κv = `diapycnal`·κh across them (the
# closure's own κv = κh when `diapycnal = nothing`), tapered above
# `slope_max` and isotropic where N² ≤ 0. It is option-only: the flagship is
# a daytime convective case where the isentropic direction is ill-defined
# (N² ≤ 0) over most of the domain. It wraps the stability-corrected
# closure, so its across-isentrope κv is the corrected one (κv < κh in
# stable air) unless `diapycnal` is given. It lives in Operators (exempt),
# so no coverage list changes.
#
# A new SGS provider goes in its own file, couplings/sgs_<name>.jl (the
# couplings load in sorted order, after this file, which defines the slot).

"""
    sgs_step_options(closure, stress) -> NamedTuple

The step keywords of every `:sgs` provider: the closure as `sgs` and the
form of its momentum divergence as `sgs_stress` (`:symmetric` by default,
the stress form; `:laplacian` the Laplacian form).
"""
sgs_step_options(closure, stress::Symbol) =
    (sgs = closure, sgs_stress = Aeolus.Operators._check_sgs_stress(stress))

"The option every `:sgs` provider takes: the form of the SGS momentum divergence."
const SGS_STRESS_DEFAULT = (stress = :symmetric,)

"""
The scalar-flux options every `:sgs` provider takes (P55-c, #181):
`scalar_tensor` (`:diagonal`, the contract's diagonal operator, or
`:isentropic`, the rotated tensor of `Operators.IsentropicDiffusion`),
and, for `:isentropic`, `diapycnal` (κv/κh across the isentropes;
`nothing` = the closure's own κv) and `slope_max` (the taper slope).
"""
const SGS_SCALAR_DEFAULT = (scalar_tensor = :diagonal, diapycnal = nothing, slope_max = 1.0)

"""
    sgs_stability_scheme(name::Symbol) -> AbstractRiStability

The TurbulenceStability family of `CaseConfig.sgs_stability` (`:brown94`,
`:lilly`, `:long_tail`, `:short_tail`, `:most`; the names of the
`turbulence_diag` option, with `:brown94` for its `:brown`).
"""
sgs_stability_scheme(name::Symbol) =
    turbulence_stability_family(name === :brown94 ? :brown : name)

"""
    sgs_stability_closure(closure, case) -> AbstractSGSClosure

`closure` wrapped in `Operators.StabilityCorrected` (P55-e, #187) with the
scheme of `case.cfg.sgs_stability`, the anisotropic length scales of the
case grid and the θ_v of the `qv` tracer — or `closure` itself for
`sgs_stability = :none`.
"""
function sgs_stability_closure(closure, case)
    s = case.cfg.sgs_stability
    s === :none && return closure
    return StabilityCorrected(closure, case.mesh; scheme = sgs_stability_scheme(s),
                              length_scales = :anisotropic, qv = tracer(case, :qv))
end

"""
    sgs_provider_closure(base, o, case) -> AbstractSGSClosure

The closure every `:sgs` provider hands the dynamics: the eddy-viscosity
closure `base`, stability-corrected ([`sgs_stability_closure`](@ref)),
then with the scalar options `o` ([`sgs_scalar_closure`](@ref)).
"""
sgs_provider_closure(base, o, case) = sgs_scalar_closure(sgs_stability_closure(base, case), o, case)

"""
    stability_closure(c) -> Union{Nothing, StabilityCorrected}

The `StabilityCorrected` inside a provider's closure (itself, or the inner
closure of an `IsentropicDiffusion`), or `nothing`.
"""
stability_closure(c::StabilityCorrected) = c
stability_closure(c::IsentropicDiffusion) = stability_closure(c.inner)
stability_closure(c) = nothing

"""
    sgs_base_closure(c) -> AbstractSGSClosure

The provider's own closure inside the wrappers (`IsentropicDiffusion`,
`StabilityCorrected`): e.g. the `LagrangianDynamicSmagorinsky` of `:dynamic`.
"""
sgs_base_closure(c::IsentropicDiffusion) = sgs_base_closure(c.inner)
sgs_base_closure(c::StabilityCorrected) = sgs_base_closure(c.base)
sgs_base_closure(c) = c

"""
    stability_stats(c::StabilityCorrected, case) -> NamedTuple

From the closure's last coefficients, over the fluid cells: the fraction
with `Ri_g ≥ Ri_c` (`supercritical`; Ri_c of the scheme, 0.25 for Brown
and Lilly, `NaN` for a scheme without one), the means of νh, νv, κh, κv
and the mean κv/κh where κh > 0.
"""
function stability_stats(c::StabilityCorrected, case; levels = nothing)
    m = case.mesh
    sm = case.geom.sm.mask
    ks = levels === nothing ? (1:m.Nz) : levels
    fluid = [!sm[m.Hx+i, m.Hy+j, m.Hz+k] for i in 1:m.Nx, j in 1:m.Ny, k in ks]
    v(f) = view(interior_view(f), :, :, ks)[fluid]
    Ri = v(c.Ri); νh = v(case.sim.νt); νv = v(c.νv); κh = v(case.sim.κt); κv = v(c.κv)
    Ric = hasproperty(c.scheme, :Ri_c) ? c.scheme.Ri_c : NaN
    pos = κh .> 0
    return (supercritical = isnan(Ric) ? NaN : count(>=(Ric), Ri) / length(Ri),
            νh = mean(νh), νv = mean(νv), κh = mean(κh), κv = mean(κv),
            κv_over_κh = any(pos) ? mean(κv[pos] ./ κh[pos]) : NaN)
end

# the stability-correction checks every provider adds to its smoke_check
function sgs_stability_checks(closure, case)
    c = stability_closure(closure)
    c === nothing && return Pair{String,Bool}[]
    Ri = interior_view(c.Ri)
    νh = interior_view(case.sim.νt); νv = interior_view(c.νv)
    κh = interior_view(case.sim.κt); κv = interior_view(c.κv)
    r = c.ℓv / c.ℓh                      # νv/νh of a neutral cell
    tol(x) = 1e-12 * max(abs(x), eps())
    st = Ri .>= 0; un = Ri .<= 0
    checks = ["stability-corrected SGS: κv ≥ 0 and νv ≥ 0" => all(>=(0), κv) && all(>=(0), νv),
              "stability-corrected SGS: κv ≤ κh where Ri ≥ 0" =>
                  all(i -> κv[i] <= κh[i] + tol(κh[i]), findall(st)),
              "stability-corrected SGS: νv ≤ νh ℓv/ℓh where Ri ≥ 0 (damped)" =>
                  all(i -> νv[i] <= νh[i] * r + tol(νh[i]), findall(st))]
    if !dynamic_sgs_coefficient(c.base)
        push!(checks, "stability-corrected SGS: νv ≥ νh ℓv/ℓh where Ri ≤ 0 (convective enhancement)" =>
                          all(i -> νv[i] >= νh[i] * r - tol(νh[i]), findall(un)))
    end
    if hasproperty(c.scheme, :Ri_c) && c.scheme isa Union{BrownEtAl1994,LillyCutoff}
        push!(checks, "stability-corrected SGS: κv = 0 where Ri ≥ Ri_c" =>
                          all(i -> κv[i] == 0, findall(>=(c.scheme.Ri_c), Ri)))
    end
    return checks
end

function sgs_stability_report(closure, case)
    c = stability_closure(closure)
    c === nothing && return nothing
    s = stability_stats(c, case)
    return @sprintf("stability-corrected SGS (%s): %.1f %% of fluid cells Ri ≥ Ri_c, mean νv %.3g νh %.3g m²/s, mean κv/κh %.3f",
                    case.cfg.sgs_stability, 100 * s.supercritical, s.νv, s.νh, s.κv_over_κh)
end

"""
    sgs_scalar_closure(closure, o, case) -> AbstractSGSClosure

The closure a provider hands the dynamics for the scalar options `o`
(`SGS_SCALAR_DEFAULT`): `closure` itself for `scalar_tensor =
:diagonal`, or `IsentropicDiffusion(closure, case.mesh; slope_max,
diapycnal)` for `:isentropic`.
"""
function sgs_scalar_closure(closure, o, case)
    o.scalar_tensor === :diagonal && return closure
    o.scalar_tensor === :isentropic ||
        throw(ArgumentError("scalar_tensor must be :diagonal or :isentropic, got $(repr(o.scalar_tensor))"))
    return IsentropicDiffusion(closure, case.mesh; slope_max = o.slope_max, diapycnal = o.diapycnal)
end

"""
    isentropic_fraction(c::IsentropicDiffusion) -> Float64

Fraction of the interior cells whose isentropic tensor is rotated
(`K13² + K23² > 0`: stably stratified with a horizontal θ gradient, and not
fully tapered), from the closure's last coefficients.
"""
function isentropic_fraction(c::IsentropicDiffusion)
    a = interior_view(c.K.K13); b = interior_view(c.K.K23)
    return count(i -> a[i]^2 + b[i]^2 > 0, eachindex(a)) / length(a)
end

# the scalar-tensor checks every provider adds to its smoke_check: every
# cell's K is positive semi-definite (the down-gradient condition)
function sgs_scalar_checks(closure, case)
    closure isa IsentropicDiffusion || return Pair{String,Bool}[]
    K = closure.K
    v(f) = interior_view(f)
    psd = all(i -> (k11 = v(K.K11)[i]; k22 = v(K.K22)[i]; k33 = v(K.K33)[i];
                    k12 = v(K.K12)[i]; k13 = v(K.K13)[i]; k23 = v(K.K23)[i];
                    tol = 1e-12 * max(k11, k22, k33, eps());
                    k11 >= -tol && k22 >= -tol && k33 >= -tol &&
                    k11 * k22 - k12^2 >= -tol^2 && k11 * k33 - k13^2 >= -tol^2 &&
                    k22 * k33 - k23^2 >= -tol^2), eachindex(v(K.K11)))
    return ["isentropic scalar tensor K positive semi-definite in every cell" => psd]
end

"""
    mean_sgs_dissipation(case, closure = nothing) -> W kg⁻¹

Domain-mean resolved-to-subgrid kinetic-energy transfer of the stress-form
closure, `Σ 2νᵢⱼSᵢⱼSᵢⱼ ΔV / V` (`sgs_stress_dissipation` with the current
`νh = sim.νt` and the vertical `νv` of a stability-corrected closure, or
`νv = νh`). Fills the velocity halos.
"""
function mean_sgs_dissipation(case, closure = nothing)
    sim = case.sim
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    m = case.mesh
    c = stability_closure(closure)
    νv = c === nothing ? sim.νt : c.νv
    return sgs_stress_dissipation(sim.u, sim.v, sim.w, sim.νt, νv) / (m.Lx * m.Ly * m.Lz)
end

# the stress-form checks every provider adds to its smoke_check
sgs_stress_checks(stress, case, closure = nothing) =
    stress === :symmetric ?
        ["stress-form SGS dissipation Σ2νᵢⱼSᵢⱼSᵢⱼ ≥ 0" => mean_sgs_dissipation(case, closure) >= 0] :
        Pair{String,Bool}[]

# -------------------------------------------------------------------------
# A-priori metrics at the test-filter scale (P56-e, #192), shared by the
# prognostic providers (:tke, :reynolds_stress)
# -------------------------------------------------------------------------

"""
    sgs_canopy_cells(case) -> (cells, solid)

The interior fluid cells `(i, j, k)` of the canopy levels (more than 5 % of
the level solid) and the interior solid mask.
"""
function sgs_canopy_cells(case)
    m = case.mesh
    solid = view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
    kr = something(findlast(k -> count(view(solid, :, :, k)) > 0.05 * m.Nx * m.Ny, 1:m.Nz), 0)
    return [(i, j, k) for k in 1:kr for j in 1:m.Ny for i in 1:m.Nx if !solid[i, j, k]], solid
end

# the resolved strain rate at cell centres (Operators.filtered_strain of the
# centred velocity)
function _sgs_centre_strain(case)
    sim, m = case.sim, case.mesh
    U = map((sim.u, sim.v, sim.w)) do f
        F = Field(CellCenter(), m)
        interior_view(F) .= Aeolus.Operators._to_center(f, sim.bcs)   # BC-aware (#262)
        F
    end
    return filtered_strain(U...)
end

"""
    sgs_test_filter_metrics(case, τ_model::SymmetricTensorField, e) -> NamedTuple

A-priori metrics of a closure's subfilter stress against the resolved
stress at the test-filter scale (Germano et al. 1991),
``L_{ij} = \\widehat{\\bar u_i \\bar u_j} − \\hat{\\bar u}_i \\hat{\\bar u}_j``
(`subfilter_stress!` of the LES velocity with the 3-cell masked box test
filter, `box_filter!`, P56-c), over the fluid cells of the canopy levels:

* `k_test` — ``½⟨L_{kk}⟩``, the resolved energy between the grid and the
  test-filter scale; `e` — the closure's mean subfilter energy;
  `ratio = e/k_test` (scale similarity: a fixed ratio in an inertial range);
* `ρ` — the correlation of the deviatoric model stress with the
  deviatoric ``L_{ij}``, averaged over the three off-diagonal components
  (the scale-similarity correlation of Liu, Meneveau & Katz 1994);
* `II_L`, `II_model` — median Lumley invariants II (`lumley_invariants`)
  of ``L`` and of the model stress (cells with a positive trace), and
  `outside`, the fraction of cells whose model stress has II > 2/3, i.e.
  lies outside the Lumley triangle (an eddy-viscosity stress with a small
  ``e`` is not realizable; the Reynolds-stress closure's never is).
"""
function sgs_test_filter_metrics(case, τm, e)
    sim, m = case.sim, case.mesh
    cells, _ = sgs_canopy_cells(case)
    isempty(cells) && return (k_test = NaN, e = NaN, ratio = NaN, ρ = NaN, II_L = NaN, II_model = NaN, outside = NaN)
    mask = case.geom.sm.mask
    bcs = case_scalar_bcs(case.cfg)
    L = SymmetricTensorField(m)
    subfilter_stress!(L, sim.u, sim.v, sim.w, (o, f) -> box_filter!(o, f, 3; mask = mask); bcs = bcs,
                      velocity_bcs = sim.bcs)
    pick(a) = [a[i, j, k] for (i, j, k) in cells]
    trL = pick(L.xx) .+ pick(L.yy) .+ pick(L.zz)
    k_test = mean(trL) / 2
    em = mean(pick(e))
    ρs = map(((L.xy, τm.xy), (L.xz, τm.xz), (L.yz, τm.yz))) do (a, b)
        x = pick(a); y = pick(b)
        x = x .- mean(x); y = y .- mean(y)
        d = sqrt(sum(abs2, x) * sum(abs2, y))
        d > 0 ? sum(x .* y) / d : NaN
    end
    IIL, _ = lumley_invariants(L); IIm, _ = lumley_invariants(τm)
    med(a) = (v = sort!(filter(isfinite, pick(a))); isempty(v) ? NaN : v[(length(v) + 1) ÷ 2])
    vm = filter(isfinite, pick(IIm))
    return (k_test = k_test, e = em, ratio = em / k_test, ρ = mean(ρs),
            II_L = med(IIL), II_model = med(IIm),
            outside = isempty(vm) ? NaN : count(>(2 / 3), vm) / length(vm))
end

sgs_apriori_line(a) =
    @sprintf("a priori at the test filter (3-cell box, canopy): k_test %.4f m²/s², e/k_test %.2f, ρ(τ, L) %.2f, median II: L %.3f, model %.3f (%.0f %% of cells II > 2/3)",
             a.k_test, a.ratio, a.ρ, a.II_L, a.II_model, 100a.outside)

"Highest k (1-based interior level) with a solid cell, 0 without buildings."
function highest_solid_level(case)
    m = case.mesh
    mask = view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
    ks = [k for k in 1:m.Nz if any(view(mask, :, :, k))]
    return isempty(ks) ? 0 : maximum(ks)
end

"""
    SGSSmagorinsky

Slot `:sgs`, provider `:smagorinsky` (the default before #182; now
`:tke`, couplings/sgs_tke.jl): `StaticSmagorinsky(C_s)`
with the core's `C_s` (`cfg.C_s`) and turbulent Prandtl number,
stability-corrected (`StabilityCorrected`, `cfg.sgs_stability`, P55-e;
with `sgs_stability = :none` the closure the cores run built in, bit for
bit), passed as the `sgs` of every step with the stress-form momentum
divergence (option `stress = :symmetric`; `:laplacian` is the cores'
built-in form). With `C_s = 0` there is no closure (no SGS terms), as
before.
"""
struct SGSSmagorinsky{C} <: AbstractCoupling
    stress::Symbol
    closure::C                 # nothing when C_s = 0
end
coupling_name(::Type{<:SGSSmagorinsky}) = :sgs_smagorinsky
slot(::Type{<:SGSSmagorinsky}) = :sgs
provider(::Type{<:SGSSmagorinsky}) = :smagorinsky
is_default(::Type{<:SGSSmagorinsky}) = false
modules_used(::Type{<:SGSSmagorinsky}) = (:Simulations, :Operators, :TurbulenceStability)
default_options(::Type{<:SGSSmagorinsky}) = (; SGS_STRESS_DEFAULT..., SGS_SCALAR_DEFAULT...)
build(::Type{<:SGSSmagorinsky}, case, o) =
    SGSSmagorinsky(o.stress, case.sim.C_s == 0 ? nothing :
                             sgs_provider_closure(StaticSmagorinsky(case.sim.C_s), o, case))
step_options(c::SGSSmagorinsky, case) =
    c.closure === nothing ? (;) : sgs_step_options(c.closure, c.stress)
# (each on the :smagorinsky provider: the slot default is :tke since #182,
# which takes none of these options)
smoke_variants(::Type{<:SGSSmagorinsky}) =
    ((label = "SGS Laplacian form", physics = Dict(:sgs => :smagorinsky),
      options = Dict(:sgs_smagorinsky => (stress = :laplacian,))),
     (label = "SGS isentropic scalar tensor", physics = Dict(:sgs => :smagorinsky),
      options = Dict(:sgs_smagorinsky => (scalar_tensor = :isentropic, diapycnal = 0.1))),
     (label = "SGS isentropic scalar tensor, corrected κv", physics = Dict(:sgs => :smagorinsky),
      options = Dict(:sgs_smagorinsky => (scalar_tensor = :isentropic,))),
     (label = "SGS stability correction off", physics = Dict(:sgs => :smagorinsky),
      sgs_stability = :none),
     (label = "SGS stability correction, Lilly cutoff", physics = Dict(:sgs => :smagorinsky),
      sgs_stability = :lilly))
smoke_check(c::SGSSmagorinsky, case) =
    ["ν_t ≥ 0" => minimum(interior_view(case.sim.νt)) >= 0,
     "ν_t > 0 where the flow is strained" => core_max_νt(case.sim) > 0,
     sgs_stress_checks(c.stress, case, c.closure)..., sgs_scalar_checks(c.closure, case)...,
     sgs_stability_checks(c.closure, case)...]
function report(c::SGSSmagorinsky, case)
    lines = String[]
    c.closure isa IsentropicDiffusion &&
        push!(lines, @sprintf("isentropic SGS scalar tensor: %.1f %% of the cells rotated",
                              100 * isentropic_fraction(c.closure)))
    r = sgs_stability_report(c.closure, case)
    r === nothing || push!(lines, r)
    return isempty(lines) ? nothing : join(lines, "; ")
end
# the built-in closure of either core (the sound-proof cores, #133, have no max_νt)
core_max_νt(sim::BoussinesqSimulation) = Aeolus.Simulations.max_νt(sim)
core_max_νt(sim) = maximum(interior_view(sim.νt))

"""
    SGSDynamic

Slot `:sgs`, provider `:dynamic`: the dynamic Smagorinsky coefficient as
the `sgs` of every step, on any core, with the buildings' solid mask
(`case.geom.sm.mask`): `ν_t = 0` in buildings, the test filter and the
averages run over air cells only, and nothing stored inside a building
reaches the coefficient. Options:

* `averaging = :lagrangian` (default) — `LagrangianDynamicSmagorinsky`,
  the Germano identity averaged along pathlines (Meneveau, Lund & Cabot
  1996); its memory is committed once per accepted step and rolled back
  with a rejected adaptive step;
* `scale_dependent = true` (default; Lagrangian only) — the
  scale-dependent model (LASD) of Bou-Zeid, Meneveau & Parlange (2005):
  a second test filter (5-cell box) and the power law between the two
  test-filter scales give ``C_s^2`` at the grid scale and ``β``, the
  coefficient ratio per doubling of the filter width, ``β ∈ [1/8, 8]``;
  `false` is the scale-invariant model (LASI);
* `cadence = 4` (Lagrangian only) — the Germano identities every 4th
  accepted step, the memories transported along the pathlines at every
  step (common Lagrangian-dynamic practice). On the 64 × 64 × 32 flagship
  mesh LASD with `cadence = 1` costs about 28 % of a step, with 4 about
  15 %; the memory time ``T`` (10th percentile to median) is 50-100 s
  in the smoke case, so 4 steps of 2 s
  are well inside it. `cadence = 1` is the exact model;
* `averaging = :plane` / `:local` — `DynamicSmagorinsky` (horizontal
  average over the air cells of each level / pointwise);
* `filter_ratio = 3.0`, `cs_max = 0.5`;
* `stress = :symmetric` — the form of the momentum divergence, as for
  `SGSSmagorinsky`.
* `scalar_tensor = :diagonal` (`:isentropic`, with `diapycnal`,
  `slope_max`) — the scalar flux, as for `SGSSmagorinsky`.

`report` prints the mean `C_s` (and, with LASD, the geometric-mean `β`)
over the air cells in the canopy (the levels with buildings) and in the
free stream (between the highest roof and the sponge).
"""
struct SGSDynamic{W, O, C} <: AbstractCoupling
    ws::W
    opts::O
    closure::C
    stress::Symbol
end
coupling_name(::Type{<:SGSDynamic}) = :sgs_dynamic
slot(::Type{<:SGSDynamic}) = :sgs
provider(::Type{<:SGSDynamic}) = :dynamic
is_default(::Type{<:SGSDynamic}) = false
modules_used(::Type{<:SGSDynamic}) = (:Operators, :TurbulenceStability)
default_options(::Type{<:SGSDynamic}) =
    (averaging = :lagrangian, scale_dependent = true, cadence = 4, filter_ratio = 3.0,
     cs_max = 0.5, SGS_STRESS_DEFAULT..., SGS_SCALAR_DEFAULT...)
function build(::Type{<:SGSDynamic}, case, o)
    sd = o.averaging === :lagrangian && o.scale_dependent
    opts = (averaging = o.averaging, scale_dependent = sd, cadence = o.cadence,
            filter_ratio = o.filter_ratio, cs_max = o.cs_max)
    mask = case.geom.sm.mask
    inner = if o.averaging === :lagrangian
        LagrangianDynamicSmagorinsky(case.mesh; filter_ratio = o.filter_ratio,
                                     cs_max = o.cs_max, mask, scale_dependent = sd,
                                     cadence = o.cadence)
    else
        DynamicSmagorinsky(DynamicSmagorinskyWorkspace(case.mesh); averaging = o.averaging,
                           filter_ratio = o.filter_ratio, cs_max = o.cs_max, mask)
    end
    return SGSDynamic(inner.workspace, opts, sgs_provider_closure(inner, o, case), o.stress)
end

step_options(c::SGSDynamic, case) = sgs_step_options(c.closure, c.stress)

"Interior view of the buildings' solid mask."
function interior_solid(case)
    m = case.mesh
    return view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
end

"""
    dynamic_cs_profile(c::SGSDynamic, case) -> (canopy, free, β_canopy, β_free)

Mean `C_s = √(C_s²)` of the last coefficient evaluation over the air cells
of the levels with buildings and of the free-stream levels (above the
highest roof + 1 and below the sponge). Building cells are excluded. With
the scale-dependent model also the geometric mean of `β` over the same
cells, ``2^{⟨\\log_2 β⟩}`` (the mean power-law exponent; the arithmetic
mean of a noisy ratio is biased high), `NaN` otherwise.
"""
function dynamic_cs_profile(c::SGSDynamic, case)
    m = case.mesh
    kr = highest_solid_level(case)
    ksp = floor(Int, case.cfg.sponge_frac * m.Nz)
    Cs = sqrt.(c.ws.cs2)
    air = .!interior_solid(case)
    function lvl(A, ks; f = identity, inv = identity)
        isempty(ks) && return NaN
        a = view(air, :, :, ks)
        return inv(sum(f, view(A, :, :, ks)[a]) / count(a))
    end
    inner = sgs_base_closure(c.closure)
    sd = inner isa LagrangianDynamicSmagorinsky && inner.scale_dependent
    βl(ks) = sd ? lvl(inner.beta, ks; f = log2, inv = exp2) : NaN
    return (canopy = lvl(Cs, 1:kr), free = lvl(Cs, kr+2:ksp),
            β_canopy = βl(1:kr), β_free = βl(kr+2:ksp))
end

function report(c::SGSDynamic, case)
    p = dynamic_cs_profile(c, case)
    name = c.opts.scale_dependent ? "lagrangian, scale dependent" : String(c.opts.averaging)
    s = @sprintf("dynamic SGS (%s): mean C_s canopy %.3f, free stream %.3f (air cells)",
                 name, p.canopy, p.free)
    c.opts.scale_dependent &&
        (s *= @sprintf("; β canopy %.3f, free stream %.3f", p.β_canopy, p.β_free))
    r = sgs_stability_report(c.closure, case)
    return r === nothing ? s : string(s, "; ", r)
end

smoke_variants(::Type{<:SGSDynamic}) =
    ((label = "plane averaging", physics = Dict(:sgs => :dynamic),
      options = Dict(:sgs_dynamic => (averaging = :plane,))),
     (label = "scale-invariant Lagrangian (LASI)", physics = Dict(:sgs => :dynamic),
      options = Dict(:sgs_dynamic => (scale_dependent = false,))))

function smoke_check(c::SGSDynamic, case)
    solid = interior_solid(case)
    checks = ["dynamic C_s² within the clip [0, cs_max²]" =>
                  all(x -> 0 <= x <= c.opts.cs_max^2, c.ws.cs2),
              "ν_t ≥ 0 (no backscatter)" => minimum(interior_view(case.sim.νt)) >= 0,
              "ν_t = 0 in the buildings" => all(==(0), interior_view(case.sim.νt)[solid])]
    inner = sgs_base_closure(c.closure)
    if inner isa LagrangianDynamicSmagorinsky
        push!(checks, "Lagrangian memory I_LM ≥ 0, I_MM ≥ 0" =>
                          minimum(inner.I_LM) >= 0 && minimum(inner.I_MM) >= 0)
        push!(checks, "Lagrangian memory initialised" => inner.initialised[])
        if inner.scale_dependent
            β = inner.beta; air = .!solid
            push!(checks, "LASD β ∈ [β_min, β_max] in the air" =>
                              all(b -> inner.beta_min <= b <= inner.beta_max, β[air]))
            push!(checks, "LASD memory I_QN ≥ 0, I_NN ≥ 0, 0 in the buildings" =>
                              minimum(inner.I_QN) >= 0 && minimum(inner.I_NN) >= 0 &&
                              all(==(0), inner.I_QN[solid]))
        end
    end
    append!(checks, sgs_stress_checks(c.stress, case, c.closure))
    append!(checks, sgs_scalar_checks(c.closure, case))
    append!(checks, sgs_stability_checks(c.closure, case))
    return checks
end

"""
    SGSMLSGS

Slot `:sgs`, provider `:mlsgs`: an `MLSGS` closure (an
`AbstractEddyViscosityClosure`, `κt = νt/Prₜ`) as the `sgs` of every step.
Option `fn` (`fn(νt, u, v, w, mesh)`); `nothing` (default) means the
Smagorinsky closure with `cfg.C_s`, which reproduces the default run bit
for bit. Option `stress = :symmetric` (as for `SGSSmagorinsky`).
"""
struct SGSMLSGS{M, C} <: AbstractCoupling
    ml::M
    C_s::Float64
    stress::Symbol
    closure::C                 # ml, stability-corrected and/or wrapped for scalar_tensor = :isentropic
end
coupling_name(::Type{<:SGSMLSGS}) = :sgs_mlsgs
slot(::Type{<:SGSMLSGS}) = :sgs
provider(::Type{<:SGSMLSGS}) = :mlsgs
is_default(::Type{<:SGSMLSGS}) = false
modules_used(::Type{<:SGSMLSGS}) = (:ML, :TurbulenceStability)
default_options(::Type{<:SGSMLSGS}) = (fn = nothing, SGS_STRESS_DEFAULT..., SGS_SCALAR_DEFAULT...)
function build(::Type{<:SGSMLSGS}, case, o)
    C_s = case.cfg.C_s
    fn = o.fn === nothing ? ((νt, u, v, w, m) -> compute_smagorinsky_νt!(νt, u, v, w, C_s)) : o.fn
    ml = MLSGS(fn)
    return SGSMLSGS(ml, C_s, o.stress, sgs_provider_closure(ml, o, case))
end

step_options(c::SGSMLSGS, case) = sgs_step_options(c.closure, c.stress)

function smoke_check(c::SGSMLSGS, case)
    sim = case.sim
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    a = Field(CellCenter(), case.mesh)
    b = Field(CellCenter(), case.mesh)
    c.ml.fn(a, sim.u, sim.v, sim.w, case.mesh)
    compute_smagorinsky_νt!(b, sim.u, sim.v, sim.w, c.C_s)
    return ["MLSGS ν_t ≥ 0" => minimum(interior_view(sim.νt)) >= 0,
            "default MLSGS(Smagorinsky) ν_t == the built-in ν_t (bitwise)" =>
                c.ml.fn isa Function && interior_view(a) == interior_view(b),
            sgs_stress_checks(c.stress, case, c.closure)..., sgs_scalar_checks(c.closure, case)...,
            sgs_stability_checks(c.closure, case)...]
end

register!(SGSSmagorinsky)
register!(SGSDynamic)
register!(SGSMLSGS)
