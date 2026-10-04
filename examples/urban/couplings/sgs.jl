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
#   :smagorinsky (default) Operators.StaticSmagorinsky(cfg.C_s) — the
#                Smagorinsky–Lilly closure, C_s = cfg.C_s (0.1); bit for
#                bit the built-in closure of the cores.
#   :dynamic     the Germano–Lilly dynamic coefficient from the resolved
#                flow, obstacle aware (the buildings' solid mask, #173).
#                Default averaging :lagrangian —
#                Operators.LagrangianDynamicSmagorinsky (Meneveau, Lund &
#                Cabot 1996: averaged along pathlines, no homogeneous
#                planes needed); `averaging = :plane` / `:local` run
#                Operators.DynamicSmagorinsky (moved to src from the
#                example's dead turbulence_addons.jl by #132).
#   :mlsgs       ML.MLSGS — a user closure `fn(νt, u, v, w, mesh)`. There
#                is no trained model: the default `fn` is the Smagorinsky
#                closure with cfg.C_s, which reproduces the default run
#                bit for bit (the hook is the test).
#
# Every provider runs its closure with the stress-form momentum divergence,
# `sgs_stress = :symmetric` (∂ⱼ(2νᵢⱼSᵢⱼ), P55-b #175; option `stress`,
# `:laplacian` for the pre-Era-VI ∂ⱼ(νₜ∂ⱼuᵢ)), through the one helper
# `sgs_step_options`. The cores' default stays `:laplacian`; the flagship
# opts in. νh = νv (the closures are isotropic) until P55-e (#187).
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
    mean_sgs_dissipation(case) -> W kg⁻¹

Domain-mean resolved-to-subgrid kinetic-energy transfer of the stress-form
closure, `Σ 2νᵢⱼSᵢⱼSᵢⱼ ΔV / V` (`sgs_stress_dissipation` with the current
`ν_t`, `νh = νv`). Fills the velocity halos.
"""
function mean_sgs_dissipation(case)
    sim = case.sim
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    m = case.mesh
    return sgs_stress_dissipation(sim.u, sim.v, sim.w, sim.νt) / (m.Lx * m.Ly * m.Lz)
end

# the stress-form checks every provider adds to its smoke_check
sgs_stress_checks(stress, case) =
    stress === :symmetric ?
        ["stress-form SGS dissipation Σ2νSᵢⱼSᵢⱼ ≥ 0" => mean_sgs_dissipation(case) >= 0] :
        Pair{String,Bool}[]

"Highest k (1-based interior level) with a solid cell, 0 without buildings."
function highest_solid_level(case)
    m = case.mesh
    mask = view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
    ks = [k for k in 1:m.Nz if any(view(mask, :, :, k))]
    return isempty(ks) ? 0 : maximum(ks)
end

"""
    SGSSmagorinsky

Slot `:sgs`, provider `:smagorinsky` (default): `StaticSmagorinsky(C_s)`
with the core's `C_s` (`cfg.C_s`) and turbulent Prandtl number, passed as
the `sgs` of every step — the coefficients of the closure the cores run
built in, bit for bit — with the stress-form momentum divergence (option
`stress = :symmetric`; `:laplacian` is the cores' built-in form). With
`C_s = 0` there is no closure (no SGS terms), as before.
"""
struct SGSSmagorinsky <: AbstractCoupling
    stress::Symbol
end
coupling_name(::Type{SGSSmagorinsky}) = :sgs_smagorinsky
slot(::Type{SGSSmagorinsky}) = :sgs
provider(::Type{SGSSmagorinsky}) = :smagorinsky
modules_used(::Type{SGSSmagorinsky}) = (:Simulations, :Operators)
default_options(::Type{SGSSmagorinsky}) = SGS_STRESS_DEFAULT
build(::Type{SGSSmagorinsky}, case, o) = SGSSmagorinsky(o.stress)
step_options(c::SGSSmagorinsky, case) =
    case.sim.C_s == 0 ? (;) : sgs_step_options(StaticSmagorinsky(case.sim.C_s), c.stress)
smoke_variants(::Type{SGSSmagorinsky}) =
    ((label = "SGS Laplacian form", options = Dict(:sgs_smagorinsky => (stress = :laplacian,))),)
smoke_check(c::SGSSmagorinsky, case) =
    ["ν_t ≥ 0" => minimum(interior_view(case.sim.νt)) >= 0,
     "ν_t > 0 where the flow is strained" => core_max_νt(case.sim) > 0,
     sgs_stress_checks(c.stress, case)...]
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
* `averaging = :plane` / `:local` — `DynamicSmagorinsky` (horizontal
  average over the air cells of each level / pointwise);
* `filter_ratio = 3.0`, `cs_max = 0.5`;
* `stress = :symmetric` — the form of the momentum divergence, as for
  `SGSSmagorinsky`.

`report` prints the mean `C_s` over the air cells in the canopy (the
levels with buildings) and in the free stream (between the highest roof
and the sponge).
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
modules_used(::Type{<:SGSDynamic}) = (:Operators,)
default_options(::Type{<:SGSDynamic}) =
    (averaging = :lagrangian, filter_ratio = 3.0, cs_max = 0.5, SGS_STRESS_DEFAULT...)
function build(::Type{<:SGSDynamic}, case, o)
    opts = (averaging = o.averaging, filter_ratio = o.filter_ratio, cs_max = o.cs_max)
    mask = case.geom.sm.mask
    closure = if o.averaging === :lagrangian
        LagrangianDynamicSmagorinsky(case.mesh; filter_ratio = o.filter_ratio,
                                     cs_max = o.cs_max, mask)
    else
        DynamicSmagorinsky(DynamicSmagorinskyWorkspace(case.mesh); averaging = o.averaging,
                           filter_ratio = o.filter_ratio, cs_max = o.cs_max, mask)
    end
    return SGSDynamic(closure.workspace, opts, closure, o.stress)
end

step_options(c::SGSDynamic, case) = sgs_step_options(c.closure, c.stress)

"Interior view of the buildings' solid mask."
function interior_solid(case)
    m = case.mesh
    return view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
end

"""
    dynamic_cs_profile(c::SGSDynamic, case) -> (canopy, free)

Mean `C_s = √(C_s²)` of the last coefficient evaluation over the air cells
of the levels with buildings and of the free-stream levels (above the
highest roof + 1 and below the sponge). Building cells are excluded.
"""
function dynamic_cs_profile(c::SGSDynamic, case)
    m = case.mesh
    kr = highest_solid_level(case)
    ksp = floor(Int, case.cfg.sponge_frac * m.Nz)
    Cs = sqrt.(c.ws.cs2)
    air = .!interior_solid(case)
    function lvl(ks)
        isempty(ks) && return NaN
        a = view(air, :, :, ks)
        return sum(view(Cs, :, :, ks)[a]) / count(a)
    end
    return (canopy = lvl(1:kr), free = lvl(kr+2:ksp))
end

function report(c::SGSDynamic, case)
    p = dynamic_cs_profile(c, case)
    return @sprintf("dynamic SGS (%s): mean C_s canopy %.3f, free stream %.3f (air cells)",
                    c.opts.averaging, p.canopy, p.free)
end

smoke_variants(::Type{<:SGSDynamic}) =
    ((label = "plane averaging", physics = Dict(:sgs => :dynamic),
      options = Dict(:sgs_dynamic => (averaging = :plane,))),)

function smoke_check(c::SGSDynamic, case)
    solid = interior_solid(case)
    checks = ["dynamic C_s² within the clip [0, cs_max²]" =>
                  all(x -> 0 <= x <= c.opts.cs_max^2, c.ws.cs2),
              "ν_t ≥ 0 (no backscatter)" => minimum(interior_view(case.sim.νt)) >= 0,
              "ν_t = 0 in the buildings" => all(==(0), interior_view(case.sim.νt)[solid])]
    if c.closure isa LagrangianDynamicSmagorinsky
        push!(checks, "Lagrangian memory I_LM ≥ 0, I_MM ≥ 0" =>
                          minimum(c.closure.I_LM) >= 0 && minimum(c.closure.I_MM) >= 0)
        push!(checks, "Lagrangian memory initialised" => c.closure.initialised[])
    end
    append!(checks, sgs_stress_checks(c.stress, case))
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
struct SGSMLSGS{M} <: AbstractCoupling
    ml::M
    C_s::Float64
    stress::Symbol
end
coupling_name(::Type{<:SGSMLSGS}) = :sgs_mlsgs
slot(::Type{<:SGSMLSGS}) = :sgs
provider(::Type{<:SGSMLSGS}) = :mlsgs
is_default(::Type{<:SGSMLSGS}) = false
modules_used(::Type{<:SGSMLSGS}) = (:ML,)
default_options(::Type{<:SGSMLSGS}) = (fn = nothing, SGS_STRESS_DEFAULT...)
function build(::Type{<:SGSMLSGS}, case, o)
    C_s = case.cfg.C_s
    fn = o.fn === nothing ? ((νt, u, v, w, m) -> compute_smagorinsky_νt!(νt, u, v, w, C_s)) : o.fn
    return SGSMLSGS(MLSGS(fn), C_s, o.stress)
end

step_options(c::SGSMLSGS, case) = sgs_step_options(c.ml, c.stress)

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
            sgs_stress_checks(c.stress, case)...]
end

register!(SGSSmagorinsky)
register!(SGSDynamic)
register!(SGSMLSGS)
