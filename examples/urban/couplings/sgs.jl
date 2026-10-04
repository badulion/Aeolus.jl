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
#   :dynamic     Operators.DynamicSmagorinsky — the Germano–Lilly dynamic
#                coefficient, plane-averaged, from the resolved flow
#                (moved to src from the example's dead
#                turbulence_addons.jl by #132).
#   :mlsgs       ML.MLSGS — a user closure `fn(νt, u, v, w, mesh)`. There
#                is no trained model: the default `fn` is the Smagorinsky
#                closure with cfg.C_s, which reproduces the default run
#                bit for bit (the hook is the test).
#
# A new SGS provider goes in its own file, couplings/sgs_<name>.jl (the
# couplings load in sorted order, after this file, which defines the slot).

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
the `sgs` of every step — the closure the cores run built in, bit for
bit. With `C_s = 0` there is no closure (no SGS terms), as before.
"""
struct SGSSmagorinsky <: AbstractCoupling end
coupling_name(::Type{SGSSmagorinsky}) = :sgs_smagorinsky
slot(::Type{SGSSmagorinsky}) = :sgs
provider(::Type{SGSSmagorinsky}) = :smagorinsky
modules_used(::Type{SGSSmagorinsky}) = (:Simulations, :Operators)
step_options(c::SGSSmagorinsky, case) =
    case.sim.C_s == 0 ? (;) : (sgs = StaticSmagorinsky(case.sim.C_s),)
smoke_check(c::SGSSmagorinsky, case) =
    ["ν_t ≥ 0" => minimum(interior_view(case.sim.νt)) >= 0,
     "ν_t > 0 where the flow is strained" => core_max_νt(case.sim) > 0]
# the built-in closure of either core (the sound-proof cores, #133, have no max_νt)
core_max_νt(sim::BoussinesqSimulation) = Aeolus.Simulations.max_νt(sim)
core_max_νt(sim) = maximum(interior_view(sim.νt))

"""
    SGSDynamic

Slot `:sgs`, provider `:dynamic`: the `DynamicSmagorinsky` closure
(`compute_dynamic_smagorinsky_νt!`) as the `sgs` of every step, on any
core. Options `averaging = :plane`, `filter_ratio = 3.0`, `cs_max = 0.5`.
`report` prints the mean `C_s` in the free stream (the levels between the
highest roof and the sponge) and in the canopy (the levels with buildings).
"""
struct SGSDynamic{W, O, C} <: AbstractCoupling
    ws::W
    opts::O
    closure::C
end
coupling_name(::Type{<:SGSDynamic}) = :sgs_dynamic
slot(::Type{<:SGSDynamic}) = :sgs
provider(::Type{<:SGSDynamic}) = :dynamic
is_default(::Type{<:SGSDynamic}) = false
modules_used(::Type{<:SGSDynamic}) = (:Operators,)
default_options(::Type{<:SGSDynamic}) = (averaging = :plane, filter_ratio = 3.0, cs_max = 0.5)
function build(::Type{<:SGSDynamic}, case, o)
    ws = DynamicSmagorinskyWorkspace(case.mesh)
    opts = (averaging = o.averaging, filter_ratio = o.filter_ratio, cs_max = o.cs_max)
    return SGSDynamic(ws, opts, DynamicSmagorinsky(ws; opts...))
end

step_options(c::SGSDynamic, case) = (sgs = c.closure,)

"""
    dynamic_cs_profile(c::SGSDynamic, case) -> (canopy, free)

Mean `C_s = √(C_s²)` of the last step over the levels with buildings and
over the free-stream levels (above the highest roof + 1 and below the
sponge).
"""
function dynamic_cs_profile(c::SGSDynamic, case)
    m = case.mesh
    kr = highest_solid_level(case)
    ksp = floor(Int, case.cfg.sponge_frac * m.Nz)
    Cs = sqrt.(c.ws.cs2)
    lvl(ks) = isempty(ks) ? NaN : mean(view(Cs, :, :, ks))
    return (canopy = lvl(1:kr), free = lvl(kr+2:ksp))
end

function report(c::SGSDynamic, case)
    p = dynamic_cs_profile(c, case)
    return @sprintf("dynamic SGS: mean C_s canopy %.3f, free stream %.3f", p.canopy, p.free)
end

smoke_check(c::SGSDynamic, case) =
    ["dynamic C_s² within the clip [0, cs_max²]" =>
         all(x -> 0 <= x <= c.opts.cs_max^2, c.ws.cs2),
     "ν_t ≥ 0 (no backscatter)" => minimum(interior_view(case.sim.νt)) >= 0]

"""
    SGSMLSGS

Slot `:sgs`, provider `:mlsgs`: an `MLSGS` closure (an
`AbstractEddyViscosityClosure`, `κt = νt/Prₜ`) as the `sgs` of every step.
Option `fn` (`fn(νt, u, v, w, mesh)`); `nothing` (default) means the
Smagorinsky closure with `cfg.C_s`, which reproduces the default run bit
for bit.
"""
struct SGSMLSGS{M} <: AbstractCoupling
    ml::M
    C_s::Float64
end
coupling_name(::Type{<:SGSMLSGS}) = :sgs_mlsgs
slot(::Type{<:SGSMLSGS}) = :sgs
provider(::Type{<:SGSMLSGS}) = :mlsgs
is_default(::Type{<:SGSMLSGS}) = false
modules_used(::Type{<:SGSMLSGS}) = (:ML,)
default_options(::Type{<:SGSMLSGS}) = (fn = nothing,)
function build(::Type{<:SGSMLSGS}, case, o)
    C_s = case.cfg.C_s
    fn = o.fn === nothing ? ((νt, u, v, w, m) -> compute_smagorinsky_νt!(νt, u, v, w, C_s)) : o.fn
    return SGSMLSGS(MLSGS(fn), C_s)
end

step_options(c::SGSMLSGS, case) = (sgs = c.ml,)

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
                c.ml.fn isa Function && interior_view(a) == interior_view(b)]
end

register!(SGSSmagorinsky)
register!(SGSDynamic)
register!(SGSMLSGS)
