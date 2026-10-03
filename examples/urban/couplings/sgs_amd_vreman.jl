# Two more providers of the `:sgs` slot (P55-d, #176), both closures of the
# Era VI contract from src/operators/sgs_amd_vreman.jl, passed as the `sgs`
# of every step like the providers in sgs.jl (which defines the slot):
#
#   :amd     Operators.AMDClosure — the anisotropic minimum-dissipation
#            model (Rozema et al. 2015) with its own scalar diffusivity
#            (Abkar, Bae & Moin 2016) and buoyancy term (Abkar & Moin
#            2017), C = 0.3, g/θ₀ of the core. No separate Richardson
#            correction: the buoyancy term is its stratification response.
#   :vreman  Operators.VremanClosure — Vreman (2004), c = 2.5 C_s² with the
#            case's C_s (cfg.C_s), κ_t = ν_t/Pr_t.
#
# Both weight each direction with its own spacing (the Berlin grid is
# 15.6 × 15.6 × 6.25 m) and vanish in laminar flow, where the Smagorinsky
# ν_t = (C_s Δ)²|S| does not. Not the default: flipping the default is the
# maintainer's call (#176).

"""
    sgs_level_means(case, ν) -> (canopy, free)

Mean of the cell-centred field `ν` over the fluid cells of the canopy levels
(those with buildings) and of the free stream (the levels above the
highest roof + 1 and below the sponge), the split of
[`dynamic_cs_profile`](@ref).
"""
function sgs_level_means(case, ν)
    m = case.mesh
    kr = highest_solid_level(case)
    ksp = floor(Int, case.cfg.sponge_frac * m.Nz)
    fluid = .!view(case.geom.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
    νi = interior_view(ν)
    lvl(ks) = (n = count(view(fluid, :, :, ks)); n == 0 ? NaN :
               sum(view(νi, :, :, ks)[view(fluid, :, :, ks)]) / n)
    return (canopy = lvl(1:kr), free = lvl(kr+2:ksp))
end

# The Smagorinsky ν_t of the current flow with the case's C_s, for comparison.
function smagorinsky_reference(case)
    sim = case.sim
    foreach(f -> fill_halos!(f, sim.bcs), (sim.u, sim.v, sim.w))
    ν = Field(CellCenter(), case.mesh)
    compute_smagorinsky_νt!(ν, sim.u, sim.v, sim.w, case.cfg.C_s)
    return ν
end

"""
    laminar_reference_νt(case, closure) -> (ν, cells)

The closure's `ν_t` on a laminar version of the current flow: `u` and `v`
replaced by their horizontal means at each level (a horizontally uniform
shear flow `u(z), v(z)`), `w = 0`, the case's θ. `cells` are the interior
cells above the inversion (`z > cfg.θ_inversion_z`), above the highest roof
+ 1 and below the sponge: the laminar free stream. AMD and Vreman must give
`ν_t = 0` there exactly (pure shear has no AMD production and a rank-one
Vreman `β`), where Smagorinsky gives `(C_s Δ)² |∂u/∂z|`.
"""
function laminar_reference_νt(case, closure)
    sim, m = case.sim, case.mesh
    u = deepcopy(sim.u); v = deepcopy(sim.v); w = deepcopy(sim.w)
    for f in (u, v)
        fi = interior_view(f)
        for k in axes(fi, 3)
            fi[:, :, k] .= mean(view(fi, :, :, k))
        end
    end
    w.data .= 0
    foreach(f -> fill_halos!(f, sim.bcs), (u, v, w))
    fill_halos!(sim.θ, sim.θ_bcs)
    ν = Field(CellCenter(), m); κ = Field(CellCenter(), m)
    if closure isa AMDClosure
        compute_amd_coefficients!(ν, κ, u, v, w, sim.θ; C = closure.C,
                                  buoyancy = sgs_buoyancy_parameter(sim))
    elseif closure isa VremanClosure
        compute_vreman_νt!(ν, u, v, w, closure.C_s)
    else
        compute_smagorinsky_νt!(ν, u, v, w, closure)
    end
    kr = highest_solid_level(case)
    ksp = floor(Int, case.cfg.sponge_frac * m.Nz)
    cells = falses(m.Nx, m.Ny, m.Nz)
    for k in kr+2:ksp
        (k - 0.5) * m.dz > case.cfg.θ_inversion_z && (cells[:, :, k] .= true)
    end
    return interior_view(ν), cells
end

function sgs_switchoff_checks(case, closure)
    sim = case.sim
    ν, cells = laminar_reference_νt(case, closure)
    νs, _ = laminar_reference_νt(case, case.cfg.C_s)
    return ["ν_t ≥ 0" => minimum(interior_view(sim.νt)) >= 0,
            "κ_t ≥ 0" => minimum(interior_view(sim.κt)) >= 0,
            "ν_t = 0 in the laminar free stream above the inversion ($(count(cells)) cells; Smagorinsky there: max $(round(maximum(νs[cells]), sigdigits = 3)) m²/s)" =>
                count(cells) > 0 && all(==(0), ν[cells]) && maximum(νs[cells]) > 0]
end

"""
    SolidMaskedSGS(closure, solid)

`closure` with every eddy coefficient zero in the solid cells `solid` (the
IBM mask, halo-padded): `νh, νv, κh, κv` and the diffusivity of any other
scalar. Every other generic of the closure contract is forwarded to
`closure` unchanged (storage, memory commit and rollback, the momentum and
scalar tendencies), so a stateful or anisotropic closure can be wrapped
too; only the coefficients in solid cells change, the fluid cells are bit
for bit those of `closure` (`smoke_check` of [`SGSAMD`](@ref)).

Why AMD needs it: its buoyancy term `(g/θ₀)(δ_k∂_k w)(δ_k∂_k θ)` is divided
by `∂_l u_m ∂_l u_m`, so `ν_t` grows like `1/|∇u|` where the velocity
gradient vanishes while θ still varies, which is the state inside a
building (IBM forces u ≈ 0, θ keeps diffusing). On `:smoke` after 24 s
that gave `ν_t` up to 19 m²/s in solid cells against a fluid maximum of
1.7 m²/s, which only shortened the adaptive step (24 s in 20 steps
instead of 12). This is a property of the published model (Abkar & Moin
2017), not of a cell being solid: a quiescent, unstably stratified fluid
region gets the same large `ν_t`, and the mask does not touch fluid cells.
"""
struct SolidMaskedSGS{C<:AbstractSGSClosure, M} <: AbstractSGSClosure
    closure::C
    solid::M
end
const _OPS = Aeolus.Operators
_OPS.sgs_fields(c::SolidMaskedSGS, sim) = _OPS.sgs_fields(c.closure, sim)
function _OPS.sgs_coefficients!(f::SGSFields, c::SolidMaskedSGS, sim)
    _OPS.sgs_coefficients!(f, c.closure, sim)
    for x in unique!(Any[f.νh, f.νv, f.κh, f.κv])     # aliased fields once
        x.data[c.solid] .= 0
    end
    return f
end
_OPS.advance_sgs!(c::SolidMaskedSGS, sim, dt) = (_OPS.advance_sgs!(c.closure, sim, dt); c)
_OPS.sgs_state(c::SolidMaskedSGS) = _OPS.sgs_state(c.closure)
_OPS.restore_sgs_state!(c::SolidMaskedSGS, s) = (_OPS.restore_sgs_state!(c.closure, s); c)
_OPS.add_sgs_momentum!(tu, tv, tw, c::SolidMaskedSGS, f::SGSFields, sim) =
    _OPS.add_sgs_momentum!(tu, tv, tw, c.closure, f, sim)
_OPS.add_sgs_scalar!(tϕ, ϕ, c::SolidMaskedSGS, f::SGSFields, sim) =
    _OPS.add_sgs_scalar!(tϕ, ϕ, c.closure, f, sim)
function _OPS.sgs_tracer_diffusivity!(κ, c::SolidMaskedSGS, f::SGSFields, sim, ϕ)
    k = _OPS.sgs_tracer_diffusivity!(κ, c.closure, f, sim, ϕ)
    k === f.κh || (k.data[c.solid] .= 0)              # f.κh is masked already
    return k
end

function sgs_compare_report(name, case)
    a = sgs_level_means(case, case.sim.νt)
    s = sgs_level_means(case, smagorinsky_reference(case))
    return @sprintf("%s SGS: mean ν_t canopy %.3e, free stream %.3e m²/s (Smagorinsky C_s = %.2f: %.3e, %.3e)",
                    name, a.canopy, a.free, case.cfg.C_s, s.canopy, s.free)
end

"""
    SGSAMD

Slot `:sgs`, provider `:amd`: the anisotropic minimum-dissipation closure
`AMDClosure(C = 0.3)` with its own κ_t and the core's buoyancy parameter, as
the `sgs` of every step, on any core, with ν_t = κ_t = 0 in the building
cells ([`SolidMaskedSGS`](@ref)). Option `C` (default 0.3, the value of
Rozema et al. 2015 for second-order central differences). `report` prints
the canopy and free-stream mean ν_t against Smagorinsky with `cfg.C_s`.
"""
struct SGSAMD{C, S} <: AbstractCoupling
    closure::C          # the AMDClosure
    sgs::S              # what the steps run: it, zero in the solid cells
end
coupling_name(::Type{<:SGSAMD}) = :sgs_amd
slot(::Type{<:SGSAMD}) = :sgs
provider(::Type{<:SGSAMD}) = :amd
is_default(::Type{<:SGSAMD}) = false
modules_used(::Type{<:SGSAMD}) = (:Operators,)
default_options(::Type{<:SGSAMD}) = (C = 0.3,)
function build(::Type{<:SGSAMD}, case, o)
    c = AMDClosure(; C = o.C)
    return SGSAMD(c, SolidMaskedSGS(c, copy(case.geom.sm.mask)))
end
step_options(c::SGSAMD, case) = (sgs = c.sgs,)
report(c::SGSAMD, case) = sgs_compare_report("AMD", case)
function smoke_check(c::SGSAMD, case)
    sim = case.sim
    foreach(f -> fill_halos!(f, sim.bcs), (sim.u, sim.v, sim.w))
    fill_halos!(sim.θ, sim.θ_bcs)
    # the wrapper against the bare closure on the current state, into
    # scratch fields (sim.νt / sim.κt are left as the last step wrote them)
    fm = SGSFields(Field(CellCenter(), case.mesh), Field(CellCenter(), case.mesh))
    fb = SGSFields(Field(CellCenter(), case.mesh), Field(CellCenter(), case.mesh))
    sgs_coefficients!(fm, c.sgs, sim); sgs_coefficients!(fb, c.closure, sim)
    fluid = .!c.sgs.solid
    in_interior = falses(size(fluid)); m = case.mesh
    in_interior[m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz] .= true
    fl = fluid .& in_interior; so = c.sgs.solid .& in_interior
    return [sgs_switchoff_checks(case, c.closure);
            "ν_t = κ_t = 0 in the solid cells" =>
                all(==(0), sim.νt.data[c.sgs.solid]) && all(==(0), sim.κt.data[c.sgs.solid]);
            "fluid cells unchanged by the solid mask (bare AMD in solids: max ν_t $(round(maximum(fb.νh.data[so]; init = 0.0), sigdigits = 3)) m²/s, fluid max $(round(maximum(fb.νh.data[fl]), sigdigits = 3)))" =>
                fm.νh.data[fl] == fb.νh.data[fl] && fm.κh.data[fl] == fb.κh.data[fl]]
end

"""
    SGSVreman

Slot `:sgs`, provider `:vreman`: `VremanClosure(C_s)` (`c = 2.5 C_s²`,
`κ_t = ν_t/Pr_t`) as the `sgs` of every step, on any core. Option `C_s`
(`nothing`, the default, takes the case's `cfg.C_s`). `report` as for
[`SGSAMD`](@ref).
"""
struct SGSVreman{C} <: AbstractCoupling
    closure::C
end
coupling_name(::Type{<:SGSVreman}) = :sgs_vreman
slot(::Type{<:SGSVreman}) = :sgs
provider(::Type{<:SGSVreman}) = :vreman
is_default(::Type{<:SGSVreman}) = false
modules_used(::Type{<:SGSVreman}) = (:Operators,)
default_options(::Type{<:SGSVreman}) = (C_s = nothing,)
build(::Type{<:SGSVreman}, case, o) =
    SGSVreman(VremanClosure(o.C_s === nothing ? case.cfg.C_s : o.C_s))
step_options(c::SGSVreman, case) = (sgs = c.closure,)
report(c::SGSVreman, case) = sgs_compare_report("Vreman", case)
smoke_check(c::SGSVreman, case) = sgs_switchoff_checks(case, c.closure)

register!(SGSAMD)
register!(SGSVreman)
