# The `:sgs` slot (ONE-a, #450): the subgrid-scale closure the core runs.

"""
    SGS(closure)

The provider of the `:sgs` slot: an
[`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure) the core
runs inside stage 6 (its coefficients from the substage state into
`sim.νt`/`sim.κt`, its momentum and scalar tendencies, and its memory
committed once per step, `advance_sgs!`). The closures are those of
`Operators`; this component only selects and builds one from the `[sgs]`
section ([`schema`](@ref)), a slot section (ONE-m, #476): `model`
(`smagorinsky`, the default, `dynamic_smagorinsky`, `vreman`, `sigma`,
`amd`, `tke`) and that closure's options only: its coefficient `c_s`,
`pr_t`, the Richardson-number `stability` correction
(`StabilityCorrected`; the eddy-viscosity closures but `amd`) and the
form of the momentum divergence, `stress` (`laplacian`, the default, or
`symmetric`: the stress form, through which an anisotropic vertical
viscosity such as the stability correction's reaches the momentum
equation; `Operators.add_sgs_stress_divergence!`). `model = "none"`
enables no component (no SGS terms). Its checkpoint
state is the closure's memory (`sgs_state`).

`model = "tke"` (ONE-l, #475) is the Deardorff (1980) / Moeng (1984)
prognostic SGS TKE closure (`PrognosticSGS.TKEClosure`): over a z wall
Moeng's `surface_factor` multiplies ``c_ε`` in the first level, and with
`wall_cap` the mixing length is capped at ``κ`` × the distance to the
wall (`IBM.wall_distance`); ``e`` is seeded from the local equilibrium of
the initial flow, at least `e_min` (`initialize_tke!`: ``e = 0`` is a
fixed point). Its buoyant production at the walls takes the θ fluxes
that the domain's components apply there ([`wall_theta_flux`](@ref)),
gathered in stage 3 (`:surface_flux`, after the components that compute
them); its memory ``e`` is checkpointed. On the anelastic core its
buoyancy reference is the base state's θ̄ at the first level. On a
`StretchedGrid` the wall distance is the exact height of each level
centre above the bottom face (and below the top face): `IBM.wall_distance`
assumes uniform levels (NEST-p1, #481). `wall_cap_top = false` caps at the
bottom only (a free-slip lid under a sponge is no rough wall: the urban
nest parent). With `surface_value`, ``e`` in the first level is the
surface value of the u* of the domain's `:surface` component
([`surface_friction_velocity`](@ref); `PrognosticSGS.tke_surface_value`),
copied in stage 3 after that component has computed it.
"""
struct SGS{C<:AbstractSGSClosure, F} <: AbstractComponent
    closure::C
    stress::Symbol
    wall_flux::F            # nothing, or the TKE closure's (bottom, top) θ fluxes (ONE-l)
end
SGS(closure::AbstractSGSClosure) = SGS(closure, :laplacian, nothing)
SGS(closure::AbstractSGSClosure, stress::Symbol) = SGS(closure, stress, nothing)

"""
    wall_theta_flux(c, domain) -> (bottom, top)

The kinematic θ fluxes (K m s⁻¹, upward positive; plane means) that
component `c` applies through the bottom and the top wall of `domain` in
the current step; `(0.0, 0.0)` (the default) for one that applies none.
The TKE closure of [`SGS`](@ref) takes their buoyant production from the
sum over the domain's components.
"""
wall_theta_flux(::AbstractComponent, domain) = (0.0, 0.0)

"""
    surface_friction_velocity(c, domain) -> Union{Nothing,Matrix{Float64}}

The friction velocity u* (m s⁻¹) per column of the surface that component
`c` computes in the current step (stage 3, `:surface_flux`), or `nothing`
(the default). The TKE closure of [`SGS`](@ref) with `surface_value` sets
its first-level ``e`` from that of the domain's `:surface` component.
"""
surface_friction_velocity(::AbstractComponent, domain) = nothing

_sum_wall_flux(::Tuple{}, d) = (0.0, 0.0)
_sum_wall_flux(cs::Tuple, d) = (a = wall_theta_flux(first(cs), d); b = _sum_wall_flux(Base.tail(cs), d);
                                (a[1] + b[1], a[2] + b[2]))

stages(c::SGS) = c.wall_flux === nothing ? () : (Stage(:surface_flux; order = 1),)
function run_stage!(c::SGS, d, ::Val{:surface_flux}, dt)
    b, t = _sum_wall_flux(d.components, d)
    c.wall_flux[1] = b; c.wall_flux[2] = t
    _copy_surface_ustar!(c.closure, d)
    return nothing
end

# the first-level e of the TKE closure from the :surface component's u* (surface_value)
_copy_surface_ustar!(::AbstractSGSClosure, d) = nothing
function _copy_surface_ustar!(c::TKEClosure, d)
    c.surface === nothing && return nothing
    s = slot_component(d, :surface)
    us = s === nothing ? nothing : surface_friction_velocity(s, d)
    us === nothing &&
        throw(ArgumentError("[sgs] surface_value: the domain $(d.id) has no :surface component that computes u*"))
    copyto!(c.surface.u_star, us)
    return nothing
end

component_name(::Type{<:SGS}) = :sgs
section(::Type{<:SGS}) = "sgs"
slot(::Type{<:SGS}) = :sgs
enabled(::Type{<:SGS}, opts) = opts.model !== :none

"""
The default coefficient `c_s` of each closure that has one (Smagorinsky-Lilly
0.17; Vreman 0.17; σ model 1.35, Nicoud et al. 2011; AMD 0.3, Rozema et al.
2015): the `[sgs]` schema's defaults and `build_closure`'s for options
without a `c_s`.
"""
const SGS_DEFAULT_C_S = (smagorinsky = 0.17, vreman = 0.17, sigma = 1.35, amd = 0.3)

function schema(::Type{<:SGS})
    return Config.slot_section("sgs"; key = "model", doc = "The subgrid-scale closure.") do s
        stab(p) = Config.option!(p, "stability", Symbol; default = :none,
                                 allowed = [:none, :brown_et_al_1994, :lilly_cutoff, :short_tail, :long_tail],
                                 doc = "Richardson-number stability correction of the eddy viscosity (StabilityCorrected); none: off")
        stress(p) = Config.option!(p, "stress", Symbol; default = :laplacian, allowed = [:laplacian, :symmetric],
                                   doc = "the SGS momentum divergence: laplacian (ν_t ∇²u) or symmetric (the stress form ∇·(2ν_t S), " *
                                         "with which the stability correction's vertical viscosity reaches the momentum)")
        prt(p) = Config.option!(p, "pr_t", Float64; units = "1", default = 1 / 3, range = (0, Inf), open_lower = true,
                                doc = "turbulent Prandtl number")
        cs(p, m) = Config.option!(p, "c_s", Float64; units = "1", default = SGS_DEFAULT_C_S[m], range = (0, Inf), open_lower = true,
                                  doc = "the closure's coefficient")
        Config.provider!(s, "none"; doc = "No SGS terms.")
        Config.provider!(s, "smagorinsky"; default = [:model], doc = "Smagorinsky-Lilly (StaticSmagorinsky).") do p
            cs(p, :smagorinsky); prt(p); stab(p); stress(p)
        end
        Config.provider!(s, "dynamic_smagorinsky"; doc = "Germano-Lilly dynamic Smagorinsky, plane averaged (DynamicSmagorinsky).") do p
            prt(p); stab(p); stress(p)
        end
        Config.provider!(s, "vreman"; doc = "Vreman (2004) (VremanClosure).") do p
            cs(p, :vreman); prt(p); stab(p); stress(p)
        end
        Config.provider!(s, "sigma"; doc = "σ model, Nicoud et al. (2011) (SigmaModel).") do p
            cs(p, :sigma); prt(p); stab(p); stress(p)
        end
        Config.provider!(s, "amd"; doc = "Anisotropic minimum dissipation (AMDClosure).") do p
            cs(p, :amd); prt(p); stress(p)
        end
        Config.provider!(s, "tke"; doc = "Deardorff (1980) / Moeng (1984) prognostic SGS TKE (PrognosticSGS.TKEClosure; " *
                                         "its own Prandtl number 1/(1 + 2l/Δ)).") do p
            Config.option!(p, "e_min", Float64; default = 1e-3, range = (0, Inf), units = "m^2 s^-2",
                           doc = "the initial SGS TKE is the local equilibrium of the initial flow, at least e_min (e = 0 is a fixed point)")
            Config.option!(p, "surface_factor", Float64; default = 3.9, range = (0, Inf), open_lower = true, units = "1",
                           doc = "Moeng's (1984) factor of c_ε in the first level above a z wall")
            Config.option!(p, "wall_cap", Bool; default = false,
                           doc = "cap the mixing length at κ × the distance to the z walls")
            Config.option!(p, "wall_cap_top", Bool; default = true,
                           doc = "with wall_cap: the top wall caps too; false: the bottom only")
            Config.option!(p, "surface_value", Bool; default = false,
                           doc = "e in the first level from the u* of the domain's :surface component (its surface value)")
            stress(p)
        end
    end
end

"""
    build_closure(opts, mesh) -> AbstractSGSClosure

The closure of the resolved `[sgs]` options `opts` on `mesh` (the one
selection of the closure from configuration).
"""
function build_closure(s, mesh)
    m = s.model
    c = option(s, :c_s, nothing)
    cs = c === nothing ? get(SGS_DEFAULT_C_S, m, nothing) : c; pr = option(s, :pr_t, 1 / 3)
    base = m === :smagorinsky ? StaticSmagorinsky(cs; Pr_t = pr) :
           m === :dynamic_smagorinsky ? DynamicSmagorinsky(mesh; Pr_t = pr) :
           m === :vreman ? VremanClosure(cs; Pr_t = pr) :
           m === :sigma ? SigmaModel(cs; Pr_t = pr) :
           m === :amd ? AMDClosure(; C = cs, Pr_t = pr) :
           throw(ArgumentError("build_closure: no closure :$m"))
    st = option(s, :stability, :none)
    st === :none && return base
    scheme = st === :brown_et_al_1994 ? BrownEtAl1994() :
             st === :lilly_cutoff ? LillyCutoff() :
             st === :short_tail ? ShortTail() : LongTail()
    return StabilityCorrected(base, mesh; scheme = scheme)
end

function build(::Type{<:SGS}, opts, domain)
    stress = option(opts, :stress, :laplacian)
    opts.model === :tke || return SGS(build_closure(opts, grid(domain)), stress)
    return _tke_sgs(opts, domain, stress)
end

# the prognostic TKE closure on the domain (ONE-l): the first level and
# the wall distance from its z walls, its buoyancy reference, and e seeded
# from the initial flow
function _tke_sgs(opts, domain, stress)
    sim = domain.state; m = grid(domain)
    bottom = !(sim.bcs.bottom isa Periodic); top = !(sim.bcs.top isa Periodic)
    sm = m isa StretchedGrid ? nothing : SolidMask(m)     # no obstacles on a stretched grid
    fl = bottom ? _first_level(m, sm) : nothing
    top = top && option(opts, :wall_cap_top, true)
    wd = option(opts, :wall_cap, false) && (bottom || top) ? _wall_distance(m, sm, bottom, top) : nothing
    flux = zeros(2)
    θ₀ = sim isa BoussinesqSimulation ? nothing : sim.ref.θ̄[1]
    sv = option(opts, :surface_value, false) ? (u_star = fill(NaN, m.Nx, m.Ny), k = ones(Int, m.Nx, m.Ny)) : nothing
    c = TKEClosure(m; wall_distance = wd, first_level = fl, surface_factor = option(opts, :surface_factor, 3.9),
                   θ₀ = θ₀, g_accel = sim.g_accel, surface_flux = flux, surface = sv)
    initialize_tke!(c, sim; e_min = option(opts, :e_min, 1e-3))
    return SGS(c, stress, flux)
end

# the distance (m) of every cell centre to the nearest z wall: IBM's on a
# regular grid; on a stretched grid (no obstacles on a model domain) the
# exact heights of the level centres above the bottom face and below the top
_wall_distance(m, sm, bottom, top) = wall_distance(sm; bottom = bottom, top = top)
function _wall_distance(m::StretchedGrid, ::Nothing, bottom, top)
    Hz = m.Hz
    zb, zt = m.zf[Hz+1], m.zf[Hz+m.Nz+1]
    d(k) = min(bottom ? m.zc[Hz+k] - zb : Inf, top ? zt - m.zc[Hz+k] : Inf)
    return [d(k) for _ in 1:m.Nx, _ in 1:m.Ny, k in 1:m.Nz]
end

# the cells of the first level above the bottom wall
_first_level(m, sm) = first_level_cells(sm; bottom = true)
_first_level(m::StretchedGrid, ::Nothing) = (f = zeros(Bool, m.Nx, m.Ny, m.Nz); f[:, :, 1] .= true; f)

# the closure's memory, under its type's name (a restart into another closure is refused)
checkpoint_state(c::SGS) = (closure = string(nameof(typeof(c.closure))), state = checkpoint_state(c.closure))
function restore_checkpoint_state!(c::SGS, s)
    n = string(nameof(typeof(c.closure)))
    s.closure == n ||
        throw(ArgumentError("restart!: the checkpoint's SGS closure is $(s.closure), restoring into a $n"))
    restore_checkpoint_state!(c.closure, s.state)
    return c
end

# the prognostic closure's SGS TKE in the output catalogue (ONE-l)
output_names(::Type{<:SGS}, opts) = option(opts, :model, :smagorinsky) === :tke ? ("sgs_tke", "sgs_tke_profile") : ()
function outputs(c::SGS)
    c.closure isa TKEClosure || return OutputVariable[]
    e = c.closure.e
    return OutputVariable[
        OutputVariable("sgs_tke", "m2 s-2", "subgrid-scale turbulent kinetic energy (prognostic)", :xyz, _ -> interior_view(e)),
        OutputVariable("sgs_tke_profile", "m2 s-2", "horizontal mean of the subgrid-scale turbulent kinetic energy", :z,
                       _ -> _profile(e))]
end
