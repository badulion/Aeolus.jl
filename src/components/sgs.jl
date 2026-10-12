# The `:sgs` slot (ONE-a, #450): the subgrid-scale closure the core runs.

"""
    SGS(closure)

The provider of the `:sgs` slot: an
[`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure) the core
runs inside stage 6 (its coefficients from the substage state into
`sim.νt`/`sim.κt`, its momentum and scalar tendencies, and its memory
committed once per step, `advance_sgs!`). The closures are those of
`Operators`; this component only selects and builds one from the `[sgs]`
section ([`schema`](@ref)): `closure` (`smagorinsky`, the default,
`dynamic_smagorinsky`, `vreman`, `sigma`, `amd`), its coefficient `c_s`,
`pr_t`, the Richardson-number `stability` correction
(`StabilityCorrected`) and the form of the momentum divergence, `stress`
(`laplacian`, the default, or `symmetric`: the stress form, through which
an anisotropic vertical viscosity such as the stability correction's
reaches the momentum equation; `Operators.add_sgs_stress_divergence!`).
`closure = "none"` enables no component (no SGS terms). Its checkpoint
state is the closure's memory (`sgs_state`).

`closure = "tke"` (ONE-l, #475) is the Deardorff (1980) / Moeng (1984)
prognostic SGS TKE closure (`PrognosticSGS.TKEClosure`): over a z wall
Moeng's `surface_factor` multiplies ``c_ε`` in the first level, and with
`wall_cap` the mixing length is capped at ``κ`` × the distance to the
wall (`IBM.wall_distance`); ``e`` is seeded from the local equilibrium of
the initial flow, at least `e_min` (`initialize_tke!`: ``e = 0`` is a
fixed point). Its buoyant production at the walls takes the θ fluxes
that the domain's components apply there ([`wall_theta_flux`](@ref)),
gathered in stage 3 (`:surface_flux`, after the components that compute
them); its memory ``e`` is checkpointed. On the anelastic core its
buoyancy reference is the base state's θ̄ at the first level.
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

_sum_wall_flux(::Tuple{}, d) = (0.0, 0.0)
_sum_wall_flux(cs::Tuple, d) = (a = wall_theta_flux(first(cs), d); b = _sum_wall_flux(Base.tail(cs), d);
                                (a[1] + b[1], a[2] + b[2]))

stages(c::SGS) = c.wall_flux === nothing ? () : (Stage(:surface_flux; order = 1),)
function run_stage!(c::SGS, d, ::Val{:surface_flux}, dt)
    b, t = _sum_wall_flux(d.components, d)
    c.wall_flux[1] = b; c.wall_flux[2] = t
    return nothing
end

component_name(::Type{<:SGS}) = :sgs
section(::Type{<:SGS}) = "sgs"
slot(::Type{<:SGS}) = :sgs
enabled(::Type{<:SGS}, opts) = opts.closure !== :none

function schema(::Type{<:SGS})
    return Config.section("sgs"; doc = "The subgrid-scale closure.") do s
        Config.option!(s, "closure", Symbol; default = :smagorinsky,
                       allowed = [:none, :smagorinsky, :dynamic_smagorinsky, :vreman, :sigma, :amd, :tke],
                       doc = "none, smagorinsky (Smagorinsky-Lilly), dynamic_smagorinsky (Germano-Lilly, plane averaged), vreman, sigma (Nicoud et al. 2011), amd (anisotropic minimum dissipation), tke (Deardorff 1980 / Moeng 1984 prognostic SGS TKE)")
        Config.option!(s, "c_s", Config.Optional{Float64}; units = "1", range = (0, Inf), open_lower = true,
                       doc = "the closure's coefficient; \"none\": its default (smagorinsky 0.17, vreman 0.17, sigma 1.35, amd 0.3); not used by dynamic_smagorinsky and tke")
        Config.option!(s, "pr_t", Float64; units = "1", default = 1 / 3, range = (0, Inf), open_lower = true,
                       doc = "turbulent Prandtl number (tke: its own, 1/(1 + 2l/Δ))")
        Config.option!(s, "e_min", Float64; default = 1e-3, range = (0, Inf), units = "m^2 s^-2",
                       doc = "tke: the initial SGS TKE is the local equilibrium of the initial flow, at least e_min (e = 0 is a fixed point)")
        Config.option!(s, "surface_factor", Float64; default = 3.9, range = (0, Inf), open_lower = true, units = "1",
                       doc = "tke: Moeng's (1984) factor of c_ε in the first level above a z wall")
        Config.option!(s, "wall_cap", Bool; default = false,
                       doc = "tke: cap the mixing length at κ × the distance to the z walls")
        Config.option!(s, "stability", Symbol; default = :none,
                       allowed = [:none, :brown_et_al_1994, :lilly_cutoff, :short_tail, :long_tail],
                       doc = "Richardson-number stability correction of the eddy viscosity (StabilityCorrected); none: off")
        Config.option!(s, "stress", Symbol; default = :laplacian, allowed = [:laplacian, :symmetric],
                       doc = "the SGS momentum divergence: laplacian (ν_t ∇²u) or symmetric (the stress form ∇·(2ν_t S), " *
                             "with which the stability correction's vertical viscosity reaches the momentum)")
    end
end

"""
    build_closure(opts, mesh) -> AbstractSGSClosure

The closure of the resolved `[sgs]` options `opts` on `mesh` (the one
selection of the closure from configuration).
"""
function build_closure(s, mesh)
    cs = s.c_s; pr = s.pr_t
    base = s.closure === :smagorinsky ? StaticSmagorinsky(something(cs, 0.17); Pr_t = pr) :
           s.closure === :dynamic_smagorinsky ? DynamicSmagorinsky(mesh; Pr_t = pr) :
           s.closure === :vreman ? VremanClosure(something(cs, 0.17); Pr_t = pr) :
           s.closure === :sigma ? SigmaModel(something(cs, 1.35); Pr_t = pr) :
           s.closure === :amd ? AMDClosure(; C = something(cs, 0.3), Pr_t = pr) :
           throw(ArgumentError("build_closure: no closure :$(s.closure)"))
    s.stability === :none && return base
    scheme = s.stability === :brown_et_al_1994 ? BrownEtAl1994() :
             s.stability === :lilly_cutoff ? LillyCutoff() :
             s.stability === :short_tail ? ShortTail() : LongTail()
    return StabilityCorrected(base, mesh; scheme = scheme)
end

function build(::Type{<:SGS}, opts, domain)
    stress = option(opts, :stress, :laplacian)
    opts.closure === :tke || return SGS(build_closure(opts, grid(domain)), stress)
    return _tke_sgs(opts, domain, stress)
end

# the prognostic TKE closure on the domain (ONE-l): the first level and
# the wall distance from its z walls, its buoyancy reference, and e seeded
# from the initial flow
function _tke_sgs(opts, domain, stress)
    sim = domain.state; m = grid(domain)
    bottom = !(sim.bcs.bottom isa Periodic); top = !(sim.bcs.top isa Periodic)
    sm = SolidMask(m)
    fl = bottom ? first_level_cells(sm; bottom = true) : nothing
    wd = option(opts, :wall_cap, false) && (bottom || top) ? wall_distance(sm; bottom = bottom, top = top) : nothing
    flux = zeros(2)
    θ₀ = sim isa BoussinesqSimulation ? nothing : sim.ref.θ̄[1]
    c = TKEClosure(m; wall_distance = wd, first_level = fl, surface_factor = option(opts, :surface_factor, 3.9),
                   θ₀ = θ₀, g_accel = sim.g_accel, surface_flux = flux)
    initialize_tke!(c, sim; e_min = option(opts, :e_min, 1e-3))
    return SGS(c, stress, flux)
end

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
output_names(::Type{<:SGS}, opts) = option(opts, :closure, :smagorinsky) === :tke ? ("sgs_tke", "sgs_tke_profile") : ()
function outputs(c::SGS)
    c.closure isa TKEClosure || return OutputVariable[]
    e = c.closure.e
    return OutputVariable[
        OutputVariable("sgs_tke", "m2 s-2", "subgrid-scale turbulent kinetic energy (prognostic)", :xyz, _ -> interior_view(e)),
        OutputVariable("sgs_tke_profile", "m2 s-2", "horizontal mean of the subgrid-scale turbulent kinetic energy", :z,
                       _ -> _profile(e))]
end
