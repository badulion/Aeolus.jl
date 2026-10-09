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
"""
struct SGS{C<:AbstractSGSClosure} <: AbstractComponent
    closure::C
    stress::Symbol
end
SGS(closure::AbstractSGSClosure) = SGS(closure, :laplacian)

component_name(::Type{<:SGS}) = :sgs
section(::Type{<:SGS}) = "sgs"
slot(::Type{<:SGS}) = :sgs
enabled(::Type{<:SGS}, opts) = opts.closure !== :none

function schema(::Type{<:SGS})
    return Config.section("sgs"; doc = "The subgrid-scale closure.") do s
        Config.option!(s, "closure", Symbol; default = :smagorinsky,
                       allowed = [:none, :smagorinsky, :dynamic_smagorinsky, :vreman, :sigma, :amd],
                       doc = "none, smagorinsky (Smagorinsky-Lilly), dynamic_smagorinsky (Germano-Lilly, plane averaged), vreman, sigma (Nicoud et al. 2011), amd (anisotropic minimum dissipation)")
        Config.option!(s, "c_s", Config.Optional{Float64}; units = "1", range = (0, Inf), open_lower = true,
                       doc = "the closure's coefficient; \"none\": its default (smagorinsky 0.17, vreman 0.17, sigma 1.35, amd 0.3); not used by dynamic_smagorinsky")
        Config.option!(s, "pr_t", Float64; units = "1", default = 1 / 3, range = (0, Inf), open_lower = true,
                       doc = "turbulent Prandtl number")
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

build(::Type{<:SGS}, opts, domain) = SGS(build_closure(opts, grid(domain)), opts.stress)

# the closure's memory, under its type's name (a restart into another closure is refused)
checkpoint_state(c::SGS) = (closure = string(nameof(typeof(c.closure))), state = checkpoint_state(c.closure))
function restore_checkpoint_state!(c::SGS, s)
    n = string(nameof(typeof(c.closure)))
    s.closure == n ||
        throw(ArgumentError("restart!: the checkpoint's SGS closure is $(s.closure), restoring into a $n"))
    restore_checkpoint_state!(c.closure, s.state)
    return c
end
