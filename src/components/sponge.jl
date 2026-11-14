# The Rayleigh sponge below the top (ONE-a, #450), retargeted at the
# geostrophic wind or the plane mean (ONE-g, #456: the sponge of the former
# `sbl` run type, P55-f #191, P58-d #194).

"""
    Sponge(α_max, z_start, target, kind, ut, vt)

Rayleigh damping `−α(z)(u − u_target)` of `u, v, w` above `z_start`, with
`α(z) = α_max sin²(π/2 (z − z_start)/(z_top − z_start))`
(`Simulations.add_velocity_sponge!` → `Operators.add_sponge!`; Klemp &
Lilly 1978), in stage 7 (`:body_force`). `[sponge] alpha_max` (s⁻¹) and
`z_start` (m) configure it; it is enabled when `α_max ≠ 0` and `z_start`
is set (the model's configuration requires it below the top). Its
`target` (`kind`):

  * `"rest"` (the default) — zero: the target of a closed top
    (`Simulations.sponge_target`);
  * `"geostrophic"` — `u, v` toward the geostrophic wind `U_g(z, t)`,
    `V_g(z, t)` of the domain's [`LargeScale`](@ref) forcing (its step
    mean per level), `w` toward zero: the free atmosphere above a
    boundary layer in geostrophic balance;
  * `"mean"` — `u, v` toward their plane mean per level (only the
    departures are damped), `w` toward zero: for a mean wind above the
    boundary layer that evolves under advective tendencies (GABLS3).

The retargeted profiles are level vectors (`ut`, `vt`) refreshed in every
(sub)step.
"""
struct Sponge{G<:NamedTuple} <: AbstractComponent
    α_max::Float64
    z_start::Float64
    target::G
    kind::Symbol
    ut::Vector{Float64}
    vt::Vector{Float64}
end
Sponge(α_max, z_start, target) = Sponge(α_max, z_start, target, :rest, Float64[], Float64[])

component_name(::Type{<:Sponge}) = :sponge
section(::Type{<:Sponge}) = "sponge"
schema(::Type{<:Sponge}) =
    Config.section("sponge"; doc = "Rayleigh damping of the velocity below the top (Klemp & Lilly 1978).") do s
        Config.option!(s, "alpha_max", Float64; default = 0.0, range = (0, Inf), units = "s^-1",
                       doc = "Rayleigh-damping rate at the top; 0: no sponge")
        Config.option!(s, "z_start", Config.Optional{Float64}; units = "m",
                       doc = "height where the sponge starts, below the top; \"none\": no sponge")
        Config.option!(s, "target", Symbol; default = :rest, allowed = [:rest, :geostrophic, :mean],
                       doc = "what u, v are damped toward: rest (zero), geostrophic (the geostrophic wind of " *
                             "[large_scale_forcing]) or mean (their plane mean: only the departures are damped); w: zero")
    end
enabled(::Type{<:Sponge}, opts) = opts.alpha_max != 0 && opts.z_start !== nothing
function build(::Type{<:Sponge}, opts, domain)
    s = domain.state
    z_top = grid(domain).z0 + grid(domain).Lz
    opts.z_start < z_top ||
        throw(ArgumentError("Sponge: z_start = $(opts.z_start) m is not below the top ($z_top m)"))
    α, z0 = opts.alpha_max, Float64(opts.z_start)
    kind = option(opts, :target, :rest)
    kind === :rest && return Sponge(α, z0, Simulations._sponge_target(nothing, s.mesh, s.bcs, true))
    nz = grid(domain).Nz
    ut = zeros(nz); vt = zeros(nz)
    return Sponge(α, z0, (u = ut, v = vt, w = 0), kind, ut, vt)
end
stages(::Sponge) = (Stage(:body_force),)
# the velocity sponge: the budgets measure it on the momentum only (#539)
budget_writes(::Sponge, ::Val{:body_force}) = (:momentum_x, :momentum_y)

# The large-scale forcing component of the domain (nothing without one).
_large_scale(::Tuple{}) = nothing
_large_scale(cs::Tuple) = first(cs) isa LargeScale ? first(cs) : _large_scale(Base.tail(cs))

# the level targets of a retargeted sponge for the (sub)step from t0 over dt
function _retarget!(c::Sponge, d, dt)
    s = d.state; m = s.mesh
    if c.kind === :mean
        c.ut .= _profile(s.u); c.vt .= _profile(s.v)
    else
        ls = _large_scale(d.components)
        t0 = ls.t0[]
        z = damping_coordinates(m, CellCenter())
        for k in 1:m.Nz
            c.ut[k] = forcing_mean(ls.U_g, z[k], t0, t0 + dt)
            c.vt[k] = forcing_mean(ls.V_g, z[k], t0, t0 + dt)
        end
    end
    return c
end

function run_stage!(c::Sponge, d, ::Val{:body_force}, dt)
    c.kind === :rest || _retarget!(c, d, dt)
    add_velocity_sponge!(d.state, c.α_max, c.z_start, c.target)
    return nothing
end

# --- the fused sweep (#533) ---------------------------------------------------
# The damping of `add_velocity_sponge!` as a sweep term
# (`FiniteVolume.sponge_term`); a retargeted sponge refreshes its level
# targets from the stage-start state before the sweep, as `run_stage!` does
# before its pass.
struct SpongeSweep{S<:Sponge, D, X<:FiniteVolume.SweepTerm} <: FiniteVolume.SweepTerm
    c::S
    d::D
    term::X
end
FiniteVolume.sweep_term(c::Sponge, d) =
    SpongeSweep(c, d, FiniteVolume.sponge_term(c.α_max, c.z_start, c.target, d.state.mesh))
FiniteVolume.footprint(t::SpongeSweep) = FiniteVolume.footprint(t.term)
FiniteVolume.sweep_inputs(t::SpongeSweep) = FiniteVolume.sweep_inputs(t.term)
FiniteVolume.prepare!(t::SpongeSweep, sim, Δt) = (t.c.kind === :rest || _retarget!(t.c, t.d, Δt); nothing)
@inline FiniteVolume.add_row!(t::SpongeSweep, buf, a, b, s, j, k, ii, st) =
    FiniteVolume.add_row!(t.term, buf, a, b, s, j, k, ii, st)
@inline FiniteVolume.add_tendency!(t::SpongeSweep, tb, i, j, k, st) =    # the per-point form (#533 step 8)
    FiniteVolume.add_tendency!(t.term, tb, i, j, k, st)
