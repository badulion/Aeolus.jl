# Virtual towers and the hockey-stick threshold of the stable boundary layer
# (ONE-g, #456; the `[towers]` of the former `sbl` run type, P58-d #194):
# the adapter of `Transition.TowerSampler`.

"""
    Towers(sampler, hours)

Virtual towers in every column at the cell centres below `[towers] below`
(m): after every step (stage 13, `:output`) the state is added to the
current `window`-second segment of `Transition.TowerSampler`
(`sample_towers!`), whose samples of the mean wind speed ``V`` and the
turbulence velocity scale ``V_{TKE}`` are binned in ``V`` per level. The
towers record only in the local `hours` `(from, to)` of the model time
(hours of the day of `t mod 24 h`; `(0, 24)`: always). Its output
`tower_threshold` is the hockey-stick threshold wind speed ``V_s`` of
Sun et al. (2012) per level (`Transition.tower_thresholds`; `NaN` above
the towers and where fewer than four bins hold 20 samples), the profile
a `[checks.reference]` comparison sets against the CASES-99 observations.
"""
struct Towers <: AbstractComponent
    sampler::TowerSampler
    hours::NTuple{2,Float64}
end

component_name(::Type{Towers}) = :towers
section(::Type{Towers}) = "towers"
schema(::Type{Towers}) =
    Config.section("towers"; doc = "Virtual towers: V_TKE against the wind speed and the hockey-stick threshold " *
                                   "(Transition.TowerSampler; Sun et al. 2012).") do s
        Config.option!(s, "below", Float64; default = 0.0, range = (0, Inf), units = "m",
                       doc = "towers at every cell centre below this height; 0: no towers")
        Config.option!(s, "window", Float64; default = 300.0, range = (0, Inf), open_lower = true, units = "s",
                       doc = "averaging segment")
        Config.option!(s, "hours", NTuple{2,Float64}; units = "h", default = (0.0, 24.0), range = (0, 24),
                       doc = "local hours (from, to) of the model time in which the towers record; (0, 24): always")
    end
enabled(::Type{Towers}, opts) = opts.below > 0
function build(::Type{Towers}, opts, domain)
    m = grid(domain)
    hs = [(k - 0.5) * m.dz for k in 1:m.Nz if (k - 0.5) * m.dz < opts.below]
    return Towers(tower_sampler(m, hs; window = opts.window), opts.hours)
end
stages(::Towers) = (Stage(:output),)

_in_hours(t, (h0, h1)) = (h = mod(t / 3600, 24); h0 <= h1 ? h0 <= h < h1 : (h >= h0 || h < h1))

function run_stage!(c::Towers, d, ::Val{:output}, dt)
    s = d.state
    _in_hours(clock(d), c.hours) || (reset_segment!(c.sampler); return nothing)
    sample_towers!(c.sampler, s.u, s.v, s.w, s.νt, dt)
    return nothing
end

output_names(::Type{Towers}, opts) = ("tower_threshold",)
function outputs(c::Towers)
    function vs(d)
        out = fill(NaN, grid(d).Nz)
        out[c.sampler.ks] .= tower_thresholds(c.sampler)
        return out
    end
    return OutputVariable[OutputVariable("tower_threshold", "m s-1",
                                         "hockey-stick threshold wind speed V_s of the tower samples (Sun et al. 2012)",
                                         :z, vs)]
end

checkpoint_state(c::Towers) =
    (τ = c.sampler.τ, acc = copy(c.sampler.acc), count = copy(c.sampler.count), sumV = copy(c.sampler.sumV),
     sumE = copy(c.sampler.sumE), sumE2 = copy(c.sampler.sumE2), nseg = c.sampler.nseg)
function restore_checkpoint_state!(c::Towers, s)
    T = c.sampler
    T.τ = s.τ; T.acc .= s.acc; T.count .= s.count; T.sumV .= s.sumV; T.sumE .= s.sumE; T.sumE2 .= s.sumE2
    T.nseg = s.nseg
    return c
end
