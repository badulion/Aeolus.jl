# The boundary statistics a refined child's synthetic inflow needs, from a
# resolved convective parent (ONE-l, #475; the `[inflow_statistics]` of the
# former `cbl` run type, N-inflow-b #258): the adapter of
# `Transition.ParentResolvedStatistics`.

"""
    ParentStatistics(start, τ_avg, refine, nlev)

The domain as a nest parent: after every step whose end lies after
`[inflow_statistics] start` (s; stage 13, `:output`) the state is folded
into a `Transition.ParentResolvedStatistics` (`update_inflow_statistics!`)
over the whole plane, at the west face, in scale-gap mode for a child
refined `refine`, with the running-mean time scale `tau_avg` (s), at the
levels below `depth` (m). The statistics read the domain's SGS closure
(its ``e`` and ``ν_t``), so they are created at the first fold. Its
outputs (`inflow_*`, profiles; `NaN` above `depth` and before the first
fold) are the parts of `Transition.inflow_stress_parts`: the parent's
resolved stress, the scale-gap band and the SGS part, and what the
generator must add (`generated` = SGS + band); in the face frame, `uu`
the face-normal variance, `ww` the vertical one, `uw` their covariance.
Its running sums are checkpointed (a restart folds on bit for bit).
"""
mutable struct ParentStatistics <: AbstractComponent
    start::Float64
    τ_avg::Float64
    refine::NTuple{3,Int}
    nlev::Int
    stats::Any                       # nothing until the first fold
    pending::Any                     # a restored state, applied at the first fold
end

component_name(::Type{ParentStatistics}) = :inflow_statistics
section(::Type{ParentStatistics}) = "inflow_statistics"
schema(::Type{ParentStatistics}) =
    Config.section("inflow_statistics"; doc = "The domain as a nest parent: the boundary statistics of a refined " *
                                              "child's synthetic inflow (Transition.ParentResolvedStatistics, " *
                                              "west face, whole plane, scale-gap mode).") do s
        Config.option!(s, "enabled", Bool; default = false, doc = "fold every step after start into the statistics")
        Config.option!(s, "start", Float64; default = 0.0, range = (0, Inf), units = "s",
                       doc = "the steps that end after this model time are folded in")
        Config.option!(s, "tau_avg", Float64; default = 600.0, range = (0, Inf), open_lower = true, units = "s",
                       doc = "running-mean time scale τ_avg")
        Config.option!(s, "refine", NTuple{3,Int}; default = (2, 2, 4), range = (1, 64),
                       doc = "the child's refinement (sets the band k_c < k ≤ r k_c)")
        Config.option!(s, "depth", Float64; default = 500.0, range = (0, Inf), open_lower = true, units = "m",
                       doc = "the levels below this height (at least one)")
    end
enabled(::Type{ParentStatistics}, opts) = opts.enabled
function build(::Type{ParentStatistics}, opts, domain)
    m = grid(domain)
    nlev = clamp(round(Int, opts.depth / m.dz), 1, m.Nz)
    return ParentStatistics(opts.start, opts.tau_avg, opts.refine, nlev, nothing, nothing)
end
stages(::ParentStatistics) = (Stage(:output),)

# the running sums of ParentResolvedStatistics (what a restart needs)
const _IS_SUMS = (:Um, :Rres, :Rband, :Cres, :Cband, :Rsgs, :esgs, :Cx, :Cy, :Cz, :Vz, :Bx, :By, :Bz, :BVz)

function run_stage!(c::ParentStatistics, d, ::Val{:output}, dt)
    clock(d) > c.start || return nothing
    if c.stats === nothing
        c.stats = ParentResolvedStatistics(d.state; closure = sgs_closure(d), τ_avg = c.τ_avg, refine = c.refine,
                                           levels = 1:c.nlev)
        c.pending === nothing || (_restore_sums!(c.stats, c.pending); c.pending = nothing)
    end
    update_inflow_statistics!(c.stats, dt)
    return nothing
end

const _IS_OUTPUTS = ("inflow_uu_resolved", "inflow_uu_band", "inflow_uu_sgs", "inflow_uu_generated",
                     "inflow_ww_resolved", "inflow_ww_generated", "inflow_uw_resolved", "inflow_uw_generated",
                     "inflow_e_sgs", "inflow_L_z_full_u", "inflow_L_z_u", "inflow_L_s_u")
output_names(::Type{ParentStatistics}, opts) = _IS_OUTPUTS

function outputs(c::ParentStatistics)
    # (part, row): a row of a 6 × nl stress (uu 1, ww 3, uw 5) or of a 3 × nl length (u 1)
    pick = Dict("inflow_uu_resolved" => (:resolved, 1), "inflow_uu_band" => (:band, 1), "inflow_uu_sgs" => (:sgs, 1),
                "inflow_uu_generated" => (:generated, 1), "inflow_ww_resolved" => (:resolved, 3),
                "inflow_ww_generated" => (:generated, 3), "inflow_uw_resolved" => (:resolved, 5),
                "inflow_uw_generated" => (:generated, 5), "inflow_e_sgs" => (:e_sgs, 0),
                "inflow_L_z_full_u" => (:L_z_full, 1), "inflow_L_z_u" => (:L_z, 1), "inflow_L_s_u" => (:L_s, 1))
    units = Dict(:e_sgs => "m2 s-2", :L_z_full => "m", :L_z => "m", :L_s => "m")
    function profile(d, part, row)
        out = fill(NaN, grid(d).Nz)
        (c.stats === nothing || c.stats.nsamples == 0) && return out
        a = getproperty(inflow_stress_parts(c.stats), part)
        out[1:c.nlev] .= row == 0 ? a : a[row, :]
        return out
    end
    return OutputVariable[OutputVariable(n, get(units, pick[n][1], "m2 s-2"),
                                         "inflow statistics of a refined child: $(pick[n][1]) " *
                                         (pick[n][2] == 0 ? "" : "row $(pick[n][2]) ") * "(Transition.inflow_stress_parts)",
                                         :z, d -> profile(d, pick[n]...)) for n in _IS_OUTPUTS]
end

function checkpoint_state(c::ParentStatistics)
    s = c.stats
    s === nothing && return c.pending
    return merge((W = s.W, t = s.t, nsamples = s.nsamples), NamedTuple{_IS_SUMS}(map(k -> copy(getfield(s, k)), _IS_SUMS)))
end
function _restore_sums!(s, st)
    s.W = st.W; s.t = st.t; s.nsamples = st.nsamples
    for k in _IS_SUMS
        getfield(s, k) .= getproperty(st, k)
    end
    return s
end
function restore_checkpoint_state!(c::ParentStatistics, st)
    st === nothing && return c
    c.stats === nothing ? (c.pending = st) : _restore_sums!(c.stats, st)
    return c
end
