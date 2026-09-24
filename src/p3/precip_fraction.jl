# P3 part-file: sub-grid precipitation fraction (Phase 44, issue #8).
#
# A grid cell that is only partly cloudy, or that rain falls into over
# only part of its area, is not the same as a cell uniformly filled with
# the grid-mean amounts: the process rates are nonlinear in the mixing
# ratios, and rain evaporates only where the air around it is
# subsaturated. Two pieces fix that.
#
#  1. The PRECIPITATION FRACTION f_p of each level is carried down the
#     column: the area covered by rain at level k is the union of the
#     rain area arriving from above and the cloud area of level k (new
#     rain forms in cloud), combined with a vertical overlap assumption
#     — maximum, random, or maximum-random (Geleyn & Hollingsworth 1979;
#     used for precipitation by Jakob & Klein 2000 and the ECMWF IFS,
#     Tiedtke 1993). Where the rain above has evaporated completely, the
#     rain area is reset (IFS convention).
#
#  2. The PROCESS RATES are then evaluated per sub-grid region and
#     weighted by area, the approach of the P3 code in E3SM/SCREAM
#     (cld_frac_l, cld_frac_r; Morrison & Milbrandt 2015 with the
#     sub-grid extension of Morrison et al.), with cloud and rain assumed
#     maximally overlapped inside the cell. Four regions:
#
#         cloud ∩ rain     area min(f_c, f_p)   accretion, riming, autoconversion
#         cloud only       f_c - min            autoconversion
#         rain only        f_p - min            rain evaporation in clear air
#         clear            the rest             —
#
#     In-cloud cloud water is q_c/f_c (and N_c/f_c), in-precipitation
#     rain q_r/f_p (N_r/f_p). Cloudy air is at liquid saturation, the
#     clear air holds the rest of the vapour (Tiedtke 1993):
#
#         q_v,cld = min(q_s, q_v/f_c),  q_v,clr = (q_v - f_c q_v,cld)/(1 - f_c),
#
#     so the grid-mean vapour is kept exactly. Ice is taken uniform over
#     the cell (ice fraction 1). Each region's rates come from the
#     unchanged `p3_process_rates`, so every process — including any
#     added later — is automatically partitioned, and the tendency map
#     `p3_tendencies` (and with it water conservation) is untouched.

"""
    p3_precipitation_overlap(P_above, c_above, c_here, overlap = :maximum_random) -> P

Fraction of a level covered by precipitation, given the precipitation
fraction `P_above` arriving from the level above, the cloud fraction
`c_above` of that level and the cloud fraction `c_here` of this level
(new precipitation forms under all of it). `overlap`:

  * `:maximum` — `P = max(P_above, c_here)`;
  * `:random`  — `P = P_above + c_here - P_above c_here`;
  * `:maximum_random` — Geleyn & Hollingsworth (1979), maximum overlap
    between vertically adjacent cloudy layers, random across clear gaps:

        P = 1 - (1 - P_above)(1 - max(c_here, c̃))/(1 - c̃),   c̃ = min(c_above, P_above).

    (`c̃ ≤ P_above` because only cloud that fed the arriving rain can
    overlap it; for `P_above ≤ c_above` this reduces to maximum overlap.)

For every input `max ≤ maximum_random ≤ random`, all three are
`≥ max(P_above, c_here)`, and all are nondecreasing in `P_above`, so the
ordering holds all the way down a column. Inputs are clamped to `[0, 1]`.

References: Geleyn & Hollingsworth (1979, Beitr. Phys. Atmos. 52, 1-16);
Jakob & Klein (2000, QJRMS 126, 2525-2544); Tiedtke (1993, MWR 121,
3040-3061).
"""
function p3_precipitation_overlap(P_above::Real, c_above::Real, c_here::Real,
                                  overlap::Symbol = :maximum_random)
    P, ca, c = promote(clamp(float(P_above), 0, 1), clamp(float(c_above), 0, 1),
                       clamp(float(c_here), 0, 1))
    # Written as "P plus the newly covered part of the uncovered area", so
    # the three forms coincide bit for bit where they should (random and
    # maximum-random across a clear gap; maximum-random and maximum when
    # the new cloud hides under the old).
    if overlap === :maximum
        return max(P, c)
    elseif overlap === :random
        return P + (1 - P) * c
    elseif overlap === :maximum_random
        ct = min(ca, P)
        (ct < 1 && c > ct) || return P
        return clamp(P + (1 - P) * ((c - ct) / (1 - ct)), max(P, c), one(P))
    end
    throw(ArgumentError("overlap must be :maximum, :random or :maximum_random"))
end

"""
    p3_precipitation_fraction(cloud_fraction, q_c, q_r; overlap = :maximum_random, q_small = 1e-14) -> f_p

Precipitation fraction of every level of a column (vectors, `k = 1` at
the surface, `k = nz` at the top, as for [`p3_sediment!`](@ref)),
carried downward from the top:

  * a level's cloud counts as a precipitation source where it holds cloud
    water (`q_c ≥ q_small`), with its `cloud_fraction`;
  * rain arriving from above is inherited only if the level above holds
    rain (`q_r ≥ q_small`): where it has all evaporated the precipitation
    area is reset, as in the IFS (Tiedtke 1993);
  * the two are combined by [`p3_precipitation_overlap`](@ref);
  * rain with neither a cloud nor rain above it (e.g. advected in from a
    neighbouring column) is given fraction 1, i.e. grid-mean behaviour.

Levels without rain can carry a nonzero fraction (the area rain forming
there would cover); it multiplies no rain.
"""
function p3_precipitation_fraction(cloud_fraction::AbstractVector, q_c::AbstractVector,
                                   q_r::AbstractVector;
                                   overlap::Symbol = :maximum_random,
                                   q_small::Real = 1e-14)
    nz = length(q_r)
    (length(cloud_fraction) == nz && length(q_c) == nz) ||
        throw(DimensionMismatch("cloud_fraction, q_c and q_r must have the same length"))
    T = float(promote_type(eltype(cloud_fraction), eltype(q_c), eltype(q_r)))
    fp = zeros(T, nz)
    csrc(k) = q_c[k] >= q_small ? clamp(T(cloud_fraction[k]), zero(T), one(T)) : zero(T)
    for k in nz:-1:1
        rain_above = k < nz && q_r[k + 1] >= q_small
        P_in = rain_above ? fp[k + 1] : zero(T)
        c_in = rain_above ? csrc(k + 1) : zero(T)
        P = p3_precipitation_overlap(P_in, c_in, csrc(k), overlap)
        fp[k] = (P == 0 && q_r[k] >= q_small) ? one(T) : P
    end
    return fp
end

"""
    p3_fractional_process_rates(s::P3ParcelState, cloud_fraction, precip_fraction,
                                pp = P3_DEFAULT_PROCESS; f_min = 0.01, dt = nothing) -> NamedTuple

The rates of [`p3_process_rates`](@ref) for a cell only partly covered by
cloud (`cloud_fraction` `f_c`) and by precipitation (`precip_fraction`
`f_p`, e.g. from [`p3_precipitation_fraction`](@ref)). The cell is split
into four regions, cloud and rain maximally overlapped inside it
(E3SM/SCREAM P3): cloud ∩ rain (area `min(f_c, f_p)`), cloud only, rain
only and clear. Each region gets the in-cloud `q_c/f_c, N_c/f_c` and/or
in-precipitation `q_r/f_p, N_r/f_p`, vapour `q_v,cld = min(q_s, q_v/f_c)`
in cloud and `q_v,clr = (q_v - f_c q_v,cld)/(1 - f_c)` in clear air
(grid-mean vapour kept), and the grid-mean ice; its rates are
`p3_process_rates` of that state, and the cell rate is the area-weighted
sum. The rime density of cloud rime is combined so that the rime
*volume* rate is the sum of the regions'.

Consequences, all asserted in the tests:

  * `f_c = f_p = 1` returns `p3_process_rates(s)` exactly;
  * KK2000 accretion `67 (q_c q_r)^1.15` over overlap area `a` changes by
    the factor `a (f_c f_p)^{-1.15}` relative to the grid mean
    (`f^{-0.15}` = 1.231 for `f_p = 0.25`, `f_c = 1`);
    autoconversion `∝ q_c^2.47 N_c^-1.79` by `f_c^{0.32}`;
  * rain under cloud does not evaporate (the cloudy air is saturated);
    rain in the clear part evaporates at the clear-sky deficit.

A species present with a fraction below `f_min` is given `f_min`, which
bounds the in-region mixing ratio at `q/f_min`. Fractions are ignored
for absent species. Water conservation is unaffected: the rates still
act through [`p3_tendencies`](@ref).

With `dt` given, each region's rates are limited by
[`p3_limit_rates`](@ref) over `dt` against *that region's* contents and
saturation state before they are weighted (with `f_c`, `f_p` ∈ {0, 1}:
the grid-mean `p3_limit_rates` itself). The result is then a complete,
limited set of rates and must not be limited again at the grid mean,
which would be wrong twice over: a grid-mean limiter lets rain collect,
within one step, cloud water lying outside the rain area and evaporate
rain that sits in saturated cloudy air, and, where the grid mean is
subsaturated with respect to ice, it switches off the deposition that the
cloudy (water-saturated) region does have. [`p3_step`](@ref) uses this
form. Without `dt` (the default) the rates are unlimited.
"""
function p3_fractional_process_rates(s::P3ParcelState{FT}, cloud_fraction::Real,
                                     precip_fraction::Real,
                                     pp::P3ProcessParameters = P3_DEFAULT_PROCESS;
                                     f_min::Real = 0.01,
                                     dt::Union{Nothing,Real} = nothing) where {FT}
    # rates of one region (or of the whole cell), limited over dt if given
    rates(st) = dt === nothing ? p3_process_rates(st, pp) :
                p3_limit_rates(st, p3_process_rates(st, pp), dt, pp)
    fc = s.q_c >= pp.q_small ? clamp(FT(cloud_fraction), FT(f_min), one(FT)) : zero(FT)
    fp = s.q_r >= pp.q_small ? clamp(FT(precip_fraction), FT(f_min), one(FT)) : zero(FT)
    # A species that is absent or fills the cell leaves nothing sub-grid:
    # the grid-mean rates, bit for bit.
    (fc == 0 || fc == 1) && (fp == 0 || fp == 1) && return rates(s)

    qs = saturation_specific_humidity(s.T, s.p, pp.thermo)
    qv_cld = fc < 1 ? min(qs, s.q_v / fc) : s.q_v
    qv_clr = fc < 1 ? max((s.q_v - fc * qv_cld) / (1 - fc), zero(FT)) : s.q_v
    a_cp = min(fc, fp)
    z = zero(FT)
    regions = ((a_cp,          true,  true),
               (fc - a_cp,     true,  false),
               (fp - a_cp,     false, true),
               (1 - fc - fp + a_cp, false, false))
    total = nothing
    rime_vol = z
    for (a, cloudy, rainy) in regions
        a > 0 || continue
        sub = _p3_with(s; q_v = cloudy ? qv_cld : qv_clr,
                       q_c = cloudy ? s.q_c / fc : z, N_c = cloudy ? s.N_c / fc : z,
                       q_r = rainy ? s.q_r / fp : z, N_r = rainy ? s.N_r / fp : z)
        r = rates(sub)
        rime_vol += a * r.rime_c / r.ρ_rime_c
        w = map(x -> a * x, r)
        total = total === nothing ? w : map(+, total, w)
    end
    ρ_rime = total.rime_c > 0 ? total.rime_c / rime_vol : FT(pp.ice.ρ_rim_default)
    return merge(total, (ρ_rime_c = ρ_rime,))
end

"""
    p3_step_column(column::AbstractVector{<:P3ParcelState}, dt, pp = P3_DEFAULT_PROCESS;
                   cloud_fraction, overlap = :maximum_random, adjust = true, f_min = 0.01)
        -> (new_column, precip_fraction)

[`p3_step`](@ref) for every level of a column (`k = 1` at the surface)
with sub-grid cloud and precipitation fractions: the precipitation
fraction is first carried down the column by
[`p3_precipitation_fraction`](@ref) from the given `cloud_fraction`
vector and the start-of-step `q_c`, `q_r`, then each level is stepped
with [`p3_fractional_process_rates`](@ref). Sedimentation is separate
([`p3_sediment_rain!`](@ref)); it needs no fraction, because the fall
speeds depend on `q_r/N_r` only, which the in-precipitation scaling
leaves unchanged. Returns the new states and the fractions used.
"""
function p3_step_column(column::AbstractVector{<:P3ParcelState}, dt::Real,
                        pp::P3ProcessParameters = P3_DEFAULT_PROCESS;
                        cloud_fraction::AbstractVector,
                        overlap::Symbol = :maximum_random,
                        adjust::Bool = true, f_min::Real = 0.01)
    fp = p3_precipitation_fraction(cloud_fraction, [s.q_c for s in column],
                                   [s.q_r for s in column];
                                   overlap = overlap, q_small = pp.q_small)
    new = [p3_step(column[k], dt, pp; adjust = adjust, f_min = f_min,
                   cloud_fraction = cloud_fraction[k], precip_fraction = fp[k])
           for k in eachindex(column)]
    return (new, fp)
end
