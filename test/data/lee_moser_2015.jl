# Bulk velocity of turbulent plane channel flow: the DNS of Lee & Moser
# (2015) and the correlation of Dean (1978). Used by
# test/era6_wall_models_channel.jl and examples/validation/log_layer_mismatch.jl
# (P57-b, #183).
#
# Lee, M. & Moser, R. D. (2015). Direct numerical simulation of turbulent
# channel flow up to Re_τ ≈ 5200. J. Fluid Mech. 774, 395-415.
#
# The values are not typed in from memory. They come from the headers of
# the published mean-profile files LM_Channel_{0180,0550,1000,5200}_mean_prof.dat
# (https://turbulence.oden.utexas.edu/channel2015/data/, last updated
# 6/15/2015; fetched 2026-10-03). Each case was run at a constant flow
# rate, U_mean = 1 with δ = 1, so
#
#     U_b⁺ = U_mean/u_τ = U_mean δ / (ν Re_τ)
#
# from the header's ν and Re_τ. As a check, the trapezoidal integral of
# each file's U⁺(y/δ) over its tabulated points (y/δ up to 0.992-0.999)
# came out within 0.15 % of these values (15.668, 18.388, 19.982, 24.101).
#
# The issue (#183) quoted from memory about 15.7 / 18.4 / 20.0 / 24.1. The
# published values are 15.69 / 18.40 / 19.99 / 24.10.

"Lee & Moser (2015) channel DNS: Re_τ, ν (with U_mean = δ = 1) and U_b⁺ = 1/(ν Re_τ)."
const LM2015_CHANNEL = map(((182.088, 3.5e-4), (543.496, 1.0e-4),
                            (1000.512, 5.0e-5), (5185.897, 8.0e-6))) do (Re, ν)
    (Re_τ = Re, ν = ν, Ub_plus = 1 / (ν * Re))
end

"""
    dean_ub_plus(Re_τ) -> U_b⁺

The bulk velocity of a plane channel from Dean's (1978) skin-friction
correlation ``C_f = τ_w/(½ρU_b^2) = 0.073\\,Re_m^{-1/4}`` with
``Re_m = 2δU_b/ν``. Because ``C_f = 2/U_b^{+2}`` and
``Re_m = 2Re_τU_b^+``, this gives
``U_b^{+\\,7/8} = (2/0.073)^{1/2}(2Re_τ)^{1/8}``.

Dean, R. B. (1978). Reynolds number dependence of skin friction and other
bulk flow variables in two-dimensional rectangular duct flow.
J. Fluids Eng. 100, 215-223. (Also quoted in Pope 2000, *Turbulent
Flows*, §7.1.)
"""
dean_ub_plus(Re_τ::Real) = (sqrt(2 / 0.073) * (2 * Re_τ)^(1 / 8))^(8 / 7)

"U_b⁺ of Lee & Moser (2015) for the case whose Re_τ is nearest `Re_τ`."
lm2015_ub_plus(Re_τ::Real) = LM2015_CHANNEL[argmin([abs(c.Re_τ - Re_τ) for c in LM2015_CHANNEL])].Ub_plus
