# Shared by test/era5_ice_thermodynamics.jl and
# validation/era5_ice_thermodynamics.jl: the MU71 forcing, the BL99 reference
# driver protocol and the BL99 parameter sets.

using Aeolus.IceThermodynamics
using Aeolus.SurfaceExchange: SurfaceConstants, surface_albedo
using Aeolus.MoistThermodynamics: ThermodynamicConstants

_mean(x) = sum(x) / length(x)

# ---------------------------------------------------------------------------
# MU71 forcing (test/data/mu71_forcing.csv) and the BL99 driver protocol
# ---------------------------------------------------------------------------
function _read_mu71()
    rows = Vector{Vector{Float64}}()
    for l in eachline(joinpath(@__DIR__, "..", "data", "mu71_forcing.csv"))
        (startswith(l, "#") || startswith(l, "day")) && continue
        push!(rows, parse.(Float64, split(l, ',')))
    end
    return reduce(hcat, rows)'
end

# MU71 snowfall (m of snow per day at 330 kg m⁻³): 30 cm Aug 20 - Oct 30,
# 5 cm Nov - Apr, 5 cm in May (as coded in the BL99 reference driver).
_mu71_snow_depth(d) = (d <= 119 || d >= 302) ? 2.79e-4 :
                      (120 <= d <= 150) ? 1.61e-3 : (d > 230 ? 4.16e-3 : 0.0)

# The BL99 reference driver: forcing interpolated linearly from the previous
# day's to the current day's value over `nday` sub-steps; initial state of
# the reference code (2.53 m ice, 28.27 cm snow, its layer temperatures).
function _run_mu71(nyrs; N = 10, nday = 2, params, salinity = :bl99, Tf = -1.96,
                   lw_pert = 0.0, F_ocean = 2.0)
    D = _read_mu71()
    T0 = 273.15
    Tl = [-15.7925, -14.1042, -12.4534, -10.8423, -9.2777, -7.7687, -6.3255,
          -4.9605, -3.6879, -2.5230] .+ T0
    col = sea_ice_column(; h_ice = 2.53, h_snow = 0.2827, N, params, salinity,
                         T_bottom = T0 + Tf, T_surface = T0 - 29.2894,
                         T_layers = N == 10 ? Tl : nothing, T_snow = T0 - 23.1128,
                         F_ocean)
    daily = Float64[]; maxres = 0.0; maxseb = 0.0
    σ, ε = params.constants.σ_SB, params.ε
    U0 = ice_column_enthalpy(col); ∑src = 0.0
    prev = D[1, :]
    for _ in 1:nyrs, d in 1:365
        cur = D[d, :]
        for k in 1:nday
            f = prev .+ (cur .- prev) .* (k / nday)
            bc = PrescribedSurfaceFluxes(sw_down = f[2], lw_down = f[3] + lw_pert,
                                         H = f[4], LE = f[5],
                                         snowfall = _mu71_snow_depth(d) * 330 / 86400,
                                         T_snowfall = T0)
            dt = 86400 / nday
            α = surface_albedo(col)
            b = step_ice!(col, bc, dt)
            maxres = max(maxres, abs(b.energy_residual))
            # Surface balance, recomputed here from the forcing at the new skin
            # temperature: the energy that entered the column at the top must
            # equal (1-α)SW↓ - I₀ + εLW↓ - εσT_s⁴ - H - LE. (The enthalpy
            # residual alone cannot see an unconverged surface Newton.)
            Fatm = (1 - α) * bc.sw_down - b.F_penetrating + ε * bc.lw_down -
                   ε * σ * b.T_surface^4 - bc.H - bc.LE
            maxseb = max(maxseb, abs(Fatm - b.F_surface))
            ∑src += dt * (b.F_surface + b.F_penetrating - b.F_transmitted + b.F_ocean -
                          b.F_to_water) + b.enthalpy_in - b.enthalpy_out
        end
        prev = cur
        push!(daily, col.h_ice)
    end
    return (; daily, maxres, maxseb, col, drift = (ice_column_enthalpy(col) - U0 - ∑src) /
                                          (nyrs * 365 * 86400))
end
_year(h, y) = h[(y-1)*365+1:y*365]

# BL99 / MU71 standard parameters: k₀ = 2.034 W m⁻¹ K⁻¹ (BL99 §3.1) via the
# contract constants, emissivity 1 (as in the BL99 code). c₀ stays the
# contract's 2108 J kg⁻¹ K⁻¹, i.e. the paper's 2110 (BL99 §2).
const K_BL99 = SurfaceConstants(k_ice = 2.034)
# The reference *code* (init_column.m) uses c₀ = 2054 J kg⁻¹ K⁻¹ instead of
# the paper's 2110; the same-configuration comparison with the code uses it.
const K_BL99_CODE = SurfaceConstants(k_ice = 2.034,
                                     thermo = ThermodynamicConstants(cp_ice = 2054.0))
