# Phase 52 — wave-dependent sea-surface roughness (issue #39).
#
# COARE stability functions and gustiness, Charnock / COARE / wave-age /
# wave-steepness momentum roughness, scalar roughness, the COARE-style
# bulk algorithm, and the SeaSurface implementer of the surface shared core.
#
# Reference values come from (i) the NOAA-PSL COARE reference
# implementation (github.com/NOAA-PSL/COARE-algorithm, commit bd1cac8),
# whose own test data and output are in test/data/coare35_*.txt, and
# (ii) numbers printed in Fairall et al. (2003, J. Climate 16) and
# Edson et al. (2013, J. Phys. Oceanogr. 43). See test/data/README_coare.md.

using Test
using Aeolus
using Aeolus.SurfaceExchange
using Aeolus.AirSeaFlux

const ASF = Aeolus.AirSeaFlux

const MT = Aeolus.MoistThermodynamics

# ---- Input conversion of the COARE reference code (NOT model physics) ----
# The reference test files give air humidity as RH and water temperature;
# the reference code turns them into specific humidities with Buck (1981)
# and a 2 % seawater reduction, and uses latitude-dependent gravity. To
# compare like with like, the test feeds the model the same q_a, q_s and g.
_bucksat(T, P) = 6.1121 * exp(17.502T / (T + 240.97)) * (1.0007 + 3.46e-6P)   # hPa, T in °C
_qsea(T, P) = (es = 0.98 * _bucksat(T, P); 0.622es / (P - 0.378es))
_qair(T, P, rh) = (em = 0.01rh * _bucksat(T, P); 0.622em / (P - 0.378em))
_grv(lat) = (x = sind(lat); 9.7803267715 * (1 + 0.0052790414x^2 + 0.0000232718x^4 +
                                             0.0000001262x^6 + 0.0000000007x^8))
_readtab(f) = [parse.(Float64, split(l)) for l in eachline(f)
               if !startswith(l, "#") && !startswith(l, "u")]
const DATA = joinpath(@__DIR__, "data")

# Two user-defined Charnock laws for the extension-point test.
struct _UserCharnock <: Aeolus.AirSeaFlux.AbstractCharnockRoughness
    β::Float64
end
Aeolus.AirSeaFlux.charnock_parameter(::_UserCharnock, u_star; kwargs...) = 0.02
struct _NoAlphaCharnock <: Aeolus.AirSeaFlux.AbstractCharnockRoughness
    β::Float64
end

@testset "Phase 52 — AirSeaFlux: wave-dependent roughness (#39)" begin

    @testset "COARE stability: φ–ψ consistency, reference code, free convection" begin
        for v in (:coare30, :coare35)
            sf = COAREStability(v)
            # dψ/dζ = (1 - φ)/ζ across the blended unstable branch — this
            # checks the analytic φ of the blend, including the f′ term.
            worst = 0.0
            for ζ in (-10.0, -1.0, -0.05, 0.3, 3.0), (ψ, φ) in ((psi_m, phi_m), (psi_h, phi_h))
                h = 1e-4 * max(abs(ζ), 0.1)
                dψ = (-ψ(sf, ζ + 2h) + 8ψ(sf, ζ + h) - 8ψ(sf, ζ - h) + ψ(sf, ζ - 2h)) / (12h)
                worst = max(worst, abs(dψ - (1 - φ(sf, ζ)) / ζ))
            end
            @test worst < 1e-7
        end
        # NOAA-PSL COARE 3.5 reference (meteo.psiu_26 / psit_26); it uses
        # x^0.3333 and rounded stable coefficients, so agreement is ~1e-3
        ζs = (-10.0, -1.0, -0.05, 0.3, 3.0)
        ref_u = (2.7054530992, 1.11042086012, 0.154980480191, -1.4805245613, -9.85231262359)
        ref_t = (3.70791670101, 1.86535940766, 0.299540089787, -1.44810646799, -11.0897487826)
        sf = COAREStability(:coare35)
        @test maximum(abs(psi_m(sf, ζ) - r) for (ζ, r) in zip(ζs, ref_u)) < 2e-3
        @test maximum(abs(psi_h(sf, ζ) - r) for (ζ, r) in zip(ζs, ref_t)) < 6e-3
        # free-convection limit: φ → (1 - γ_c ζ)^{-1/3}
        @test phi_h(sf, -1e4) ≈ (1 + sf.γc_h * 1e4)^(-1 / 3) rtol = 1e-6
        # convective gustiness w_g = β (B z_i)^{1/3}
        g = ConvectiveGustiness()
        @test gustiness_velocity(g, 5e-3) ≈ 1.2 * cbrt(5e-3 * 600)
    end

    @testset "Roughness: Charnock, smooth flow, wave age, scalar" begin
        g, ν = Aeolus.SurfaceExchange.grav(), 1.5e-5
        # air viscosity delegates to Sutherland and matches the COARE polynomial
        coare_ν(t) = 1.326e-5 * (1 + 6.542e-3t + 8.301e-6t^2 - 4.84e-9t^3)
        @test maximum(abs(ASF.air_kinematic_viscosity(273.15 + t, 101325.0) / coare_ν(t) - 1) for t in 0:10:30) < 0.005
        # deep-water dispersion ω² = g k
        s = deep_water_sea_state(8.0, 2.0)
        @test s.c_p^2 ≈ g * s.λ_p / 2π
        # smooth-flow limit (Smith 1988): R_r → 0.11; fully rough: z0 g/u*² → α
        z = sea_roughness_length(COARE35Charnock(), 1e-3; ν = ν, g = g, U10N = 1.0)
        @test roughness_reynolds(z, 1e-3, ν) ≈ 0.11 rtol = 1e-3
        @test sea_roughness_length(CharnockRoughness(), 2.0; ν = ν, g = g) * g / 4 ≈ 0.011 rtol = 1e-3
        # COARE 3.5 α (Edson et al. 2013): 0.0017 U − 0.005
        @test charnock_parameter(COARE35Charnock(), 0.3; U10N = 10.0) ≈ 0.012
        # Edson et al. (2013): wave-age α = 0.013 at c_p/u* = 32
        @test abs(charnock_parameter(WaveAgeCharnock(), 0.4; sea = SeaState(c_p = 32 * 0.4, H_s = 2.0, λ_p = 100.0)) - 0.013) < 5e-4
        # Oost et al. (2002) α = 50 (c_p/u*)^(-2.5), exact for deep-water waves
        s = deep_water_sea_state(6.0, 1.5)
        @test charnock_parameter(OostRoughness(), 0.5; g = g, ν = ν, sea = s) ≈ 50 * (s.c_p / 0.5)^(-2.5) rtol = 1e-10
        # user Charnock law: z0 = α u*²/g + β ν/u* from the abstract type;
        # one without α fails with a message, not a stack overflow (#100)
        @test sea_roughness_length(_UserCharnock(0.11), 2.0; ν = ν, g = g) ≈ 0.02 * 4 / g + 0.11 * ν / 2 rtol = 1e-12
        @test_throws ArgumentError sea_roughness_length(_NoAlphaCharnock(0.11), 2.0; ν = ν)
        # scalar roughness: z0q = A R_r^(−B) in rough flow
        z0, u = 2e-3, 0.8
        Rr = roughness_reynolds(z0, u, ν)
        @test sea_scalar_roughness(COARE35ScalarRoughness(), z0, u; ν = ν).z0q ≈ 5.8e-5 * Rr^-0.72
    end

    @testset "Neutral C_D10N / C_E10N against published COARE numbers" begin
        c30 = (COARE30Charnock(), COARE30ScalarRoughness())
        c35 = (COARE35Charnock(), COARE35ScalarRoughness())
        ntc(r, U) = neutral_transfer_coefficients(r..., U)
        n = ntc(c35, 12.0)
        @test n.u_star ≈ 0.4 * 12 / log(10 / n.z0m) rtol = 1e-10
        # Fairall et al. (2003): 10³ C_d10n = 2.06 at 20 m/s; C_e10n 1.08e-3 at 5 m/s
        @test 1e3 * ntc(c30, 20.0).C_D ≈ 2.06 rtol = 0.01
        @test 1e3 * ntc(c30, 5.0).C_E ≈ 1.08 rtol = 0.04
        # Edson et al. (2013) eq. 22: u* = 0.062 U10N − 0.28 for U10N > 8.5 m/s
        @test maximum(U -> abs(ntc(c35, U).u_star / (0.062U - 0.28) - 1), 10.0:5.0:25.0) < 0.08
        # younger seas are rougher at the same wind
        U = 12.0
        cd(sea) = neutral_transfer_coefficients(WaveAgeCharnock(), COARE35ScalarRoughness(), U; sea = sea).C_D
        @test cd(deep_water_sea_state(4.0, 2.0)) > 1.1 * cd(deep_water_sea_state(8.0, 2.0))
    end

    @testset "Bulk algorithm: neutral and free-convection limits" begin
        b = coare35_bulk(gustiness = ConvectiveGustiness(w_min = 0.0, w_init = 0.0))
        r = air_sea_fluxes(b; U = 8.0, z_m = 10.0, θ_a = 290.0, θ_s = 290.0,
                           q_a = 0.01, q_s = 0.01, ν = 1.5e-5)
        n = neutral_transfer_coefficients(COARE35Charnock(), COARE35ScalarRoughness(), 8.0; ν = 1.5e-5)
        @test r.converged && r.surface.ζ == 0
        @test r.surface.u_star ≈ n.u_star rtol = 1e-8
        # free convection: U = 0, the converged w_g is β (B z_i)^{1/3} of its own B
        r0 = air_sea_fluxes(coare35_bulk(); U = 0.0, z_m = 10.0, θ_a = 299.0, θ_s = 302.0,
                            q_a = 0.016, q_s = 0.027)
        @test r0.converged && r0.B > 0 && r0.surface.ζ < -1
        @test r0.w_g ≈ 1.2 * cbrt(r0.B * 600) rtol = 1e-6
    end

    @testset "COARE 3.5 reference test data (NOAA-PSL, 116 hourly records)" begin
        # test/data/coare35_test_{input,output}.txt: the input and output
        # files shipped with the NOAA-PSL COARE 3.5 Python reference.
        # The reference was run with its cool-skin model on (jcool = 1);
        # the cool-skin depressions dter, dqer are in its output, so the
        # interface values θ_s = ts − dter, q_s = q_sea(ts) − dqer are
        # exactly what its bulk loop used.
        inp = _readtab(joinpath(DATA, "coare35_test_input.txt"))
        out = _readtab(joinpath(DATA, "coare35_test_output.txt"))
        keys_ = (:u_star, :τ, :H, :LE, :θ_star, :q_star, :z0t, :L, :C_D, :C_D10N, :H_dry_cp)
        err = Dict(k => 0.0 for k in keys_)
        bias = Dict(k => 0.0 for k in keys_)
        nconv = 0
        for (a, o) in zip(inp, out)
            u, zu, t, zt, rh, zq, P, ts, _, _, lat = a[1:11]
            usr, tau, hsb, hlb, _, tsr, qsr, zot, _, Cd, _, _, L, _, dter, dqer = o[1:16]
            Cdn = o[19]
            c = SurfaceConstants(thermo = MT.ThermodynamicConstants(grav = _grv(lat)))
            qa = _qair(t, P, rh)
            r = air_sea_fluxes(coare35_bulk(); U = u, z_m = zu,
                               θ_a = t + 273.16 + 0.0098zt, θ_s = ts - dter + 273.16,
                               q_a = qa, q_s = _qsea(ts, P) - dqer,
                               T_air = t + 273.16, p = 100P, c = c)
            nconv += r.converged
            sl = r.surface
            pp = MT.PhasePartition(qa)
            ρ = MT.air_density(t + 273.16, 100P, pp, c.thermo)
            cpm = MT.heat_capacity_moist(pp, c.thermo)
            Lv = MT.latent_heat_vapor(ts - dter + 273.16, c.thermo)
            G = sl.U / u                              # gustiness factor S/U
            mine = Dict(:u_star => sl.u_star / usr, :τ => ρ * sl.u_star^2 / G / tau,
                        :H => -ρ * cpm * sl.u_star * sl.θ_star / hsb,
                        :LE => -ρ * Lv * sl.u_star * sl.q_star / hlb,
                        :θ_star => sl.θ_star / tsr, :q_star => 1000sl.q_star / qsr,
                        :z0t => r.z0h / zot, :L => sl.L / L,
                        :C_D => sl.u_star^2 / G / (sl.U * u) / Cd,
                        :C_D10N => 1000r.C_D10N / Cdn,
                        # with COARE's dry-air c_p = 1004.67 instead of c_pm
                        :H_dry_cp => -ρ * 1004.67 * sl.u_star * sl.θ_star / hsb)
            for k in keys_
                err[k] = max(err[k], abs(mine[k] - 1))
                bias[k] += (mine[k] - 1) / length(inp)
            end
        end
        @test nconv == 116
        @test maximum(err[k] for k in (:u_star, :τ, :LE, :θ_star, :q_star, :C_D, :C_D10N)) < 0.01
        @test err[:z0t] < 0.02 && err[:L] < 0.025
        # H is 1.5 % high throughout: COARE uses the dry-air c_p = 1004.67,
        # the model the moist c_pm (q ≈ 18 g/kg here ⇒ +1.5 %). With the
        # dry c_p the difference collapses:
        @test 0.012 < bias[:H] < 0.02
        @test err[:H_dry_cp] < 0.005
    end

    @testset "COARE 3.5 reference model: stable/unstable × wind / wave-age / sea-state" begin
        # test/data/coare35_synthetic.txt: the NOAA-PSL COARE 3.5 Python
        # reference run by test/data/coare35_generate.py on a synthetic grid
        # (jcool = 0; 2–25 m/s; fully developed seas). Its sea-state option
        # omits the smooth-flow term, which SeaStateRoughness(β = 0) reproduces.
        c = SurfaceConstants(thermo = MT.ThermodynamicConstants(grav = _grv(45.0)))
        worst = Dict(k => 0.0 for k in (:u_star, :θ_star, :q_star, :L, :C_D10N))
        n = 0; ok = true
        for l in eachline(joinpath(DATA, "coare35_synthetic.txt"))
            startswith(l, "#") && continue
            w = split(l)
            U, t, rh, ts, cp, Hs, usr, _, _, _, tsr, qsr, _, _, _, _, _, L, _, Cdn =
                parse.(Float64, w[3:22])
            b = w[2] == "none" ? coare35_bulk() :
                w[2] == "age"  ? coare35_bulk(roughness = WaveAgeCharnock()) :
                                 coare35_bulk(roughness = SeaStateRoughness(β = 0.0))
            sea = SeaState(c_p = cp, H_s = Hs, λ_p = 2π * cp^2 / _grv(45.0))
            r = air_sea_fluxes(b; U = U, z_m = 10.0, θ_a = t + 273.16 + 0.098,
                               θ_s = ts + 273.16, q_a = _qair(t, 1010.0, rh),
                               q_s = _qsea(ts, 1010.0), T_air = t + 273.16,
                               p = 101000.0, c = c, sea = sea)
            s = r.surface
            ok &= r.converged && sign(s.ζ) == (w[1] == "stable" ? 1 : -1)
            for (k, v) in ((:u_star, s.u_star / usr), (:θ_star, s.θ_star / tsr),
                           (:q_star, 1000s.q_star / qsr), (:L, s.L / L),
                           (:C_D10N, 1000r.C_D10N / Cdn))
                worst[k] = max(worst[k], abs(v - 1))
            end
            n += 1
        end
        @test ok && n == 60
        @test max(worst[:u_star], worst[:θ_star], worst[:q_star]) < 0.005
        # C_D10N: 0.6 % at U = 2 m/s, where the COARE 3.5 Charnock fit is
        # negative in the reference and floored at 0 here
        @test worst[:C_D10N] < 0.01 && worst[:L] < 0.015
    end

    @testset "SeaSurface implements the SurfaceExchange surface interface" begin
        c = DEFAULT_SURFACE_CONSTANTS
        s = SeaSurface(T_skin = 301.0)
        @test s isa AbstractSurface
        @test surface_emissivity(s) == Aeolus.SurfaceOptics.material_emissivity(:water)
        f = SurfaceForcing(sw_down = 600.0, lw_down = 420.0, T_air = 298.0, q_air = 0.016,
                           U = 7.0, p = 101000.0, z_ref = 10.0)
        # seawater: e = 0.98 e_sat ⇒ q_s ≈ 0.98 q_sat (to O(e/p))
        @test surface_specific_humidity(s, f) / MT.saturation_specific_humidity(301.0, 101000.0) ≈ 0.98 rtol = 1e-3
        fl = turbulent_fluxes(s, f)
        @test fl.Rn ≈ 0.945 * 600 + s.emissivity * (420 - c.σ_SB * 301.0^4)
        @test fl.LE ≈ MT.latent_heat_vapor(301.0) * fl.E
        r = sea_surface_solution(s, f)
        ρ = MT.air_density(298.0, 101000.0, MT.PhasePartition(0.016))
        # COARE stress: τ = ρ C_D S U
        @test fl.τ ≈ ρ * r.surface.C_D * r.surface.U * 7.0 rtol = 1e-12
        # the contract's default MOST solve agrees at the solution's roughness
        s.z0m, s.z0h, s.z0q = r.z0m, r.z0h, r.z0q
        fd = invoke(turbulent_fluxes, Tuple{AbstractSurface, SurfaceForcing}, s, f;
                    sf = s.bulk.stability, gustiness = r.w_g)
        @test fd.H ≈ fl.H rtol = 1e-9
        @test fd.LE ≈ fl.LE rtol = 1e-9
    end
end
