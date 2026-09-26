# Phase 53 — blending-height formulation (issue #42).
#
# Effective roughness at a blending height (Wieringa 1986; Mason 1988),
# blending-height closures (Mason 1988; Claussen 1991; Bou-Zeid et al.
# 2004, 2007), internal-boundary-layer growth laws (Savelyev & Taylor
# 2005, Table I) and tiles forced at the blending height (BlendedMosaic).
#
# What is checked against what:
#   * analytic identities and limits of the effective roughness (defining
#     equation, bounds, the l_b → ∞ log-average limit with its leading
#     correction, the neutral exactness of the blended profile);
#   * the published LES values of Bou-Zeid, Parlange & Meneveau (2007,
#     J. Atmos. Sci. 64, Table 2) for two-stripe surfaces;
#   * the published spread of IBL formulae at 300 m fetch (Savelyev &
#     Taylor 2005, Sect. 2 and Fig. 3);
#   * conservation and single-tile consistency of the blended mosaic.
# No field observations are used.

using Test, Printf
using Aeolus
using Aeolus.SurfaceTiles
import Aeolus.SurfaceExchange: surface_temperature, roughness_lengths, surface_albedo,
                               surface_emissivity, moisture_availability, step_surface!,
                               psi_m, psi_h, phi_m, phi_h, AbstractStabilityFunctions
using Aeolus.MoistThermodynamics: PhasePartition, air_density, dry_adiabatic_lapse_rate,
                                  heat_capacity_moist,
                                  saturation_specific_humidity

# ---- test surfaces -------------------------------------------------------

mutable struct BHTestSlab <: AbstractSurface
    T::Float64
    C::Float64
    β::Float64
    z0m::Float64
    z0h::Float64
end
BHTestSlab(; T = 293.0, C = 2e5, β = 0.0, z0m = 0.1, z0h = 0.01) = BHTestSlab(T, C, β, z0m, z0h)
surface_temperature(s::BHTestSlab) = s.T
roughness_lengths(s::BHTestSlab) = (z0m = s.z0m, z0h = s.z0h, z0q = s.z0h)
surface_albedo(::BHTestSlab) = 0.2
surface_emissivity(::BHTestSlab) = 0.98
moisture_availability(s::BHTestSlab) = s.β
function step_surface!(s::BHTestSlab, f::SurfaceForcing, dt; kwargs...)
    fl = turbulent_fluxes(s, f; kwargs...)
    s.T += fl.G * dt / s.C
    return fl
end

# Neutral similarity (ψ ≡ 0): the blended profile is then closed-form.
struct BHTestNeutral <: AbstractStabilityFunctions end
psi_m(::BHTestNeutral, ζ) = zero(ζ)
psi_h(::BHTestNeutral, ζ) = zero(ζ)
phi_m(::BHTestNeutral, ζ) = one(ζ)
phi_h(::BHTestNeutral, ζ) = one(ζ)

_bh_logmean(z, f) = exp(sum(f .* log.(z)) / sum(f))
_bh_rel(a, b) = abs(a - b) / max(abs(a), abs(b))

const κ_bh = Aeolus.SurfaceExchange.DEFAULT_SURFACE_CONSTANTS.κ
_θ(f::SurfaceForcing, z = f.z_ref) =
    f.T_air + dry_adiabatic_lapse_rate(PhasePartition(f.q_air)) * z

@testset "Era V — blending height (#42)" begin

    @testset "Effective roughness: defining identity, bounds, limits" begin
        # Single tile / identical tiles: z0_eff = z0.
        @test effective_roughness([0.05], [1.0], 50.0) ≈ 0.05 rtol = 8eps()
        @test effective_roughness([0.05, 0.05, 0.05], [0.2, 0.3, 0.5], 50.0) ≈ 0.05 rtol = 8eps()
        # Zero-fraction tiles are ignored (and may even exceed l_b).
        @test effective_roughness([0.01, 0.5, 200.0], [0.4, 0.6, 0.0], 60.0) ==
              effective_roughness([0.01, 0.5], [0.4, 0.6], 60.0)
        @test_throws ArgumentError effective_roughness([0.01, 2.0], [0.5, 0.5], 1.5)
        @test_throws DimensionMismatch effective_roughness([0.01, 2.0], [1.0], 10.0)

        # The defining identity (neutral drag with a common wind at l_b),
        # bounds, and monotonicity in l_b, over random patchworks.
        worst_id = 0.0
        for k in 1:500
            n = 2 + k % 4
            z = 10 .^ (-4 .+ 4 .* rand(n))           # 1e-4 … 1 m
            f = rand(n); f ./= sum(f)
            lb = 5.0 + 500rand()
            ze = effective_roughness(z, f, lb)
            worst_id = max(worst_id, _bh_rel(1 / log(lb / ze)^2, sum(f ./ log.(lb ./ z) .^ 2)))
            @test _bh_logmean(z, f) < ze < maximum(z)
            @test effective_roughness(z, f, 2lb) < ze          # smaller patches → rougher
        end
        @test worst_id < 1e-13

        # l_b → ∞: tends to the log average (Taylor 1987) with the leading
        # correction ln z0_eff − ⟨ln z0⟩ = (3/2) Var(ln z0) / ln(l_b/z0_lm).
        z, f = [0.01, 0.1], [0.4, 0.6]
        x̄ = sum(f .* log.(z)); v = sum(f .* (log.(z) .- x̄) .^ 2)
        ratio(lb) = (log(effective_roughness(z, f, lb)) - x̄) * log(lb / exp(x̄)) / (1.5v)
        r = [ratio(10.0^p) for p in (2, 4, 8, 16)]
        # the remainder is O(1/ln l_b): measured 0.917, 0.949, 0.971, 0.984
        @test all(abs.(r .- 1) .< [0.1, 0.06, 0.035, 0.02])
        @test issorted(abs.(r .- 1), rev = true)

        # Scalar roughness: with common z0m, ln(l_b/z0h_eff) is the harmonic
        # mean of ln(l_b/z0h,i); with z0h = z0m everywhere it equals z0m_eff.
        lb = 40.0
        zh = [1e-4, 1e-2, 1e-3]; fz = [0.2, 0.5, 0.3]; zm = fill(0.05, 3)
        zhe = effective_scalar_roughness(zm, zh, fz, lb)
        @test log(lb / zhe) ≈ 1 / sum(fz ./ log.(lb ./ zh)) rtol = 1e-13
        zm2 = [0.01, 0.3, 0.05]
        @test effective_scalar_roughness(zm2, zm2, fz, lb) ≈ effective_roughness(zm2, fz, lb) rtol = 1e-13

        # Mosaic method: the three effective lengths from the tiles.
        m = SurfaceMosaic([BHTestSlab(z0m = 0.02, z0h = 0.002), BHTestSlab(z0m = 0.5, z0h = 0.05)], [0.7, 0.3])
        e = effective_roughness(m, 30.0)
        @test e.z0m == effective_roughness([0.02, 0.5], [0.7, 0.3], 30.0)
        @test e.z0h == effective_scalar_roughness([0.02, 0.5], [0.002, 0.05], [0.7, 0.3], 30.0)
        @test e.z0q == e.z0h
        # l_b → ∞ joins the log-averaged parameter aggregate of #41, but only
        # logarithmically: the gap is the (3/2) Var(ln z0)/ln(l_b/z0) term.
        zl = roughness_lengths(m).z0m
        vm = 0.7 * 0.3 * log(0.5 / 0.02)^2
        for (lb, tol) in ((1e3, 0.25), (1e30, 0.05))
            @test log(effective_roughness(m, lb).z0m / zl) * log(lb / zl) / (1.5vm) ≈ 1 atol = tol
        end
    end

    @testset "Blending-height closures solve their own equations" begin
        z, f = [0.01, 0.1], [0.25, 0.75]
        lnz0(lb) = log(lb / effective_roughness(z, f, lb))
        for L in (500.0, 3140.0, 20_000.0)
            lm = blending_height(MasonBlendingHeight(L), z, f)
            @test (lm / L) * lnz0(lm)^2 ≈ 2κ_bh^2 rtol = 1e-12      # Kramm & Dlugi Eq. 5.4
            lc = blending_height(ClaussenBlendingHeight(L), z, f)
            @test (lc / L) * lnz0(lc) ≈ 1.75κ_bh rtol = 1e-12       # Kramm & Dlugi Eq. 5.5
            lz = blending_height(BouZeidBlendingHeight(L), z, f)
            # Bou-Zeid et al. (2007) Eq. (2) and Eq. (3) hold together.
            @test (lz / (1.7κ_bh * L + lz))^2 ≈ sum(f ./ log.(lz ./ z) .^ 2) rtol = 1e-12
            @test effective_roughness(z, f, lz) ≈ lz * exp(-1.7κ_bh * L / lz - 1) rtol = 1e-10
            # Claussen is Bou-Zeid without the −1: with the same C it needs a
            # smaller ln(l_b/z0_eff), i.e. blends lower.
            @test blending_height(ClaussenBlendingHeight(L; C = 1.7), z, f) < lz
        end
        # Longer patches blend higher; Mason's l_b is O(L/100).
        ls = [blending_height(MasonBlendingHeight(L), z, f) for L in (300.0, 1e3, 3e3, 1e4, 3e4)]
        @test issorted(ls)
        @test all(0.002 .< ls ./ [300.0, 1e3, 3e3, 1e4, 3e4] .< 0.05)
        @test blending_height(FixedBlendingHeight(42.0), z, f) == 42.0
        @test_throws ArgumentError FixedBlendingHeight(0.0)
        @test_throws ArgumentError MasonBlendingHeight(-1.0)

        # Mosaic forms agree with the vector forms.
        m = SurfaceMosaic([BHTestSlab(z0m = 0.01), BHTestSlab(z0m = 0.1)], f)
        @test blending_height(BouZeidBlendingHeight(3140.0), m) == blending_height(BouZeidBlendingHeight(3140.0), z, f)
        @test effective_roughness(m, BouZeidBlendingHeight(3140.0)).z0m ==
              effective_roughness(z, f, blending_height(BouZeidBlendingHeight(3140.0), z, f))
    end

    @testset "Bou-Zeid et al. (2007) Table 2: two-stripe LES" begin
        # Bou-Zeid, Parlange & Meneveau (2007), J. Atmos. Sci. 64, 216-227,
        # Table 2: stripes of z0 = 0.01 m and 0.1 m, the smooth fraction in
        # the case name (S07R93 = 7 % smooth), variability scale Lp = mean
        # patch length = 3140 m (their Sect. 3); z0,e from a log-law fit to
        # the LES mean profile, hb from where the velocity quartiles merge.
        table2 = [("S07R93", 0.07, 0.090, 225.0), ("S12R88", 0.12, 0.080, 230.0),
                  ("S25R75", 0.25, 0.063, 240.0), ("S33R67", 0.33, 0.058, 240.0),
                  ("S67R33", 0.67, 0.028, 260.0), ("S75R25", 0.75, 0.023, 265.0),
                  ("S88R12", 0.88, 0.017, 305.0), ("S93R07", 0.93, 0.014, 310.0)]
        z = [0.01, 0.1]
        err = Dict(k => Float64[] for k in (:bz, :cl, :ma, :lm, :hb))
        for (name, fs, z0les, hbles) in table2
            f = [fs, 1 - fs]
            lb = blending_height(BouZeidBlendingHeight(3140.0), z, f)
            zbz = effective_roughness(z, f, lb)
            zcl = effective_roughness(z, f, blending_height(ClaussenBlendingHeight(3140.0), z, f))
            zma = effective_roughness(z, f, blending_height(MasonBlendingHeight(3140.0), z, f))
            push!(err[:bz], log(zbz / z0les)); push!(err[:cl], log(zcl / z0les))
            push!(err[:ma], log(zma / z0les)); push!(err[:lm], log(_bh_logmean(z, f) / z0les))
            push!(err[:hb], lb / hbles - 1)
            @printf("    %s  z0e LES %.3f  BZ %.4f (%+5.1f%%)  Claussen %.4f  Mason %.4f  log-mean %.4f | hb LES %3.0f  BZ %5.1f m\n",
                    name, z0les, zbz, 100 * (zbz / z0les - 1), zcl, zma, _bh_logmean(z, f), hbles, lb)
            # Bou-Zeid et al. report the model within 25 % of their LES.
            @test abs(zbz / z0les - 1) < 0.15
            @test abs(zcl / z0les - 1) < 0.15
        end
        rms(v) = sqrt(sum(abs2, v) / length(v))
        @printf("    rms ln-error of z0e: BZ %.3f  Claussen %.3f  Mason %.3f  log-mean %.3f;  max |hb error| %.2f\n",
                rms(err[:bz]), rms(err[:cl]), rms(err[:ma]), rms(err[:lm]), maximum(abs, err[:hb]))
        # The blending-height models beat the log-average, which is biased
        # low in every case (the Bou-Zeid et al. and Taylor-1987 finding).
        @test rms(err[:bz]) < rms(err[:lm]) / 2
        @test rms(err[:cl]) < rms(err[:lm]) / 2
        @test all(<(0), err[:lm])
        # The blending height itself is the right size (within 35 %) but its
        # trend with the smooth fraction is not reproduced (see PR).
        @test maximum(abs, err[:hb]) < 0.35
    end

    @testset "Internal-boundary-layer growth" begin
        # Savelyev & Taylor (2005, Sect. 2, Fig. 3): for the Jegede & Foken
        # (1999) roughness changes, the published formulae give IBL heights
        # of 8-34 m at 300 m fetch, Elliott (1958) the highest in both cases.
        for (zu, zd) in ((0.008, 0.032), (0.053, 0.014))
            δe = ibl_height(ElliottIBL(), 300.0; z0_up = zu, z0_down = zd)
            δw = ibl_height(PowerLawIBL(), 300.0; z0_up = zu, z0_down = zd)
            δd = ibl_height(DiffusionIBL(), 300.0; z0_up = zu, z0_down = zd)
            @printf("    IBL at 300 m, z0 %.3f → %.3f: Elliott %.1f  Wood %.1f  Panofsky-Dutton %.1f m\n",
                    zu, zd, δe, δw, δd)
            @test all(8 .<= (δe, δw, δd) .<= 34.5)
            @test δe >= max(δw, δd)
        end
        # Elliott: explicit formula; a smooth-to-rough step grows a deeper
        # layer at the same fetch than rough-to-smooth onto the same z0D.
        @test ibl_height(ElliottIBL(), 100.0; z0_up = 0.01, z0_down = 0.1) ≈
              0.1 * (0.75 - 0.03log(10)) * 1000^0.8 rtol = 1e-14
        # Diffusion analogy: δ(0) = z0D, the implicit law holds to rounding
        # and dδ/dx = Cκ/ln(δ/z0D) (Panofsky & Dutton 1984).
        d = DiffusionIBL()
        @test ibl_height(d, 0.0; z0_up = 0.1, z0_down = 0.05) == 0.05
        worst, worst_ode = 0.0, 0.0
        for x in (1e-3, 0.1, 1.0, 10.0, 300.0, 1e4, 1e6), z0 in (1e-4, 0.03, 1.0)
            δ = ibl_height(d, x; z0_up = 0.1, z0_down = z0)
            y = δ / z0
            worst = max(worst, abs(y * (log(y) - 1) + 1 - 1.25κ_bh * x / z0) / max(1.25κ_bh * x / z0, 1.0))
            h = 1e-6x
            dδ = (ibl_height(d, x + h; z0_up = 0.1, z0_down = z0) -
                  ibl_height(d, x - h; z0_up = 0.1, z0_down = z0)) / 2h
            worst_ode = max(worst_ode, _bh_rel(dδ, 1.25κ_bh / log(y)))
        end
        @test worst < 1e-12
        @test worst_ode < 1e-5
        # Rule of thumb δ ≈ x/10 (Brutsaert 1998, quoted by Bou-Zeid et al.
        # 2007) at a kilometre fetch over grassland-to-crop roughness.
        @test 0.05 < ibl_height(d, 1000.0; z0_up = 0.01, z0_down = 0.1) / 1000 < 0.2
        @test_throws ArgumentError ibl_height(d, -1.0; z0_up = 0.1, z0_down = 0.1)

        # IBL-based blending height: the deepest pairwise IBL at 2L.
        z, f = [0.01, 0.1, 0.5], [0.3, 0.3, 0.4]
        bh = IBLBlendingHeight(ElliottIBL(), 1000.0)
        pairs = [ibl_height(ElliottIBL(), 2000.0; z0_up = z[j], z0_down = z[i]) for i in 1:3, j in 1:3 if i != j]
        @test blending_height(bh, z, f) == maximum(pairs)
        @test blending_height(bh, [0.1], [1.0]) == ibl_height(ElliottIBL(), 2000.0; z0_up = 0.1, z0_down = 0.1)
        @test blending_height(IBLBlendingHeight(DiffusionIBL(), 1000.0), z, [0.5, 0.5, 0.0]) ==
              blending_height(IBLBlendingHeight(DiffusionIBL(), 1000.0), z[1:2], [0.5, 0.5])
    end

    @testset "Blended forcing: neutral limit is exactly z0_eff" begin
        # ψ ≡ 0, all tiles at one skin temperature and saturated (β = 1):
        # every tile has the same surface-to-l_b scalar difference, so the
        # blended cell must exchange exactly like one surface with the
        # effective roughness lengths forced at z.
        tiles = [BHTestSlab(T = 296.0, β = 1.0, z0m = 0.01, z0h = 1e-3),
                 BHTestSlab(T = 296.0, β = 1.0, z0m = 0.8, z0h = 0.02),
                 BHTestSlab(T = 296.0, β = 1.0, z0m = 0.1, z0h = 1e-4)]
        m = SurfaceMosaic(tiles, [0.5, 0.2, 0.3])
        f = SurfaceForcing(T_air = 290.0, q_air = 0.006, U = 8.0, z_ref = 60.0)
        worst = 0.0
        for lb in (2.0, 10.0, 30.0, 59.0)
            sol = blended_forcing(m, f, lb; sf = BHTestNeutral())
            @test sol.converged && sol.blended
            e = effective_roughness(m, lb)
            u★ = κ_bh * f.U / log(f.z_ref / e.z0m)
            θ★ = κ_bh * (_θ(f) - 296.0) / log(f.z_ref / e.z0h)
            q★ = κ_bh * (f.q_air - saturation_specific_humidity(296.0, f.p)) / log(f.z_ref / e.z0q)
            worst = max(worst, _bh_rel(sol.u_star, u★), _bh_rel(sol.θ_star, θ★), _bh_rel(sol.q_star, q★))
            # and the blended wind sits on the log profile of z0_eff
            @test sol.U_b ≈ f.U * log(lb / e.z0m) / log(f.z_ref / e.z0m) rtol = 1e-8
        end
        @printf("    neutral blended vs z0_eff closed form: max rel. error %.1e\n", worst)
        @test worst < 1e-8
    end

    @testset "Blended forcing: single tile, stability, consistency" begin
        # One tile split at l_b into two surface layers vs the tile forced
        # directly at z. Kinematic fluxes are compared (ρ differs between
        # l_b and z; H is divided by ρ c_p, c_p moist). With ψ ≡ 0 the split
        # is exact. With stability it is
        # exact only up to the level at which MOST takes its reference θv
        # (the contract uses the forcing level, so l_b for the lower layer
        # and z for the direct solve): O(Δθ/θ) in L, which in the stable
        # case is amplified by ψ. Verified by temporarily fixing θv_ref =
        # 300 K in both solvers: the BDBH errors then drop to 3e-11 and
        # 1e-13. β = 0 and 1 only: the β hook defines
        # q_s from q_a at the forcing level, so a 0 < β < 1 tile is itself
        # level-dependent.
        errs = Dict{Symbol,Float64}()
        for (sf, key) in ((BHTestNeutral(), :neutral), (Aeolus.SurfaceExchange.BusingerDyerBeljaarsHoltslag(), :bdbh))
            for β in (0.0, 1.0), (Ts, Ta) in ((300.0, 290.0), (285.0, 293.0), (293.0, 293.0))
                s = BHTestSlab(T = Ts, β = β)
                f = SurfaceForcing(T_air = Ta, q_air = 0.007, U = 4.0, z_ref = 40.0)
                direct = turbulent_fluxes(s, f; sf = sf)
                sol = blended_forcing(SurfaceMosaic([s], [1.0]), f, 8.0; sf = sf)
                @test sol.converged
                ρb = air_density(sol.T_b, f.p, PhasePartition(sol.q_b))
                ρz = air_density(f.T_air, f.p, PhasePartition(f.q_air))
                b = turbulent_fluxes(s, sol.forcings[1]; sf = sf)
                e = _bh_rel(b.τ / ρb, direct.τ / ρz)
                cpb = heat_capacity_moist(PhasePartition(sol.q_b))
                cpz = heat_capacity_moist(PhasePartition(f.q_air))
                abs(direct.H) > 1 && (e = max(e, _bh_rel(b.H / (ρb * cpb), direct.H / (ρz * cpz))))
                abs(direct.E) > 1e-7 && (e = max(e, _bh_rel(b.E / ρb, direct.E / ρz)))
                k = Symbol(key, Ts < Ta ? :_stable : :_other)
                errs[k] = max(get(errs, k, 0.0), e)
            end
        end
        @printf("    single tile split at l_b vs forced at z: ψ≡0 %.1e; BDBH unstable/neutral %.1e, stable %.1e\n",
                max(errs[:neutral_stable], errs[:neutral_other]), errs[:bdbh_other], errs[:bdbh_stable])
        @test max(errs[:neutral_stable], errs[:neutral_other]) < 1e-8
        @test errs[:bdbh_other] < 2e-3
        @test errs[:bdbh_stable] < 0.03

        # Heterogeneous, unstable and stable: converged, the blended layer
        # carries exactly the tile-aggregate flux, profile residual tiny.
        tiles() = [BHTestSlab(T = 305.0, β = 0.1, z0m = 0.05, z0h = 0.005),
                   BHTestSlab(T = 295.0, β = 1.0, z0m = 1.0, z0h = 0.1)]
        for (Ta, U) in ((292.0, 3.0), (310.0, 6.0), (292.0, 0.5))
            m = SurfaceMosaic(tiles(), [0.6, 0.4])
            f = SurfaceForcing(T_air = Ta, q_air = 0.008, U = U, z_ref = 50.0)
            sol = blended_forcing(m, f, 15.0)
            @test sol.converged
            @test sol.residual < 1e-10
            agg = turbulent_fluxes(m, sol.forcings)
            ρ = air_density(sol.T_b, f.p, PhasePartition(sol.q_b))
            @test sol.u_star ≈ sqrt(agg.τ / ρ) rtol = 1e-12
            @test sign(sol.L) == sign(Ta - 300.0)       # cell-mean skin ≈ 301 K
        end

        # l_b ≥ z: no blending, identical to the plain mosaic.
        m = SurfaceMosaic(tiles(), [0.6, 0.4])
        f = SurfaceForcing(T_air = 292.0, U = 5.0, z_ref = 10.0)
        sol = blended_forcing(m, f, 12.0)
        @test !sol.blended
        a, b = turbulent_fluxes(m, sol.forcings), turbulent_fluxes(m, f)
        @test (a.τ, a.H, a.LE) == (b.τ, b.H, b.LE)
        # l_b → z⁻: continuous with the plain mosaic.
        d(lb) = _bh_rel(turbulent_fluxes(m, blended_forcing(m, f, lb).forcings).H, b.H)
        @test d(9.99) < d(9.9) < d(9.0)
        @test d(9.99) < 1e-3
        @test_throws ArgumentError blended_forcing(m, f, 0.5)     # l_b < z0 of the rough tile
    end

    @testset "BlendedMosaic: contract, conservation, time stepping" begin
        mk() = SurfaceMosaic([BHTestSlab(T = 300.0, C = 1e5, β = 0.2, z0m = 0.02, z0h = 0.002),
                              BHTestSlab(T = 296.0, C = 3e5, β = 0.9, z0m = 0.6, z0h = 0.06)],
                             [0.55, 0.45])
        bh = BouZeidBlendingHeight(2000.0)
        b = BlendedMosaic(mk(), bh)
        @test b isa AbstractSurface
        @test last_blending(b) === nothing
        lb = blending_height(b)
        @test lb == blending_height(bh, b.mosaic)
        @test roughness_lengths(b) == effective_roughness(b.mosaic, lb)
        @test surface_temperature(b) == surface_temperature(b.mosaic)
        @test surface_albedo(b) == surface_albedo(b.mosaic)
        @test ntiles(b) == 2 && mosaic_fractions(b) == mosaic_fractions(b.mosaic)

        # Energy conservation over a diurnal cycle: Σ f C ΔT = Σ G Δt.
        dt, stored, nconv, nstep = 300.0, 0.0, 0, 0
        e0 = sum(w * t.C * t.T for (t, w) in zip(b.mosaic.tiles, b.mosaic.fractions))
        for k in 0:287
            sw = max(0.0, 700sin(2π * (k * dt / 86400 - 0.25)))
            f = SurfaceForcing(sw_down = sw, lw_down = 330.0, T_air = 294.0 + 3sin(2π * (k * dt / 86400 - 0.35)),
                               q_air = 0.008, U = 4.0, z_ref = 80.0)
            fl = step_surface!(b, f, dt)
            stored += fl.G * dt
            nconv += last_blending(b).converged; nstep += 1
            @test fl.Rn ≈ fl.H + fl.LE + fl.G rtol = 1e-12 atol = 1e-9
            @test last_tile_fluxes(b) == last_tile_fluxes(b.mosaic)
        end
        e1 = sum(w * t.C * t.T for (t, w) in zip(b.mosaic.tiles, b.mosaic.fractions))
        @printf("    diurnal cycle: %d/%d blended solves converged; |ΣfCΔT − ΣGΔt| / ΣGΔt = %.1e\n",
                nconv, nstep, abs(e1 - e0 - stored) / max(abs(stored), 1.0))
        @test nconv == nstep
        @test abs(e1 - e0 - stored) <= 1e-9 * max(abs(stored), sum(abs, (e0, e1)) * 1e-6)

        # Single tile: a BlendedMosaic is the bare tile (to the l_b vs z
        # difference in air density).
        s1, s2 = BHTestSlab(T = 299.0), BHTestSlab(T = 299.0)
        bm = BlendedMosaic([s1], [1.0], FixedBlendingHeight(5.0))
        f = SurfaceForcing(sw_down = 500.0, T_air = 293.0, U = 3.0, z_ref = 30.0)
        for _ in 1:20
            step_surface!(bm, f, 600.0); step_surface!(s2, f, 600.0)
        end
        d1, d2 = s1.T - 299.0, s2.T - 299.0
        @printf("    single-tile BlendedMosaic vs bare tile after 20 steps: warming %.3f vs %.3f K\n", d1, d2)
        @test abs(d1 - d2) < 0.01 * abs(d2)

        # Neutral stress of a blended cell equals that of z0_eff; forcing
        # the tiles at z instead (plain mosaic) gives a different stress.
        m = mk()
        for t in m.tiles; t.T = _θ(SurfaceForcing(T_air = 295.0, z_ref = 50.0)); t.β = 0.0; end
        f = SurfaceForcing(T_air = 295.0, U = 6.0, z_ref = 50.0, q_air = 0.0)
        bn = BlendedMosaic(m, FixedBlendingHeight(20.0))
        τb = turbulent_fluxes(bn, f; sf = BHTestNeutral()).τ
        e = effective_roughness(m, 20.0)
        ρb = air_density(last_blending(bn).T_b, f.p, PhasePartition(0.0))
        @test τb ≈ ρb * (κ_bh * 6.0 / log(50.0 / e.z0m))^2 rtol = 1e-8
        τp = turbulent_fluxes(m, f; sf = BHTestNeutral()).τ
        @printf("    neutral cell stress: blended (l_b = 20 m) %.4f, plain mosaic at z %.4f N m⁻²\n", τb, τp)
        @test abs(τb - τp) / τp > 1e-3
    end
end
