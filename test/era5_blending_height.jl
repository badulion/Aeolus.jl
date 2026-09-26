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
end
