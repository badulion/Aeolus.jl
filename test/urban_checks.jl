# The urban site's checks of the nest parent's helpers on synthetic setups
# (formerly part of the by-hand examples/urban/berlin_checks.jl, #320):
#
#   zi      the parent's boundary-layer height (`parent_zi`) and its heat-flux
#           profile (`mean_heat_flux`) on a synthetic unstepped state with a
#           known flux minimum;
#   forcing the synthetic offline forcing of the parent surface
#           (`parent_offline_forcing`): the clear-sky day in closed form.
#
# The Berlin smoke matrix, the parent smoke run and the end-to-end validation
# runs (the flagship past step 16, encroachment, Wangara Day 33, the 24 h
# parent mosaic against the child) are in validation/urban_checks.jl; the
# synthetic city's run is test/urban_run.jl.

using Test
using Aeolus
using Aeolus.Config: load_run_config

const BC = Aeolus.Urban
isdefined(@__MODULE__, :berlin_default) || include(joinpath(@__DIR__, "support", "berlin_checks.jl"))

"""
A one-domain model of the core (8 × 8 × 16, Δz = 100 m, the TKE closure),
built and not stepped: the state the `parent_zi` checks write by hand.
"""
function zi_probe_domain()
    text = """
        schema_version = 2
        [run]
        name = "zi_probe"
        [time]
        dt = 1.0
        duration = 10.0
        [grid]
        size = [8, 8, 16]
        extent = [800.0, 800.0, 1600.0]
        halo = [2, 2, 2]
        [dynamics]
        theta0 = 300.0
        [boundaries]
        z = "free_slip"
        [initial]
        theta = {kind = "constant", value = 300.0}
        u = {kind = "constant", value = 0.0}
        [sgs]
        model = "tke"
        [output]
        variables = []
        """
    m = build_config_model(load_run_config(IOBuffer(text); overrides = ["fftw.planner=\"estimate\""]))
    return only(m.domains)
end

"zi: the parent's boundary-layer height on a synthetic state with a known heat-flux minimum."
function parent_zi_checks()
    @testset "parent_zi and mean_heat_flux on a synthetic state" begin
        d = zi_probe_domain()
        s = d.state; msh = s.mesh
        Hx, Hy, H, Nx, Ny, Nz, dz = msh.Hx, msh.Hy, msh.Hz, msh.Nx, msh.Ny, msh.Nz, msh.dz
        I, J = Hx+1:Hx+Nx, Hy+1:Hy+Ny
        # θ' = a(x), w = h(z) a(x) with a = cos(2πx/L): plane mean 0, mean(a²) = 1/2,
        # so the resolved flux at face k is h(k)/2 exactly; the minimum, −0.15
        # K m/s, is at face 6 (500 m). θ: mixed (300 K) up to 700 m, 5 K/km above.
        a = [cos(2π * (i - 0.5) / Nx) for i in 1:Nx, j in 1:Ny]
        hk(k) = k == 6 ? -0.3 : 0.1 * sin(π * k / Nz)
        θbar(k) = 300.0 + max(0.0, 5e-3 * ((k - 0.5) * dz - 650.0))
        fill!(s.w.data, 0.0)
        for k in 1:Nz
            s.θ.data[I, J, H+k] .= θbar(k) .+ a
        end
        for k in 2:Nz
            s.w.data[I, J, H+k] .= hk(k) .* a
        end
        κ = Aeolus.Operators.sgs_fields(Aeolus.Components.sgs_closure(d), s).κv
        fill!(κ.data, 0.0)                     # no SGS part: the resolved flux alone
        zf, fl = BC.mean_heat_flux(s, κ)
        @test zf ≈ [(k - 1) * dz for k in 2:Nz] rtol = 1e-14
        @test maximum(abs.(fl .- [hk(k) / 2 for k in 2:Nz])) < 1e-12     # measured 3e-15
        @test BC.parent_zi(d) == 500.0                                  # the face of the minimum
        @test BC.parent_zi(d; method = :min_flux) == BC.parent_zi(d)
        # :parcel is the parcel height of the plane-mean θ: the top of the mixed layer
        @test BC.parent_zi(d; method = :parcel) == Aeolus.Components.parcel_zi(s)
        @test BC.parent_zi(d; method = :parcel) ≈ 650.0 atol = 1e-9
        @test_throws ArgumentError BC.parent_zi(d; method = :unknown)
        # the SGS part: w = 0, θ at 3 K/km, κ = 2 m²/s: −κ ∂θ/∂z = −0.006 K m/s at every face
        fill!(s.w.data, 0.0)
        for k in 1:Nz
            s.θ.data[:, :, H+k] .= 300.0 + 3e-3 * (k - 0.5) * dz
        end
        fill!(κ.data, 2.0)
        _, fl2 = BC.mean_heat_flux(s, κ)
        @test maximum(abs.(fl2 .+ 0.006)) < 1e-12                       # measured 1.1e-15
    end
end

@testset "urban run type: the parent's helpers on synthetic setups" begin
    parent_zi_checks()
    PCFG = berlin_default()
    @testset "parent_offline_forcing: the synthetic clear-sky day" begin
        # one evaluation at 13 h (sun up) and at 0 h (Berlin in June: below
        # the horizon); the air temperature T_mean + T_amp sin(2π(t − 9)/24)
        # and Brutsaert's (1975) clear-sky longwave in closed form
        f13 = BC.parent_offline_forcing(PCFG, 13.0)
        f0 = BC.parent_offline_forcing(PCFG, 0.0)
        for f in (f13, f0)
            @test all(isfinite, (f.sw_down, f.lw_down, f.T_air, f.q_air, f.U, f.p))
            @test f.U == 3.0 && f.q_air == 0.008
            @test 9.0e4 < f.p < 1.05e5
            e = f.q_air * f.p / (0.622 + 0.378 * f.q_air) / 100
            @test f.lw_down ≈ 1.24 * (e / f.T_air)^(1 / 7) * 5.670374419e-8 * f.T_air^4 rtol = 1e-6
        end
        @test f13.sw_down > 300 && f0.sw_down == 0
        @test f13.T_air ≈ 295 + 6 * sin(2π * 4 / 24) rtol = 1e-14
        @test f0.T_air ≈ 295 + 6 * sin(-2π * 9 / 24) rtol = 1e-14
        @test 250 < f0.lw_down < f13.lw_down < 450
    end
end
