# P55-f (#191), ONE-g (#456): GABLS1 (Beare et al. 2006, BLM 118, 247) on
# the one model — the configuration examples/sbl/gabls1.toml builds the
# protocol from its components: the geostrophic forcing (`LargeScale`), the
# Monin-Obukhov surface layer (`SurfaceLayer`), the retargeted sponge and
# Smagorinsky with the Brown et al. (1994) stability correction in the
# stress form. The reduced run (32³, 1.25 h: θ budget, u*, heat flux, the
# correction in the SBL) is in validation/era6_gabls1.jl.

using Test
using Aeolus
using Aeolus.Components: Components, LargeScale, SurfaceLayer, Sponge, SGS
using Aeolus.Fields: interior_view
using Statistics: mean

# gabls1_domain, _comp: shared with validation/era6_gabls1.jl
isdefined(@__MODULE__, :gabls1_domain) || include(joinpath(@__DIR__, "support", "sbl_domains.jl"))

@testset "GABLS1 on the one model: configuration (P55-f, ONE-g)" begin

@testset "The configuration builds the GABLS1 protocol from components" begin
    s = BusingerDyerLogLinear(β_m = 4.8, β_h = 7.8)
    @test phi_m(s, 0.1) ≈ 1.48 && phi_h(s, 0.1) ≈ 1.78     # GABLS1: β_m = 4.8, β_h = 7.8
    @test psi_m(s, 0.5) ≈ -2.4 && psi_h(s, 0.5) ≈ -3.9
    # the momentum wall law with the Obukhov length of the heat-flux solve
    # returns the solver's u* (one Monin-Obukhov solution)
    sol = solve_surface_layer(s; U = 7.3, z_m = 6.25, θ_a = 264.8, θ_s = 263.9, z0m = 0.1, z0h = 0.1)
    @test sol.L > 0
    @test values_agree(friction_velocity(MOSTWallLaw(s, 0.1, sol.L), 7.3, 6.25), sol.u_star)
    @test kinematic_heat_flux(sol) == -sol.u_star * sol.θ_star < 0
    d = gabls1_domain(8)
    @test Components.core(d) === :boussinesq && d.scheme === :rk3 && d.cfl.C_advective == 1.2
    @test d.state.f_coriolis == 0                                # the Coriolis term is a component
    @test _comp(d, Components.Coriolis).f == 1.39e-4
    ls = _comp(d, LargeScale)
    @test ls.geostrophic isa GeostrophicPressureGradient && ls.U_g == 8.0 && ls.V_g == 0.0 && isempty(ls.extra)
    sl = _comp(d, SurfaceLayer)
    @test surface_temperature(sl.Ts, 3600.0) ≈ 265 - 0.25           # 0.25 K/h
    @test sl.sf === s && sl.z0m == sl.z0h == 0.1
    @test Components.slot(sl) === :surface
    sp = _comp(d, Sponge)
    @test sp.kind === :geostrophic && sp.z_start == 300.0 && sp.α_max == 0.01
    c = _comp(d, SGS)
    @test c.closure isa StabilityCorrected && c.closure.scheme isa BrownEtAl1994 && c.stress === :symmetric
    @test _comp(gabls1_domain(8, "none"), SGS).closure isa StaticSmagorinsky
    # the initial state: θ of the protocol, ±0.1 K noise below 50 m, u = U_g
    θ = interior_view(d.state.θ); m = d.state.mesh
    zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    for k in 1:m.Nz
        θ0 = zc[k] < 100 ? 265.0 : 265.0 + 0.01 * (zc[k] - 100)
        dev = maximum(abs, θ[:, :, k] .- θ0)
        zc[k] < 50 ? (@test 0 < dev <= 0.1) : (@test dev < 1e-12)
    end
    @test all(==(8.0), interior_view(d.state.u)) && all(iszero, interior_view(d.state.v))
    # BL height: 5 % of the surface stress, linearly interpolated, / 0.95
    @test stress_bl_height([0.0, 10.0, 20.0, 30.0], [1.0, 0.5, 0.02, 0.0]) ≈
          (10 + (0.05 - 0.5) / (0.02 - 0.5) * 10) / 0.95
end

@testset "The retargeted sponge relaxes toward the geostrophic wind" begin
    # a sponge-only layer at rest relaxes toward U_g: one Euler step of the
    # sponge component alone adds α(z) U_g to tu in the sponge, nothing below
    d = gabls1_domain(8)
    s = d.state; sp = _comp(d, Sponge); ls = _comp(d, LargeScale)
    interior_view(s.u) .= 0; interior_view(s.v) .= 0; interior_view(s.w) .= 0
    fill!(s.tu.data, 0); fill!(s.tv.data, 0); fill!(s.tw.data, 0)
    Components.run_stage!(ls, d, Val(:clock), 1.0)
    Components.run_stage!(sp, d, Val(:body_force), 1.0)
    m = s.mesh; z = [(k - 0.5) * m.dz for k in 1:m.Nz]
    tu = [mean(interior_view(s.tu)[:, :, k]) for k in 1:m.Nz]
    for k in 1:m.Nz
        α = z[k] >= 300 ? 0.01 * sin(0.5π * (z[k] - 300) / 100)^2 : 0.0
        @test tu[k] ≈ α * 8.0 atol = 1e-14
    end
    @test all(iszero, interior_view(s.tv))
end

end # testset
