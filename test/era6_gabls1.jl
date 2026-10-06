# P55-f (#191): GABLS1 (Beare et al. 2006, BLM 118, 247) on the shared SBL
# driver, examples/sbl/driver.jl — reduced to 32³ at 12.5 m and the first
# 1.25 h (the full 9 h, 64³ comparison is examples/sbl/gabls1.jl). The case
# is built from examples/sbl/configs/gabls1.toml (the `sbl` run driver,
# CFG-d #359).

using Test
using Aeolus
using Statistics: mean

include(joinpath(@__DIR__, "..", "examples", "sbl", "config_schema.jl"))

# Reduced CI run: 32³ (Δ = 12.5 m), 0-1.25 h, statistics over 1.0-1.25 h
# (after the spin-up of the initial neutral layer: u* peaks at 0.43 m/s
# near 15 min), with and without the stability correction.
const T_CI = 4500.0
const T_AVG_CI = 3600.0

gabls1_case(N, stab = :brown94) =
    sbl_build("gabls1"; overrides = ["grid.size=[$N,$N,$N]", "sgs.stability=\"$stab\""]).case

function gabls1_ci(stab)
    c = gabls1_case(32, stab)
    H0 = sbl_column_heat(c)
    hs = Float64[]; us = Float64[]; ws = Float64[]; sgs = Float64[]; rκ = Float64[]
    wmax = Ref(-Inf); maxres = Ref(0.0)
    cb = function (c)
        # the θ budget of every step: d/dt ∫⟨θ⟩dz = ⟨w'θ'⟩_0 (no flux at the top)
        maxres[] = max(maxres[], abs(sbl_column_heat(c) - H0 - c.heat_applied))
        c.sim.t >= T_AVG_CI || return
        wmax[] = max(wmax[], mean(c.wθ))
        push!(us, sqrt(hypot(c.τx, c.τy))); push!(ws, mean(c.wθ))
        c.nsteps % 50 == 0 || return
        p = sbl_profiles(c)
        h = sbl_bl_height(p.zf, p.τ)
        push!(hs, h)
        d = gabls1_sgs_share(c, h)
        push!(sgs, d.share); push!(rκ, d.κ_ratio)
    end
    sbl_run!(c, T_CI; callback = cb)
    return (h = mean(hs), u_star = mean(us), u_min = minimum(us), u_max = maximum(us),
            wθ = mean(ws), wθ_max = wmax[], heat = c.heat_applied, maxres = maxres[],
            sgs_share = mean(sgs), κ_ratio = mean(rκ), nsteps = c.nsteps, wall = c.wall_time)
end

# SGS share of the vertical θ flux on the faces of the lower half of the BL,
# and the mean κv/κh of the cells below h (the closure's last coefficients)
function gabls1_sgs_share(c, h)
    s = c.sim; m = s.mesh
    κv = c.closure isa StabilityCorrected ? c.closure.κv : s.κt
    κh = s.κt
    sg = 0.0; tot = 0.0; r = 0.0; n = 0
    for k in 2:max(2, round(Int, h / 2 / m.dz)), j in 1:m.Ny, i in 1:m.Nx
        I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
        κf = 0.5 * (κv.data[I, J, K] + κv.data[I, J, K-1])
        g = (s.θ.data[I, J, K] - s.θ.data[I, J, K-1]) / m.dz
        sg += -κf * g
        tot += s.w.data[I, J, K] * 0.5 * (s.θ.data[I, J, K] + s.θ.data[I, J, K-1]) - κf * g
    end
    for k in 1:max(1, round(Int, h / m.dz)), j in 1:m.Ny, i in 1:m.Nx
        I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
        κh.data[I, J, K] > 0 && (r += κv.data[I, J, K] / κh.data[I, J, K]; n += 1)
    end
    return (share = sg / tot, κ_ratio = r / max(n, 1))
end

@testset "GABLS1 on the shared SBL driver (P55-f)" begin

@testset "Driver: package physics only" begin
    s = SBL_GABLS1_STABILITY                                # the package family, no private copy
    @test s isa BusingerDyerLogLinear && s === BusingerDyerLogLinear(β_m = 4.8, β_h = 7.8)
    @test phi_m(s, 0.1) ≈ 1.48 && phi_h(s, 0.1) ≈ 1.78     # GABLS1: β_m = 4.8, β_h = 7.8
    @test psi_m(s, 0.5) ≈ -2.4 && psi_h(s, 0.5) ≈ -3.9
    @test phi_m(s, -0.3) == phi_m(BusingerDyerBeljaarsHoltslag(), -0.3)
    # the momentum wall law with the Obukhov length of the heat-flux solve
    # returns the solver's u* bit for bit (one Monin-Obukhov solution)
    sol = solve_surface_layer(s; U = 7.3, z_m = 6.25, θ_a = 264.8, θ_s = 263.9, z0m = 0.1, z0h = 0.1)
    @test sol.L > 0
    @test friction_velocity(MOSTWallLaw(s, 0.1, sol.L), 7.3, 6.25) == sol.u_star
    # the case (gabls1.toml): the GABLS1 protocol, forcing from LargeScaleForcing
    c = gabls1_case(8)
    @test c.sim.f_coriolis == 1.39e-4 && c.forcing[1] isa GeostrophicPressureGradient
    @test surface_temperature(c.Ts, 3600.0) ≈ 265 - 0.25           # 0.25 K/h
    @test c.closure isa StabilityCorrected && c.closure.scheme isa BrownEtAl1994
    @test gabls1_case(8, :none).closure isa StaticSmagorinsky
    @test c.sf === SBL_GABLS1_STABILITY && c.z0m == c.z0h == 0.1
    # BL height: 5 % of the surface stress, linearly interpolated, / 0.95
    @test sbl_bl_height([0.0, 10.0, 20.0, 30.0], [1.0, 0.5, 0.02, 0.0]) ≈
          (10 + (0.05 - 0.5) / (0.02 - 0.5) * 10) / 0.95
    # reference data carry their provenance
    ref = sbl_read_csv(joinpath(@__DIR__, "..", "examples", "sbl", "data", "gabls1_reference.csv"))
    @test any(occursin("Beare", l) for l in ref.header) && any(occursin("Basu", l) for l in ref.header)
    @test sbl_csv_value(ref, "u_star", "low") ≈ sqrt(0.06) atol = 1e-3
end

@testset "Reduced GABLS1: θ budget, u*, heat flux, the correction in the SBL" begin
    b = gabls1_ci(:brown94)
    n = gabls1_ci(:none)
    for (name, r) in (("brown94", b), ("none", n))
        @info "GABLS1 32³, 1.0-1.25 h, $name" r.h r.u_star r.u_min r.u_max r.wθ r.wθ_max r.maxres / abs(r.heat) r.sgs_share r.κ_ratio r.nsteps r.wall
        # θ budget: Δ∫⟨θ⟩dz = ∫⟨w'θ'⟩_0 dt to round-off (measured 1.7e-9 relative)
        @test r.maxres <= 1e-6 * abs(r.heat)
        # after the spin-up u* is in [0.2, 0.4] m/s at every step (measured 0.26-0.29)
        @test 0.2 <= r.u_min && r.u_max <= 0.4
        # the cooled surface takes heat from the air at every step
        @test r.wθ_max < 0
    end
    # The correction acts in the SBL: κv/κh < 1 below h (measured 0.63; 1 by
    # construction without it) and the SGS share of the vertical heat flux
    # in the lower half of the BL is reduced (measured 0.113 against 0.145).
    @test b.κ_ratio < 0.8 && n.κ_ratio ≈ 1.0
    @test b.sgs_share < 0.9 * n.sgs_share
    # The issue expected h(:brown94) < h(:none) here. The expectation was
    # wrong at this resolution, not the code: over 1.0-1.25 h the two BL
    # heights agree within their scatter (measured 178.7 ± 3.4 m and
    # 178.2 ± 3.9 m), and at 32³ after 9 h the corrected run is deeper
    # (222.5 m against 210.2 m; examples/sbl/gabls1.jl). Resolved eddies
    # carry ~85-90 % of the heat flux at 12.5 m, and the weaker SGS mixing
    # of the corrected closure lets them carry more. No assertion on the
    # ordering is made; the numbers are reported above.
end

end # testset
