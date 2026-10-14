# Era VI, Phase 56 (P56-b, #182): the Deardorff (1980) 1.5-order prognostic
# SGS TKE closure (src/PrognosticSGS.jl) with the Moeng (1984) coefficients,
# and the wall distance it caps its mixing length with (IBM.wall_distance).
#
# Numbers measured on this branch are quoted next to each assertion.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.IBM: SolidMask, mark_solid!, wall_distance, first_level_cells
using Aeolus.Operators: advance_sgs!, sgs_state, restore_sgs_state!, add_sgs_scalar!
using Aeolus.Simulations: kinetic_energy
using Random: Xoshiro
using Statistics: mean
using Printf

# the CBL harness tke_cbl (a test fixture; the model's configuration is examples/cbl_tke/cbl_tke.toml)
isdefined(@__MODULE__, :TKECBL) || include(joinpath(@__DIR__, "fixtures", "tke_cbl.jl"))
using .TKECBL: tke_cbl, tke_cbl_summary
isdefined(@__MODULE__, :moist_adiabat_column!) ||
    include(joinpath(@__DIR__, "support", "moist_adiabat.jl"))

@testset "Era VI — prognostic SGS TKE (P56-b, #182)" begin

@testset "wall_distance: vector distance transform" begin
    # floor only: d = Δz/2 + (k-1)Δz exactly
    m = RegularGrid(; size = (8, 8, 12), extent = (16.0, 16.0, 6.0))
    d = wall_distance(SolidMask(m); bottom = true)
    @test size(d) == (8, 8, 12)
    @test all(d[i, j, k] == 0.25 + (k - 1) * 0.5 for i in 1:8, j in 1:8, k in 1:12)
    # floor and lid: the nearer of the two
    d2 = wall_distance(SolidMask(m); bottom = true, top = true)
    @test all(d2[1, 1, k] ≈ min(0.25 + (k - 1) * 0.5, 0.25 + (12 - k) * 0.5) for k in 1:12)
    # no surface at all: infinite
    @test all(isinf, wall_distance(SolidMask(m); bottom = false))

    # a cube on a 2:2:1 grid, against the Euclidean distance to its box
    m = RegularGrid(; size = (24, 24, 24), extent = (48.0, 48.0, 24.0))
    sm = SolidMask(m)
    mark_solid!(sm, (x, y, z) -> 16 < x < 32 && 16 < y < 32 && z < 12)
    d = wall_distance(sm; bottom = false, periodic = (false, false, false))
    err = 0.0
    nsolid = 0
    for k in 1:24, j in 1:24, i in 1:24
        if sm.mask[m.Hx+i, m.Hy+j, m.Hz+k]
            nsolid += 1
            @test d[i, j, k] == 0
            continue
        end
        x, y, z = (i - 0.5) * 2, (j - 0.5) * 2, (k - 0.5)
        ex = sqrt(max(16 - x, 0, x - 32)^2 + max(16 - y, 0, y - 32)^2 + max(z - 12, 0)^2)
        err = max(err, abs(d[i, j, k] - ex) / ex)
    end
    @test nsolid == 8 * 8 * 12
    @test err < 1e-12                       # measured 0.0 (a plain chamfer: 0.236)

    # periodic seam: a wall at the east edge is seen from the west edge
    m = RegularGrid(; size = (16, 4, 4), extent = (16.0, 4.0, 4.0))
    sm = SolidMask(m)
    mark_solid!(sm, (x, y, z) -> x > 15)
    d = wall_distance(sm; bottom = false)
    @test d[1, 2, 2] ≈ 0.5 && d[3, 2, 2] ≈ 2.5 && d[14, 2, 2] ≈ 1.5
    dnp = wall_distance(sm; bottom = false, periodic = (false, false, false))
    @test dnp[1, 2, 2] ≈ 14.5

    # first level above a surface: level 1 and the cells above a roof
    m = RegularGrid(; size = (8, 8, 8), extent = (8.0, 8.0, 8.0))
    sm = SolidMask(m)
    mark_solid!(sm, (x, y, z) -> x < 4 && z < 3)
    f = first_level_cells(sm)
    @test all(f[i, j, 4] for i in 1:4, j in 1:8)           # above the roof (3 solid levels)
    @test all(f[i, j, 1] for i in 5:8, j in 1:8)           # on the floor
    @test count(f) == 4 * 8 + 4 * 8
    @test !any(f[i, j, 1] for i in 1:4, j in 1:8)          # solid cells are not fluid
end


# a box with linear shear u = S z and θ = θ₀ + Γ z between free-slip walls
function shear_box(S, Ri; Nz = 16, Δ = 10.0)
    m = RegularGrid(; size = (4, 4, Nz), extent = (4Δ, 4Δ, Nz * Δ))
    sim = BoussinesqSimulation(m; θ₀ = 300.0, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                               θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    set!(sim.u, (x, y, z) -> S * z)
    Γ = Ri * S^2 * 300.0 / 9.81
    set!(sim.θ, (x, y, z) -> 300.0 + Γ * z)
    return sim
end

@testset "1. free decay at l = Δ (c_ε = 0.70)" begin
    m = RegularGrid(; size = (4, 4, 4), extent = (40.0, 40.0, 40.0))
    sim = BoussinesqSimulation(m)
    Δ, e0, T = 10.0, 1.0, 600.0
    @test tke_dissipation_coefficient(Δ, Δ) ≈ 0.70
    exact(t) = (e0^-0.5 + 0.70 * t / (2Δ))^-2           # de/dt = −c_ε e^{3/2}/l
    errs = Dict(:exact => Float64[], :semi_implicit => Float64[])
    for diss in (:exact, :semi_implicit), n in (60, 120, 240, 480)
        c = TKEClosure(m; e0 = e0, dissipation = diss)
        for _ in 1:n
            advance_sgs!(c, sim, T / n)
        end
        push!(errs[diss], maximum(abs.(interior_view(c.e) .- exact(T))) / exact(T))
        @test tke_statistics(c).clips == 0
    end
    @test maximum(errs[:exact]) < 1e-6                     # measured ≤ 1.7e-14
    r = errs[:semi_implicit][1:end-1] ./ errs[:semi_implicit][2:end]
    @test all(x -> 1.9 < x < 2.1, r)                       # first order: 2.03, 2.01, 2.01
    @test errs[:semi_implicit][end] < 0.01                 # 6.2e-3 at Δt = 1.25 s
end

@testset "2. neutral shear equilibrium: C_s = c_k^{3/4} c_ε^{-1/4}" begin
    S, Δ = 0.05, 10.0
    Cs = 0.1^0.75 * 0.70^-0.25
    @test abs(Cs - 0.1944) < 1e-4                          # 0.19441 (the issue's "0.194" is rounded)
    sim = shear_box(S, 0.0)
    c = TKEClosure(sim.mesh; e0 = 1e-3, transport = false)
    for _ in 1:4000
        advance_sgs!(c, sim, 5.0)
    end
    eq = 0.1 / 0.7 * Δ^2 * S^2                            # e = (c_k/c_ε) l² S²
    e = interior_view(c.e)[:, :, 3:14]                     # away from the walls
    @test maximum(abs.(e .- eq)) / eq < 1e-10              # measured 4.7e-15
    @test tke_equilibrium(S^2, 0.0, Δ) ≈ eq rtol = 1e-12
    f = compute_sgs!(c, sim)
    νt = interior_view(sim.νt)[:, :, 3:14]
    Cs_eff = sqrt(mean(νt) / (Δ^2 * S))
    @test abs(Cs_eff - Cs) < 1e-4                          # 0.194413 vs 0.194413
    smag = Field(CellCenter(), sim.mesh)
    for q in (sim.u, sim.v, sim.w)
        fill_halos!(q, sim.bcs)
    end
    compute_smagorinsky_νt!(smag, sim.u, sim.v, sim.w, Cs)
    @test maximum(abs.(νt .- interior_view(smag)[:, :, 3:14])) / maximum(νt) < 1e-6
    # κ_t = (1 + 2l/Δ) ν_t = 3 ν_t at l = Δ
    @test interior_view(sim.κt)[:, :, 3:14] ≈ 3 .* νt rtol = 1e-14
end

@testset "3. critical Richardson number (l → 0 limit)" begin
    Ric = tke_critical_richardson()
    @test Ric ≈ 1 / (1 + 0.19 / (0.1 * 0.76^2)) rtol = 1e-14
    @test abs(Ric - 0.233) < 5e-4                          # 0.23313
    S, Δ = 0.05, 10.0
    for Ri in (0.0, 0.1, 0.2)
        sim = shear_box(S, Ri)
        c = TKEClosure(sim.mesh; e0 = 0.05, transport = false)
        for _ in 1:20000
            advance_sgs!(c, sim, 5.0)
        end
        e = interior_view(c.e)[2, 2, 8]
        root = tke_equilibrium(S^2, Ri * S^2, Δ)
        @test abs(e - root) / root < 1e-6                  # 1.4e-15, 1.1e-12, 4.1e-12
        # and the closed form: below l = Δ, P_s + P_b = ε is linear in x = l/Δ
        if Ri > 0
            a = 0.1 * 0.76^2
            x = (1 / Ri - 1 - 0.19 / a) / (2 + 0.51 / a)
            @test 0 < x < 1
            @test root ≈ (x * Δ * sqrt(Ri) * S / 0.76)^2 rtol = 1e-10
        end
    end
    @test tke_equilibrium(S^2, 0.25 * S^2, Δ) == 0
    sim = shear_box(S, 0.25)
    c = TKEClosure(sim.mesh; e0 = 0.05, transport = false)
    for _ in 1:20000
        advance_sgs!(c, sim, 5.0)
    end
    @test maximum(interior_view(c.e)[:, :, 3:14]) < 1e-20 # decays to 0 (7.5e-28 after 1e5 s)
    @test tke_statistics(c).clips == 0
end

@testset "4. energy exchange with the resolved flow" begin
    Ek(k) = 0.05 * k^4 * exp(-2 * (k / 4)^2)
    function box(; c_ε0 = 0.19, c_ε1 = 0.51, wall = false)
        local m, sim, c
        m = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        sim = wall ? BoussinesqSimulation(m; θ₀ = 300.0,
                                          bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                                          θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux)) :
                     BoussinesqSimulation(m)
        isotropic_random_field!(sim, Ek; rng = Xoshiro(1))
        if wall
            sim.w.data[:, :, m.Hz+1] .= 0
            set!(sim.θ, (x, y, z) -> 300.0 + 3z + 0.1 * sin(3x) * cos(2y))
        end
        c = TKEClosure(m; c_ε0 = c_ε0, c_ε1 = c_ε1)
        interior_view(c.e) .= 0.02 .+ 0.01 .* sin.((1:16) .* 0.4) .* cos.((1:16)' .* 0.3)
        return sim, c
    end
    # Σ P_s V = −Σ u·T_sgs V (the dual), periodic and with walls
    for wall in (false, true)
        sim, c = box(; wall)
        b = kinetic_energy_budget(sim; sgs = c)
        tb = tke_budget(c, sim)
        @test b.sgs < 0
        @test abs(b.sgs + tb.shear) / tb.shear < 1e-12      # 2.0e-16 / 1.5e-15
        if wall
            # Σ P_b V = (g/θ₀) Σ z T_θ V: the potential energy of the SGS heat flux
            f = compute_sgs!(c, sim)
            tθ = Field(CellCenter(), sim.mesh)
            add_sgs_scalar!(tθ, sim.θ, c, f, sim)
            z = reshape([(k - 0.5) * sim.mesh.dz for k in 1:16], 1, 1, :)
            pe = 9.81 / 300.0 * sum(interior_view(tθ) .* z) * sim.mesh.dx^3
            @test tb.buoyancy < 0
            @test abs(tb.buoyancy - pe) / abs(pe) < 1e-12
        end
    end
    # c_ε = 0, ν = 0: d/dt(KE + Σ e V) = 0 semi-discretely ...
    sim, c = box(; c_ε0 = 0.0, c_ε1 = 0.0)
    b = kinetic_energy_budget(sim; sgs = c); tb = tke_budget(c, sim)
    V = sim.mesh.dx^3
    rate = b.total + tb.shear + tb.transport
    @test abs(rate) / (kinetic_energy(sim) + tb.tke) < 1e-10
    # ... and per Euler step the residual is the O(Δt²) time truncation
    function resid(dt)
        local sim, c, E0, E1, r
        sim, c = box(; c_ε0 = 0.0, c_ε1 = 0.0)
        E0 = kinetic_energy(sim) + sum(interior_view(c.e)) * V
        r = 0.0
        for _ in 1:5
            step!(sim, dt; sgs = c)
            E1 = kinetic_energy(sim) + sum(interior_view(c.e)) * V
            r = max(r, abs(E1 - E0) / E0); E0 = E1
        end
        return r
    end
    rs = [resid(dt) for dt in (0.02, 0.01, 0.005)]
    @info @sprintf("  energy-exchange residual per step: %.2e, %.2e, %.2e (Δt = 0.02, 0.01, 0.005)", rs...)
    @test all(x -> 3.5 < x < 4.5, rs[1:end-1] ./ rs[2:end])   # second order per step
    # with c_ε > 0 the total decays at exactly Σ ε V
    sim, c = box()
    b = kinetic_energy_budget(sim; sgs = c); tb = tke_budget(c, sim)
    rate = b.total + tb.shear + tb.buoyancy + tb.transport - tb.dissipation
    @test tb.dissipation > 0
    @test abs(rate + tb.dissipation) / tb.dissipation < 1e-10
end

@testset "5. positivity over 500 steps, zero clips" begin
    Ek(k) = 0.05 * k^4 * exp(-2 * (k / 4)^2)
    for (H, strat) in ((2, false), (3, true))
        m = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (H, H, H))
        sim = strat ? BoussinesqSimulation(m; θ₀ = 300.0,
                                           bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                                           θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux)) :
                      BoussinesqSimulation(m)
        isotropic_random_field!(sim, Ek; rng = Xoshiro(2))
        strat && (sim.w.data[:, :, m.Hz+1] .= 0; set!(sim.θ, (x, y, z) -> 300.0 + 3z))
        c = TKEClosure(m)
        @test c.advection === (H >= 3 ? :weno_z5 : :centered)
        initialize_tke!(c, sim)
        mine, minν, minΔκ = Inf, Inf, Inf
        for _ in 1:500
            step_rk3!(sim, 0.01; sgs = c)
            mine = min(mine, minimum(interior_view(c.e)))
            minν = min(minν, minimum(interior_view(sim.νt)))
            minΔκ = min(minΔκ, minimum(interior_view(sim.κt) .- interior_view(sim.νt)))
        end
        st = tke_statistics(c)
        @test mine >= 0 && minν >= 0 && minΔκ >= 0            # e ≥ 0, ν_t ≥ 0, κ_t ≥ ν_t
        @test st.clips == 0 && st.clipped_energy == 0
        @test st.steps == 500                                  # one commit per RK3 step
        @test abs(st.transport) < 1e-12 * st.dissipation       # flux form, conservative
        strat && @test st.buoyancy < 0 && st.buoyant_destruction > 0
    end
end

@testset "contract: pure coefficients, exact rollback" begin
    Ek(k) = 0.05 * k^4 * exp(-2 * (k / 4)^2)
    function mk()     # (locals renamed: an inner `c = …` would rebind the testset's `c`)
        m_ = RegularGrid(; size = (12, 12, 12), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        s_ = BoussinesqSimulation(m_)
        isotropic_random_field!(s_, Ek; rng = Xoshiro(5))
        c_ = TKEClosure(m_)
        initialize_tke!(c_, s_)
        return s_, c_
    end
    sim, c = mk()
    e0 = copy(c.e.data)
    compute_sgs!(c, sim); compute_sgs!(c, sim)
    @test c.e.data == e0                                   # coefficients change no memory
    snap = sgs_state(c)
    advance_sgs!(c, sim, 0.01)
    @test c.e.data != e0
    restore_sgs_state!(c, snap)
    @test c.e.data == e0 && tke_statistics(c).steps == 0
    # a rejected adaptive step: the first try commits, then is poisoned and
    # rejected; the retry at Δt/2 must equal a direct Δt/2 step bit for bit
    sim, c = mk(); simB, cB = mk()
    tries = Ref(0)
    stepper = (s, h; kw...) -> begin
        step!(s, h; kw..., sgs = c)
        tries[] += 1
        tries[] == 1 && (s.u.data[5, 5, 5] = NaN)
        s
    end
    _, dt_taken, nretry = step_adaptive!(sim, CFLConstraints(); dt = 0.02, stepper = stepper, sgs = c)
    step!(simB, 0.01; sgs = cB)
    @test nretry == 1 && dt_taken == 0.01
    @test c.e.data == cB.e.data
    @test interior_view(sim.u) == interior_view(simB.u) && interior_view(sim.θ) == interior_view(simB.θ)
    @test tke_statistics(c).steps == 1
end

@testset "6. reduced CBL: 24³, Δ = 50 m, Q_s = 0.1 K m/s, 1 h" begin
    r = tke_cbl(24; Δ = 50.0, Q = 0.1, T_end = 3600.0, avg = 900.0)
    @info "  " * tke_cbl_summary(r)
    @test r.clips == 0 && r.min_e >= 0
    # heat-flux ratio minimum (at z_i) in the entrainment range
    @test -0.25 < r.min_flux_ratio < -0.10
    # SGS share of the TKE below 25 % in the mixed layer
    @test r.max_sgs_fraction < 0.25
    # SGS TKE peaks in the surface layer (z ≤ 0.1 z_i)
    @test r.z_epeak <= 0.1 * r.zi
    # the boundary layer grew by encroachment: z_i² − z_i0² ≈ 2(1 + 2A) Q t / Γ
    @test 750 < r.zi < 1000
    # N-parent-a (#247): the harness's z_i is the one diagnostic,
    # boundary_layer_height(:min_flux); it equals the former local code
    # (zf[argmin(flux)]) bit for bit, and the parcel method on the mean θ
    # profile agrees within 10 %
    @test r.zi === r.zf[argmin(r.flux)]
    zi_parcel = boundary_layer_height(r.zc, r.θ; method = :parcel)
    @info @sprintf("  z_i: min flux %.0f m, parcel (mean θ) %.0f m (%.1f %%)", r.zi, zi_parcel,
                   100 * (zi_parcel / r.zi - 1))
    @test abs(zi_parcel - r.zi) < 0.1 * r.zi
    # With the mixing length also capped at κ·d_wall (on top of Moeng's 3.9)
    # the first levels are over-damped: measured min flux ratio −0.08, SGS
    # fraction 0.31, e peak at 125 m — outside the three criteria above,
    # which come from Moeng-type LES without a wall cap (see sgs_tke.md).
    rc = tke_cbl(24; Δ = 50.0, Q = 0.1, T_end = 3600.0, avg = 900.0, wall_cap = true)
    @info "  wall cap: " * tke_cbl_summary(rc)
    @test rc.clips == 0
    @test rc.e[1] < r.e[1]                                  # the cap lowers e at the first level
end


@testset "7. saturated column: Deardorff's length scale sees the cloudy-air N² (#239)" begin
    g = RegularGrid(; size = (4, 4, 20), extent = (100.0, 100.0, 500.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(g; θ₀ = 285.0, g_accel = 9.81)
    qv = Field(CellCenter(), g); qc = Field(CellCenter(), g)
    pv = moist_adiabat_column!(sim.θ, qv, qc)
    e0 = 0.01; Δ = 25.0; K = 2:g.Nz-1
    v(x) = view(interior_view(x), :, :, K)
    dry = TKEClosure(g; e0 = e0, qv = qv)
    f = compute_sgs!(dry, sim)
    # θ_v: N² ≈ 1.45e-4, l = 0.76 √e / N ≈ 6.3 m < Δ
    @test minimum(v(dry.N2)) > 1.4e-4
    @test maximum(v(f.νh)) < 0.1 * 0.3 * Δ * sqrt(e0)
    wet = TKEClosure(g; e0 = e0, qv = qv, qc = qc, p = pv)
    f = compute_sgs!(wet, sim)
    @test maximum(abs, v(wet.N2)) < 1e-8                 # moist-neutral
    @test all(νt -> νt ≈ 0.1 * Δ * sqrt(e0), v(f.νh))    # l = Δ: ν_t = c_k Δ √e
    rsm = ReynoldsStressClosure(g; e0 = e0, qv = qv, qc = qc, p = pv)
    Aeolus.PrognosticSGS._stratification!(rsm, sim)
    @test interior_view(rsm.N2) == interior_view(wet.N2)   # the one N² of both closures
    @test_throws ArgumentError TKEClosure(g; qv = qv, qc = qc)          # no pressure
    @info "#239 TKE in a saturated column" N2_θv = minimum(v(dry.N2)) N2_cloudy = maximum(abs, v(wet.N2))
end

end
