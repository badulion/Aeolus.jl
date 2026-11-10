# Phase 56 (P56-b, #182): the Deardorff (1980) 1.5-order prognostic
# SGS TKE closure (src/PrognosticSGS.jl) with the Moeng (1984) coefficients,
#
# The 500-step positivity run and the reduced CBL are validation runs, in
# validation/tke.jl.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Operators: advance_sgs!, sgs_state, restore_sgs_state!
using Aeolus.Backends: CPUBackend, with_backend
using Random: Xoshiro, randn

# The update of e (#554): row-parallel, branch-free and vectorised. Checked
# against a plain scalar transcription of the branching update it replaced
# (#304): the same e within round-off (the expressions are the same, only
# the compiler may contract a multiply-add differently in the vectorised
# loop), the budget sums within the round-off of a reordered sum.
function tke_update_reference(c, Δt, pb1, pb2)
    g = c.e.mesh
    T = Float64
    e = copy(c.e.data)
    sums = zeros(8)
    for K in g.Hz+1:g.Hz+g.Nz, J in g.Hy+1:g.Hy+g.Ny, I in g.Hx+1:g.Hx+g.Nx
        i, j, k = I - g.Hx, J - g.Hy, K - g.Hz
        if c.mask !== nothing && c.mask[I, J, K]
            e[I, J, K] = 0
            continue
        end
        Δ = c.Δk[k]; vr = c.vrel[k]; V = Aeolus.PrognosticSGS._tke_volume0(g) * vr
        ee = c.e.data[I, J, K]; N2 = c.N2.data[I, J, K]; ν = c.ν.data[I, J, K]
        lw = c.l_wall === nothing ? Inf : c.l_wall[i, j, k]
        sq = sqrt(max(ee, 0.0)); lmax = min(Δ, lw)
        if N2 > 0
            l = min(lmax, c.c_l * sq / sqrt(N2)); s = max(sq / lmax, sqrt(N2) / c.c_l)
        else
            l = lmax; s = sq / lmax
        end
        fac = c.first_level !== nothing && c.first_level[i, j, k] ? c.surface_factor : 1.0
        cε = (c.c_ε0 + c.c_ε1 * l / Δ) * fac
        Pw = c.Pw.data[I, J, K]
        Ps = ν * c.D.data[I, J, K] + Pw
        κv = (1 + 2l / Δ) * ν
        Psf = (k == 1 ? pb1 : 0.0) + (k == g.Nz ? pb2 : 0.0)
        Pb = -κv * N2 + Psf
        src = Ps + (N2 < 0 ? -κv * N2 : 0.0) + max(Psf, 0.0)
        rb = (N2 > 0 && s > 0 ? (1 + 2l / Δ) * c.c_k * N2 / s : 0.0) + (Psf < 0 ? -Psf / max(ee, floatmin(T)) : 0.0)
        tr = c.tend.data[I, J, K]
        estar = ee + Δt * (tr + src)
        nclip = 0.0; eclip = 0.0
        if estar < 0
            nclip = 1.0; eclip = -(estar * V); estar = 0.0
        end
        rε = cε * s
        if c.dissipation === :exact && l > 0 && estar > 0
            e1 = inv(inv(sqrt(estar)) + cε * Δt / (2l))^2
            enew = e1 / (1 + Δt * rb); dε = (estar - e1) / Δt; dB = (e1 - enew) / Δt
        else
            R = rε + rb
            enew = estar / (1 + Δt * R); rem = (estar - enew) / Δt
            dε = R > 0 ? rem * (rε / R) : 0.0; dB = R > 0 ? rem * (rb / R) : 0.0
        end
        e[I, J, K] = enew
        sums .+= (Ps * vr, Pb * vr, tr * vr, Pw * vr, dε * vr, dB * vr, nclip, eclip)
    end
    return e, sums
end

# The dual strain rate (#554: a bulk loop without checks, the k faces with
# the k checks only, the rest of the shell with the halo fold) against the
# scatter it replaced (#304): every velocity point of the interior adds
# each of its 18 flux shares to the cells it borders, the first halo layer
# is folded onto its images (`_fold_halos!`), and the sum is divided by the
# cell thickness on a StretchedGrid. The same shares added in a different
# order: agreement within round-off of a sum of ~100 terms.
const _DUAL_TARGETS = (
    u = ((:_ux_e, ((0,0,0),)), (:_ux_w, ((1,0,0),)),
         (:_uy_n, ((1,0,0), (0,0,0), (1,-1,0), (0,-1,0))), (:_uy_s, ((1,1,0), (0,1,0), (1,0,0), (0,0,0))),
         (:_uz_t, ((1,0,0), (0,0,0), (1,0,-1), (0,0,-1))), (:_uz_b, ((1,0,1), (0,0,1), (1,0,0), (0,0,0)))),
    v = ((:_vy_n, ((0,0,0),)), (:_vy_s, ((0,1,0),)),
         (:_vx_e, ((0,1,0), (-1,1,0), (0,0,0), (-1,0,0))), (:_vx_w, ((1,1,0), (0,1,0), (1,0,0), (0,0,0))),
         (:_vz_t, ((0,1,0), (0,0,0), (0,1,-1), (0,0,-1))), (:_vz_b, ((0,1,1), (0,0,1), (0,1,0), (0,0,0)))),
    w = ((:_wz_t, ((0,0,0),)), (:_wz_b, ((0,0,1),)),
         (:_wx_e, ((0,0,1), (-1,0,1), (0,0,0), (-1,0,0))), (:_wx_w, ((1,0,1), (0,0,1), (1,0,0), (0,0,0))),
         (:_wy_n, ((0,0,1), (0,-1,1), (0,0,0), (0,-1,0))), (:_wy_s, ((0,1,1), (0,0,1), (0,1,0), (0,0,0)))))

function dual_strain_scatter(sim, g, bcs)
    PS = Aeolus.PrognosticSGS
    m = PS._dual_metrics(g)
    d = zeros(size(sim.θ.data))
    for (q, list) in ((sim.u.data, _DUAL_TARGETS.u), (sim.v.data, _DUAL_TARGETS.v), (sim.w.data, _DUAL_TARGETS.w))
        for K in g.Hz+1:g.Hz+g.Nz, J in g.Hy+1:g.Hy+g.Ny, I in g.Hx+1:g.Hx+g.Nx, (f, offs) in list
            x = getfield(PS, f)(q, m, I, J, K)
            for o in offs
                d[I-o[1], J-o[2], K-o[3]] += x
            end
        end
    end
    PS._fold_halos!(d, bcs, g)
    for K in g.Hz+1:g.Hz+g.Nz
        d[:, :, K] .= PS._per_volume.(Ref(m), d[:, :, K], K)
    end
    return d
end

@testset "prognostic SGS TKE (P56-b, #182)" begin

@testset "1. free decay at l = Δ (c_ε = 0.70)" begin
    m = RegularGrid(; size = (4, 4, 4), extent = (40.0, 40.0, 40.0))
    sim = BoussinesqSimulation(m)
    Δ, e0, T = 10.0, 1.0, 600.0
    @test tke_dissipation_coefficient(Δ, Δ) ≈ 0.70
    exact(t) = (e0^-0.5 + 0.70 * t / (2Δ))^-2           # de/dt = −c_ε e^{3/2}/l
    c = TKEClosure(m; e0 = e0, dissipation = :exact)
    for _ in 1:60
        advance_sgs!(c, sim, T / 60)
    end
    @test maximum(abs.(interior_view(c.e) .- exact(T))) / exact(T) < 1e-6   # measured ≤ 1.7e-14
    @test tke_statistics(c).clips == 0
end

@testset "2. shear equilibrium: C_s = c_k^{3/4} c_ε^{-1/4}; critical Ri" begin
    S, Δ = 0.05, 10.0
    eq = 0.1 / 0.7 * Δ^2 * S^2                            # e = (c_k/c_ε) l² S²
    @test tke_equilibrium(S^2, 0.0, Δ) ≈ eq rtol = 1e-12
    # ν_t = c_k Δ √e at that e is Smagorinsky's with C_s = c_k^{3/4} c_ε^{-1/4} = 0.19441
    @test sqrt(0.1 * Δ * sqrt(eq) / (Δ^2 * S)) ≈ 0.1^0.75 * 0.70^-0.25 rtol = 1e-12
    Ric = tke_critical_richardson()
    @test Ric ≈ 1 / (1 + 0.19 / (0.1 * 0.76^2)) rtol = 1e-14
    @test tke_equilibrium(S^2, 0.25 * S^2, Δ) == 0          # above Ri_c: no equilibrium
    # below l = Δ, P_s + P_b = ε is linear in x = l/Δ: the closed-form root
    Ri = 0.2; a = 0.1 * 0.76^2
    x = (1 / Ri - 1 - 0.19 / a) / (2 + 0.51 / a)
    @test tke_equilibrium(S^2, Ri * S^2, Δ) ≈ (x * Δ * sqrt(Ri) * S / 0.76)^2 rtol = 1e-10
end

@testset "contract: pure coefficients, exact rollback" begin
    m = RegularGrid(; size = (8, 8, 8), extent = (2π, 2π, 2π), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m)
    rng = Xoshiro(5)
    for f in (sim.u, sim.v, sim.w)
        interior_view(f) .= 0.3 .* randn(rng, 8, 8, 8)
    end
    c = TKEClosure(m)
    initialize_tke!(c, sim)
    e0 = copy(c.e.data)
    compute_sgs!(c, sim); compute_sgs!(c, sim)
    @test c.e.data == e0                                   # coefficients change no memory
    νt = interior_view(sim.νt)
    @test minimum(νt) >= 0 && all(interior_view(sim.κt) .>= νt)   # κ_t = (1 + 2l/Δ) ν_t
    snap = sgs_state(c)
    advance_sgs!(c, sim, 0.01)
    @test c.e.data != e0
    restore_sgs_state!(c, snap)
    @test c.e.data == e0 && tke_statistics(c).steps == 0   # copied back, not computed
end


@testset "the update of e: rows, branch-free, vectorised (#554)" begin
    # the branch-free mixing length is the branching one, value for value
    for (e, N2) in ((0.3, 1e-4), (0.3, -1e-4), (0.3, 0.0), (0.0, 1e-4), (0.0, -1e-4), (0.0, 0.0),
                    (-1e-9, 1e-4), (2.0, 1e-2), (NaN, 1e-4), (0.3, NaN))
        l, s = tke_length_scale(e, N2, 10.0; l_wall = 4.0)
        sq = sqrt(max(e, 0.0))
        lr, sr = N2 > 0 ? (min(4.0, 0.76 * sq / sqrt(N2)), max(sq / 4.0, sqrt(N2) / 0.76)) : (4.0, sq / 4.0)
        @test isequal(l, lr) && isequal(s, sr)          # the same two expressions, selected
    end
    # its square root (Base.sqrt_llvm, an unexported Base internal) is
    # `sqrt`, bit for bit (both correctly rounded), on 0, subnormals, the
    # extremes, Inf, NaN and random values, in both precisions; a Julia
    # that drops or changes the internal fails here, not in the physics
    for T in (Float32, Float64)
        xs = T[0, -zero(T), nextfloat(zero(T)), floatmin(T), floatmax(T), Inf, NaN, 1, 2, 1e-30, 0.76]
        xs = vcat(xs, exp.(T(40) .* (rand(Xoshiro(10), T, 1000) .- T(0.5))))
        @test all(x -> isequal(Aeolus.PrognosticSGS._sqrt_nonneg(x), sqrt(x)), xs)
        @test isnan(Aeolus.PrognosticSGS._sqrt_nonneg(-one(T)))   # no DomainError: NaN, discarded by ifelse
    end
    n = (24, 20, 12)
    rg = RegularGrid(; size = n, extent = (240.0, 200.0, 120.0), halo = (3, 3, 3))
    sg = StretchedGrid(; size = n, extent = (240.0, 200.0, 360.0), halo = (3, 3, 3), stretching = (:tanh, 1.5))
    for (gname, g) in (("regular", rg), ("stretched", sg)), diss in (:semi_implicit, :exact)
        sim = BoussinesqSimulation(g; θ₀ = 300.0, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        rng = Xoshiro(554)
        for f in (sim.u, sim.v, sim.w)
            interior_view(f) .= 0.5 .* randn(rng, n)
        end
        masked = g isa RegularGrid                       # a mask is RegularGrid-only
        mask = masked ? falses(size(sim.θ.data)) : nothing
        masked && (mask[8:14, 6:12, 4:9] .= true; mask[18:22, 15:20, 4:6] .= true)
        fl = falses(n); fl[:, :, 1] .= true
        c = TKEClosure(g; mask = mask, dissipation = diss, first_level = fl,
                       wall_distance = 1.0 .+ 30 .* rand(rng, n...))
        initialize_tke!(c, sim; e_min = 1e-4)
        # ν, N², D, the transport current: the separate loops keep them in the
        # scratch fields (the blocked pass of #554 does not store ν or the transport)
        Aeolus.PrognosticSGS._advance_separate!(c, sim, 0.5)
        # every case of the update: N² of both signs, e = 0 (l = 0 in stable
        # air), a large negative transport (clipping), surface fluxes of
        # both signs, a wall-model production
        interior_view(c.e) .= 0.2 .* rand(rng, n...) .* (rand(rng, n...) .> 0.1)
        interior_view(c.N2) .= 1e-3 .* randn(rng, n)
        interior_view(c.tend) .= 0.05 .* randn(rng, n) .- 5.0 .* (rand(rng, n...) .< 0.05)
        interior_view(c.Pw) .= 1e-3 .* rand(rng, n...)
        ref_e, ref = tke_update_reference(c, 0.5, 2e-3, -1e-3)
        e0 = copy(c.e.data)
        for (bname, b) in (("serial", CPUBackend(serial = true)), ("threaded", CPUBackend(threshold = 0)))
            copyto!(c.e.data, e0); fill!(c.stats, 0)
            with_backend(b) do
                Aeolus.PrognosticSGS._update!(c, 0.5, (2e-3, -1e-3))
            end
            # e: the same expressions; ≤ 4 ulp from a contracted multiply-add
            @test maximum(abs.(c.e.data .- ref_e) ./ max.(abs.(ref_e), floatmin())) <= 4eps()  # measured 0
            masked && @test all(iszero, c.e.data[mask])
            V0 = Aeolus.PrognosticSGS._tke_volume0(g)
            # the sums: reordered, so a relative round-off of a few eps (measured
            # ≤ 3.1e-15; 1e-11 leaves room for other vector widths); the
            # transport sum cancels
            # (Σ T V ≈ 0): compared on the scale of Σ P_s V (measured 3.4e-14)
            scale = abs(ref[1]) * V0
            @test c.stats[1] == ref[7] && ref[7] > 0      # clip count: integers
            @test c.stats[2] ≈ ref[8] rtol = 1e-11
            for (sn, rn) in ((4, 1), (5, 2), (6, 5), (7, 6), (10, 4))
                @test c.stats[sn] ≈ ref[rn] * V0 rtol = 1e-11
            end
            @test abs(c.stats[8] - ref[3] * V0) < 1e-11 * scale
        end
    end
end


@testset "dual strain: bulk, k faces and shell against the scatter (#554)" begin
    for (gname, mk) in (("regular", n -> RegularGrid(; size = n, extent = (120.0, 100.0, 80.0), halo = (2, 2, 2))),
                        ("stretched", n -> StretchedGrid(; size = n, extent = (120.0, 100.0, 240.0), halo = (2, 2, 2),
                                                         stretching = (:tanh, 1.5)))),
        n in ((12, 10, 8), (3, 1, 4)),
        bcs in (FieldBoundaryConditions(:periodic, :periodic, :wall), FieldBoundaryConditions(:noflux, :noflux, :wall))
        g = mk(n)
        sim = BoussinesqSimulation(g; θ₀ = 300.0, bcs = bcs)
        rng = Xoshiro(17)
        for f in (sim.u, sim.v, sim.w)
            interior_view(f) .= randn(rng, size(interior_view(f)))
            fill_halos!(f, bcs)
        end
        ref = dual_strain_scatter(sim, g, bcs)
        for b in (CPUBackend(serial = true), CPUBackend(threshold = 0))
            D = Aeolus.Fields.Field(Aeolus.Grids.CellCenter(), g)
            D.data .= NaN                                 # every cell written, the halos zeroed
            with_backend(b) do
                dual_strain!(D, sim.u, sim.v, sim.w, bcs)
            end
            # round-off of a reordered sum: 1e-13 of the largest |D| (measured ≤ 4e-16)
            @test maximum(abs.(D.data .- ref)) <= 1e-13 * maximum(abs.(ref))
        end
    end
end

end
