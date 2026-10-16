# KA-l (#304): the SGS closures on launch! (Operators: the dynamic
# Smagorinsky passes, the box filter, the Lagrangian closure and its second
# test filter, AMD / Vreman / σ, the stability correction, N² / Ri / S²;
# PrognosticSGS: the TKE and Reynolds-stress closures).
#
#   1. against the pre-port serial loops (a transcript:
#      test/support/legacy_sgs.jl) for every ported kernel, on random
#      fields with random halos, with and without a solid mask, periodic
#      and bounded; serially, threaded with every launch on the KA path
#      (`threshold = 0`), and on the default backend. The fused budget sums
#      of the TKE / Reynolds-stress updates moved to the canonical
#      plane-reduction order (a named one-time change): the fields they
# update are compared, the sums to round-off;
#   2. the named one-time change of the dual strain (the owner-computes
#      gather replaces the scatter and halo fold): equal to the scatter to
#      round-off on a RegularGrid (periodic and walls) and a StretchedGrid,
#      and still the exact dual of the momentum operator;
#   3. the second test filter (three full 3-D passes) is the plane-by-plane
# filter, and allocates nothing per call.

using Test
using Random: Xoshiro, randn, rand
using Aeolus
using Aeolus.Grids: FaceX, FaceY, FaceZ, CellCenter, interior
using Aeolus.Fields: Field, interior_view
using Aeolus.StretchedGrids: StretchedGrid
const OPS = Aeolus.Operators
const PSG = Aeolus.PrognosticSGS

isdefined(@__MODULE__, :LegacySGS) || include(joinpath(@__DIR__, "support", "legacy_sgs.jl"))

const SGS_BACKENDS = ("serial" => CPUBackend(serial = true),
                      "threaded, threshold 0" => CPUBackend(threshold = 0),
                      "default" => CPUBackend())
sgs_bits(A) = collect(Float64, A)      # a copy, compared within ROUNDOFF_RTOL (#495)
sgs_data(x::Field) = x.data
sgs_data(x::AbstractArray) = x
sgs_data(x) = x

# Run `new!()` on every backend, each on a fresh copy of the state built by
# `mk()`, and `old!()` once on its own copy; `outs(state)` lists the arrays
# that must agree.
function sgs_check(name, mk, new!, old!, outs)
    ref = mk()
    old!(ref)
    for (bname, b) in SGS_BACKENDS
        got = mk()
        with_backend(b) do
            new!(got)
        end
        ok = all(agree(sgs_data(a), sgs_data(r)) for (a, r) in zip(outs(got), outs(ref)))
        ok || @info "KA-l bit mismatch" name bname
        @test ok
    end
end

function sgs_field(rng, loc, g; scale = 1.0, offset = 0.0)
    f = Field(loc, g)
    f.data .= offset .+ scale .* randn(rng, size(f.data))
    return f
end

sgs_grid() = RegularGrid(Float64; size = (13, 11, 9), extent = (1.3, 1.7, 0.8), halo = (3, 3, 3))

@testset "KA-l: SGS closures on launch! (#304)" begin
    rng = Xoshiro(304)
    g = sgs_grid()
    u = sgs_field(rng, FaceX(), g); v = sgs_field(rng, FaceY(), g); w = sgs_field(rng, FaceZ(), g)
    θ = sgs_field(rng, CellCenter(), g; scale = 2.0, offset = 300.0)
    qv = sgs_field(rng, CellCenter(), g; scale = 1e-3, offset = 8e-3)
    sz = size(θ.data)
    cell = Array{Bool,3}(rand(rng, sz...) .< 0.12)
    cellb = BitArray(cell)
    cc() = Field(CellCenter(), g)
    rarr() = randn(rng, sz)

    @testset "solid mask: halo fill and masked velocities" begin
        for per in ((true, true, false), (false, false, false), (true, false, true))
            sgs_check("mask halos $per", () -> copy(cell),
                      m -> OPS._fill_mask_halos!(m, g, per), m -> LegacySGS.fill_mask_halos!(m, g, per),
                      m -> (m,))
        end
        sm = OPS._SGSMask(cell, g, (true, true, false))
        @test sm.cell isa Array{Bool,3} && sm.fx isa Array{Bool,3}
        sgs_check("masked velocities", () -> (OPS._SGSMask(cell, g, nothing), map(copy, (rarr(), rarr(), rarr()))),
                  s -> OPS._masked_velocities!(s[1], u.data, v.data, w.data),
                  s -> LegacySGS.masked_velocities!(s[1].ub, s[1].vb, s[1].wb, s[1].fx, s[1].fy, s[1].fz,
                                                    u.data, v.data, w.data),
                  s -> (s[1].ub, s[1].vb, s[1].wb))
    end

    @testset "dynamic Smagorinsky (plane / local, masked / not)" begin
        for plane in (true, false), masked in (false, true)
            name = "dynamic $(plane ? :plane : :local)$(masked ? ", masked" : "")"
            mk() = (cc(), OPS.DynamicSmagorinskyWorkspace(g))
            function new!(s)
                OPS.compute_dynamic_smagorinsky_νt!(s[1], u, v, w; workspace = s[2],
                                                    averaging = plane ? :plane : :local,
                                                    filter_ratio = 3, cs_max = 0.4,
                                                    mask = masked ? cellb : nothing)
            end
            function old!(s)
                lm = masked ? (cell, map(l -> Array{Bool,3}(OPS.staggered_solid_mask(cell, l)),
                                         (FaceX(), FaceY(), FaceZ()))..., zeros(sz), zeros(sz), zeros(sz)) : nothing
                LegacySGS.dynamic_smagorinsky!(s[1].data, u.data, v.data, w.data, s[2], g, plane, float(3)^2, float(0.4)^2, lm)
            end
            sgs_check(name, mk, new!, old!,
                      s -> (s[1], s[2].cs2, s[2].LM, s[2].MM, s[2].magS, s[2].Uc, s[2].S13))
        end
    end

    @testset "box filter" begin
        f = sgs_field(rng, CellCenter(), g)
        for n in (3, (3, 5, 1)), masked in (false, true)
            rx, ry, rz = OPS._box_halfwidths(n)
            sgs_check("box filter n=$n masked=$masked", cc,
                      o -> OPS.box_filter!(o, f, n; mask = masked ? cellb : nothing),
                      o -> LegacySGS.box_filter!(o.data, f.data, g, rx, ry, rz, masked ? cellb : nothing),
                      o -> (o,))
        end
    end

    @testset "Lagrangian closure kernels" begin
        N = (g.Nx, g.Ny, g.Nz)
        ws0 = OPS.DynamicSmagorinskyWorkspace(g)
        OPS._germano_identity!(ws0, u.data, v.data, w.data, g, 9.0)
        for per in ((true, true, false), (false, false, false), (true, true, true)), masked in (false, true)
            cl = masked ? cell : nothing
            pairs() = ((rand(rng, N...), rand(rng, N...), rand(rng, N...), rand(rng, N...),
                        rand(rng, N...), rand(rng, N...)),
                       (rand(rng, N...), rand(rng, N...), rand(rng, N...), rand(rng, N...),
                        rand(rng, N...), rand(rng, N...)))
            P = pairs()
            U, V, W = 0.3 .* randn(rng, N...), 0.3 .* randn(rng, N...), 0.3 .* randn(rng, N...)
            for mt in (nothing, 2.5)
                sgs_check("Lagrangian relax per=$per masked=$masked memory_time=$mt",
                          () -> map(p -> map(copy, p), P),
                          s -> OPS._lagrangian_relax!(s, U, V, W, 0.05, g, 1.5, mt, cl, per, 0.07),
                          s -> LegacySGS.lagrangian_relax!(s, U, V, W, 0.05, g, 1.5, mt, cl, per, 0.07),
                          s -> (s[1][1], s[1][2], s[2][1], s[2][2]))
            end
            sgs_check("second test filter per=$per masked=$masked",
                      () -> (OPS._SecondTestFilter(g), LegacySGS.SecondTestFilter(g)),
                      s -> OPS._second_test_filter!(s[1], ws0, g, 25.0, cl, per),
                      s -> (LegacySGS.second_test_filter!(s[2], ws0, g, 25.0, cl, per);
                            s[1].QN .= s[2].QN; s[1].NN .= s[2].NN),
                      s -> (s[1].QN, s[1].NN))
        end
        # several planes per work item (Nz + 2 > 12 work items) and rows too
        # short for the fused bulk sums (N < 5)
        for gg in (RegularGrid(Float64; size = (9, 4, 30), extent = (0.9, 0.4, 3.0), halo = (2, 2, 2)),
                   RegularGrid(Float64; size = (4, 9, 3), extent = (0.4, 0.9, 0.3), halo = (2, 2, 2)))
            uu = sgs_field(rng, FaceX(), gg); vv = sgs_field(rng, FaceY(), gg); ww = sgs_field(rng, FaceZ(), gg)
            wsg = OPS.DynamicSmagorinskyWorkspace(gg)
            OPS._germano_identity!(wsg, uu.data, vv.data, ww.data, gg, 9.0)
            cg = Array{Bool,3}(rand(rng, size(uu.data)...) .< 0.1)
            for per in ((true, true, false), (false, true, true)), cl in (nothing, cg)
                sgs_check("second test filter $(gg.Nx)×$(gg.Ny)×$(gg.Nz) per=$per masked=$(cl !== nothing)",
                          () -> (OPS._SecondTestFilter(gg), LegacySGS.SecondTestFilter(gg)),
                          s -> OPS._second_test_filter!(s[1], wsg, gg, 25.0, cl, per),
                          s -> (LegacySGS.second_test_filter!(s[2], wsg, gg, 25.0, cl, per);
                                s[1].QN .= s[2].QN; s[1].NN .= s[2].NN),
                          s -> (s[1].QN, s[1].NN))
            end
        end
        sf = OPS._SecondTestFilter(g)
        OPS._second_test_filter!(sf, ws0, g, 25.0, nothing, (true, true, false))
        with_backend(CPUBackend(serial = true)) do
            @test (@allocated OPS._second_test_filter!(sf, ws0, g, 25.0, nothing, (true, true, false))) == 0
        end
        # the closure's coefficient paths
        for (sd, fixed) in ((false, nothing), (true, nothing), (true, 1.7))
            c = LagrangianDynamicSmagorinsky(g; scale_dependent = sd, beta = fixed)
            c.I_LM .= rand(rng, N...) .- 0.2; c.I_MM .= rand(rng, N...) .* (rand(rng, N...) .> 0.1)
            if sd
                c.I_QN .= rand(rng, N...) .- 0.1; c.I_NN .= rand(rng, N...) .* (rand(rng, N...) .> 0.1)
                sgs_check("scale-dependent C² beta=$fixed", () -> (zeros(N...), zeros(N...)),
                          s -> OPS._scale_dependent_cs2!(s[1], s[2], c, 0.16),
                          s -> LegacySGS.scale_dependent_cs2!(s[1], s[2], c, 0.16), s -> s)
            else
                sgs_check("Lagrangian C²", () -> zeros(N...),
                          s -> launch!(Aeolus.Backends.backend(s), OPS._local_cs2_body!,
                                       (1:N[1], 1:N[2], 1:N[3]), s, c.I_LM, c.I_MM, 0.16),
                          s -> LegacySGS.lagrangian_cs2!(s, c.I_LM, c.I_MM, 0.16, nothing, g), s -> (s,))
            end
        end
        sgs_check("cell velocity", () -> (zeros(N...), zeros(N...), zeros(N...)),
                  s -> OPS._cell_velocity!(s..., nothing, (u = u, v = v, w = w), g),
                  s -> LegacySGS.cell_velocity!(s..., u.data, v.data, w.data, g), s -> s)
    end

    @testset "AMD, Vreman, σ" begin
        θ̄ = 300 .+ randn(rng, g.Nz)
        for buoy in (0.0, 9.81 / 300, OPS._LevelBuoyancy(9.81, θ̄), k -> 9.81 / θ̄[k])
            sgs_check("AMD buoyancy $(typeof(buoy))", () -> (cc(), cc()),
                      s -> OPS._amd_kernel!(s[1].data, s[2].data, u.data, v.data, w.data, θ.data, g,
                                            0.3, 1 / 3, buoy),
                      s -> LegacySGS.amd_kernel!(s[1].data, s[2].data, u.data, v.data, w.data, θ.data, g,
                                                 0.3, 1 / 3, buoy), s -> s)
        end
        # the sound-proof cores' buoyancy parameter: the callable struct is the closure it replaced
        @test OPS._LevelBuoyancy(9.81, θ̄)(3) === (k -> 9.81 / θ̄[k])(3)
        ν0 = sgs_field(rng, CellCenter(), g; scale = 0.01, offset = 0.02)
        sgs_check("AMD scalar", cc,
                  s -> OPS._amd_scalar_kernel!(s.data, ν0.data, u.data, v.data, w.data, qv.data, g, 0.3, 1 / 3),
                  s -> LegacySGS.amd_scalar_kernel!(s.data, ν0.data, u.data, v.data, w.data, qv.data, g, 0.3, 1 / 3),
                  s -> (s,))
        sgs_check("Vreman", cc, s -> OPS._vreman_kernel!(s.data, u.data, v.data, w.data, g, 0.07),
                  s -> LegacySGS.vreman_kernel!(s.data, u.data, v.data, w.data, g, 0.07), s -> (s,))
        sgs_check("σ-model", cc, s -> compute_sigma_νt!(s, u, v, w, 1.35),
                  s -> LegacySGS.sigma_νt!(s.data, u.data, v.data, w.data, 1.35, g), s -> (s,))
    end

    @testset "stability correction, N², Ri, S², TKE estimate" begin
        Ri = sgs_field(rng, CellCenter(), g; scale = 0.3)
        for dynamic in (false, true)
            sgs_check("stability correction dynamic=$dynamic",
                      () -> (sgs_field(Xoshiro(1), CellCenter(), g; scale = 0.01, offset = 0.03), cc(), cc(), cc(), cc()),
                      s -> OPS._stability_correct!(map(sgs_data, s)..., Ri.data, Aeolus.TurbulenceStability.BrownEtAl1994(),
                                                   1.84, 0.29, 0.7, dynamic, g),
                      s -> LegacySGS.stability_correct!(map(sgs_data, s)..., Ri.data, Aeolus.TurbulenceStability.BrownEtAl1994(),
                                                        1.84, 0.29, 0.7, dynamic, g), s -> s)
        end
        qc = sgs_field(rng, CellCenter(), g; scale = 1e-4); qc.data .= max.(qc.data, 0)
        θ₀v = 300 .+ 0.1 .* randn(rng, sz[3]); pv = 1e5 .- 10 .* (1:sz[3])
        for (nm, hum, θ₀) in (("dry", nothing, 300.0), ("moist", qv, θ₀v),
                              ("cloudy", OPS._Cloudy(qv.data, qc.data, pv), 300.0))
            sgs_check("N² $nm", cc, s -> OPS._buoyancy_frequency_squared!(s, θ, hum, θ₀, 9.81, g),
                      s -> LegacySGS.n2!(s.data, θ.data, hum, θ₀, 9.81, g), s -> (s,))
        end
        N2 = sgs_field(rng, CellCenter(), g; scale = 1e-4)
        sgs_check("Ri", cc, s -> gradient_richardson!(s, N2, u, v, w; S2_min = 1e-8),
                  s -> LegacySGS.ri!(s.data, N2.data, u.data, v.data, w.data, 1e-8, g), s -> (s,))
        sgs_check("S²", cc, s -> strain_rate_squared!(s, u, v, w),
                  s -> LegacySGS.s2!(s.data, u.data, v.data, w.data, g), s -> (s,))
        ν0 = sgs_field(rng, CellCenter(), g; scale = 0.01)
        sgs_check("TKE from ν", cc, s -> OPS._tke_from_viscosity!(s, ν0, 0.1),
                  s -> LegacySGS.tke_from_viscosity!(s.data, ν0.data, 0.1, g), s -> (s,))
        sgs_check("κ = ν/Prₜ over the storage", cc, s -> OPS._divide_storage!(s.data, ν0.data, 0.7),
                  s -> (s.data .= ν0.data ./ 0.7), s -> (s,))
    end

    @testset "TKE closure: coefficients, update, initialisation" begin
        N = (g.Nx, g.Ny, g.Nz)
        lw = 0.05 .+ rand(rng, N...); fl = rand(rng, N...) .< 0.2
        for masked in (false, true), dissipation in (:semi_implicit, :exact)
            c0 = TKEClosure(g; wall_distance = lw, first_level = fl, mask = masked ? cell : nothing,
                            dissipation = dissipation, transport = false)
            c0.e.data .= abs.(randn(rng, sz)) .* 1e-2
            c0.ν.data .= abs.(randn(rng, sz)) .* 1e-2
            c0.N2.data .= 1e-4 .* randn(rng, sz)
            c0.D.data .= abs.(randn(rng, sz))
            c0.tend.data .= 0.05 .* randn(rng, sz)       # some cells clip
            c0.Pw.data .= 1e-3 .* abs.(randn(rng, sz))
            t = "masked=$masked $dissipation"
            sgs_check("TKE coefficients $t", () -> (cc(), cc()),
                      s -> PSG._coefficients!(s[1], s[2], c0, c0.N2),
                      s -> LegacySGS.tke_coefficients!(s[1].data, s[2].data, c0, c0.N2.data), s -> s)
            ref = deepcopy(c0)
            LegacySGS.tke_update!(ref, 0.3, (1e-4, -2e-5))
            for (bname, b) in SGS_BACKENDS
                c = deepcopy(c0)
                with_backend(b) do
                    PSG._update!(c, 0.3, (1e-4, -2e-5))
                end
                @test agree(c.e.data, ref.e.data)          # e:
                @test c.stats[1] == ref.stats[1] && c.stats[9] == ref.stats[9]   # counts: exact
                @test ref.stats[1] > 0                                   # (the clip path ran)
                for n in (2, 4, 5, 6, 7, 8, 10)                          # sums: canonical order
                    @test isapprox(c.stats[n], ref.stats[n]; rtol = 1e-12, atol = 1e-300)
                end
            end
            sgs_check("TKE initialisation $t", () -> deepcopy(c0),
                      s -> launch!(Aeolus.Backends.backend(s.e.data), PSG._init_tke_body!, interior(g),
                                   s.e.data, s.D.data, s.N2.data, s.Δk, s.mask, s.first_level, s.l_wall,
                                   g.Hx, g.Hy, g.Hz, (c_k = s.c_k, c_ε0 = s.c_ε0, c_ε1 = s.c_ε1, c_l = s.c_l,
                                                      sfac = s.surface_factor, e_min = 1e-6)),
                      s -> LegacySGS.init_tke!(s, 1e-6), s -> (s.e,))
        end
    end

    @testset "dual strain: owner-computes gather (named one-time change)" begin
        sg = StretchedGrid(Float64; size = (13, 11, 9), extent = (1.3, 1.7, 0.8), halo = (3, 3, 3),
                           stretching = (:geometric, 1.08))
        for (gn, gg) in (("RegularGrid", g), ("StretchedGrid", sg)),
            bc in (FieldBoundaryConditions(:periodic, :periodic, :wall),
                   FieldBoundaryConditions(:periodic, :periodic, :periodic),
                   FieldBoundaryConditions(:wall, :wall, :wall))
            gg isa StretchedGrid && bc.bottom isa Aeolus.BoundaryConditions.Periodic && continue
            uu = sgs_field(rng, FaceX(), gg); vv = sgs_field(rng, FaceY(), gg); ww = sgs_field(rng, FaceZ(), gg)
            ref = Field(CellCenter(), gg)
            LegacySGS.dual_strain!(ref.data, uu.data, vv.data, ww.data, bc, gg)
            scale = maximum(abs, interior_view(ref))
            got = Dict{String,Array{Float64,3}}()
            for (bname, b) in SGS_BACKENDS
                D = Field(CellCenter(), gg); D.data .= randn(rng, size(D.data))
                with_backend(b) do
                    dual_strain!(D, uu, vv, ww, bc)
                end
                got[bname] = sgs_bits(D.data)
                # the scatter's shares in another order: equal to round-off,
                # halos zero as before
                @test maximum(abs, D.data .- ref.data) <= 64 * eps(scale)
            end
            @test agree(got["threaded, threshold 0"], got["serial"]) && agree(got["default"], got["serial"])
        end
        # still the exact discrete dual of the Laplacian momentum operator:
        # Σ ν D V = −Σ u·T(ν) V for a random ν ≥ 0 (periodic box)
        bc = FieldBoundaryConditions(:periodic, :periodic, :periodic)
        for f in (u, v, w)
            fill_halos!(f, bc)
        end
        ν = sgs_field(rng, CellCenter(), g; scale = 0.01, offset = 0.05); fill_halos!(ν, bc)
        D = Field(CellCenter(), g); dual_strain!(D, u, v, w, bc)
        tu, tv, tw = Field(FaceX(), g), Field(FaceY(), g), Field(FaceZ(), g)
        add_eddy_viscosity_u!(tu, u, ν); add_eddy_viscosity_v!(tv, v, ν); add_eddy_viscosity_w!(tw, w, ν)
        lhs = sum(interior_view(ν) .* interior_view(D))
        rhs = -(sum(interior_view(u) .* interior_view(tu)) + sum(interior_view(v) .* interior_view(tv)) +
                sum(interior_view(w) .* interior_view(tw)))
        @test isapprox(lhs, rhs; rtol = 1e-12)
    end

    @testset "Reynolds-stress closure: coefficients, edges, update" begin
        N = (g.Nx, g.Ny, g.Nz)
        for masked in (false, true)
            sim = BoussinesqSimulation(g; ν = 1e-3, θ₀ = 300.0, f_coriolis = 1e-4,
                                       bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
            sim.u.data .= u.data; sim.v.data .= v.data; sim.w.data .= w.data; sim.θ.data .= θ.data
            c0 = ReynoldsStressClosure(g; mask = masked ? cell : nothing, transport = true,
                                       wall_distance = 0.05 .+ rand(rng, N...),
                                       first_level = rand(rng, N...) .< 0.2)
            for m in 1:3
                c0.τ[m].data .= 1e-2 .* abs.(randn(rng, sz))
            end
            for m in 4:6
                c0.τ[m].data .= 6e-3 .* randn(rng, sz)            # some unrealizable
            end
            for m in 1:6
                c0.tend[m].data .= 1e-3 .* randn(rng, sz)
            end
            c0.N2.data .= 1e-4 .* randn(rng, sz)
            t = "masked=$masked"
            Kd(c) = (c.K.K11.data, c.K.K22.data, c.K.K33.data, c.K.K12.data, c.K.K13.data, c.K.K23.data)
            sgs_check("RSM coefficients $t", () -> (deepcopy(c0), cc(), cc()),
                      s -> launch!(Aeolus.Backends.backend(s[2].data), PSG._rsm_coefficients_body!, interior(g),
                                   s[2].data, s[3].data, Kd(s[1]), map(x -> x.data, s[1].τ), s[1].N2.data,
                                   s[1], g.Hx, g.Hy, g.Hz),
                      s -> LegacySGS.rsm_coefficients!(s[2].data, s[3].data, Kd(s[1]), s[1]),
                      s -> (s[2], s[3], Kd(s[1])...))
            E = PSG._Edges(c0.mask, (false, false, true), g)
            Sarr(c) = (c.S.τ11, c.S.τ22, c.S.τ33, c.S.τ12, c.S.τ13, c.S.τ23)
            sgs_check("RSM edge stresses $t", () -> deepcopy(c0),
                      s -> PSG._stress_edges!(s.S, map(x -> x.data, s.τ), E, g),
                      s -> LegacySGS.stress_edges!(Sarr(s), map(x -> x.data, s.τ), E, g), Sarr)
            sgs_check("RSM viscosity $t", () -> deepcopy(c0), PSG._rsm_viscosity!,
                      s -> LegacySGS.rsm_viscosity!(s.ν.data, s), s -> (s.ν,))
            ref = deepcopy(c0)
            LegacySGS.rsm_update!(ref, sim, 0.4)
            for (bname, b) in SGS_BACKENDS
                c = deepcopy(c0)
                with_backend(b) do
                    PSG._rsm_update!(c, sim, 0.4)
                end
                @test all(agree(c.τ[m].data, ref.τ[m].data) for m in 1:6)
                @test c.stats[[1, 7, 8, 10, 12]] == ref.stats[[1, 7, 8, 10, 12]]   # counts: exact
                @test ref.stats[8] > 0                                          # (projections ran)
                for n in (2, 3, 4, 5, 6, 9)
                    @test isapprox(c.stats[n], ref.stats[n]; rtol = 1e-12, atol = 1e-300)
                end
            end
            # the stack-allocated 6 × 6 solve and 3 × 3 Jacobi are the heap ones
            τu = (1.0, 0.5, 0.2, 0.9, -0.3, 0.4)
            @test values_agree(PSG.project_realizable(τu), LegacySGS.project_realizable(τu))
            G = Tuple(randn(rng, 9)); dθ = (0.01, -0.02, 0.003)
            @test reynolds_stress_step(τu, G, dθ, 0.8, 1.2, 0.03, 0.7) ==
                  LegacySGS.rsm_substeps!(τu, G, dθ, 0.8, 1.2, 0.03, 0.7, 1.8, 0.6, 0.5,
                                          zeros(6, 6), zeros(6, 6), zeros(6))[1]
        end
    end
end
