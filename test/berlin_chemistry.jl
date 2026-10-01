# The Berlin chemistry coupling (examples/urban/couplings/aq_chem.jl, #135):
# the src NOx–O3–VOC mechanism on advected tracers, emissions and dry
# deposition in the first air cell (roof-aware), r_a from SurfaceExchange,
# rain scavenging of HNO3.

using Test
using Aeolus
using Aeolus.Fields: interior_view

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase

# domain sums of the species arrays (ppb × cells)
_aq_sums(c) = Dict(s => sum(BC._aq_sp(c, s)) for s in BC.AQ_SPECIES)

@testset "Berlin chemistry coupling (#135)" begin
    cfg = BC.CaseConfig(:smoke; verbose = false, output_path = nothing)
    case = BC.build_case(cfg; io = devnull)
    c = BC.find_coupling(case, :aq_chem)
    mesh = case.mesh
    Nx, Ny, Nz, dz = mesh.Nx, mesh.Ny, mesh.Nz, mesh.dz
    A = Nx * Ny
    sfc = case.geom.sfc
    kair = c.k_air

    @testset "species are advected, positive tracers in ppb" begin
        for s in BC.AQ_SPECIES
            t = case.tracers.list[case.tracers.index[s]]
            @test t.advect && t.positive && t.units == "ppb"
        end
        @test BC.slot(BC.AQChem) === :chemistry && BC.is_default(BC.AQChem)
        @test (:Chemistry in BC.modules_used(BC.AQChem)) && (:SurfaceExchange in BC.modules_used(BC.AQChem))
    end

    @testset "roof-aware first air level" begin
        roofs = findall(==(BC.SURFACE_ROOF), sfc)
        @test !isempty(roofs)
        @test all(I -> kair[I] > 1, roofs)                               # above the roof
        @test all(I -> kair[I] == 1, findall(==(BC.SURFACE_STREET), sfc))
        sm = case.geom.sm.mask
        H = (mesh.Hx, mesh.Hy, mesh.Hz)
        @test all(I -> kair[I] > Nz || !sm[H[1]+I[1], H[2]+I[2], H[3]+kair[I]], CartesianIndices(kair))
        @test all(I -> kair[I] == 1 || sm[H[1]+I[1], H[2]+I[2], H[3]+kair[I]-1], CartesianIndices(kair))
    end

    @testset "initial state: photostationary at the start hour" begin
        NO, NO2, O3 = BC._aq_sp(c, :NO), BC._aq_sp(c, :NO2), BC._aq_sp(c, :O3)
        @test c.J > 5e-3                                                   # noon, 20 June
        for k in (1, Nz)
            k1 = ppb_rate_constant(k_NO_O3(c.T_col[k]), c.p_col[k], c.T_col[k])
            @test leighton_ratio(NO[1, 1, k], NO2[1, 1, k], O3[1, 1, k], c.J, k1) ≈ 1 rtol = 1e-10
            @test NO[1, 1, k] + NO2[1, 1, k] ≈ c.opts.background.NOx
            @test O3[1, 1, k] + NO2[1, 1, k] ≈ c.opts.background.Ox
        end
    end

    @testset "r_a from SurfaceExchange, v_d = 1/(r_a + r_b + r_c)" begin
        u, v, θ = (interior_view(getfield(case.sim, f)) for f in (:u, :v, :θ))
        qv = interior_view(BC.tracer(case, :qv))
        for I in (first(findall(==(BC.SURFACE_PARK), sfc)), first(findall(==(BC.SURFACE_ROOF), sfc)))
            i, j = Tuple(I); k = kair[I]
            U = hypot((u[i, j, k] + u[mod1(i + 1, Nx), j, k]) / 2, (v[i, j, k] + v[i, mod1(j + 1, Ny), k]) / 2)
            sol = solve_surface_layer(; U = U, z_m = dz / 2, θ_a = θ[i, j, k],
                                      θ_s = case.bus.Ts[I] / BC.ref_exner(case, k),
                                      q_a = qv[i, j, k], q_s = qv[i, j, k],
                                      z0m = c.z0m[I], z0h = c.z0m[I] / 10)
            @test c.r_a[I] ≈ 1 / (sol.C_H * sol.U)
            kind = BC._AQ_SURFACE_KIND[sfc[I]]
            for s in BC.AQ_DEPOSITING
                rb = quasi_laminar_resistance(sol.u_star; Sc = schmidt_number(s))
                @test c.v_d[s][I] ≈ 1 / (c.r_a[I] + rb + surface_resistance(s, kind))
            end
        end
        # A warm surface under the same air is unstable: smaller r_a, faster
        # deposition. The smoke case starts with the slab 8 K below the air
        # (Ts0 = 285 K), which is very stable (see the PR).
        ra_cold = copy(c.r_a)
        Ts0 = copy(case.bus.Ts)
        case.bus.Ts .= maximum(interior_view(case.sim.θ)) + 5
        BC._aq_deposition_velocities!(c, case)
        open_ = kair .<= Nz
        @test all(c.r_a[open_] .< ra_cold[open_])
        park = findall(==(BC.SURFACE_PARK), sfc); street = findall(==(BC.SURFACE_STREET), sfc)
        # vegetation takes up O3 through stomata; paving hardly does
        @test sum(c.v_d[:O3][park]) / length(park) > 3 * sum(c.v_d[:O3][street]) / length(street)
        case.bus.Ts .= Ts0
        BC._aq_deposition_velocities!(c, case)
    end

    @testset "emissions go into the first air cell of street/wall and park columns only" begin
        before = Dict(s => copy(BC._aq_sp(c, s)) for s in BC.AQ_SPECIES)
        BC._aq_sun!(c, case)
        BC._aq_emit!(c, case, 2.0)
        f = BC.traffic_profile(case.clock.hour)
        for s in (:NO, :NO2, :VOC)
            d = BC._aq_sp(c, s) .- before[s]
            nbad = 0
            for j in 1:Ny, i in 1:Nx, k in 1:Nz
                cl = sfc[i, j]
                exp_ = 0.0
                if k == kair[i, j]
                    if (cl == BC.SURFACE_STREET || cl == BC.SURFACE_WALL) && s !== :VOC
                        exp_ = (s === :NO ? c.F_NO : c.F_NO2) * f * 2.0 / dz
                    elseif cl == BC.SURFACE_PARK && s === :VOC
                        exp_ = c.F_VOC * c.J / j_NO2(1.0) * 2.0 / dz
                    end
                end
                nbad += !(d[i, j, k] ≈ exp_) && !(abs(d[i, j, k] - exp_) < 1e-12)
            end
            @test nbad == 0
            @test sum(d) > 0
        end
        @test BC._aq_sp(c, :O3) == before[:O3]
        # 2 µg m⁻² s⁻¹ of NOx as NO2 at the morning peak, 10 % primary NO2
        @test (c.F_NO + c.F_NO2) ≈ emission_ppb_flux(2e-9, 0.046)
        @test c.F_NO2 / (c.F_NO + c.F_NO2) ≈ 0.1
        for s in BC.AQ_SPECIES; BC._aq_sp(c, s) .= before[s]; end
        foreach(s -> c.emitted[s] = 0.0, BC.AQ_SPECIES)
    end

    @testset "night titration: Ox and NOx accounted for exactly in the chemistry stage" begin
        nc = BC.build_case(BC.CaseConfig(:smoke; start_h = 0.5, verbose = false, output_path = nothing);
                           io = devnull)
        cn = BC.find_coupling(nc, :aq_chem)
        @test cn.J == 0 && cn.OH == 0
        # the night start is titrated: min(NOx, Ox) = NOx is all NO2
        @test maximum(BC._aq_sp(cn, :NO)) < 1e-9
        BC._aq_sp(cn, :NO) .+= 3.0                         # a fresh NO plume everywhere
        s0 = _aq_sums(cn)
        BC.run_stage!(cn, nc, Val(:chemistry), 2.0)
        s1 = _aq_sums(cn)
        cells = A / dz                                      # ppb m (area mean) → ppb × cells
        eNO2 = cn.emitted[:NO2] * cells; eNO = cn.emitted[:NO] * cells
        dry = Dict(s => cn.dry[s] * cells for s in BC.AQ_DEPOSITING)
        ΔOx = (s1[:O3] + s1[:NO2]) - (s0[:O3] + s0[:NO2])
        ΔNOx = (s1[:NO] + s1[:NO2]) - (s0[:NO] + s0[:NO2])
        # Ox: + primary NO2 − ozonolysis − deposition; NOx: + emission − deposition
        # (to round-off of the domain sums, ~1e-14 of the total)
        tot = s0[:O3] + s0[:NO2]
        @test ΔOx ≈ eNO2 - cn.last.L_Ox - dry[:O3] - dry[:NO2] atol = 1e-12 * tot
        @test ΔNOx ≈ eNO + eNO2 - dry[:NO] - dry[:NO2] atol = 1e-12 * tot
        @test cn.last.L_Ox / (s0[:O3]) < 1e-5                # ozonolysis is slow
        # titration consumed O3 and NO one-for-one
        @test s1[:O3] < s0[:O3] - 0.9 * 3.0 * A * Nz * 0.01
        @test cn.last.HNO3 == 0                              # no OH at night
    end

    @testset "wet deposition: rain removes HNO3 at Λ = s·qr, nothing else" begin
        qr = interior_view(BC.tracer(case, :qr))
        fill!(qr, 0.0)
        qr[:, :, 2:4] .= 1e-4
        H0 = copy(BC._aq_sp(c, :HNO3))
        O30 = sum(BC._aq_sp(c, :O3))
        w0 = c.wet[:HNO3]
        # isolate the wet step: no photochemistry change of HNO3 matters, so
        # compare with the prediction from the HNO3 after the photochemistry
        BC.run_stage!(c, case, Val(:chemistry), 2.0)
        removed = (c.wet[:HNO3] - w0) * A / dz               # ppb × cells
        @test removed > 0
        @test c.last.rain
        # Λ = 1e-4 s⁻¹ at qr = 1e-4: ≈ 2e-4 of the HNO3 in the rainy levels per 2 s step
        rainy = sum(view(BC._aq_sp(c, :HNO3), :, :, 2:4))
        @test removed ≈ rainy * expm1(1e-4 * 2.0) rtol = 1e-6
        @test c.wet[:O3] == 0 && c.wet[:NO] == 0 && c.wet[:NO2] == 0
        fill!(qr, 0.0)
    end

    @testset "physics[:chemistry] = :off registers no species" begin
        cfg_off = BC.CaseConfig(:smoke; verbose = false, output_path = nothing,
                                physics = Dict{Symbol,Any}(:chemistry => :off))
        sel = BC.selected_couplings(cfg_off)
        @test BC.NoChem in sel && !(BC.AQChem in sel)
        off = BC.build_case(cfg_off; io = devnull)
        @test !any(s -> haskey(off.tracers, s), BC.AQ_SPECIES)
    end

    @testset "a noon run keeps the photostationary state and closes the budgets" begin
        BC.run_case!(case; nsteps = 12)
        checks = BC.smoke_check(c, case)
        @test all(last, checks)
        L = BC.aq_leighton(c, case)
        @test 0.95 < L.φ < 1.05
        @test 0.95 < L.φ_ext < 1.05
    end
end
