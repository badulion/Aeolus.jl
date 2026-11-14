# The urban nest parent's physics as components of a domain (NEST-p1, #481),
# on both dynamical cores:
#
#   * TiledUrbanSurface: the heat and vapour the air receives over a step are the
#     tiles' area-mean fluxes × the step, exactly (the canopy levels' depth
#     weights sum to one); u* = √(Σ fᵢ τᵢ / ρ̄₁); the drag is the implicit
# deceleration u/(1 + Δt a/U); a checkpoint restores the tiles;
#   * TiledUrbanSurface temperatures (#536): the area-weighted skin
#     temperature and per class (NaN where a class is absent), the 2 m
#     Monin–Obukhov value against a hand assembly, the interpolation branch;
#   * PrescribedSurfaceFlux: Q Δt/Δz₁ into the first level of every column;
#   * MeanColumnRadiation: the sun of the site's clock (sw ≈ 0 at night, a
#     clear-sky noon sun by day), the heating applied as Ṫ Δt / Π̄, the
#     downwelling flux handed to the tiles, one call per period;
#   * LargeScale: the boundary-layer tracker (the parcel z_i of a mixed layer
#     every `every` steps, the floor), the nudging weight zero below
#     factor × z_i, the two clocks of an SSP-RK3 step (the step mean versus
#     the core's substage clock, which leads by 3Δt/2);
#   * [sgs] model = "tke": the exact wall distance of a stretched grid
#     (bottom only, or both walls), and the surface value u*²/√(c_k c_ε) of
#     the :surface component's u*;
#   * (the parent as a [[domain]] of the model's config, NEST-p3 #483:
#     test/config_parent_domain.jl; the tiled Berlin parent is in
#     urban_checks.jl, test and validation).

using Test
using Aeolus
using Aeolus.Components
using Aeolus.Fields: interior_view, set!
using Aeolus.Anelastic: AnelasticSimulation, isentropic_reference
using Aeolus.TimeIntegration: CFLConstraints
using Aeolus.Radiation: SlabLand
using Aeolus.SurfaceTiles: SurfaceMosaic, last_tile_fluxes, mosaic_fractions
using Aeolus.LargeScaleForcing: AdvectiveTendencies, ProfileNudging, NudgingWeight
using Aeolus.IO: checkpoint_state, restore_checkpoint_state!
using Aeolus.SoilColumn: SoilThermalColumn, SoilWaterColumn, clm_exponential_soil_grid, carsel_parrish_texture,
                         clm_soil_thermal_properties
using Serialization: serialize, deserialize
using Random: MersenneTwister
const PU = Aeolus.Urban
const PC = Aeolus.Components

include(joinpath(@__DIR__, "support", "berlin_configs.jl"))

const PM = RegularGrid(; size = (6, 4, 8), extent = (600.0, 400.0, 800.0), halo = (2, 2, 2))
pbcs() = (bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
          θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
function p_core(core::Symbol; mesh = PM, moist = true)
    core === :boussinesq && return BoussinesqSimulation(mesh; θ₀ = 300.0, moisture = moist, pbcs()...)
    return AnelasticSimulation(mesh; ref = isentropic_reference(mesh; θ_const = 300.0), pbcs()...)
end
"The reference the components convert with: the isentropic base state (the anelastic core's own)."
p_ref(sim) = sim isa BoussinesqSimulation ? isentropic_reference(sim.mesh; θ_const = 300.0) : nothing
"θ = 300 K, a uniform wind (5, 1) m/s, qv = 8 g/kg on a moist core."
function p_state!(sim)
    interior_view(sim.θ) .= 300.0
    interior_view(sim.u) .= 5.0; interior_view(sim.v) .= 1.0
    PU._moist(sim) && (interior_view(sim.qv) .= 0.008)
    return sim
end
zcentres(m) = [m.z0 + (k - 0.5) * m.dz for k in 1:m.Nz]

"""
A tiled surface on `m`: every column a street tile (hot, dry, rough) and a
park tile (cooler, moist), 70/30; the street of the odd columns forced at
level 3 (its canopy the two levels below), of the even ones at level 1.
"""
function p_tiles(m; soil::Bool = false)
    nx, ny = m.Nx, m.Ny
    street() = SlabLand(; T = 308.0, C_s = 1e6, albedo = 0.12, emissivity = 0.95, z0m = 0.5, β = 0.0)
    park() = soil ? SoilThermalColumn(SoilWaterColumn(clm_exponential_soil_grid(6), carsel_parrish_texture(:loam);
                                                      θ = 0.25), clm_soil_thermal_properties(40, 20); T_soil = 292.0) :
                    SlabLand(; T = 299.0, C_s = 2e5, albedo = 0.2, emissivity = 0.97, z0m = 0.1, β = 0.6)
    cols = Matrix{Union{Nothing,PU.ParentColumn}}(nothing, nx, ny)
    for I in CartesianIndices((nx, ny))
        kt = isodd(I[1]) ? 3 : 1
        cols[I] = PU.ParentColumn(SurfaceMosaic([street(), park()], [0.7, 0.3]), UInt8[1, 3], [0.0, 0.0], [kt, 1], kt)
    end
    z = zeros(nx, ny)
    pl = PU.ParentLanduse(nx, ny, m.dx, 0.0, 0.0, 1, zeros(Int, nx, ny, 5), z, z, z, String[])
    return PU.ParentSurface(pl, zeros(nx, ny, 5), z, z, z, z, z, zcentres(m), nothing, cols)
end
p_tiles_opts(sim; soil::Bool = false) = (; surface = p_tiles(sim.mesh; soil), period = 60.0, u_min = 0.5, reference = p_ref(sim))

"Σ_k w_k Σ_ij (a − b) over the interior levels (J m⁻² with the heat weights)."
wsum(w, a, b) = sum(w[k] * sum(view(a, :, :, k) .- view(b, :, :, k)) for k in eachindex(w))

const NOON = (start_h = 13.0, utc_offset = 2.0, date = (2019, 6, 29), lat0 = 52.522, lon0 = 13.41, elevation = 34.0)
const NIGHT = merge(NOON, (start_h = 1.0,))

@testset "Parent forcing level: a valid Monin–Obukhov height over a closed roof (#568)" begin
    SE = Aeolus.SurfaceExchange
    # the unit: the first centre at or above h, raised until z_k − d ≥ 2 z0
    zc = collect(10.0:20.0:2990.0)
    @test PU.parent_forcing_level(zc, 9.96) == 1                       # d = z0 = 0: the old rule
    @test PU.parent_forcing_level(zc, 0.0) == 1
    @test PU.parent_forcing_level(zc, 9.96; d = 9.961, z0 = 0.05) == 2  # 10 − 9.961 < 0.1: up to 30 m
    @test PU.parent_forcing_level(zc, 9.96; d = 9.0, z0 = 0.05) == 1    # 1 m ≥ 0.1 m: unchanged
    @test PU.parent_forcing_level(zc, 31.0; d = 25.0, z0 = 1.0) == 3    # 50 − 25 ≥ 2
    @test_throws ErrorException PU.parent_forcing_level(zc, 3000.0)
    # the Berlin 20 m case (#568) in miniature: 20 m cells of a 10 m raster
    # (r = 2), a block of closed roof (λp = 1 over the whole morphometry
    # window, so Macdonald's d = h and z0 = 0) at h = 9.96 m, just below the
    # first level centre (10 m), next to a block of half-built cells
    nx, ny, r = 12, 4, 2
    counts = zeros(Int, nx, ny, 5)
    counts[1:6, :, PU.SURFACE_ROOF] .= r^2
    counts[7:12, :, PU.SURFACE_ROOF] .= 2; counts[7:12, :, PU.SURFACE_STREET] .= 2
    hm = fill(9.96, nx, ny); λf = fill(0.05, nx, ny)
    pl = PU.ParentLanduse(nx, ny, 20.0, 0.0, 0.0, r, counts, λf, hm, copy(hm), String[])
    tc = PU.TileClasses((;))
    # by default the build refuses, naming the nearest tile (the closed roof:
    # first level 10 m, 0.04 m above d) and the option that allows the raise
    err = try
        PU.build_parent_surface(tc, pl; z_centres = zc, window = 2); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("20 built tile(s) in 20 column(s)", err.msg)       # columns 1–5 × 4 rows have λp = 1
    @test occursin("raise_forcing_level = true", err.msg) && occursin("h = 9.96 m", err.msg)
    ps = PU.build_parent_surface(tc, pl; z_centres = zc, window = 2, raise_forcing_level = true)
    I = CartesianIndex(2, 2)                      # deep inside the roof block
    # the record: exactly the λp = 1 columns' roof tiles, each from level 1 to 2
    @test length(ps.raised) == 20
    @test all(r -> ps.λp[r.I, r.J] == 1.0 && r.class == PU.SURFACE_ROOF && (r.k_first, r.k) == (1, 2), ps.raised)
    @test ps.λp[I] == 1.0
    @test ps.d[I] ≈ 9.96 rtol = 1e-14            # d/h = 1 at λp = 1 (round-off in A^(−λp))
    @test ps.z0[I] == 0.0                        # exactly: the drag term s is 0
    # every built tile is forced at a height ≥ 2 max(z0m, z0h) above its
    # displacement: the condition of the surface-layer solve with PALM's margin
    for J in CartesianIndices(ps.columns)
        col = ps.columns[J]
        for (i, t) in enumerate(PU._mosaic(col).tiles)
            PU.is_built_class(col.classes[i]) || continue
            z0 = roughness_lengths(t)
            @test zc[col.ktile[i]] - col.d[i] >= 2max(z0.z0m, z0.z0h)
            @test zc[col.ktile[i]] >= ps.h[J]
        end
    end
    @test ps.columns[I].ktile == [2]              # the roof tile, raised to 30 m
    # the half-built block keeps the first centre above its roofs (d < h)
    J = CartesianIndex(11, 2)
    @test ps.λp[J] == 0.5 && all(==(1), ps.columns[J].ktile)
    # and the closed-roof column now steps: at the old level (z_ref =
    # 10 − 9.96 = 0.04 m < z0m = 0.05 m) the surface-layer solve refuses
    f = SurfaceForcing(; sw_down = 600.0, lw_down = 350.0, T_air = 295.0, q_air = 0.008, U = 3.0,
                       p = 101325.0, z_ref = 10.0)
    col = ps.columns[I]
    t = PU._mosaic(col).tiles[1]
    @test_throws ArgumentError SE.solve_surface_layer(BusingerDyerBeljaarsHoltslag(); U = 3.0,
        z_m = zc[1] - col.d[1], θ_a = 296.0, θ_s = 300.0, z0m = roughness_lengths(t).z0m,
        z0h = roughness_lengths(t).z0h)
    fl = PU.step_parent_column!(ps, col, f, 60.0)
    @test isfinite(fl.H) && isfinite(fl.LE) && isfinite(fl.τ)
    # the component reports the raised tiles in results.toml (and warns at build)
    mesh = RegularGrid(; size = (nx, ny, 6), extent = (240.0, 80.0, 120.0), halo = (2, 2, 2))
    sim = p_state!(p_core(:boussinesq; mesh = mesh))
    d0 = Domain(sim; validate = false)
    ts = @test_logs (:warn, r"20 built tile\(s\) raised") PC.build(PU.TiledUrbanSurface,
        (; surface = ps, period = 60.0, u_min = 0.5, reference = p_ref(sim)), d0)
    res = Components.run_results(ts, d0)["tiled_surface_main"]
    @test res["forcing_levels_raised"] == 20
    @test res["worst_raised"]["z_ref_first"] ≈ 10.0 - 9.96 rtol = 1e-9          # 0.04 m, below z0m = 0.05 m
    @test res["worst_raised"]["z_ref"] ≈ 30.0 - 9.96 rtol = 1e-12
    # a surface with nothing raised reports 0
    ts0 = PC.build(PU.TiledUrbanSurface, p_tiles_opts(sim), d0)
    @test Components.run_results(ts0, d0)["tiled_surface_main"] == Dict{String,Any}("forcing_levels_raised" => 0)
end

@testset "TiledUrbanSurface: the air receives the tiles' mean fluxes × Δt, u*, the implicit drag ($core)" for
        core in (:boussinesq,)
    sim = p_state!(p_core(core))
    d0 = Domain(sim; validate = false)
    ts = PC.build(PU.TiledUrbanSurface, p_tiles_opts(sim), d0)
    d = Domain(sim; components = (CentredAdvection(), ts))
    @test Components.slot(ts) === :surface && Components.slot_component(d, :surface) === ts
    @test size(ts.θsrc, 3) == 2                          # the canopy of a tile forced at level 3
    θ0 = copy(interior_view(sim.θ))
    dt = 10.0
    Components.run_stage!(ts, d, Val(:surface_flux), dt)
    H = ts.sfc_mean[:H] / ts.period                      # the tiles' area-mean H over the surface step (W m⁻²)
    @test H > 50                                         # hot street, cool air
    got = wsum(ts.wθ, interior_view(sim.θ), θ0)          # what the air received, J m⁻²
    @test got ≈ H * dt rtol = 1e-9                      # the round-off of θ ≈ 300 K
    @test ts.sfc_mean[:delivered_H] ≈ H * dt rtol = 1e-12
    # only the canopy levels are heated: the street's two below level 3, the park's first
    @test all(interior_view(sim.θ)[:, :, 3:end] .== θ0[:, :, 3:end])
    @test all(interior_view(sim.θ)[2:2:end, :, 2] .== θ0[2:2:end, :, 2])
    if PU._moist(sim)
        E = ts.sfc_mean[:E] / ts.period
        @test E > 0
        @test wsum(ts.wq, interior_view(sim.qv), fill(0.008, size(θ0))) ≈ E * dt rtol = 1e-9
    end
    # u* of every column from its tiles' stress
    ref = ts.ref
    ustar(col, tf = last_tile_fluxes(col.surface), fr = mosaic_fractions(col.surface)) =
        sqrt(max(sum(fr[n] * tf[n].τ for n in eachindex(tf)), 0.0) / ref.ρ̄[1])
    @test all(I -> ts.ustar[I] == ustar(ts.surface.columns[I]), CartesianIndices(ts.ustar))
    @test Components.surface_friction_velocity(ts, d) === ts.ustar
    # the drag: u ← u / (1 + Δt a / U) at the west face of column (1, 1) (a
    # the mean of the two columns it separates, U the speed there, the
    # stress along the wind); every factor from the pre-drag velocities
    # (Jacobi, #538), so the v neighbours are the predictor's
    m = sim.mesh
    I, J, K = m.Hx + 1, m.Hy + 1, m.Hz + 1
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    u0 = sim.u.data[I, J, K]
    vf = 0.25 * (sim.v.data[I-1, J, K] + sim.v.data[I, J, K] + sim.v.data[I-1, J+1, K] + sim.v.data[I, J+1, K])
    a = 0.5 * (ts.drag[m.Nx, 1, 1] + ts.drag[1, 1, 1])
    Components.run_stage!(ts, d, Val(:post_predictor), dt)
    @test sim.u.data[I, J, K] ≈ u0 / (1 + dt * a / max(hypot(u0, vf), 0.5)) rtol = 1e-14
    @test sim.u.data[I, J, K] < u0
    @test sim.u.data[I, J, m.Hz+3] == 5.0              # above every canopy: no drag
end

# --- the canopy drag of the tiled surface (#538) -----------------------------

"A tiled surface on the core `sim` with a drag `a` (m s⁻², per column and canopy level) of our choosing."
function drag_surface(sim)
    d0 = Domain(sim; validate = false)
    ts = PC.build(PU.TiledUrbanSurface, p_tiles_opts(sim), d0)
    return ts, Domain(sim; components = (CentredAdvection(), ts))
end
drag!(ts, d, Δt) = Components.run_stage!(ts, d, Val(:post_predictor), Δt)

"""
The drag as an explicit Jacobi reference: both factors of every face from
copies of the pre-drag u, v (periodic wrap), applied in the reverse loop
order (k, j, i descending).
"""
function jacobi_reference!(ts, sim, Δt)
    m = sim.mesh; Hx, Hy, Hz, Nx, Ny = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    u0 = copy(sim.u.data); v0 = copy(sim.v.data); a = ts.drag
    for k in reverse(axes(a, 3)), j in Ny:-1:1, i in Nx:-1:1
        I, J, K = Hx + i, Hy + j, Hz + k
        au = 0.5 * (a[mod1(i - 1, Nx), j, k] + a[i, j, k])
        vf = 0.25 * (v0[I-1, J, K] + v0[I, J, K] + v0[I-1, J+1, K] + v0[I, J+1, K])
        sim.u.data[I, J, K] = u0[I, J, K] / (1 + Δt * au / max(hypot(u0[I, J, K], vf), ts.U_min))
        av = 0.5 * (a[i, mod1(j - 1, Ny), k] + a[i, j, k])
        uf = 0.25 * (u0[I, J-1, K] + u0[I+1, J-1, K] + u0[I, J, K] + u0[I+1, J, K])
        sim.v.data[I, J, K] = v0[I, J, K] / (1 + Δt * av / max(hypot(v0[I, J, K], uf), ts.U_min))
    end
    return sim
end

"The former drag loop (before #538): in place, Gauss–Seidel ordering k, j, i, periodic wrap."
function gauss_seidel_reference!(ts, sim, Δt)
    m = sim.mesh; Hx, Hy, Hz, Nx, Ny = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs)
    ud, vd, a = sim.u.data, sim.v.data, ts.drag
    for k in axes(a, 3), j in 1:Ny, i in 1:Nx
        im = i == 1 ? Nx : i - 1; jm = j == 1 ? Ny : j - 1
        I, J, K = Hx + i, Hy + j, Hz + k
        au = 0.5 * (a[im, j, k] + a[i, j, k])
        vf = 0.25 * (vd[I-1, J, K] + vd[I, J, K] + vd[I-1, J+1, K] + vd[I, J+1, K])
        ud[I, J, K] /= 1 + Δt * au / max(hypot(ud[I, J, K], vf), ts.U_min)
        av = 0.5 * (a[i, jm, k] + a[i, j, k])
        uf = 0.25 * (ud[I, J-1, K] + ud[I+1, J-1, K] + ud[I, J, K] + ud[I+1, J, K])
        vd[I, J, K] /= 1 + Δt * av / max(hypot(vd[I, J, K], uf), ts.U_min)
    end
    return sim
end

@testset "TiledUrbanSurface canopy drag: the analytic decay of a uniform wind (#538)" begin
    sim = p_state!(p_core(:boussinesq))
    ts, d = drag_surface(sim)
    m = sim.mesh; Hz = m.Hz
    kc = size(ts.drag, 3)
    canopy(f) = view(interior_view(f), :, :, 1:kc)
    # (1) a drag the surface recomputes from the wind, a = c U² (τ = ρ C_D U²
    # over the canopy depth): du/dt = -c |U| u, whose solution for a
    # uniform u₀ is u = u₀/(1 + c u₀ t). The implicit step
    # u ← u/(1 + Δt c u) is 1/u ← 1/u + c Δt: exact at every step, so the
    # tolerance is round-off (≈ 100 steps × a few ulp)
    interior_view(sim.u) .= 6.0; interior_view(sim.v) .= 0.0
    c = 2e-3; Δt = 2.0; n = 100
    for _ in 1:n
        ts.drag .= c * canopy(sim.u) .^ 2      # uniform, so the face mean is the column value
        drag!(ts, d, Δt)
    end
    @test all(isapprox.(canopy(sim.u), 6.0 / (1 + c * 6.0 * n * Δt); rtol = 1e-12))
    @test all(interior_view(sim.u)[:, :, kc+1:end] .== 6.0)   # above the canopy: untouched (copied)
    @test maximum(abs, canopy(sim.v)) < 1e-14
    # (2) the drag along the wind: a uniform (u, v) keeps its direction
    # (one factor for both components), to round-off
    interior_view(sim.u) .= 4.0; interior_view(sim.v) .= 3.0
    ts.drag .= 0.01
    drag!(ts, d, 5.0)
    @test all(isapprox.(canopy(sim.v) ./ canopy(sim.u), 0.75; rtol = 1e-14))
    @test all(isapprox.(canopy(sim.u), 4.0 / (1 + 5.0 * 0.01 / 5.0); rtol = 1e-14))
    # (3) a held deceleration a (between two surface steps): du/dt = -a,
    # u = u₀ - a t. The implicit form is first order in Δt: its error at
    # t = 20 s is ≈ t Δt a²/u (0.16 m/s at Δt = 4 s, 8 % of the decrease)
    # and halves with Δt (ratio 2 ± 10 %, the Δt² remainder)
    err(Δt) = begin
        interior_view(sim.u) .= 6.0; interior_view(sim.v) .= 0.0
        ts.drag .= 0.1
        for _ in 1:round(Int, 20 / Δt); drag!(ts, d, Δt); end
        maximum(abs, canopy(sim.u) .- (6.0 - 0.1 * 20))
    end
    e1, e2, e3 = err(4.0), err(2.0), err(1.0)
    @test 0 < e1 < 0.1 * 2.0                            # within 10 % of the 2 m/s decrease
    @test 1.8 < e1 / e2 < 2.2 && 1.8 < e2 / e3 < 2.2
end

@testset "TiledUrbanSurface canopy drag: Jacobi, independent of loop order (and threads, $(Threads.nthreads()) here) (#538)" begin
    rng = MersenneTwister(538)
    m = RegularGrid(; size = (40, 30, 8), extent = (4000.0, 3000.0, 800.0), halo = (2, 2, 2))
    sim = p_core(:boussinesq; mesh = m)
    ts, d = drag_surface(sim)
    interior_view(sim.u) .= 5 .+ 2 .* randn(rng, size(interior_view(sim.u)))
    interior_view(sim.v) .= 1 .+ 2 .* randn(rng, size(interior_view(sim.v)))
    ts.drag .= 0.05 .* rand(rng, size(ts.drag))          # Δt a/U ≤ 0.2 (Δt = 2 s, U ≥ U_min = 0.5)
    u0, v0 = copy(sim.u.data), copy(sim.v.data)
    reset!() = (copyto!(sim.u.data, u0); copyto!(sim.v.data, v0))
    jacobi_reference!(ts, sim, 2.0)
    uj, vj = copy(sim.u.data), copy(sim.v.data)
    B = Aeolus.Backends
    # serial, and the threaded backend: threaded only when this process has
    # more than one thread (`test/run.jl --threads 4`); at test/run.jl's
    # default of 1 thread both run serially. Across 1/4/12 processes the
    # drag is checked by the thread-invariance harness, section `canopy`
    # (AEOLUS_THREAD_TESTS=1, test/thread_invariance.jl)
    for b in (B.CPUBackend(; serial = true), B.CPUBackend(; threshold = 1))
        reset!()
        B.with_backend(b) do
            drag!(ts, d, 2.0)
        end
        # the same arithmetic per face as the reference, in any order: round-off only
        @test isapprox(interior_view(sim.u), view(uj, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz); rtol = 1e-14)
        @test isapprox(interior_view(sim.v), view(vj, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz); rtol = 1e-14)
    end
    # against the former Gauss–Seidel loop: they differ where a face read an
    # already-damped neighbour, by the second order of the drag increment,
    # |Δq| ≲ ε² max|U| with ε = max Δt a/U over the faces
    reset!()
    gauss_seidel_reference!(ts, sim, 2.0)
    I3 = (m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+size(ts.drag, 3))
    du = maximum(abs, view(sim.u.data, I3...) .- view(uj, I3...))
    dv = maximum(abs, view(sim.v.data, I3...) .- view(vj, I3...))
    ε = 2.0 * maximum(ts.drag) / ts.U_min
    Umax = maximum(hypot.(view(u0, I3...), view(v0, I3...)))
    @test 0 < max(du, dv) <= ε^2 * Umax
    # the change the drag itself made is far larger: the difference is the
    # ordering's, not the drag's
    @test max(du, dv) < 0.1 * maximum(abs, view(uj, I3...) .- view(u0, I3...))
end

@testset "TiledUrbanSurface canopy drag at non-periodic faces: no wrap across the domain (#538)" begin
    m = RegularGrid(; size = (6, 4, 8), extent = (600.0, 400.0, 800.0), halo = (2, 2, 2))
    Hx, Hy, Hz, Nx, Ny = m.Hx, m.Hy, m.Hz, m.Nx, m.Ny
    obdata = ProfileBoundaryData(u = 5.0, v = 1.0, w = 0.0, θ = 300.0)
    box(kind) = kind === :periodic ? BoussinesqSimulation(m; θ₀ = 300.0, pbcs()...) :
                kind === :noflux ? BoussinesqSimulation(m; θ₀ = 300.0, bcs = FieldBoundaryConditions(:noflux, :noflux, :wall),
                                                        θ_bcs = FieldBoundaryConditions(:noflux, :noflux, :noflux)) :
                BoussinesqSimulation(m; θ₀ = 300.0, bcs = open_boundary_conditions(m, obdata))
    # a strong drag in the last column and row only: the west face of
    # column 1 and the south face of row 1 see it only through a periodic wrap
    function run(kind, Δt = 10.0; edge = false)
        sim = box(kind)
        interior_view(sim.u) .= 5.0; interior_view(sim.v) .= 1.0
        # an open box's boundary data loaded, as in a run (its faces hold
        # them; before this call they hold zeros)
        kind === :open && update_open_boundaries!(sim, 0.0, 0.0)
        ts, d = drag_surface(sim)
        fill!(ts.drag, 0.0)
        ts.drag[Nx, :, :] .= 1.0; ts.drag[:, Ny, :] .= 1.0
        edge && (ts.drag[1, :, :] .= 1.0; ts.drag[:, 1, :] .= 1.0)
        drag!(ts, d, Δt)
        return sim, ts
    end
    K = Hz + 1
    # periodic: the face between column Nx and column 1 is a real face, damped
    sp, _ = run(:periodic)
    @test all(sp.u.data[Hx+1, Hy+1:Hy+Ny-1, K] .< 4.0)
    @test all(sp.v.data[Hx+1:Hx+Nx-1, Hy+1, K] .< 1.0)
    # open faces: the west / south boundary face of the normal velocity is the
    # boundary's (pinned_face_indices; the open-boundary update overwrites it):
    # no drag, so it keeps the boundary value (u = 5, v = 1), not the
    # opposite edge's drag
    for kind in (:open, :noflux)
        s, ts = run(kind)
        if kind === :open
            @test all(s.u.data[Hx+1, Hy+1:Hy+Ny-1, K] .== 5.0)    # untouched (copied)
            @test all(s.v.data[Hx+1:Hx+Nx-1, Hy+1, K] .== 1.0)
        else
            # a no-flux face is no boundary data: one-sided, the drag of
            # the column it bounds (zero there; u/1 is exact)
            @test all(s.u.data[Hx+1, Hy+1:Hy+Ny-1, K] .== 5.0)
            @test all(s.v.data[Hx+1:Hx+Nx-1, Hy+1, K] .== 1.0)
        end
        # the interior faces next to the strong column are damped as in the
        # periodic box (the mean of the two columns, a = 0.5)
        @test isapprox(s.u.data[Hx+Nx, Hy+1, K], sp.u.data[Hx+Nx, Hy+1, K]; rtol = 1e-14)
        @test all(s.u.data[Hx+Nx, Hy+1:Hy+Ny-1, K] .< 4.0)
    end
    # with a strong drag in column 1 and row 1 as well, the open and the
    # no-flux rule part (above, both leave the face alone because the
    # bounding column has no drag): the open face keeps its boundary value
    # (no drag at a pinned face), the no-flux face takes the bounding
    # column's drag (a = 1, Δt a/U ≈ 2 for u, ≈ 2-10 for v)
    so, _ = run(:open; edge = true)
    @test all(so.u.data[Hx+1, Hy+1:Hy+Ny, K] .== 5.0)          # untouched (copied)
    @test all(so.v.data[Hx+1:Hx+Nx, Hy+1, K] .== 1.0)
    sn, _ = run(:noflux; edge = true)
    @test all(sn.u.data[Hx+1, Hy+1:Hy+Ny, K] .< 2.0)           # 5/(1 + 10/5.1) ≈ 1.7
    @test all(sn.v.data[Hx+1:Hx+Nx, Hy+1, K] .< 0.5)           # 1/(1 + 10/U), U ≤ 5.1
    # one-sided at a no-flux face: the strong drag in column 1 alone acts in
    # full (a = a₁, not the periodic mean (a_Nx + a₁)/2)
    sim = box(:noflux)
    interior_view(sim.u) .= 5.0; interior_view(sim.v) .= 0.0
    ts, d = drag_surface(sim)
    fill!(ts.drag, 0.0); ts.drag[1, :, :] .= 1.0
    drag!(ts, d, 10.0)
    @test all(isapprox.(sim.u.data[Hx+1, Hy+1:Hy+Ny, K], 5.0 / (1 + 10.0 * 1.0 / 5.0); rtol = 1e-14))
end

@testset "TiledUrbanSurface: a step through the pipeline on both cores, and its checkpoint ($core, $sch)" for
        core in (:anelastic,), sch in (:rk3,)
    mk() = (s = p_state!(p_core(core)); d0 = Domain(s; scheme = sch, validate = false);
            Domain(s; scheme = sch, components = (CentredAdvection(), PC.build(PU.TiledUrbanSurface, p_tiles_opts(s; soil = true), d0))))
    d = mk()
    step!(d, 5.0)
    @test all(isfinite, interior_view(d.state.θ)) && all(isfinite, interior_view(d.state.u))
    ts = Components.slot_component(d, :surface)
    # the packed state (#526), through a serialized round trip as a checkpoint file holds it
    st = let io = IOBuffer(); serialize(io, checkpoint_state(ts)); seekstart(io); deserialize(io) end
    @test st isa PU.TiledSurfaceState
    @test st.tiles == checkpoint_state(ts).tiles       # copied, not computed
    d2 = mk()
    step!(d2, 5.0)                                      # the same air; the surface then given another
    ts2 = Components.slot_component(d2, :surface)       # history (three more tile steps; the air is
    for _ in 1:3                                        # not touched) and its scratch poisoned
        PU.step_tiles!(ts2, d2.state, 600.0)
    end
    soil(x) = [t for col in x.surface.columns if col !== nothing for t in col.surface.tiles if t isa SoilThermalColumn]
    slab(x) = [t for col in x.surface.columns if col !== nothing for t in col.surface.tiles if t isa SlabLand]
    @test all(((a, b),) -> a.T_skin != b.T_skin && a.water.T_soil != b.water.T_soil && a.H != b.H, zip(soil(ts), soil(ts2)))
    @test all(((a, b),) -> a.T != b.T, zip(slab(ts), slab(ts2)))
    for col in ts2.surface.columns
        col === nothing && continue
        for t in col.surface.tiles
            t isa SoilThermalColumn || continue          # the solver scratch: must not be read
            for ws in (t.ws, t.water.ws), f in fieldnames(typeof(ws))
                fill!(getfield(ws, f), NaN)
            end
        end
    end
    foreach(a -> fill!(a, NaN), (ts2.θsrc, ts2.qsrc, ts2.drag, ts2.ustar, ts2.wθ, ts2.wq))
    empty!(ts2.sfc_mean)
    @test any(col -> col !== nothing && any(t -> t isa SoilThermalColumn, col.surface.tiles), ts2.surface.columns)
    restore_checkpoint_state!(ts2, st)
    # every tile state back, as copied
    @test all(((a, b),) -> a.T_skin == b.T_skin && a.water.T_soil == b.water.T_soil && a.H == b.H &&
                           a.dt_sub == b.dt_sub && a.budget.surface == b.budget.surface, zip(soil(ts), soil(ts2)))
    @test all(((a, b),) -> a.T == b.T, zip(slab(ts), slab(ts2)))
    for x in (d, d2)
        step!(x, 5.0); step!(x, 70.0)                   # into the second surface interval
    end
    # the same arithmetic on the same numbers: agreement to round-off
    @test d2.state.θ.data ≈ d.state.θ.data rtol = 1e-12
    @test d2.state.u.data ≈ d.state.u.data rtol = 1e-12
    @test all(isfinite, interior_view(d2.state.θ))
    s1, s2 = ts.sfc_mean, ts2.sfc_mean
    @test keys(s1) == keys(s2) && all(k -> isapprox(s1[k], s2[k]; rtol = 1e-12), keys(s1))
    # a surface built differently is refused
    @test_throws ArgumentError restore_checkpoint_state!(ts2,
        PU.TiledSurfaceState(st.ncolumns - 1, st.offsets[1:end-1], st.tiles, st.θsrc, st.qsrc, st.drag,
                             st.ustar, st.t_surf, st.sw_down, st.lw_down, st.sfc_mean, st.wθ, st.wq))
end

@testset "TiledUrbanSurface: the surface step against a column-by-column reference (#537)" begin
    # step_tiles! (lazy forcings, the mosaic's tile fluxes filled in a loop,
    # the dynamic column launch) against the plain construction: each column's air as a vector
    # of forcings by hand, step_parent_column!, the canopy sources and u* by
    # hand. The same arithmetic per column; the area sums are folded in
    # another order (a threaded launch folds the column partials), so they
    # agree to round-off; the per-column fields to the last bits.
    mk() = (sim = p_state!(p_core(:boussinesq)); d0 = Domain(sim; validate = false);
            (sim, PC.build(PU.TiledUrbanSurface, p_tiles_opts(sim; soil = true), d0)))
    (sa, ta), (sb, tb) = mk(), mk()
    m = sa.mesh
    for s in (sa, sb)                                   # an uneven air: every column its own forcing
        for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
            I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
            s.θ.data[I, J, K] = 299.0 + 0.3i - 0.2j + 0.05k
            s.u.data[I, J, K] = 2.0 + 0.5i + 0.25j
            s.v.data[I, J, K] = -1.0 + 0.3j
            s.qv.data[I, J, K] = 0.006 + 0.0003i
        end
    end
    P = 60.0
    for _ in 1:3                                        # three surface steps
        PU.step_tiles!(ta, sa, P)
        fill_halos!(sb.u, sb.bcs); fill_halos!(sb.v, sb.bcs)
        ref, ps = tb.ref, tb.surface
        zf = PU._face_heights(m)
        θsrc, qsrc, drag = zero(tb.θsrc), zero(tb.qsrc), zero(tb.drag)
        sums = zeros(5)
        for J in 1:m.Ny, I in 1:m.Nx
            col = ps.columns[I, J]
            air = [Aeolus.SurfaceExchange.SurfaceForcing(; sw_down = tb.sw_down[], lw_down = tb.lw_down[],
                                  T_air = sb.θ.data[m.Hx+I, m.Hy+J, m.Hz+k] * ref.π̄[k],
                                  q_air = max(sb.qv.data[m.Hx+I, m.Hy+J, m.Hz+k], 0.0),
                                  U = max(hypot(0.5 * (sb.u.data[m.Hx+I, m.Hy+J, m.Hz+k] + sb.u.data[m.Hx+I+1, m.Hy+J, m.Hz+k]),
                                                0.5 * (sb.v.data[m.Hx+I, m.Hy+J, m.Hz+k] + sb.v.data[m.Hx+I, m.Hy+J+1, m.Hz+k])),
                                          tb.U_min),
                                  p = ref.p̄[k], z_ref = 10.0) for k in 1:maximum(col.ktile)]
            agg = PU.step_parent_column!(ps, col, air, P; U_min = tb.U_min)
            sums .+= (agg.Rn, agg.H, agg.LE, agg.G, agg.E)
            tf, fr = last_tile_fluxes(col.surface), mosaic_fractions(col.surface)
            for n in eachindex(tf)
                nl = max(1, col.ktile[n] - 1)
                for k in 1:min(nl, size(θsrc, 3))
                    θsrc[I, J, k] += fr[n] * tf[n].H / (zf[nl+1] * ref.ρ̄[k] * ref.cp * ref.π̄[k])
                    qsrc[I, J, k] += fr[n] * tf[n].E / (zf[nl+1] * ref.ρ̄[k])
                    drag[I, J, k] += fr[n] * tf[n].τ / (zf[nl+1] * ref.ρ̄[k])
                end
            end
        end
        # the same terms in the same order: agreement to the last bits (a tolerance, not ==: computed values)
        @test ta.θsrc ≈ θsrc rtol = 1e-14
        @test ta.qsrc ≈ qsrc rtol = 1e-14
        @test ta.drag ≈ drag rtol = 1e-14
        ustar(col) = sqrt(max(sum(mosaic_fractions(col.surface) .* getproperty.(last_tile_fluxes(col.surface), :τ)), 0.0) /
                          tb.ref.ρ̄[1])
        @test all(I -> isapprox(ta.ustar[I], ustar(tb.surface.columns[I]); rtol = 1e-14), CartesianIndices(ta.ustar))
        for (n, key) in enumerate((:Rn, :H, :LE, :G, :E))
            tb.sfc_mean[key] += sums[n] / (m.Nx * m.Ny) * P
            @test ta.sfc_mean[key] ≈ tb.sfc_mean[key] rtol = 1e-12           # the fold order only
        end
    end
    @test any(!=(0), ta.θsrc) && length(unique(ta.ustar)) > 1               # an uneven, nonzero answer
    # every tile in the same state
    tiles(t) = [x for col in t.surface.columns for x in col.surface.tiles]
    Tsfc = Aeolus.SurfaceExchange.surface_temperature
    @test all(((a, b),) -> isapprox(Tsfc(a), Tsfc(b); rtol = 1e-14), zip(tiles(ta), tiles(tb)))
end

@testset "TiledUrbanSurface temperatures: area weighting, the 2 m Monin–Obukhov value and interpolation (#536, $core)" for
        core in (:boussinesq, :anelastic)
    SE = Aeolus.SurfaceExchange
    sim = p_state!(p_core(core; moist = false))
    d0 = Domain(sim; validate = false)
    ts = PC.build(PU.TiledUrbanSurface, p_tiles_opts(sim), d0)
    d = Domain(sim; components = (CentredAdvection(), ts))
    out = Dict(v.name => v for v in Components.outputs(ts))
    @test collect(keys(out)) ⊆ collect(Components.output_names(PU.TiledUrbanSurface, nothing))
    @test length(out) == length(Components.output_names(PU.TiledUrbanSurface, nothing))
    get(n) = out[n].getter(d)
    ref = ts.ref
    zc = zcentres(sim.mesh)
    Π(z) = Components.reference_exner(ref, zc, z; g = 9.81)
    # the skin temperatures: street 308 K (70 %), park 299 K (30 %) in every
    # column; the mean is one weighted sum, to round-off
    @test all(isapprox.(get("surface_temperature"), 0.7 * 308.0 + 0.3 * 299.0; rtol = 1e-14))
    @test all(get("surface_temperature_street") .== 308.0)          # copied, not computed
    @test all(get("surface_temperature_park") .== 299.0)
    @test all(n -> all(isnan, get("surface_temperature_$n")), ("roof", "water", "wall"))   # fraction 0: NaN
    @test all(isapprox.(get("surface_theta"), (0.7 * 308.0 + 0.3 * 299.0) / Π(0.0); rtol = 1e-14))
    @test out["surface_temperature"].standard_name == "surface_temperature" &&
          out["temperature_2m"].standard_name == "air_temperature"
    # the 2 m value by hand: column (2, 1), both tiles forced at level 1
    # (z₁ = 50 m > 2 m: Monin–Obukhov), the surface-layer inputs assembled
    # as the tiles' fluxes do (θ_a = T_a + g z/c_p, θ_s = T_s, dry air),
    # solved, and θ(2 m) = [T_s + θ*/κ (ln(2/z0h) − ψh(2/L) + ψh(z0h/L))]/Π̄_s
    col = ts.surface.columns[2, 1]
    @test col.ktile == [1, 1]
    θ1 = 300.0; T1 = θ1 * ref.π̄[1]
    sf = BusingerDyerBeljaarsHoltslag()
    κ = SE.DEFAULT_SURFACE_CONSTANTS.κ
    cpd = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS.cp_dry
    g0 = SE.grav(SE.DEFAULT_SURFACE_CONSTANTS)
    θ2_tiles = map(col.surface.tiles) do t
        z0 = roughness_lengths(t)
        # the park's skin is moist (β = 0.6): q_s = β q_sat(T_s, p̄₁) over the dry air
        qs = SE.surface_specific_humidity(t, SurfaceForcing(; T_air = T1, q_air = 0.0, U = 5.0, p = ref.p̄[1], z_ref = 50.0))
        sol = solve_surface_layer(sf; U = hypot(5.0, 1.0), z_m = 50.0, θ_a = T1 + g0 / cpd * 50.0,
                                  θ_s = surface_temperature(t), q_a = 0.0, q_s = qs,
                                  z0m = z0.z0m, z0h = z0.z0h, U_min = 0.5)
        L = sol.L
        (surface_temperature(t) + sol.θ_star / κ * (log(2.0 / z0.z0h) - psi_h(sf, 2.0 / L) + psi_h(sf, z0.z0h / L))) / Π(0.0)
    end
    # the same solve and formula: agreement to round-off (the secant
    # iteration converges to its own tolerance on both sides identically)
    @test get("theta_2m")[2, 1] ≈ 0.7 * θ2_tiles[1] + 0.3 * θ2_tiles[2] rtol = 1e-12
    @test get("temperature_2m")[2, 1] ≈ (0.7 * θ2_tiles[1] + 0.3 * θ2_tiles[2]) * Π(2.0) rtol = 1e-12
    # each tile's 2 m θ between its skin's and the first level's (the street
    # unstable, 8 K above the air; the park 1 K below it, stable)
    θsk = [surface_temperature(t) / Π(0.0) for t in col.surface.tiles]
    @test θ1 < θ2_tiles[1] < θsk[1] && θsk[2] < θ2_tiles[2] < θ1
    # the 2 m air between the surface and the first level, in every column
    T2, Tsfc = get("temperature_2m"), get("surface_temperature")
    @test all(T1 .< T2 .< Tsfc)
    # interpolation: a street tile whose surface (its displacement height,
    # 49 m) puts its forcing level 1 m above it; θ = 300 K + 0.01 z (linear
    # in the level centres), so θ at 49 + 2 = 51 m is exactly 300.51 K
    set!(sim.θ, (x, y, z) -> 300.0 + 0.01z)
    t = col.surface.tiles[1]
    f = SurfaceForcing(; T_air = 300.5 * ref.π̄[1], U = 5.0, p = ref.p̄[1], z_ref = 1.0)
    θc = [300.0 + 0.01z for z in zc]
    @test PU.screen_theta(sf, t, f, θc, zc, 49.0, Π(49.0)) ≈ 300.51 rtol = 1e-14
    # ... and below the first level centre, toward the skin's θ at the surface
    @test PU.screen_theta(sf, t, f, θc, zc, 47.0, Π(47.0)) ≈ 308.0 / Π(47.0) + (300.5 - 308.0 / Π(47.0)) * 2 / 3 rtol = 1e-14
    # the cache: one evaluation per model time
    @test PU.tile_temperatures(ts, d) === PU.tile_temperatures(ts, d)
end

@testset "PrescribedSurfaceFlux: Q Δt/Δz₁ into every column of the first level ($core)" for core in (:boussinesq,)
    sim = p_state!(p_core(core))
    d0 = Domain(sim; validate = false)
    pf = PC.build(PU.PrescribedSurfaceFlux, (; q_surface = PU.ConstantForcing(0.12),
                                             e_surface = PU.SeriesForcing([0.0, 100.0], [0.0, 2e-5]),
                                             reference = p_ref(sim)), d0)
    d = Domain(sim; components = (CentredAdvection(), pf))
    θ0 = copy(interior_view(sim.θ))
    Components.run_stage!(pf, d, Val(:surface_flux), 10.0)
    @test interior_view(sim.θ)[:, :, 1] ≈ θ0[:, :, 1] .+ 0.12 * 10.0 / PM.dz rtol = 1e-15
    @test interior_view(sim.θ)[:, :, 2:end] == θ0[:, :, 2:end]
    ref = pf.ref
    @test pf.sfc_mean[:H] ≈ 0.12 * 10.0 * ref.ρ̄[1] * ref.cp * ref.π̄[1] rtol = 1e-14
    # E over [0, 10] s of the series: its exact mean, 1e-6 kg/kg m/s
    PU._moist(sim) && @test interior_view(sim.qv)[:, :, 1] ≈ fill(0.008 + 1e-6 * 10.0 / PM.dz, PM.Nx, PM.Ny) rtol = 1e-14
    @test Components.wall_theta_flux(pf, d) == (0.0, 0.0)       # none to the TKE (as the parent since #259)
end

@testset "MeanColumnRadiation: the site's sun, Ṫ Δt / Π̄, the tiles' downwelling, one call per period ($core)" for
        core in (:boussinesq,)
    for (cfg, day) in ((NOON, true), (NIGHT, false))
        sim = p_state!(p_core(core))
        d0 = Domain(sim; validate = false)
        ts = PC.build(PU.TiledUrbanSurface, p_tiles_opts(sim), d0)
        rad = PC.build(PU.MeanColumnRadiation, (; period = 600.0, sun = PU.SiteSun(cfg), reference = p_ref(sim)), d0)
        d = Domain(sim; components = (CentredAdvection(), ts, rad))
        # the surface first, then the radiation over it (stage 3, order 1)
        @test findfirst(c -> c === ts, d.components) < findfirst(c -> c === rad, d.components)
        θ0 = copy(interior_view(sim.θ))
        Components.run_stage!(rad, d, Val(:surface_flux), 10.0)
        @test rad.model.ncalls == 1 && rad.t_rad[] == 600.0
        @test (ts.sw_down[], ts.lw_down[]) == (rad.sw_down[], rad.lw_down[])
        @test 250 < rad.lw_down[] < 450
        if day
            @test rad.sw_down[] > 600                    # clear sky, 13:00 CEST, 29 June, Berlin
            @test maximum(rad.model.heat_sw) > 0         # the vapour absorbs sunlight
        else
            @test rad.sw_down[] == 0
            @test all(==(0), rad.model.heat_sw) && rad.model.heat == rad.model.heat_lw
        end
        π̄ = rad.π̄
        @test all(k -> isapprox(interior_view(sim.θ)[:, :, k], θ0[:, :, k] .+ rad.model.heat[k] * 10.0 / π̄[k];
                                rtol = 1e-15), 1:PM.Nz)
        # the next step, within the period: the heating again, no call
        sim.t = 10.0
        Components.run_stage!(rad, d, Val(:surface_flux), 10.0)
        @test rad.model.ncalls == 1
        # its checkpoint: the last call's fluxes and heating, the next call's time
        st = checkpoint_state(rad)
        h = copy(rad.model.heat); sw = rad.sw_down[]
        rad.model.heat .= 0; rad.t_rad[] = 0.0; rad.sw_down[] = -1.0
        restore_checkpoint_state!(rad, st)
        @test rad.model.heat == h && rad.t_rad[] == 600.0 && rad.sw_down[] == sw
    end
end

@testset "LargeScale: the boundary-layer tracker, the nudging weight above it, a checkpoint ($core)" for
        core in (:anelastic,)
    sim = p_core(core; moist = false)
    zi0 = 400.0
    θm(z) = z < zi0 ? 300.0 : 300.0 + 0.01 * (z - zi0)
    set!(sim.θ, (x, y, z) -> θm(z))
    tr = BoundaryLayerTracker(; floor = 150.0, every = 2)
    nudge = ProfileNudging((θ = (z, t) -> θm(z) + 1.0,), 3600.0;
                           weight = NudgingWeight(_ -> nudging_height(tr); factor = 1.2, depth = 0.0))
    ls = LargeScale(nothing, (nudge,); zi = tr)
    d = Domain(sim; components = (CentredAdvection(), ls))
    @test any(s -> s.name === :column_tendency, Components.stages(ls))
    @test nudging_height(tr) == 150.0                    # the floor before the first estimate
    Components.run_stage!(ls, d, Val(:column_tendency), 1.0)
    @test tr.zi[] == 0.0                                  # every 2 steps
    Components.run_stage!(ls, d, Val(:column_tendency), 1.0)
    @test abs(tr.zi[] - zi0) <= PM.dz                     # the parcel height of the mixed layer, to a level
    @test tr.zi[] == parcel_zi(sim)
    # the nudging acts above 1.2 z_i only
    fill!(sim.tθ.data, 0.0)
    Components.run_stage!(ls, d, Val(:body_force), 1.0)
    zc = zcentres(PM)
    tθ = interior_view(sim.tθ)
    zn = 1.2 * nudging_height(tr)
    @test all(tθ[:, :, k] == zeros(PM.Nx, PM.Ny) for k in 1:PM.Nz if zc[k] <= zn)
    @test all(all(tθ[:, :, k] .≈ 1 / 3600) for k in 1:PM.Nz if zc[k] > zn)
    # its state is the tracker's
    s = checkpoint_state(ls)
    zi = tr.zi[]; tr.zi[] = 0.0; tr.n[] = 7
    restore_checkpoint_state!(ls, s)
    @test tr.zi[] == zi && tr.n[] == 2
end

@testset "LargeScale clocks in an SSP-RK3 step: the step mean, or the core's substage clock ($core)" for
        core in (:boussinesq,)
    c = 1e-4                                              # dθ/dt = c t (K s⁻¹, t in s)
    Δθ = Dict{Symbol,Float64}()
    for clk in (:step, :core)
        sim = p_core(core; moist = false)
        interior_view(sim.θ) .= 300.0
        sim.t = 100.0
        ls = LargeScale(nothing, (AdvectiveTendencies(; θ = (z, t) -> c * t),); clock = clk)
        d = Domain(sim; scheme = :rk3, components = (CentredAdvection(), ls))
        θ0 = interior_view(sim.θ)[3, 2, 4]
        step!(d, 10.0)
        Δθ[clk] = interior_view(sim.θ)[3, 2, 4] - θ0
    end
    t0, Δt = 100.0, 10.0
    # :step — the exact step mean c (t0 + Δt/2) Δt;
    # :core — windows [t0 + (s-1)Δt, t0 + sΔt] weighted 1/6, 1/6, 2/3: c (t0 + 2Δt) Δt
    @test Δθ[:step] ≈ c * (t0 + Δt / 2) * Δt rtol = 1e-9
    @test Δθ[:core] ≈ c * (t0 + 2Δt) * Δt rtol = 1e-9
    @test_throws ArgumentError LargeScale(nothing, (); clock = :substage)
end

@testset "[sgs] model = \"tke\" on a stretched grid: the exact wall distance" begin
    faces = [0.0, 10.0, 25.0, 45.0, 75.0, 120.0, 180.0, 260.0, 360.0]
    m = StretchedGrid(; size = (4, 4, 8), extent = (100.0, 100.0, 360.0), halo = (2, 2, 2), stretching = (:custom, faces))
    zc = [(faces[k] + faces[k+1]) / 2 for k in 1:8]
    for (top, dist) in ((false, zc), (true, min.(zc, 360.0 .- zc)))
        sim = BoussinesqSimulation(m; θ₀ = 300.0, pbcs()...)
        interior_view(sim.θ) .= 300.0
        d0 = Domain(sim; validate = false)
        s = PC.build(SGS, (; model = :tke, e_min = 1e-4, surface_factor = 3.9, wall_cap = true, wall_cap_top = top,
                            surface_value = false, stress = :laplacian), d0)
        c = s.closure
        @test all(c.l_wall[:, :, k] == fill(0.4 * dist[k], 4, 4) for k in 1:8) &&
              all(c.first_level[:, :, 1]) && !any(c.first_level[:, :, 2:end])
    end
end

@testset "[sgs] surface_value: the first-level e is u*²/√(c_k c_ε) of the surface's u* ($core)" for
        core in (:anelastic,)
    sim = p_state!(p_core(core; moist = false))
    d0 = Domain(sim; validate = false)
    sl = PC.build(SurfaceLayer, (; model = :most, z0m = 0.1, z0h = nothing,
                                 temperature = Dict{String,Any}("kind" => "constant", "value" => 302.0),
                                 stability = Dict{String,Any}("family" => "businger_dyer_beljaars_holtslag"),
                                 rho_cp = 1206.0), d0)
    sgs = PC.build(SGS, (; model = :tke, e_min = 1e-4, surface_factor = 3.9, wall_cap = false, wall_cap_top = true,
                          surface_value = true, stress = :laplacian), d0)
    d = Domain(sim; components = (CentredAdvection(), sl, sgs))
    step!(d, 2.0)
    c = sgs.closure
    cε = Aeolus.PrognosticSGS.tke_dissipation_coefficient(c.Δk[1], c.Δk[1]; factor = 3.9)
    e1 = interior_view(c.e)[:, :, 1]
    @test all(sl.ustar .> 0.1)
    @test e1 ≈ Aeolus.PrognosticSGS.tke_surface_value.(sl.ustar, cε) rtol = 1e-14
    # without a surface component that computes u*, the closure says so
    sim2 = p_state!(p_core(core; moist = false))
    s2 = PC.build(SGS, (; model = :tke, e_min = 1e-4, surface_factor = 3.9, wall_cap = false, wall_cap_top = true,
                         surface_value = true, stress = :laplacian), Domain(sim2; validate = false))
    @test_throws ArgumentError step!(Domain(sim2; components = (CentredAdvection(), s2)), 2.0)
end
