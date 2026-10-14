# The example cases as configurations of the model (EX-d #441; the
# validation cases of the former run types of Aeolus.RunDrivers since
# ONE-l, #475). Each example used to be a script that built and ran its
# model; it is now a TOML file run by `aeolus run`. For each one this file
# runs the config at a small size and the old script's code at the same
# size, verbatim but for the plots and the size, and asserts that the two
# give the same bits (where the port is not a stated physics change); then
# it asserts the physics the example shows.

using Test
using Aeolus
using Aeolus.Config
using Aeolus.Fields: set!, interior_view
using Random
using TOML

include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const RD_EX = joinpath(pkgdir(Aeolus), "examples")
# the core and the couplings of a run: a `simulation` run is a Model (ONE-a, #450)
rd_sim(o) = o isa Aeolus.SimulationDriver.SimulationRun ? Aeolus.SimulationDriver.simulation(o) : o.sim
rd_couplings(o) = Aeolus.SimulationDriver.sim_couplings(o)

"Run the config `path` with `overrides` (the ESTIMATE planner, a temporary root); return the RunResult."
function rd_run(path, overrides = String[])
    cfg = load_run_config(path; overrides = vcat(overrides, "fftw.planner=\"estimate\""))
    return run_config(cfg; root = mktempdir(), console = false, repo = mktempdir())
end

"The `output/results.toml` of a finished run, parsed."
rd_results(r) = TOML.parsefile(joinpath(r.folder.path, "output", "results.toml"))

"The shipped config of the example case `name`, examples/<name>/<name>.toml."
rd_case(name) = joinpath(RD_EX, name, name * ".toml")

# ---------------------------------------------------------------------------
# examples/scalar_advection_2d/scalar_advection_2d.toml ([kinematic_flow], ONE-l)
# ---------------------------------------------------------------------------

"The NetCDF output of a finished run, opened."
rd_output(r) = NCDataset(joinpath(r.folder.path, "output", "output.nc"))

@testset "scalar advection: θ in a prescribed solid-body rotation is conserved and carried around" begin
    # a stated physics change of the port (ONE-l): the `advection` run type
    # advected a cell-centred scalar with first-order upwind fluxes in a
    # cell-centred velocity; on the model θ is carried by the core's centred
    # flux form on the staggered grid, so the old bits are not reproduced
    f = rd_case("scalar_advection_2d")
    r = rd_run(f, config_test_table(f).overrides)        # 32², a quarter revolution (64 steps)
    sim = rd_sim(r.obj); m = sim.mesh
    ds = rd_output(r)
    t = ds["time"][:]; θm = ds["theta_mean"][:]; θ = ds["theta"][:, :, 1, end]
    x = ds["x"][:]; y = ds["y"][:]
    close(ds)
    # the velocity is exactly the prescribed rotation (u = −ω(y − ½), v = ω(x − ½))
    ω = 2π
    xc = collect(Aeolus.Grids.xnodes(m, CellCenter())); yc = collect(Aeolus.Grids.ynodes(m, CellCenter()))
    @test all(interior_view(sim.u)[i, j, 1] == -ω * (yc[j] - 0.5) for i in 1:m.Nx, j in 1:m.Ny)
    @test all(interior_view(sim.v)[i, j, 1] == ω * (xc[i] - 0.5) for i in 1:m.Nx, j in 1:m.Ny)
    # the flux form on a periodic box conserves ∫θ to round-off
    @test t[end] ≈ 0.25 && all(v -> isapprox(v, θm[1]; rtol = 1e-13), θm)
    # a quarter turn counterclockwise about (½, ½) takes the peak from (¾, ½)
    # to (½, ¾) (within a cell); the centred scheme keeps it within 7 % of
    # its start (measured 0.952 of 0.963 at 32²)
    I = argmax(θ)
    @test abs(x[I[1]] - 0.5) <= m.dx && abs(y[I[2]] - 0.75) <= m.dy
    @test 0.93 < maximum(θ) <= 1
end

# ---------------------------------------------------------------------------
# the kernel demos: the simulation driver with its couplings
# ---------------------------------------------------------------------------


"The interior of a halo-padded array of `mesh`."
rd_interior(a, mesh) = a[mesh.Hx+1:mesh.Hx+mesh.Nx, mesh.Hy+1:mesh.Hy+mesh.Ny, mesh.Hz+1:mesh.Hz+mesh.Nz]

"""
Whether the state of the run `r` is bitwise that of the Boussinesq simulation
`sim` (the interiors of u, v, w, p, θ, qv: the run monitor's diagnostics
fill halos after a step, a bare loop does not).
"""
function rd_same_state(r, sim)
    f = Base.invokelatest(state_fields, r.obj)
    m = sim.mesh
    return all(bitwise_equal(rd_interior(getproperty(f, k), m), rd_interior(getproperty(sim, k).data, m))
               for k in (:u, :v, :w, :p, :θ, :qv) if haskey(f, k))
end

@testset "kernels: tgv_energy_decay.toml is the old script, bit for bit; KE decays as exp(-6νt)" begin
    N = 16; T_end = 1.0
    # examples/kernels/01_tgv_energy_decay.jl before EX-d, verbatim but for N, T_end and the plot
    L = 2π; ν = 1e-2
    mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
    sim  = BoussinesqSimulation(mesh; ν = ν)
    set!(sim.u, (x,y,z) ->  sin(x)*cos(y)*cos(z))
    set!(sim.v, (x,y,z) -> -cos(x)*sin(y)*cos(z))
    dx  = mesh.dx
    dt  = min(0.4dx / 1.0, 0.4dx^2 / (2ν))
    nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps
    ke0 = kinetic_energy(sim)
    for n in 1:nsteps
        step!(sim, dt)
    end
    r = rd_run(rd_case("tgv_energy_decay"),
               ["grid.size=[$N,$N,$N]", "time.dt=$(repr(min(0.4dx, 0.4dx^2 / (2ν))))", "time.duration=$T_end"])
    @test rd_same_state(r, sim)
    # the physics: over t = 1 the decay is the linear viscous one, KE₀ e^(−6νt),
    # to 1 % (measured +0.6 % at 16³: the discrete Laplacian's smaller
    # eigenvalue and the energy the forward-Euler step of the centred
    # advection adds both slow it down a little), and the projection keeps
    # ∇·u at round-off
    ratio = kinetic_energy(rd_sim(r.obj)) / ke0
    @test isapprox(ratio, exp(-6ν * T_end); rtol = 0.01)
    @test max_divergence(rd_sim(r.obj)) < 1e-12
end

@testset "kernels: decaying_hit.toml is the old script, bit for bit; KE decays" begin
    N = 16; T_end = 0.6
    # examples/kernels/02_decaying_hit.jl before EX-d, verbatim but for N, T_end and the plot
    L = 2π; ν = 5e-3
    mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
    sim  = BoussinesqSimulation(mesh; ν = ν, C_s = 0.17)
    Random.seed!(42)
    interior_view(sim.u) .= 0.3 .* randn(N,N,N)
    interior_view(sim.v) .= 0.3 .* randn(N,N,N)
    interior_view(sim.w) .= 0.3 .* randn(N,N,N)
    dx = mesh.dx
    dt = 0.3 * dx / 1.0
    nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps
    kes = [kinetic_energy(sim)]
    for n in 1:nsteps
        step!(sim, dt)
        push!(kes, kinetic_energy(sim))
    end
    r = rd_run(rd_case("decaying_hit"),
               ["grid.size=[$N,$N,$N]", "time.dt=$(repr(0.3 * dx))", "time.duration=$T_end"])
    @test rd_same_state(r, sim)
    # the physics: after the first projection the energy only decays
    @test issorted(kes[2:end]; rev = true) && kes[end] < 0.9kes[2]
end

@testset "kernels: channel_flow.toml is the old script, bit for bit; the walls hold w = 0" begin
    Nx = Ny = 8; Nz = 16; T_end = 0.25
    # examples/kernels/03_channel_flow.jl before EX-d, verbatim but for the size, T_end and the plot
    L = 1.0; ν = 5e-4
    mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(L,L,L), halo=(2,2,2))
    wall = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim  = BoussinesqSimulation(mesh; ν=ν, C_s=0.17, Pr_t=1/3, bcs=wall)
    Random.seed!(1)
    set!(sim.u, (x,y,z) -> 1.0 + 0.05*(rand()-0.5))
    set!(sim.v, (x,y,z) -> 0.05*(rand()-0.5))
    set!(sim.w, (x,y,z) -> 0.05*(rand()-0.5))
    dx  = mesh.dx; dz = mesh.dz
    dt  = min(0.3 * dx / 1.2, 0.4 * dz^2 / (2ν))
    nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps
    for n in 1:nsteps
        step!(sim, dt)
    end
    r = rd_run(rd_case("channel_flow"),
               ["grid.size=[$Nx,$Ny,$Nz]", "time.dt=$(repr(min(0.3dx / 1.2, 0.4dz^2 / (2ν))))",
                "time.duration=$T_end"])
    @test rd_same_state(r, sim)
    # the physics: the free-slip walls are impermeable, the mean flow is kept
    w = rd_sim(r.obj).w.data; H = mesh.Hz
    @test all(iszero, w[:, :, H+1]) && max_divergence(rd_sim(r.obj)) < 1e-12
    @test 0.99 < sum(interior_view(rd_sim(r.obj).u)) / (Nx * Ny * Nz) < 1.01
end

@testset "kernels: pollution_plume.toml is the old script, bit for bit; tracer budget" begin
    T_end = 3.0
    # examples/kernels/05_pollution_plume.jl before EX-d, verbatim but for T_end and the plot
    Nx = 64; Ny = 48; Nz = 8; L = 100.0
    mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(2L, 1.5L, L), halo=(2,2,2))
    sim = BoussinesqSimulation(mesh; ν = 1e-3, C_s = 0.15)
    set!(sim.u, (x,y,z) -> 5.0)
    set!(sim.v, (x,y,z) -> 0.0)
    set!(sim.w, (x,y,z) -> 0.0)
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    c  = Field(CellCenter(), mesh)
    tc = Field(CellCenter(), mesh)
    set!(c, (x,y,z) -> 0.0)
    i_src = round(Int, 20.0 / mesh.dx)
    j_src = round(Int, (1.5L / 2) / mesh.dy)
    k_src = 4
    S_val = 1.0
    dt = 0.3 * mesh.dx / 5.0
    nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    for n in 1:nsteps
        fill_halos!(c, FieldBoundaryConditions(:periodic))
        fill!(tc.data, 0)
        advect_centered!(tc, sim.u, sim.v, sim.w, c)
        apply_linear_chemistry!(tc, c, 60.0)
        tc.data[Hx + i_src, Hy + j_src, Hz + k_src] += S_val
        interior_view(c) .+= dt .* view(tc.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
        step!(sim, dt)
    end
    r = rd_run(rd_case("pollution_plume"), ["time.duration=$T_end"])
    @test rd_same_state(r, sim)
    f = Base.invokelatest(state_fields, r.obj)
    @test bitwise_equal(rd_interior(f.tracer, mesh), rd_interior(c.data, mesh))
    @test only(rd_couplings(r.obj).tracer.sources)[1:3] == (Hx + i_src, Hy + j_src, Hz + k_src)
    # the physics: centred advection in a periodic box conserves the tracer,
    # so the cell sum obeys dM/dt = S − M/τ exactly, step by step (forward Euler)
    M = 0.0
    for n in 1:nsteps
        M += dt * (S_val - M / 60.0)
    end
    @test isapprox(sum(interior_view(rd_couplings(r.obj).tracer.c)), M; rtol = 1e-12)
    # and the plume is carried downwind: the tracer is east of the source
    ci = interior_view(rd_couplings(r.obj).tracer.c)
    @test sum(ci[i_src+1:end, :, :]) > 2sum(ci[1:i_src-1, :, :])
end

@testset "kernels: particle_trail.toml is the old script, bit for bit" begin
    N = 16; nsteps = 20
    # examples/kernels/06_particle_trail.jl before EX-d, verbatim but for N, nsteps and the plot
    L = 2π
    mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
    sim  = BoussinesqSimulation(mesh; ν=1e-2)
    set!(sim.u, (x,y,z) ->  sin(x)*cos(y)*cos(z))
    set!(sim.v, (x,y,z) -> -cos(x)*sin(y)*cos(z))
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    ps = ParticleSet(mesh)
    Random.seed!(7)
    add_particles!(ps, [(L*rand(), L*rand(), L/2) for _ in 1:24])
    dt = 0.05
    for n in 1:nsteps
        step!(sim, dt)
        fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
        advance_particles!(ps, sim, dt)
    end
    r = rd_run(rd_case("particle_trail"), ["grid.size=[$N,$N,$N]", "time.nsteps=$nsteps"])
    @test rd_same_state(r, sim)
    f = Base.invokelatest(state_fields, r.obj)
    @test f.particles == [x[d] for d in 1:3, x in positions(ps)]
    # the trajectories file: 24 particles at steps 0..nsteps, inside the box
    rows = readlines(joinpath(r.folder.path, "output", "particles.csv"))
    @test rows[1] == "step,t,id,x,y,z" && length(rows) == 1 + 24 * (nsteps + 1)
    last_ = [parse.(Float64, split(l, ',')) for l in rows[end-23:end]]
    @test all(p -> 0 <= p[4] < L && 0 <= p[5] < L, last_)
    @test [p[4:6] for p in last_] == [collect(x) for x in positions(ps)]
end

# ---------------------------------------------------------------------------
# the coupled examples
# ---------------------------------------------------------------------------

"examples/coupled/01_dcbl_radiation_lsm.jl before EX-d at N³ for `nsteps`, verbatim but for the size and the plots; `fix` sets the surface-flux flag."
function rd_dcbl_script(N, nsteps; fix::Bool)
    Lx = Ly = 3200.0; Lz = 1600.0
    Nx = Ny = N; Nz = N
    mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))
    wall = FieldBoundaryConditions(:periodic, :periodic, :wall)
    θ_bc = FieldBoundaryConditions(:periodic, :periodic, :noflux)
    sim = BoussinesqSimulation(mesh;
        θ₀ = 300.0, g_accel = 9.81,
        C_s = 0.17, Pr_t = 1/3,
        sponge_α_max = 0.01, sponge_z_start = 0.75*Lz,
        bcs = wall, θ_bcs = θ_bc)
    fix && (sim.has_surface_flux = true)
    Random.seed!(1)
    set!(sim.θ, (x,y,z) -> 300.0 + max(0, (z - 800.0)) * 3e-3 + 0.05*(rand()-0.5))
    land = Aeolus.Radiation.SlabLand(; T = 297.0, C_s = 1e5, albedo = 0.25)
    cool_profile = fill(-2.315e-5, Nz)
    T_end = 3.0 * 3600.0
    dt    = 5.0
    ρcp = 1.0 * 1005.0
    Qs = Float64[]
    for n in 1:nsteps
        sw_in = 800.0 * sin(π * sim.t / T_end)
        lw_in = 340.0
        Q_sfc = (Aeolus.Radiation.surface_temperature(land) - 300.0) * 1e-2
        sim.θ_flux_bottom = Q_sfc
        fill!(sim.tθ.data, 0)
        step!(sim, dt)
        Aeolus.Radiation.apply_radiative_cooling!(sim.θ, cool_profile .* dt)
        Aeolus.Radiation.step_land!(land, sw_in, lw_in, ρcp * Q_sfc, dt)
        push!(Qs, Q_sfc)
    end
    return sim, land, Qs
end

@testset "coupled: dcbl_radiation_lsm.toml is the old script with its surface flux applied, bit for bit" begin
    N = 16; nsteps = 30
    old, land_old, _ = rd_dcbl_script(N, nsteps; fix = false)
    fixed, land, Qs = rd_dcbl_script(N, nsteps; fix = true)
    # The script's own bug, for the record: the flux it set never reached
    # the air (the flag is frozen at construction), so its θ was that of a
    # run with no land at all, while its ground cooled by the flux it lost
    @test !old.has_surface_flux
    noland = BoussinesqSimulation(old.mesh; θ₀ = 300.0, g_accel = 9.81, C_s = 0.17, Pr_t = 1/3,
                                  sponge_α_max = 0.01, sponge_z_start = 1200.0,
                                  bcs = old.bcs, θ_bcs = old.θ_bcs)
    Random.seed!(1)
    set!(noland.θ, (x,y,z) -> 300.0 + max(0, (z - 800.0)) * 3e-3 + 0.05*(rand()-0.5))
    for n in 1:nsteps
        step!(noland, 5.0)
        Aeolus.Radiation.apply_radiative_cooling!(noland.θ, fill(-2.315e-5, N) .* 5.0)
    end
    @test interior_view(old.θ) == interior_view(noland.θ)
    @test interior_view(fixed.θ) != interior_view(old.θ)
    # the config is the script with the flux applied
    r = rd_run(rd_case("dcbl_radiation_lsm"),
               ["grid.size=[$N,$N,$N]", "time.duration=\"none\"", "time.nsteps=$nsteps"])
    @test rd_same_state(r, fixed)
    f = Base.invokelatest(state_fields, r.obj)
    @test f.T_s == [land.T]
    # the physics: the domain heat budget closes. Advection and diffusion
    # conserve Σθ in the closed box, so per step Σθ changes by the floor
    # flux Q Δt/Δz over one level plus the cooling over all cells
    m = fixed.mesh
    Σθ0 = let s = BoussinesqSimulation(m; θ₀ = 300.0)
        Random.seed!(1)
        set!(s.θ, (x,y,z) -> 300.0 + max(0, (z - 800.0)) * 3e-3 + 0.05*(rand()-0.5))
        sum(interior_view(s.θ))
    end
    expect = Σθ0 + sum(Qs) * 5.0 / m.dz * m.Nx * m.Ny - 2.315e-5 * 5.0 * nsteps * m.Nx * m.Ny * m.Nz
    @test isapprox(sum(interior_view(rd_sim(r.obj).θ)), expect; rtol = 1e-13)
    # the ground is cooler than the air (T_s < 300 K), so the flux is
    # downward; in the first 150 s the ground still cools, because its
    # longwave loss σT⁴ ≈ 441 W m⁻² exceeds the 340 W m⁻² it receives plus
    # the shortwave of the early morning (< 40 W m⁻²) and the flux (≈ 30 W m⁻²)
    @test all(<(0), Qs) && land.T < 297.0
end

@testset "coupled: moist_convection.toml is the old script, bit for bit; Kessler turns cloud into rain" begin
    Nx = Ny = 12; Nz = 16; nsteps = 24
    # examples/coupled/02_moist_convection.jl before EX-d, verbatim but for the size, nsteps and the plots
    Lx = Ly = 3200.0; Lz = 2400.0
    mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))
    sim = BoussinesqSimulation(mesh;
        θ₀ = 300.0, g_accel = 9.81,
        C_s = 0.15, Pr_t = 1/3,
        moisture = true,
        θ_flux_bottom = 0.05,
        sponge_α_max = 0.02, sponge_z_start = 0.8*Lz,
        bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall),
        θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    Random.seed!(3)
    set!(sim.θ, (x,y,z) -> 300.0 + max(0, (z - 1500)) * 4e-3 + 0.03*(rand()-0.5))
    set!(sim.qv, (x,y,z) -> 0.012 * exp(-z / 1500))
    ql = Field(CellCenter(), mesh)
    qr = Field(CellCenter(), mesh)
    set!(ql, (x,y,z) -> begin
        xc, yc = Lx/2, Ly/2
        if abs(x - xc) < Lx/4 && abs(y - yc) < Ly/4 && 1000 < z < 1400
            return 2e-3
        else
            return 0.0
        end
    end)
    ql0 = sum(interior_view(ql))
    dt = 5.0
    for n in 1:nsteps
        step!(sim, dt)
        apply_kessler_step!(sim.qv, ql, qr, dt; Vt=:spectrum)
    end
    r = rd_run(rd_case("moist_convection"),
               ["grid.size=[$Nx,$Ny,$Nz]", "time.duration=\"none\"", "time.nsteps=$nsteps"])
    @test 0.8 * Lz == 1920.0                     # the config's sponge_z_start is the script's 0.8 Lz
    @test rd_same_state(r, sim)
    f = Base.invokelatest(state_fields, r.obj)
    @test bitwise_equal(rd_interior(f.ql, mesh), rd_interior(ql.data, mesh))
    @test bitwise_equal(rd_interior(f.qr, mesh), rd_interior(qr.data, mesh))
    # the physics: autoconversion and accretion move cloud water into rain
    # (ql falls, qr appears), and the rain falls below the cloud base
    @test sum(interior_view(ql)) < ql0 && maximum(interior_view(qr)) > 0
    zc = collect(Aeolus.Grids.znodes(mesh, CellCenter()))
    qri = interior_view(qr)
    @test any(qri[:, :, k] != zeros(Nx, Ny) for k in eachindex(zc) if zc[k] < 1000)
    # without the run type's [microphysics] the water is not carried
    @test_throws ConfigError load_run_config(rd_case("moist_convection");
                                             overrides = ["microphysics.scheme=\"none\""])
end

@testset "coupled: heated_plume.toml is the old script, bit for bit; the patch heats its column" begin
    N = 16; nsteps = 30
    # examples/coupled/03_heated_plume_rain.jl before EX-d, verbatim but for N, nsteps and the plots
    Lx = Ly = 2000.0; Lz = 1500.0
    Nx = Ny = N; Nz = N
    mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))
    sim = BoussinesqSimulation(mesh;
        θ₀ = 300.0, g_accel = 9.81,
        C_s = 0.15, Pr_t = 1/3,
        sponge_α_max = 0.02, sponge_z_start = 0.8Lz,
        bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall),
        θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    Random.seed!(2)
    set!(sim.θ, (x,y,z) -> 300.0 + max(0, (z - 1000.0)) * 5e-3 + 0.02*(rand()-0.5))
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    xs = collect(Aeolus.Grids.xnodes(mesh, Aeolus.Grids.CellCenter()))
    ys = collect(Aeolus.Grids.ynodes(mesh, Aeolus.Grids.CellCenter()))
    hot_flux(x, y) = (abs(x - Lx/2) < Lx/8 && abs(y - Ly/2) < Ly/8) ? 0.3 : 0.02
    dt = 2.0
    for n in 1:nsteps
        for j in 1:Ny, i in 1:Nx
            sim.θ.data[Hx+i, Hy+j, Hz+1] += hot_flux(xs[i], ys[j]) * dt / mesh.dz
        end
        step!(sim, dt)
    end
    r = rd_run(rd_case("heated_plume"),
               ["grid.size=[$N,$N,$N]", "time.duration=\"none\"", "time.nsteps=$nsteps"])
    @test 0.8 * Lz == 1200.0
    @test rd_same_state(r, sim)
    # the physics: the patch puts 15 times the flux of its surroundings into
    # the air, so the lowest level is warmer over it than elsewhere
    θ1 = interior_view(rd_sim(r.obj).θ)[:, :, 1]
    hot = [abs(x - 1000) < 250 && abs(y - 1000) < 250 for x in xs, y in ys]
    @test sum(θ1[hot]) / count(hot) > sum(θ1[.!hot]) / count(.!hot) + 0.1
end

@testset "coupled: flow_past_cube.toml (cube in the static driver) is the old script, bit for bit" begin
    nsteps = 6
    # examples/coupled/04_flow_past_cube.jl before EX-d, verbatim but for nsteps and the plots
    Lx = 200.0; Ly = 80.0; Lz = 80.0
    Nx = 96; Ny = 48; Nz = 40
    mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))
    sim = BoussinesqSimulation(mesh;
        ν = 1e-1, C_s = 0.15,
        bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall))
    U_inf = 3.0
    set!(sim.u, (x,y,z) -> U_inf)
    cube_x0, cube_x1 = 60.0, 80.0
    cube_y0, cube_y1 = 30.0, 50.0
    cube_z0, cube_z1 = 0.0,  20.0
    sm = SolidMask(mesh)
    mark_solid!(sm, (x,y,z) ->
        cube_x0 <= x <= cube_x1 && cube_y0 <= y <= cube_y1 && cube_z0 <= z <= cube_z1)
    c  = Field(CellCenter(), mesh)
    tc = Field(CellCenter(), mesh)
    set!(c, (x,y,z) -> 0.0)
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    i_src_start = round(Int, cube_x1 / mesh.dx) + 1
    i_src_end   = i_src_start + 1
    j_src = (round(Int, cube_y0 / mesh.dy)+1) : (round(Int, cube_y1 / mesh.dy))
    k_src = round(Int, (cube_z1 - 2) / mesh.dz)
    dt = 0.15 * mesh.dx / U_inf
    T_end = 40.0
    dt = T_end / ceil(Int, T_end / dt)
    force! = (sim, Δt) -> add_ibm_penalty!(sim, sm; τ = 3*Δt)
    for n in 1:nsteps
        step!(sim, dt; body_force! = force!)
        fill_halos!(c, sim.bcs)
        fill!(tc.data, 0)
        advect_centered!(tc, sim.u, sim.v, sim.w, c)
        apply_linear_chemistry!(tc, c, 30.0)
        for i in i_src_start:i_src_end, j in j_src
            tc.data[Hx+i, Hy+j, Hz+k_src] += 1.0
        end
        interior_view(c) .+= dt .* view(tc.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    end
    r = rd_run(rd_case("flow_past_cube"), ["time.duration=\"none\"", "time.nsteps=$nsteps"])
    # the static driver's cube is the script's cube, cell for cell
    @test rd_couplings(r.obj).solid.mask == sm.mask
    @test sort(rd_couplings(r.obj).tracer.sources) ==
          sort([(Hx + i, Hy + j, Hz + k_src, 1.0) for i in i_src_start:i_src_end for j in j_src])
    @test rd_same_state(r, sim)
    f = Base.invokelatest(state_fields, r.obj)
    @test bitwise_equal(rd_interior(f.tracer, mesh), rd_interior(c.data, mesh))
    # the physics: the penalty stops the flow inside the cube (from 3 m/s,
    # e^(-6Δt/3Δt) = e^(-2) after six steps), and the tracer is released
    ui = interior_view(rd_sim(r.obj).u); solid = view(sm.mask, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    @test maximum(abs, ui[solid]) < 0.5U_inf
    @test sum(interior_view(rd_couplings(r.obj).tracer.c)) > 0
end

@testset "coupled: the cube's committed static driver" begin
    # AeolusPrep.jl writes it from its GeoJSON (ONE-h, #457; its recipes reproduce it)
    cfg = load_run_config(rd_case("flow_past_cube"))
    mesh = RegularGrid(; size = cfg.grid.size, extent = cfg.grid.extent, halo = cfg.grid.halo)
    st = Aeolus.InputDrivers.read_static_driver(cfg.domain.static_driver; mesh = mesh, origin_lat = 0.0,
                                                origin_lon = 0.0, origin_x = 0.0, origin_y = 0.0)
    @test count(>(0), st.building_height) == 9 * 12 && maximum(st.building_height) == 20.0
    @test isempty(Aeolus.InputDrivers.check_driver(cfg.domain.static_driver, Aeolus.InputDrivers.STATIC_SCHEMA))
end

# ---------------------------------------------------------------------------
# examples/cbl_tke/cbl_tke.toml, examples/cbl_inflow_statistics/cbl_inflow_statistics.toml
# ([sgs] model = "tke", [inflow_statistics]; ONE-l: the `cbl` run type's
# harness tke_cbl, a test fixture, bit for bit)
# ---------------------------------------------------------------------------

isdefined(@__MODULE__, :TKECBL) || include(joinpath(@__DIR__, "fixtures", "tke_cbl.jl"))
using .TKECBL: tke_cbl
using Aeolus.Transition: ParentResolvedStatistics, update_inflow_statistics!, inflow_stress_parts

"The interior of the halo-padded storage `a` of `m`, after a halo fill (wall faces set as the run monitor sets them)."
rd_filled(f, bcs, m) = (fill_halos!(f, bcs); rd_interior(f.data, m))

@testset "cbl: the configs on the model are the tke_cbl harness and its parent statistics, bit for bit" begin
    # the config's [test] size: 12³ on 1200 m, z_i0 = 600 m, 2 min, the
    # statistics from 60 s, four levels (400 m)
    N = 12; T = 120.0; nlev = 4
    S = Ref{Any}(); C = Ref{Any}(); src = Ref{Any}(nothing)
    function fold!(sim, c, h)
        S[] = sim; C[] = c
        sim.t > 60.0 || return
        src[] === nothing && (src[] = ParentResolvedStatistics(sim; closure = c, τ_avg = 600.0, refine = (2, 2, 4),
                                                              levels = 1:nlev))
        update_inflow_statistics!(src[], h)
    end
    r0 = tke_cbl(N; Δ = 100.0, Δz = 100.0, Q = 0.1, z_i0 = 600.0, Γ = 0.003, T_end = T, avg = 60.0, on_step = fold!)
    f = rd_case("cbl_inflow_statistics")
    r = rd_run(f, config_test_table(f).overrides)
    sim = rd_sim(r.obj); m = sim.mesh; d = first(Aeolus.Components.domains(r.obj.model))
    @test sim.t == S[].t == T
    @test all(rd_interior(getproperty(sim, k).data, m) == rd_interior(getproperty(S[], k).data, m) for k in (:u, :v, :w, :θ, :p))
    clo = Aeolus.Components.sgs_closure(d)
    @test clo isa TKEClosure && rd_interior(clo.e.data, m) == rd_interior(C[].e.data, m)
    st = only(c for c in d.components if c isa Aeolus.Components.ParentStatistics)
    p0 = inflow_stress_parts(src[]); p1 = inflow_stress_parts(st.stats)
    @test all(getproperty(p0, k) == getproperty(p1, k) for k in keys(p0))
    # the physics of the record: e stays positive, and in scale-gap mode what
    # the generator adds is exactly the SGS part plus the band; the outputs
    # hold the statistics' levels and NaN above them
    @test minimum(interior_view(clo.e)) > 0 && Aeolus.PrognosticSGS.tke_statistics(clo).clips == 0
    ds = rd_output(r)
    g = ds["inflow_uu_generated"][:, end]; sg = ds["inflow_uu_sgs"][:, end]; b = ds["inflow_uu_band"][:, end]
    close(ds)
    @test all(isfinite, g[1:nlev]) && all(isnan, g[nlev+1:end])
    @test g[1:nlev] ≈ sg[1:nlev] .+ b[1:nlev]
    @test Float32.(p1.generated[1, :]) == g[1:nlev]
    # cbl_tke.toml: the same run without the statistics; its 1-min means
    # (the [test] cadence) of the heat flux: upward through the floor at the
    # prescribed Q = 0.1 K m/s (the bottom face) and still at most Q in the
    # first cell after 2 min (measured 0.0846 K m/s), e and the resolved TKE ≥ 0
    f2 = rd_case("cbl_tke")
    r2 = rd_run(f2, config_test_table(f2).overrides)
    @test rd_interior(rd_sim(r2.obj).θ.data, m) == rd_interior(sim.θ.data, m)
    ds = rd_output(r2)
    q = ds["heat_flux_profile"][:, end]; e = ds["sgs_tke_profile"][:, end]
    kres = ds["resolved_tke_profile"][:, end]; w2 = ds["w_variance_profile"][:, end]
    close(ds)
    @test 0 < q[1] <= 0.1 && all(>(0), e) && all(>=(0), kres) && all(>=(0), w2)
end

# ---------------------------------------------------------------------------
# examples/kelvin_helmholtz/kelvin_helmholtz.toml (tanh profiles, a seeded
# mode; formerly the shear_layer run type) and flat_plate_t3a.toml ([fringe];
# formerly the flat_plate run type)
# ---------------------------------------------------------------------------

using Aeolus.Transition: turbulent_fraction, flux_richardson

@testset "kelvin_helmholtz.toml is the old script, bit for bit; the seed grows" begin
    Nx, Nz, tend = 16, 32, 12.0
    # examples/transition/kelvin_helmholtz.jl before EX-d, verbatim but for the size, tend and the figures
    δ, U, Re, J, αδ = 1.0, 1.0, 1000.0, 0.05, 0.4446
    Lx = 2π * δ / αδ; dx = Lx / Nx; Lz = Nz * dx
    mesh = RegularGrid(; size = (Nx, 1, Nz), extent = (Lx, dx, Lz),
                       origin = (0.0, 0.0, -Lz / 2), halo = (2, 2, 2))
    θ₀, g = 300.0, 9.81
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(mesh; ν = U * δ / Re, θ₀ = θ₀, g_accel = g, bcs = bcs)
    Δθ = J * U^2 * θ₀ / (g * δ)
    set!(sim.u, (x, y, z) -> U * tanh(z / δ))
    set!(sim.θ, (x, y, z) -> θ₀ + Δθ * tanh(z / δ))
    set!(sim.w, (x, y, z) -> 1e-4 * U * sin(αδ * x / δ) * exp(-(z / δ)^2))
    zc = [mesh.z0 + (k - 0.5) * mesh.dz for k in 1:Nz]
    layer = findall(z -> abs(z) < 4δ, zc)
    dt = 0.6 * dx / U; nsteps = ceil(Int, tend / dt); dt = tend / nsteps
    ts, As, γm, Bs, Ps = Float64[], Float64[], Float64[], Float64[], Float64[]
    for n in 1:nsteps
        step_rk3!(sim, dt)
        wi = interior_view(sim.w)
        push!(ts, sim.t); push!(As, sqrt(sum(abs2, wi) / length(wi)))
        γ = turbulent_fraction(sim)
        push!(γm, sum(γ[layer]) / length(layer))
        r = flux_richardson(sim)
        push!(Bs, sum(r.B[layer])); push!(Ps, sum(r.P[layer]))
    end
    f = rd_case("kelvin_helmholtz")
    r = rd_run(f, config_test_table(f).overrides)
    s2 = rd_sim(r.obj)
    # the state (the old loop's wall faces of w held the seed's 1e-102 until
    # a halo fill zeroes them; the run's output fills them)
    @test all(rd_filled(getproperty(s2, k), s2.bcs, mesh) == rd_filled(getproperty(sim, k), sim.bcs, mesh)
              for k in (:u, :v, :w)) && rd_interior(s2.θ.data, mesh) == rd_interior(sim.θ.data, mesh)
    ds = rd_output(r)
    tser = ds["time_series"][:]; A = ds["w_rms"][:]; trec = ds["time"][:]
    γo = ds["turbulent_fraction_profile"][:, :]; Bo = ds["buoyancy_production_profile"][:, :]
    Po = ds["shear_production_profile"][:, :]
    close(ds)
    # the series: rms w at every whole δ/U (the step that reaches it), the
    # profiles in the records, the layer sums those of the old diagnostics
    for (k, tq) in enumerate(tser)
        tq == 0 && continue
        n = findfirst(t -> t >= tq * (1 - 1e-9), ts)
        @test A[k] == As[n]
    end
    n = length(ts)
    @test sum(Float64.(Bo[layer, end])) ≈ Bs[n] rtol = 1e-5
    @test sum(Float64.(Po[layer, end])) ≈ Ps[n] rtol = 1e-5
    # the physics: the seeded mode grows (the layer is unstable, J < 1/4),
    # and the laminar layer has no turbulent fraction yet
    @test As[end] > As[1] && A[end] > A[2] && all(iszero, γm) && all(iszero, γo)
end

@testset "flat_plate_t3a.toml is the old script, bit for bit (small grid); c_f is Blasius at the inflow" begin
    Nx, Ny, Nz, nsteps, navg = 96, 8, 16, 6, 3
    # examples/transition/flat_plate_t3a.jl before EX-d, verbatim but for the size and the steps
    Tu = 0.033; dt = 1.0
    c = flat_plate_setup(; Re_δ0 = 300, Lx = 600, Ly = 36, H = 24, Nx = Nx, Ny = Ny, Nz = Nz,
                         fringe_length = 80)
    sim, g = c.sim, c.sim.mesh
    fst = FreeStreamDisturbance(Tu, 10.0, 200; seed = 1, period_y = g.Ly,
                                κ_max = 2π / (3 * max(g.dx, g.dy, g.dz)))
    δ99 = blasius_solution().eta99 * sqrt(c.ν * c.x_in)
    target = deepcopy(c.target)
    update! = disturbance_fringe_update(fst, c.fringe, c.target;
                                        damping = z -> Aeolus.Operators.smooth_step(z / (2δ99)))
    force! = fringe_forcing(c.fringe, target; update! = update!)
    set_blasius!(sim; x_le = c.x_le, x_min = c.x_in)
    cf = zeros(g.Nx); nacc = 0
    for n in 1:nsteps
        step_rk3!(sim, dt; body_force! = force!)
        if n > nsteps - navg
            nacc += 1
            cf .+= skin_friction(sim; U_e = 1.0)
        end
    end
    cf ./= nacc
    f = rd_case("flat_plate_t3a")
    r = rd_run(f, config_test_table(f).overrides)       # 6 steps, the means of steps 1–3 and 4–6
    s2 = rd_sim(r.obj)
    @test all(rd_filled(getproperty(s2, k), s2.bcs, g) == rd_filled(getproperty(sim, k), sim.bcs, g) for k in (:u, :v, :w))
    @test rd_interior(s2.p.data, g) == rd_interior(sim.p.data, g)
    ds = rd_output(r)
    m = ds["skin_friction"][:, :, end]; t = ds["time"][:]
    close(ds)
    cfo = vec(sum(Float64.(m); dims = 2)) ./ Ny
    @test t == [3.0, 6.0] && maximum(abs.(cfo .- cf)) < 1e-6 * maximum(cf)   # the Float32 record
    # the physics at the inflow: the Blasius layer's skin friction,
    # 0.664 Re_x^(-1/2) at Re_x = (x − x_le)/ν (measured within 4 % at the third face)
    x = collect(Aeolus.Grids.xnodes(g, Aeolus.Grids.FaceX()))
    Re_x = (x .- c.x_le) ./ c.ν
    @test isapprox(cfo[3], blasius_skin_friction(Re_x[3]); rtol = 0.15)
end
