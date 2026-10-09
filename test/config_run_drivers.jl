# The run types of the example cases (EX-d, #441; src/RunDrivers.jl and the
# simulation driver's sections added for them). Each example used to be a
# script that built and ran its model; it is now a TOML file run by
# `aeolus run`. For each one this file runs the config at a small size and
# the old script's code at the same size, verbatim but for the plots and
# the size, and asserts that the two give the same bits; then it asserts
# the physics the example shows.

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

# ---------------------------------------------------------------------------
# examples/configs/scalar_advection_2d.toml (the advection driver)
# ---------------------------------------------------------------------------

@testset "advection: scalar_advection_2d.toml is the old script, bit for bit" begin
    N = 32
    # examples/scalar_advection_2d.jl before EX-d, verbatim but for N and the prints
    Nx, Ny, Nz = N, N, 1
    L = 1.0
    mesh = RegularGrid(Float64; size=(Nx, Ny, Nz), extent=(L, L, L),
                                halo=(2, 2, 2), origin=(-L/2, -L/2, 0.0))
    ω = 2π
    u = Field(CellCenter(), mesh); v = Field(CellCenter(), mesh); w = Field(CellCenter(), mesh)
    Aeolus.Fields.set!(u, (x, y, z) -> -ω * y)
    Aeolus.Fields.set!(v, (x, y, z) ->  ω * x)
    ϕ = Field(CellCenter(), mesh)
    x0, y0, σ = 0.25, 0.0, 0.08
    Aeolus.Fields.set!(ϕ, (x, y, z) -> exp(-((x-x0)^2 + (y-y0)^2) / (2σ^2)))
    bcs = FieldBoundaryConditions(:periodic)
    fill_halos!(u, bcs); fill_halos!(v, bcs); fill_halos!(w, bcs); fill_halos!(ϕ, bcs)
    state = (ϕ = ϕ,)
    stepper = RK3(state)
    umax = ω * L / 2
    dt = 0.4 * mesh.dx / umax
    T_final = 1.0
    nsteps = ceil(Int, T_final / dt); dt = T_final / nsteps
    function tendency!(k, state, t)
        fill!(k.ϕ.data, 0)
        fill_halos!(state.ϕ, bcs)
        advect_upwind!(k.ϕ, u, v, w, state.ϕ)
        return k
    end
    mass0 = sum(interior_view(state.ϕ)) * mesh.dx * mesh.dy * mesh.dz
    t = 0.0
    for step in 1:nsteps
        Aeolus.TimeSteppers.step!(stepper, state, tendency!, dt, t)
        t += dt
    end

    r = rd_run(joinpath(RD_EX, "configs", "scalar_advection_2d.toml"), ["grid.size=[$N,$N,1]"])
    f = Base.invokelatest(state_fields, r.obj)
    @test bitwise_equal(f.ϕ, ϕ.data)
    @test r.obj.nsteps == nsteps && r.obj.dt === dt
    res = rd_results(r)
    # the physics: upwind on a periodic box conserves mass to round-off,
    # its diffusion lowers the peak monotonically, and after one revolution
    # the peak is back within two cells of where it started
    @test all(m -> isapprox(m, mass0; rtol = 1e-13), res["mass"])
    @test issorted(res["max"]; rev = true) && res["max"][end] < 0.8res["max"][1]
    @test abs(res["peak_x"] - 0.25) <= 2mesh.dx && abs(res["peak_y"]) <= 2mesh.dy
    @test res["t_end"] ≈ 1.0 && res["t"][end] <= 1.0
end

# ---------------------------------------------------------------------------
# examples/kernels/configs/: the simulation driver with its couplings
# ---------------------------------------------------------------------------

const RD_KER = joinpath(RD_EX, "kernels", "configs")
const RD_CPL = joinpath(RD_EX, "coupled", "configs")

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
    r = rd_run(joinpath(RD_KER, "tgv_energy_decay.toml"),
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
    r = rd_run(joinpath(RD_KER, "decaying_hit.toml"),
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
    r = rd_run(joinpath(RD_KER, "channel_flow.toml"),
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
    r = rd_run(joinpath(RD_KER, "pollution_plume.toml"), ["time.duration=$T_end"])
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
    r = rd_run(joinpath(RD_KER, "particle_trail.toml"), ["grid.size=[$N,$N,$N]", "time.nsteps=$nsteps"])
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
# examples/coupled/configs/
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
    r = rd_run(joinpath(RD_CPL, "dcbl_radiation_lsm.toml"),
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
    r = rd_run(joinpath(RD_CPL, "moist_convection.toml"),
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
    @test_throws ConfigError load_run_config(joinpath(RD_CPL, "moist_convection.toml");
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
    r = rd_run(joinpath(RD_CPL, "heated_plume.toml"),
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
    r = rd_run(joinpath(RD_CPL, "flow_past_cube.toml"), ["time.duration=\"none\"", "time.nsteps=$nsteps"])
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

@testset "coupled: aeolus preprocess static writes the cube's static driver from the GeoJSON" begin
    out = joinpath(mktempdir(), "cube.nc")
    @test Aeolus.main(["preprocess", "static", joinpath(RD_CPL, "flow_past_cube.toml"), "--out", out];
                      io = IOBuffer(), err = IOBuffer()) == 0
    cfg = load_run_config(joinpath(RD_CPL, "flow_past_cube.toml"))
    mesh = RegularGrid(; size = cfg.grid.size, extent = cfg.grid.extent, halo = cfg.grid.halo)
    st = Aeolus.InputDrivers.read_static_driver(out; mesh = mesh, origin_lat = 0.0, origin_lon = 0.0,
                                                origin_x = 0.0, origin_y = 0.0)
    shipped = Aeolus.InputDrivers.read_static_driver(cfg.domain.static_driver; mesh = mesh, origin_lat = 0.0,
                                                     origin_lon = 0.0, origin_x = 0.0, origin_y = 0.0)
    @test st.building_height == shipped.building_height
    @test count(>(0), st.building_height) == 9 * 12 && maximum(st.building_height) == 20.0
end

# ---------------------------------------------------------------------------
# examples/sgs/configs/cbl_tke.toml, examples/nesting/configs/cbl_inflow_statistics.toml
# (the cbl driver: tke_cbl, formerly test/tke_cbl_driver.jl, plus the parent statistics)
# ---------------------------------------------------------------------------

using Aeolus.RunDrivers.CBLDriver: tke_cbl

@testset "cbl: the config maps onto tke_cbl and the parent statistics of the old example" begin
    N = 12; Δ = 50.0; Δz = 25.0; T = 240.0; avg = 120.0; zi0 = 150.0
    # what examples/nesting/cbl_inflow_statistics.jl did, verbatim but for the size
    nlev = min(N, round(Int, 200.0 / Δz))
    src = Ref{Any}(nothing)
    function fold!(sim, c, h)
        sim.t > T - avg || return
        src[] === nothing && (src[] = ParentResolvedStatistics(sim; closure = c, τ_avg = 600.0, refine = (2, 2, 4),
                                                              levels = 1:nlev))
        update_inflow_statistics!(src[], h)
    end
    r0 = tke_cbl(N; Δ = Δ, Δz = Δz, Q = 0.1, z_i0 = zi0, Γ = 0.003, T_end = T, avg = avg, on_step = fold!)
    r = rd_run(joinpath(RD_EX, "nesting", "configs", "cbl_inflow_statistics.toml"),
               ["grid.n=$N", "grid.spacing=$Δ", "grid.dz=$Δz", "cbl.end_time=$T", "cbl.average=$avg",
                "cbl.z_i0=$zi0", "inflow_statistics.depth=200.0"])
    r1 = r.obj.result
    @test all(getproperty(r0, k) == getproperty(r1, k) for k in (:e, :kres, :w2, :flux, :sgs_flux, :θ, :zi, :steps))
    p0 = inflow_stress_parts(src[]); p1 = inflow_stress_parts(r.obj.stats)
    @test all(getproperty(p0, k) == getproperty(p1, k) for k in keys(p0))
    res = rd_results(r)
    inf = res["inflow"]
    # the physics of the record: heat goes up through the first face, at
    # most the floor's Q (the flux profile is normalised by Q; 4 min into the
    # run the first cell still stores part of it: measured 0.53), e stays
    # positive, and in scale-gap mode what the generator adds is exactly the
    # SGS part plus the band
    @test 0 < res["profiles"]["flux"][1] <= 1.0
    @test res["min_e"] > 0 && res["clips"] == 0
    @test inf["uu_generated"] ≈ inf["uu_sgs"] .+ inf["uu_band"]
    @test length(inf["z"]) == nlev && inf["updates"] > 0
    # without [inflow_statistics] there is no [inflow] table
    r2 = rd_run(joinpath(RD_EX, "sgs", "configs", "cbl_tke.toml"),
                ["grid.n=$N", "grid.spacing=$Δ", "grid.dz=$Δz", "cbl.end_time=$T", "cbl.average=$avg", "cbl.z_i0=$zi0"])
    @test !haskey(rd_results(r2), "inflow") && r2.obj.result.e == r1.e
end

# ---------------------------------------------------------------------------
# examples/nesting/configs/inflow_fetch.toml (the inflow_fetch driver; its
# harness, formerly test/inflow_fetch_harness.jl, is validated by the
# full-tier test/era6_inflow_fetch.jl)
# ---------------------------------------------------------------------------

using Aeolus.RunDrivers.InflowFetchDriver: fetch_precursor, fetch_child

@testset "inflow_fetch: the config runs the harness; the generator injects turbulence, off does not" begin
    f = joinpath(RD_EX, "nesting", "configs", "inflow_fetch.toml")
    ov = config_test_table(f).overrides
    r = rd_run(f, ov)
    # the same calls by hand
    pre = fetch_precursor(; N = (12, 6, 8), extent = (5.0, 2.5), t_spin = 0.5, t_avg = 0.5)
    df = fetch_child(pre; generator = :df, N = (24, 12, 8), extent = (5.0, 2.5), t_spin = 0.2, t_avg = 0.5,
                     sample_every = 4, lateral = :periodic)
    @test r.obj.pre.U == pre.U && Dict(r.obj.runs)["df"].p.tke == df.p.tke
    res = rd_results(r)["runs"]
    # with the mean profile only (and no initial noise) the child stays all
    # but laminar (measured: TKE ≤ 0.5 % of the reference over 0.5 δ/u*);
    # the digital filter puts turbulence in at the inflow plane
    @test maximum(res["off"]["level_1"]["tke_over_reference"]) < 0.01
    @test res["df"]["level_1"]["tke_over_reference"][1] > 0.1
    # the inflow carries the precursor's bulk velocity (Dirichlet mean profile)
    @test abs(res["off"]["bulk_ratio"] - 1) < 1e-3 && abs(res["df"]["bulk_ratio"] - 1) < 0.05
end

# ---------------------------------------------------------------------------
# examples/sgs/configs/surface_mounted_cube.toml (the obstacle_channel driver)
# ---------------------------------------------------------------------------

using Aeolus.IBM: SolidMask, mark_solid!, implicit_ibm_correction!, wall_distance, first_level_cells

"examples/sgs/surface_mounted_cube.jl before EX-d: build(model) and the loop of run(model), verbatim but for the times and the prints."
function rd_cube_script(model::Symbol, T_SPIN, T_AVG)
    NH = 16; LX, LY, LZ = 10.0, 6.0, 2.0; XC0, XC1 = 3.0, 4.0; RE_H = 40_000.0
    h = 1 / NH
    mesh = RegularGrid(; size = (round(Int, LX * NH), round(Int, LY * NH), round(Int, LZ * NH) + 1),
                       extent = (LX, LY, LZ + h), origin = (0.0, 0.0, -h), halo = (3, 3, 3))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(mesh; ν = 1 / RE_H, bcs = bcs)
    sm = SolidMask(mesh)
    yc = LY / 2
    mark_solid!(sm, (x, y, z) -> z < 0 || (XC0 < x < XC1 && abs(y - yc) < 0.5 && z < 1))
    mf = Field(CellCenter(), mesh)
    mf.data .= sm.mask
    fill_halos!(mf, FieldBoundaryConditions(:periodic, :periodic, :noflux))
    mask = mf.data .> 0.5
    d = wall_distance(sm; bottom = false)
    fl = first_level_cells(sm; bottom = false)
    closure = model === :smagorinsky ? StaticSmagorinsky(0.1) :
              model === :tke ? TKEClosure(mesh; mask = mask, wall_distance = d, first_level = fl) :
              ReynoldsStressClosure(mesh; mask = mask, wall_distance = d, first_level = fl)
    rng = Random.Xoshiro(1993)
    interior_view(sim.u) .= 1 .+ 0.1 .* randn(rng, size(interior_view(sim.u)))
    interior_view(sim.v) .= 0.1 .* randn(rng, size(interior_view(sim.v)))
    interior_view(sim.w) .= 0.1 .* randn(rng, size(interior_view(sim.w)))
    implicit_ibm_correction!(sim, sm)
    function bulk_u(sim, sm)
        m = sim.mesh
        s = 0.0; n = 0
        u = sim.u.data; mk = sm.mask
        @inbounds for k in m.Hz+1:m.Hz+m.Nz, j in m.Hy+1:m.Hy+m.Ny, i in m.Hx+1:m.Hx+m.Nx
            mk[i, j, k] && continue
            s += u[i, j, k]; n += 1
        end
        return s / n
    end
    m = sim.mesh
    model === :tke && initialize_tke!(closure, sim; e_min = 1e-4)
    model === :rsm && initialize_reynolds_stress!(closure, sim; e_min = 1e-4)
    Fi = Ref(0.0); Fp = Ref(0.0)
    force = (s, Δt) -> (interior_view(s.tu) .+= Fi[] + Fp[]; nothing)
    ibm = (s, Δt) -> (implicit_ibm_correction!(s, sm); nothing)
    kf = 2
    jc = (m.Ny ÷ 2, m.Ny ÷ 2 + 1)
    ufl = zeros(m.Nx); tavg = 0.0
    while sim.t < T_SPIN + T_AVG - 1e-9
        ub = bulk_u(sim, sm)
        umax = maximum(f -> maximum(abs, interior_view(f)), (sim.u, sim.v, sim.w))
        dt = min(0.4 * m.dx / umax, T_SPIN + T_AVG - sim.t)
        Fp[] = 2.0 * (1 - ub)
        Fi[] += 0.5 * dt * (1 - ub)
        step_rk3!(sim, dt; sgs = closure, body_force! = force, post_predictor! = ibm)
        if sim.t > T_SPIN
            u = interior_view(sim.u)
            for i in 1:m.Nx
                ip = i == m.Nx ? 1 : i + 1
                ufl[i] += dt * sum((u[i, j, kf] + u[ip, j, kf]) / 2 for j in jc) / 2
            end
            tavg += dt
        end
    end
    ufl ./= tavg
    return sim, sm, ufl
end

@testset "obstacle_channel: surface_mounted_cube.toml (cube in the static driver) is the old script, bit for bit" begin
    T = 0.04
    f = joinpath(RD_EX, "sgs", "configs", "surface_mounted_cube_closures.toml")
    for (k, model) in enumerate((:smagorinsky, :tke))
        sim, sm, ufl = rd_cube_script(model, T, T)
        cfg = last(only(load_sweep(f; member = k, overrides = ["stepper.t_spin=$T", "stepper.t_avg=$T",
                                                                "fftw.planner=\"estimate\""])))
        r = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir())
        @test r.obj.sm.mask == sm.mask                     # the static driver's cube and the floor = the script's mask
        @test rd_same_state(r, sim)
        @test r.obj.u_floor == ufl
        res = rd_results(r)
        @test res["x_front"] == 3.0 && res["x_rear"] == 4.0 && res["closure"] == String(model)
        # the physics: the direct forcing nearly stops the flow inside the cube
        # and the floor (it acts after the predictor, before the projection,
        # which leaves a residual: measured mean |u| 0.0092 U_b, max 0.088 on
        # the faces between two solid cells), and the controller keeps the bulk
        # velocity at 1 within 2 %
        m = sim.mesh
        ui = rd_sim(r.obj).u.data
        solid_faces = [ui[i, j, k] for k in m.Hz+1:m.Hz+m.Nz, j in m.Hy+1:m.Hy+m.Ny, i in m.Hx+2:m.Hx+m.Nx
                       if sm.mask[i, j, k] && sm.mask[i-1, j, k]]
        @test sum(abs, solid_faces) / length(solid_faces) < 0.02 && maximum(abs, solid_faces) < 0.1
        @test abs(res["mean_U_b"] - 1) < 0.02
    end
end

# ---------------------------------------------------------------------------
# examples/transition/configs/: kelvin_helmholtz.toml (shear_layer) and
# flat_plate_t3a.toml (flat_plate)
# ---------------------------------------------------------------------------

@testset "shear_layer: kelvin_helmholtz.toml is the old script, bit for bit; the seed grows" begin
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
    every = max(1, round(Int, 1.0 / dt))
    ts, As, γm, Bs, Ps = Float64[], Float64[], Float64[], Float64[], Float64[]
    for n in 1:nsteps
        step_rk3!(sim, dt)
        if n % every == 0 || n == nsteps
            wi = interior_view(sim.w)
            push!(ts, sim.t); push!(As, sqrt(sum(abs2, wi) / length(wi)))
            γ = turbulent_fraction(sim)
            push!(γm, sum(γ[layer]) / length(layer))
            r = flux_richardson(sim)
            push!(Bs, sum(r.B[layer])); push!(Ps, sum(r.P[layer]))
        end
    end
    r = rd_run(joinpath(RD_EX, "transition", "configs", "kelvin_helmholtz.toml"),
               ["grid.nx=$Nx", "grid.nz=$Nz", "stepper.end_time=$tend", "diagnostics.snapshots=[6.0]"])
    @test rd_same_state(r, sim)
    s = r.obj.records
    @test s.ts == ts && s.As == As && s.γm == γm && s.Bs == Bs && s.Ps == Ps
    # the physics: the seeded mode grows (the layer is unstable, J < 1/4),
    # and the laminar layer has no turbulent fraction yet
    @test As[end] > As[1] && all(iszero, γm)
    ds = NCDataset(joinpath(r.folder.path, "output", "vorticity.nc"))
    @test size(ds["omega_y"]) == (Nx, Nz, 1) && ds["t"][1] >= 6.0
    close(ds)
end

@testset "flat_plate: flat_plate_t3a.toml is the old script, bit for bit (small grid)" begin
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
    cf = zeros(g.Nx); ū = zeros(g.Nx, g.Nz); nacc = 0
    for n in 1:nsteps
        step_rk3!(sim, dt; body_force! = force!)
        if n > nsteps - navg
            nacc += 1
            cf .+= skin_friction(sim; U_e = 1.0)
            ū .+= dropdims(sum(interior_view(sim.u); dims = 2); dims = 2) ./ g.Ny
        end
    end
    cf ./= nacc; ū ./= nacc
    f = joinpath(RD_EX, "transition", "configs", "flat_plate_t3a.toml")
    r = rd_run(f, config_test_table(f).overrides)
    @test rd_same_state(r, sim)
    @test r.obj.cf == cf && r.obj.ū == ū
    res = rd_results(r)
    # the physics at the inflow: the Blasius layer's skin friction, 0.664 Re_x^(-1/2);
    # six steps are too few for transition (no onset on the plate)
    p = res["plate"]
    @test isnan(res["onset_re_theta"])
    @test isapprox(p["cf"][3], p["cf_laminar"][3]; rtol = 0.15)
    @test res["mayle"] ≈ 400 * 3.3^(-5 / 8)          # Mayle (1991): Re_θ,onset = 400 Tu^(-5/8), Tu in %
end

# ---------------------------------------------------------------------------
# examples/sgs/configs/rotating_stratified_decay.toml (the sgs_decay driver;
# its fixture, formerly test/fixtures/sgs_validation.jl, is validated by the
# full-tier test/era6_sgs_validation.jl)
# ---------------------------------------------------------------------------

using Aeolus.RunDrivers.SGSDecayDriver: validation_base, run_dns_case, run_les

@testset "sgs_decay: the config runs the fixture; the stratified continuation loses K + PE" begin
    f = joinpath(RD_EX, "sgs", "configs", "rotating_stratified_decay.toml")
    r = rd_run(f, config_test_table(f).overrides)
    base = validation_base(; N = 16, t0 = 0.2)
    d = run_dns_case(base, :stratified; nturn = 0.2, nsave = 2)
    dn = r.obj.dns[:stratified]
    @test dn.K == d.K && [s.K for s in dn.snaps] == [s.K for s in d.snaps]
    @test r.obj.les[(:stratified, :tke)].K == run_les(base, d, :tke).K
    # the physics: the continuation's kinetic plus potential energy only
    # decays (the exchange with the background is exact, viscosity removes)
    E = dn.K .+ dn.PE
    @test all(diff(E) .< 0) && all(>=(0), dn.PE)
    res = rd_results(r)["cases"]["stratified"]
    @test res["dns_K_ratio"] < 1 && Set(keys(res["models"])) == Set(["smagorinsky", "dynamic", "tke", "rsm"])
end
