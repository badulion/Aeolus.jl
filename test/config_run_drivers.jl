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
