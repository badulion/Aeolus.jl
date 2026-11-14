# Thread utilisation of the step (#566 follow-up): passes that ran on one
# thread, or on a few, now launch over work items that spread the work.
# Each is checked against the serial loop it replaced, with a backend that
# takes the threaded path at every size (threshold 0), so the work items
# are exercised even when the suite runs on one thread.
using Test, Random
using Aeolus
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Grids: RegularGrid, interior
using Aeolus.Fields: Field, interior_view
const Sim = Aeolus.Simulations

@testset "thread utilisation of the step" begin
    H = 2
    mesh = RegularGrid(; size = (21, 17, 13), extent = (1.0, 1.0, 1.0), halo = (H, H, H))
    ir, jr, kr = interior(mesh)
    tb = CPUBackend(threshold = 0)

    @testset "halo shell: every halo point combined, the interior untouched" begin
        with_backend(tb) do
            o = randn(Xoshiro(1), size(Field(mesh).data))
            x = randn(Xoshiro(2), size(o))
            ref = copy(o)
            f(a, b) = 0.75a + 0.25b
            for k in axes(o, 3), j in axes(o, 2), i in axes(o, 1)
                (i in ir && j in jr && k in kr) && continue
                ref[i, j, k] = f(o[i, j, k], x[i, j, k])
            end
            out = copy(o)
            Sim._shell_map!(f, (out,), (x,), mesh)
            # the same expression on the same inputs, point by point: equal to
            # round-off of one multiply-add (none expected)
            @test maximum(abs.(out .- ref)) <= 4eps() * maximum(abs.(ref))
            @test view(out, ir, jr, kr) == view(o, ir, jr, kr)    # not written: copied data
            # zeroing (the tendencies' shell): no `ins`
            z = copy(o)
            Sim._shell_map!(Sim._Zero(), (z,), nothing, mesh)
            @test all(iszero, z[1:H, :, :]) && all(iszero, z[:, :, end-H+1:end])
            @test view(z, ir, jr, kr) == view(o, ir, jr, kr)
        end
    end

    @testset "radiative cooling: one launch, the former loops' values" begin
        with_backend(tb) do
            prof = collect(range(-2e-5, 1e-5; length = length(kr)))
            a = Field(mesh); a.data .= randn(Xoshiro(3), size(a.data))
            ref = copy(a.data)
            for (n, k) in enumerate(kr)
                ref[:, :, k] .+= prof[n]                        # the former loop
            end
            apply_radiative_cooling!(a, prof)
            @test maximum(abs.(a.data .- ref)) <= 4eps() * maximum(abs.(ref))
            # uniform rate: every column of the interior levels, halo columns included
            u = Field(mesh); fill!(u.data, 1.0)
            apply_radiative_cooling!(u, -0.5)
            @test all(u.data[:, :, kr] .== 0.5) && all(u.data[:, :, 1:H] .== 1.0)
            # solid cells keep their value
            solid = falses(size(a.data)); solid[ir[3], jr[4], kr[2]] = true
            s = Field(mesh); fill!(s.data, 0.0)
            apply_radiative_cooling!(s, prof; solid = solid)
            @test s.data[ir[3], jr[4], kr[2]] == 0.0
            @test s.data[ir[4], jr[4], kr[2]] ≈ prof[2]
        end
    end
    # A4: the flux, the areas and the correction as threaded plane sums and
    # launches; the imbalance case of open_boundaries.jl on a backend that
    # threads every launch (u = 1 + 0.2 x/Lx: outflow Q = 0.2 Ly Lz)
    @testset "open-boundary mass-flux correction on the threaded path" begin
        Lx, Ly, Lz = 800.0, 600.0, 400.0
        Qimb = 0.2 * Ly * Lz
        with_backend(tb) do
            m = RegularGrid(size = (16, 12, 8), extent = (Lx, Ly, Lz))
            data = FunctionBoundaryData(u = (x, y, z, t) -> 1.0 + 0.2 * x / Lx, v = 0.0, w = 0.0, θ = 300.0)
            for corr in (:uniform, :outflow_only)
                s = Aeolus.Simulations.BoussinesqSimulation(m; θ₀ = 300.0,
                        bcs = open_boundary_conditions(m, data; correction = corr, record = true))
                Aeolus.BoundaryConditions.update_open_boundaries!(s, 0.0, 1.0)
                L = open_boundary_ledger(s)
                @test L.Q ≈ Qimb rtol = 1e-12                      # round-off of the sums
                @test abs(L.residual) < 1e-9 * Qimb
                corr === :uniform &&
                    @test L.δ ≈ Qimb / (2 * (Ly * Lz + Lx * Lz) + Lx * Ly) rtol = 1e-12
            end
        end
    end
    # B1/B2/B5: a box thin along i (an x-normal boundary plane) launches one
    # k-plane per work item; every point written once, as the serial loop
    @testset "thin boxes along i: one k-plane per work item" begin
        body!(i, j, k, d, a) = (@inbounds d[i, j, k] += 2 * a[i, j, k] + i - 3j + 5k; nothing)
        a = randn(Xoshiro(7), 9, 40, 30)
        for ir in (1:1, 2:4, 1:7), jr in (1:40, 3:38), kr in (1:30, 2:29)
            d1 = zeros(size(a)); d2 = zeros(size(a))
            Aeolus.Backends.launch!(tb, body!, (ir, jr, kr), d1, a)
            for k in kr, j in jr, i in ir
                body!(i, j, k, d2, a)
            end
            # the same expression once per point: equal to round-off (none expected)
            @test maximum(abs.(d1 .- d2)) <= 4eps() * maximum(abs.(d2))
        end
    end
    # A2: above 4 M points the state copy is a threaded launch
    @testset "copy_storage!: large arrays copied by a launch, exactly" begin
        n = (162, 162, 162)                              # 4.25 M points, above the threshold
        @test prod(n) >= Sim._COPY_THREADED_MIN
        src = randn(Xoshiro(11), n); dst = zeros(n)
        with_backend(tb) do
            Sim.copy_storage!(dst, src)
        end
        @test dst == src                                 # copied data: exact
        small = randn(Xoshiro(12), 8, 8, 8); d2 = zeros(8, 8, 8)
        @test Sim.copy_storage!(d2, small) == small
        @test_throws DimensionMismatch Sim.copy_storage!(zeros(2, 2, 2), small)
    end
end
