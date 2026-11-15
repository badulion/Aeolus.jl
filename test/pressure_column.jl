# #572: CosineFFTPoissonSolver with walls in z solves per horizontal
# wavenumber with the Thomas algorithm (ColumnTridiagonalSolver), not with a
# DCT along z. The transforms along all three axes (_CosineTransformSolver)
# still solve the same discrete problem and are the reference here.
using Test, Random
using Aeolus
using Aeolus.Grids: RegularGrid, CellCenter
using Aeolus.Fields: Field, interior_view
using Aeolus.BoundaryConditions: FieldBoundaryConditions, fill_halos!
using Aeolus.Operators: laplacian
const PSc = Aeolus.PressureSolvers

@testset "pressure solve: tridiagonal in z (#572)" begin
    g = RegularGrid(; size = (24, 20, 16), extent = (480.0, 400.0, 160.0), halo = (1, 1, 1))

    @testset "the implementation follows bc_z" begin
        for bx in (:periodic, :neumann), by in (:periodic, :neumann)
            s = CosineFFTPoissonSolver(g; bc_x = bx, bc_y = by, bc_z = :neumann)
            @test s.impl isa PSc.ColumnTridiagonalSolver
            @test PSc.poisson_axis_bcs(s) == (bc_x = bx, bc_y = by, bc_z = :neumann)
            sp = CosineFFTPoissonSolver(g; bc_x = bx, bc_y = by, bc_z = :periodic)
            @test sp.impl isa PSc._CosineTransformSolver
        end
    end

    # Both solve the 7-point Laplacian with the mirror closure exactly, so
    # they agree to round-off of the transforms (O(10 ε) relative here; the
    # 1e-11 bound leaves room for the growth with N: 1.9e-12 measured at
    # 1000 × 1000 × 150, benchmarks/hpc/REPORT_fft_20m.md).
    @testset "agrees with the DCT along z, ($bx, $by, :neumann)" for (bx, by) in
            ((:periodic, :periodic), (:neumann, :neumann), (:neumann, :periodic))
        src = Field(CellCenter(), g); p1 = Field(CellCenter(), g); p2 = Field(CellCenter(), g)
        raw = randn(Xoshiro(572), g.Nx, g.Ny, g.Nz); raw .-= sum(raw) / length(raw)
        interior_view(src) .= raw
        solve!(p1, src, CosineFFTPoissonSolver(g; bc_x = bx, bc_y = by, bc_z = :neumann))
        solve!(p2, src, PSc._CosineTransformSolver(g; bc_x = bx, bc_y = by, bc_z = :neumann))
        a, b = interior_view(p1), interior_view(p2)
        @test maximum(abs.(a .- b)) / maximum(abs.(b)) < 1e-11
        @test abs(sum(a)) / length(a) < 1e-12 * maximum(abs.(a))   # zero-mean solution
    end

    # An incompatible source (net mean s̄) is solved as L_h p = s − s̄: the
    # residual is the uniform −s̄ (the docstring's compatibility statement).
    @testset "incompatible source: residual is the uniform −mean" begin
        bcs = FieldBoundaryConditions(:noflux, :noflux, :noflux)
        src = Field(CellCenter(), g); p = Field(CellCenter(), g); chk = Field(CellCenter(), g)
        raw = randn(Xoshiro(5), g.Nx, g.Ny, g.Nz) .+ 0.3
        interior_view(src) .= raw
        s̄ = sum(raw) / length(raw)
        solve!(p, src, CosineFFTPoissonSolver(g; bc_x = :neumann, bc_y = :neumann, bc_z = :neumann))
        fill_halos!(p, bcs)
        laplacian(chk, p)
        r = interior_view(chk) .- raw
        # round-off of a solve whose solution is O(|s| L²): 1e-10 absolute
        @test maximum(abs.(r .+ s̄)) < 1e-10
    end

    @testset "source buffer path equals the field path" begin
        s = CosineFFTPoissonSolver(g; bc_x = :neumann, bc_y = :neumann, bc_z = :neumann)
        src = Field(CellCenter(), g); p1 = Field(CellCenter(), g); p2 = Field(CellCenter(), g)
        raw = randn(Xoshiro(9), g.Nx, g.Ny, g.Nz); raw .-= sum(raw) / length(raw)
        interior_view(src) .= raw
        solve!(p1, src, s)
        PSc.source_buffer(s) .= raw          # data copied in, not computed
        PSc.solve_source_buffer!(p2, s)
        # the same operations on the same input: identical up to round-off order
        @test maximum(abs.(interior_view(p1) .- interior_view(p2))) <= 1e-14 * maximum(abs.(interior_view(p1)))
    end

    @testset "uniform-z coefficients: the stretched builder on equal spacings" begin
        Nz = g.Nz; dz = g.dz
        c = PSc.ColumnTridiagonalSolver(Float64, (g.Nx, g.Ny, Nz), (g.dx, g.dy),
                                        fill(dz, Nz), fill(dz, Nz + 1); bc_x = :neumann, bc_y = :neumann)
        @test c.lower[1] == 0 && all(c.lower[2:end] .≈ 1 / dz^2)
        @test c.measure ≈ fill(dz, Nz)
        @test_throws DimensionMismatch PSc.ColumnTridiagonalSolver(Float64, (4, 4, 3), (1.0, 1.0),
                                                                     ones(3), ones(3))
    end

    # #575: a transform cut into blocks (levels or rows), each block a
    # serial plan, against FFTW's one plan over the whole array. Odd sizes:
    # the blocks alternate between memory alignments mod 16, which needs a
    # plan per alignment class. The same transform, other codelet order:
    # agreement to round-off (1e-13 relative, measured O(1e-16)).
    @testset "BlockTransform: blocks of serial plans equal the whole transform" begin
        O = Aeolus.Operators; F = O.FFTW
        x = randn(Xoshiro(575), 15, 13, 9)
        rel(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
        be = Aeolus.Backends.CPUBackend(; threshold = 0)
        # r2r in x-y by levels, along z by rows
        for (region, split) in (((1, 2), 3), ((3,), 2))
            kinds = ntuple(_ -> F.REDFT10, length(region))
            y = similar(x)
            t = O.BlockTransform((o, i) -> O.serial_r2r_plan(o, i, kinds, region), y, x, split;
                                 backend = be, nblocks = 4)
            @test O.nblocks(t) == 4 && length(t.plans) >= 2      # several alignments / sizes
            Aeolus.Backends.apply_transform!(be, y, t, x)
            @test rel(y, F.r2r(x, kinds, region)) < 1e-13
        end
        # r2c in x-y, complex DFT along z in place, and back
        h = zeros(ComplexF64, 8, 13, 9)
        tf = O.BlockTransform((o, i) -> O.serial_rfft_plan(o, i, (1, 2)), h, x, 3; backend = be, nblocks = 4)
        tz = O.BlockTransform((o, i) -> O.serial_fft_plan(o, i, (3,), F.FORWARD), h, h, 2; backend = be, nblocks = 5)
        # every plan before any data: a timing planner (MEASURE) overwrites its arrays
        tzi = O.BlockTransform((o, i) -> O.serial_fft_plan(o, i, (3,), F.BACKWARD), h, h, 2; backend = be, nblocks = 5)
        y = similar(x)
        ti = O.BlockTransform((o, i) -> O.serial_brfft_plan(o, i, (1, 2)), y, h, 3; backend = be, nblocks = 4)
        Aeolus.Backends.apply_transform!(be, h, tf, x)
        Aeolus.Backends.apply_transform!(be, h, tz, h)
        @test rel(h, F.rfft(x)) < 1e-13
        Aeolus.Backends.apply_transform!(be, h, tzi, h)
        Aeolus.Backends.apply_transform!(be, y, ti, h)
        @test rel(y ./ length(x), x) < 1e-13                  # the unnormalised pair: × Nx Ny Nz
        @test_throws ArgumentError O.BlockTransform((o, i) -> O.serial_r2r_plan(o, i, (F.REDFT10,), (1,)),
                                                    y, x, 4)
    end
end
