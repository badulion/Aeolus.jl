# Dependency-free NetCDF-3 reader/writer.
#
# The round trip is the main test, but note what it does and does not
# prove: reader and writer are separate code paths (the reader re-derives
# offsets from the header rather than reusing the writer's arithmetic),
# so a round trip does catch offset, padding and endianness errors. It
# cannot catch a shared misreading of the specification — for that the
# byte-level header assertions below check the format directly against
# the Unidata spec.

using Test
using Aeolus

@testset "NetCDF-3 I/O" begin
    dir = mktempdir()

    @testset "Header conforms to the specification" begin
        path = joinpath(dir, "hdr.nc")
        nc = nc_create(path; attributes = Dict("title" => "hdr test"))
        nc_defdim!(nc, "x", 3)
        nc_defvar!(nc, "x", Float32, ("x",))
        nc_endef!(nc)
        nc_write!(nc, "x", Float32[1, 2, 3])
        nc_close!(nc)

        b = read(path)
        @test b[1:3] == Vector{UInt8}("CDF")
        @test b[4] == 0x02                      # CDF-2, 64-bit offset
        # numrecs is a big-endian Int32 at byte 5
        @test ntoh(reinterpret(Int32, b[5:8])[1]) == 0
        # first structural tag after numrecs is NC_DIMENSION (0x0A)
        @test ntoh(reinterpret(Int32, b[9:12])[1]) == 0x0A
        @test length(b) % 4 == 0                # everything is 4-byte aligned
    end

    @testset "Round trip: values, shapes, dimension order" begin
        path = joinpath(dir, "rt.nc")
        nx, ny, nz, nt = 5, 4, 3, 3
        A = Float32[100i + 10j + k for i in 1:nx, j in 1:ny, k in 1:nz]
        S = Float32[1000i + j     for i in 1:nx, j in 1:ny]

        nc = nc_create(path; attributes = Dict("title" => "rt", "run" => 7.0))
        nc_defdim!(nc, "x", nx); nc_defdim!(nc, "y", ny)
        nc_defdim!(nc, "z", nz); nc_defdim!(nc, "time", nothing)
        nc_defvar!(nc, "x", Float32, ("x",); attributes = Dict("units" => "m"))
        nc_defvar!(nc, "time", Float64, ("time",))
        nc_defvar!(nc, "theta", Float32, ("x","y","z","time");
                   attributes = Dict("units" => "K"))
        nc_defvar!(nc, "Ts", Float32, ("x","y","time"))
        nc_endef!(nc)
        nc_write!(nc, "x", Float32.(1:nx))
        for t in 1:nt
            nc_append_record!(nc, Dict("time" => [Float64(10t)],
                                       "theta" => A .+ Float32(t),
                                       "Ts"    => S .+ Float32(t)))
        end
        nc_close!(nc)

        f = nc_open(path)
        @test nc_nrecords(f) == nt
        @test Set(nc_varnames(f)) == Set(["x", "time", "theta", "Ts"])
        @test nc_attributes(f)["title"] == "rt"
        @test nc_attributes(f)["Conventions"] == "CF-1.8"
        @test nc_attributes(f, "theta")["units"] == "K"

        # shape comes back in JULIA index order, not file order
        θ = nc_read(f, "theta")
        @test size(θ) == (nx, ny, nz, nt)
        for t in 1:nt
            @test θ[:, :, :, t] ≈ A .+ Float32(t)
        end
        Tsr = nc_read(f, "Ts")
        @test size(Tsr) == (nx, ny, nt)
        @test Tsr[:, :, 2] ≈ S .+ 2f0
        @test nc_read(f, "time") ≈ [10.0, 20.0, 30.0]
        @test nc_read(f, "x") ≈ Float32.(1:nx)

        # single-record reads must match the full read
        for t in 1:nt
            @test nc_read_record(f, "theta", t) ≈ θ[:, :, :, t]
        end
        @test_throws ErrorException nc_read_record(f, "theta", nt + 1)
        @test_throws ErrorException nc_read(f, "nope")
    end

    @testset "Dimension order is the CF convention" begin
        # A field declared ("x","y","z","time") in Julia index order MUST
        # be stored as (time,z,y,x) in the file, because NetCDF's last
        # dimension varies fastest while Julia's first does. Getting this
        # backwards produces a file that reads as a transposed field in
        # every external tool, which is silent and very confusing.
        path = joinpath(dir, "ord.nc")
        nc = nc_create(path)
        nc_defdim!(nc, "x", 4); nc_defdim!(nc, "y", 2)
        nc_defvar!(nc, "f", Float32, ("x", "y"))
        nc_endef!(nc)
        # distinct values so a transpose cannot go unnoticed
        F = Float32[10i + j for i in 1:4, j in 1:2]
        nc_write!(nc, "f", F); nc_close!(nc)

        f = nc_open(path)
        @test f.vars["f"].dims == ["y", "x"]     # reversed in the file
        @test nc_read(f, "f") ≈ F                # but round-trips upright
        # and the raw byte order really is x-fastest
        b = read(path)
        off = f.vars["f"].offset
        raw = [ntoh(reinterpret(Float32, b[off+1+4(i-1) : off+4i])[1]) for i in 1:8]
        @test raw == Float32[11, 21, 31, 41, 12, 22, 32, 42]
    end

    @testset "Types and edge cases" begin
        path = joinpath(dir, "ty.nc")
        nc = nc_create(path)
        nc_defdim!(nc, "n", 3)
        nc_defvar!(nc, "f32", Float32, ("n",))
        nc_defvar!(nc, "f64", Float64, ("n",))
        nc_defvar!(nc, "i32", Int32,   ("n",))
        nc_endef!(nc)
        nc_write!(nc, "f32", Float32[1.5, -2.5, 3.5])
        nc_write!(nc, "f64", [1.0e10, -1.0e-10, 0.0])
        nc_write!(nc, "i32", Int32[-7, 0, 999])
        nc_close!(nc)
        f = nc_open(path)
        @test nc_read(f, "f32") == Float32[1.5, -2.5, 3.5]
        @test nc_read(f, "f64") == [1.0e10, -1.0e-10, 0.0]
        @test nc_read(f, "i32") == Int32[-7, 0, 999]

        # a name whose length is not a multiple of 4 exercises the padding
        path2 = joinpath(dir, "pad.nc")
        nc = nc_create(path2)
        nc_defdim!(nc, "abcde", 2)               # 5 chars ⇒ 3 bytes padding
        nc_defvar!(nc, "xyz", Float32, ("abcde",);
                   attributes = Dict("long_name" => "seven!!"))
        nc_endef!(nc)
        nc_write!(nc, "xyz", Float32[1, 2]); nc_close!(nc)
        f2 = nc_open(path2)
        @test nc_read(f2, "xyz") == Float32[1, 2]
        @test nc_attributes(f2, "xyz")["long_name"] == "seven!!"

        # only one unlimited dimension is permitted by the format
        nc3 = nc_create(joinpath(dir, "bad.nc"))
        nc_defdim!(nc3, "t", nothing)
        @test_throws ErrorException nc_defdim!(nc3, "t2", nothing)
        @test_throws ErrorException nc_defvar!(nc3, "v", Float32, ("missing",))
    end

    @testset "Streaming: numrecs is patched on close" begin
        # Records are written straight to disk, so a long run never holds
        # its output in memory and the header count is fixed up at the end.
        path = joinpath(dir, "stream.nc")
        nc = nc_create(path)
        nc_defdim!(nc, "n", 2); nc_defdim!(nc, "time", nothing)
        nc_defvar!(nc, "v", Float32, ("n", "time"))
        nc_endef!(nc)
        for t in 1:17
            nc_append_record!(nc, Dict("v" => Float32[t, -t]))
        end
        nc_close!(nc)
        f = nc_open(path)
        @test nc_nrecords(f) == 17
        V = nc_read(f, "v")
        @test size(V) == (2, 17)
        @test V[1, 17] == 17f0 && V[2, 17] == -17f0
        @info "streamed records" n=nc_nrecords(f) bytes=filesize(path)
    end

    @testset "nc_write_fields convenience" begin
        path = joinpath(dir, "conv.nc")
        nc_write_fields(path,
            Dict("x" => 1.0:4.0, "y" => 1.0:3.0),
            Dict("T" => Float32[i + 10j for i in 1:4, j in 1:3]);
            attributes = Dict("title" => "conv"))
        f = nc_open(path)
        @test size(nc_read(f, "T")) == (4, 3)
        @test nc_read(f, "T") ≈ Float32[i + 10j for i in 1:4, j in 1:3]
        @test nc_attributes(f)["title"] == "conv"
    end

    rm(dir; recursive = true, force = true)
end
