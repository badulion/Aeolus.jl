# NetCDF I/O through NCDatasets (#310): the Aeolus output policy (NetCDF-4,
# deflate + shuffle, chunks of one level and one record, streamed records,
# CF time) and the reading of the NetCDF-3 files written before #310.
#
# The round trip goes through the netCDF-C library on both sides, so what
# it proves is the policy: that the file is NetCDF-4, that compression is
# lossless and actually applied, that the chunk shapes are the documented
# ones, and that `nc_append_record!` puts each array in its own record with
# Julia's dimension order.

using Test
using Aeolus
using Dates: DateTime
import NCDatasets

@testset "NetCDF I/O (NCDatasets)" begin
    dir = mktempdir()
    nx, ny, nz, nt = 64, 64, 5, 4      # a 64 x 64 level is one 4096-value chunk
    # a smooth field plus a little deterministic noise: compressible, but
    # not trivially, and every value distinct
    θ(t) = Float32[290 + 0.01k * 100 + 0.3sin(i / 3) * cos(j / 4) + 1f-3 * mod(37i + 11j + 7k, 13) + 0.5t
                   for i in 1:nx, j in 1:ny, k in 1:nz]
    Ts(t) = Float32[300 + i - j + 0.25t for i in 1:nx, j in 1:ny]
    prof(t) = Float32[k + 0.1t for k in 1:nz]

    function write_case(path; kw...)
        ds = nc_create(path; attributes = Dict("title" => "ncio test", "run" => 7.0))
        ds.dim["x"] = nx; ds.dim["y"] = ny; ds.dim["z"] = nz; ds.dim["time"] = Inf
        nc_defvar!(ds, "x", Float32, ("x",); attributes = Dict("units" => "m", "axis" => "X"), kw...)
        nc_defvar!(ds, "time", Float64, ("time",);
                   attributes = Dict("units" => cf_time_units(DateTime(2019, 6, 29, 4)), "axis" => "T"), kw...)
        nc_defvar!(ds, "building_height", Float32, ("x", "y"); attributes = Dict("units" => "m"), kw...)
        nc_defvar!(ds, "theta", Float32, ("x", "y", "z", "time");
                   attributes = Dict("units" => "K", "long_name" => "potential temperature",
                                     "coordinates" => "x y z"), kw...)
        nc_defvar!(ds, "Ts", Float32, ("x", "y", "time"); attributes = Dict("units" => "K"), kw...)
        nc_defvar!(ds, "tke_prof", Float32, ("z", "time"); attributes = Dict("units" => "m2 s-2"), kw...)
        nc_defvar!(ds, "count", Int32, ("time",); attributes = Dict("units" => "1"), kw...)
        ds["x"][:] = Float32.((1:nx) .- 0.5)
        ds["building_height"][:, :] = Float32[10i + j for i in 1:nx, j in 1:ny]
        for t in 1:nt
            nc_append_record!(ds, Dict("time" => [60.0t], "theta" => θ(t), "Ts" => Ts(t),
                                       :tke_prof => prof(t), "count" => Int32[-t]))
            @test ds.dim["time"] == t                  # one record per call
        end
        close(ds)
        return path
    end

    path = write_case(joinpath(dir, "case.nc"))

    @testset "NetCDF-4 with CF global attributes" begin
        @test read(path, 4) == UInt8[0x89, 'H', 'D', 'F']      # HDF5 signature
        NCDataset(path) do ds
            @test ds.attrib["Conventions"] == "CF-1.8"
            @test ds.attrib["title"] == "ncio test"
            @test ds.attrib["run"] == 7.0
            @test ds["theta"].attrib["long_name"] == "potential temperature"
            @test ds["theta"].attrib["coordinates"] == "x y z"
            @test ds["Ts"].attrib["units"] == "K"
        end
    end

    @testset "round trip of every variable: exact values, Julia dimension order" begin
        NCDataset(path) do ds
            @test NCDatasets.dimnames(ds["theta"]) == ("x", "y", "z", "time")
            @test size(ds["theta"]) == (nx, ny, nz, nt)
            @test ds["x"][:] == Float32.((1:nx) .- 0.5)
            @test Array(ds["building_height"]) == Float32[10i + j for i in 1:nx, j in 1:ny]
            for t in 1:nt                               # compression is lossless
                @test ds["theta"][:, :, :, t] == θ(t)
                @test ds["Ts"][:, :, t] == Ts(t)
                @test ds["tke_prof"][:, t] == prof(t)
            end
            @test eltype(ds["count"]) == Int32 && ds["count"][:] == Int32.(-(1:nt))
            # CF time: decoded to the reference date plus the seconds written
            @test ds["time"][:] == [DateTime(2019, 6, 29, 4, t) for t in 1:nt]
            @test ds["time"].var[:] == 60.0 .* (1:nt)
        end
    end

    @testset "deflate + shuffle; chunks of one level and one record" begin
        NCDataset(path) do ds
            @test NCDatasets.deflate(ds["theta"]) == (true, true, NC_DEFLATE_LEVEL)
            @test NCDatasets.chunking(ds["theta"]) == (:chunked, (nx, ny, 1, 1))
            @test NCDatasets.chunking(ds["Ts"]) == (:chunked, (nx, ny, 1))
            # small record slabs gather records to 4096 values per chunk
            @test NCDatasets.chunking(ds["tke_prof"]) == (:chunked, (nz, cld(4096, nz)))
            @test NCDatasets.chunking(ds["time"]) == (:chunked, (4096,))
            @test NCDatasets.chunking(ds["building_height"]) == (:chunked, (nx, ny))
        end
        # the compression is real: the same records uncompressed are larger
        raw = write_case(joinpath(dir, "raw.nc"); deflatelevel = 0)
        NCDataset(raw) do ds
            @test NCDatasets.deflate(ds["theta"])[2] == false
            @test ds["theta"][:, :, :, nt] == θ(nt)
        end
        @test filesize(path) < filesize(raw)
        @info "NCIO test file" deflated = filesize(path) uncompressed = filesize(raw)
    end

    @testset "nc_append_record! checks the record" begin
        p = joinpath(dir, "bad.nc")
        ds = nc_create(p)
        ds.dim["n"] = 2; ds.dim["time"] = Inf
        nc_defvar!(ds, "a", Float32, ("n", "time"))
        nc_defvar!(ds, "b", Float32, ("n", "time"))
        @test_throws ErrorException nc_append_record!(ds, Dict("a" => Float32[1, 2]))   # b missing
        @test_throws ErrorException nc_append_record!(ds, Dict("a" => Float32[1, 2, 3],
                                                               "b" => Float32[1, 2]))   # wrong size
        @test ds.dim["time"] == 0                      # nothing written by a refused record
        nc_append_record!(ds, Dict("a" => Float32[1, 2], "b" => Float32[3, 4]))
        @test ds.dim["time"] == 1
        # `at`: a resumed run rewrites from its checkpoint's record on (#269)
        nc_append_record!(ds, Dict("a" => Float32[5, 6], "b" => Float32[7, 8]))
        nc_append_record!(ds, Dict("a" => Float32[9, 9], "b" => Float32[9, 9]); at = 2)
        @test ds.dim["time"] == 2 && ds["a"].var[:, 2] == Float32[9, 9] && ds["b"].var[:, 1] == Float32[3, 4]
        nc_append_record!(ds, Dict("a" => Float32[0, 1], "b" => Float32[2, 3]); at = 3)
        @test ds.dim["time"] == 3
        @test_throws ErrorException nc_append_record!(ds, Dict("a" => Float32[0, 1], "b" => Float32[2, 3]); at = 5)
        close(ds)
        # the record dimension must be the last one
        ds = nc_create(joinpath(dir, "bad2.nc"))
        ds.dim["time"] = Inf; ds.dim["n"] = 2
        nc_defvar!(ds, "c", Float32, ("time", "n"))
        @test_throws ErrorException nc_append_record!(ds, Dict("c" => Float32[1, 2]))
        close(ds)
    end

    @testset "nc_append_record!: blocks of records, two unlimited dimensions" begin
        p = joinpath(dir, "two.nc")
        ds = nc_create(p)
        ds.dim["z"] = 3; ds.dim["time"] = Inf; ds.dim["time_costly"] = Inf
        nc_defvar!(ds, "t", Float64, ("time",))
        nc_defvar!(ds, "lim", String, ("time",); deflatelevel = 0)   # vlen: no filters
        nc_defvar!(ds, "tc", Float64, ("time_costly",))
        nc_defvar!(ds, "prof", Float64, ("z", "time_costly"))
        # with two record dimensions the dimension must be named
        @test_throws ErrorException nc_append_record!(ds, Dict("t" => [1.0], "lim" => ["a"]))
        @test_throws ErrorException nc_append_record!(ds, Dict("t" => [1.0]); dim = "z")
        # a block of three records along `time`; `time_costly` untouched
        nc_append_record!(ds, Dict("t" => [1.0, 2.0, 3.0], "lim" => ["cfl", "", "diffusive"]);
                          dim = "time", count = 3, sync = false)
        @test ds.dim["time"] == 3 && ds.dim["time_costly"] == 0
        # a block of the wrong length is refused whole
        @test_throws ErrorException nc_append_record!(ds, Dict("t" => [4.0, 5.0], "lim" => ["x"]);
                                                      dim = "time", count = 2)
        @test ds.dim["time"] == 3
        nc_append_record!(ds, Dict("tc" => [0.0], "prof" => [1.0, 2.0, 3.0]); dim = "time_costly")
        nc_append_record!(ds, Dict("tc" => [5.0, 9.0], "prof" => [4.0 7.0; 5.0 8.0; 6.0 9.0]);
                          dim = "time_costly", count = 2)
        nc_append_record!(ds, Dict("t" => [4.0], "lim" => ["advective"]); dim = "time")
        close(ds)
        NCDataset(p) do ds
            @test ds["t"][:] == [1.0, 2.0, 3.0, 4.0]
            @test ds["lim"][:] == ["cfl", "", "diffusive", "advective"]
            @test ds["tc"][:] == [0.0, 5.0, 9.0]
            @test ds["prof"][:, :] == [1.0 4.0 7.0; 2.0 5.0 8.0; 3.0 6.0 9.0]
        end
    end

    @testset "NetCDF-3 files written before #310 still read" begin
        # written by the removed in-package NetCDFIO writer (CDF-2, 64-bit
        # offsets), same values as its generator script
        legacy = joinpath(@__DIR__, "data", "netcdf3", "aeolus_netcdfio_cdf2.nc")
        @test read(legacy, 4) == UInt8['C', 'D', 'F', 0x02]
        NCDataset(legacy) do ds
            @test ds.attrib["Conventions"] == "CF-1.8"
            @test ds.dim["time"] == 3
            @test ds["x"][:] == Float32[0.5, 1.5, 2.5, 3.5]
            @test Array(ds["building_height"]) == Float32[10i + j for i in 1:4, j in 1:3]
            # its time units ("seconds since simulation start") are not CF:
            # NCDatasets warns and returns the raw seconds
            t = @test_logs (:warn, r"cannot parse time units") ds["time"][:]
            @test t == [60.0, 120.0, 180.0]
            for t in 1:3
                @test ds["theta"][:, :, :, t] ==
                      Float32[290 + i + 10j + 100k + 0.5t for i in 1:4, j in 1:3, k in 1:2]
                @test ds["Ts"][:, :, t] == Float32[300 + i - j + t for i in 1:4, j in 1:3]
            end
        end
        # the CLM/SNICAR table shipped in src/data is CDF-1 (32-bit offsets)
        @test read(Aeolus.Snowpack.SNICAR_AGING_FILE, 4) == UInt8['C', 'D', 'F', 0x01]
        tab = snicar_aging_table()
        @test size(tab.drdt0) == (8, 31, 11) && tab.T[1] ≈ 223.15
    end
end
