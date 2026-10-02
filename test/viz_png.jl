# The dependency-free PNG / APNG encoder of examples/viz/apng.jl: its
# DEFLATE (LZ77 + fixed Huffman, stored fallback) and PNG row filters
# must round-trip exactly. Decoded here by an independent minimal
# inflate (RFC 1951 BTYPE 00 and 01) and the PNG unfilter, so the test
# needs no zlib. The encoder was also checked against Python's zlib
# when it was written.

using Test
using Random

isdefined(Main, :write_apng) || include(joinpath(@__DIR__, "..", "examples", "viz", "apng.jl"))

"Minimal inflate for stored and fixed-Huffman blocks (what the encoder emits)."
function _test_inflate(z::Vector{UInt8})
    @assert z[1] == 0x78 && (UInt16(z[1]) << 8 | z[2]) % 31 == 0
    data = z[3:end-4]
    pos = Ref(0)                                    # bit position
    bit() = (b = (data[pos[] >> 3 + 1] >> (pos[] & 7)) & 1; pos[] += 1; Int(b))
    bits(n) = (v = 0; for i in 0:n-1; v |= bit() << i; end; v)
    huff(n) = (v = 0; for _ in 1:n; v = (v << 1) | bit(); end; v)   # MSB first
    function litlen()
        v = huff(7)
        v <= 0x17 && return 256 + v                 # 7-bit codes 256–279
        v = (v << 1) | bit()
        0x30 <= v <= 0xbf && return v - 0x30        # 8-bit 0–143
        0xc0 <= v <= 0xc7 && return 280 + v - 0xc0  # 8-bit 280–287
        v = (v << 1) | bit()
        return 144 + v - 0x190                      # 9-bit 144–255
    end
    out = UInt8[]
    final = 0
    while final == 0
        final = bit(); btype = bits(2)
        if btype == 0
            pos[] = cld(pos[], 8) * 8
            o = pos[] >> 3
            len = Int(data[o + 1]) | Int(data[o + 2]) << 8
            nlen = Int(data[o + 3]) | Int(data[o + 4]) << 8
            @assert len ⊻ nlen == 0xffff
            append!(out, data[o + 5:o + 4 + len]); pos[] += 8 * (4 + len)
        elseif btype == 1
            while true
                s = litlen()
                s < 256 && (push!(out, UInt8(s)); continue)
                s == 256 && break
                i = s - 256
                len = _LEN_BASE[i] + bits(_LEN_EXTRA[i])
                d = huff(5) + 1
                dist = _DIST_BASE[d] + bits(_DIST_EXTRA[d])
                for _ in 1:len
                    push!(out, out[end - dist + 1])
                end
            end
        else
            error("unexpected BTYPE $btype")
        end
    end
    @assert adler32(out) == (UInt32(z[end-3]) << 24 | UInt32(z[end-2]) << 16 | UInt32(z[end-1]) << 8 | z[end])
    return out
end

"Undo the PNG row filters (None / Sub / Up) of an RGB image of width `w`."
function _test_unfilter(raw::Vector{UInt8}, w::Int, h::Int)
    stride = 3 * w
    img = zeros(UInt8, h, w, 3)
    prev = zeros(UInt8, stride)
    p = 1
    for r in 1:h
        f = raw[p]; p += 1
        line = raw[p:p + stride - 1]; p += stride
        for j in 1:stride
            a = j > 3 ? line[j - 3] : 0x00
            f == 1 && (line[j] += a)
            f == 2 && (line[j] += prev[j])
            @assert f in (0, 1, 2)
        end
        for c in 1:w, k in 1:3
            img[r, c, k] = line[3 * (c - 1) + k]
        end
        prev = line
    end
    return img
end

@testset "examples/viz: PNG encoder (DEFLATE + row filters) round-trips" begin
    rng = Xoshiro(2026)
    Z = [sin(i / 7) * cos(j / 5) for i in 1:64, j in 1:64]
    images = ("noise" => rand(rng, UInt8, 37, 53, 3),          # incompressible: stored fallback
              "flat"  => fill(UInt8(77), 200, 300, 3),         # long runs: maximum-length matches
              "grad"  => [UInt8((r + 2c + 50k) % 256) for r in 1:120, c in 1:160, k in 1:3],
              "plot"  => scalar_to_rgb(Z, -1, 1; upscale = 4),
              "tiny"  => rand(rng, UInt8, 1, 1, 3))
    for (name, img) in images
        h, w = size(img, 1), size(img, 2)
        z = _zlib_wrap(_filter_rgb(img))
        @test _test_unfilter(_test_inflate(z), w, h) == img
        ratio = length(img) / length(z)
        @info "  png $name $(h)×$(w): $(length(img)) → $(length(z)) bytes ($(round(ratio; digits = 1))×)"
        name == "noise" && @test length(z) <= length(img) + h + 64          # never much larger
        name == "flat"  && @test ratio > 50
        name == "plot"  && @test ratio > 8                                  # measured 15×
    end
    # the empty stream and an exact 65535-byte stored block
    @test _test_inflate(_zlib_wrap(UInt8[])) == UInt8[]
    big = rand(rng, UInt8, 65535)
    @test _test_inflate(_zlib_wrap(big)) == big
    # an APNG: every frame's data decodes back to its image
    frames = [scalar_to_rgb(Z .* s, -1, 1; upscale = 2) for s in (1.0, -1.0, 0.5)]
    path = tempname() * ".png"
    write_apng(path, frames; delay_ms = 100)
    b = read(path)
    @test b[1:8] == _PNG_SIG
    i = 9; decoded = Array{UInt8,3}[]
    while i <= length(b)
        n = Int(b[i]) << 24 | Int(b[i+1]) << 16 | Int(b[i+2]) << 8 | Int(b[i+3])
        t = String(b[i+4:i+7]); d = b[i+8:i+7+n]
        @test crc32(vcat(b[i+4:i+7], d)) == (UInt32(b[i+8+n]) << 24 | UInt32(b[i+9+n]) << 16 |
                                             UInt32(b[i+10+n]) << 8 | b[i+11+n])
        t == "IDAT" && push!(decoded, _test_unfilter(_test_inflate(d), 128, 128))
        t == "fdAT" && push!(decoded, _test_unfilter(_test_inflate(d[5:end]), 128, 128))
        i += 12 + n
    end
    @test decoded == frames
end
