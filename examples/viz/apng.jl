# Pure-Julia PNG + APNG encoder.
#
# No CodecZlib (no binary dependencies): DEFLATE is implemented here,
# LZ77 with a 32 KiB window and the fixed Huffman code (RFC 1951 §3.2.6),
# after per-row PNG filtering (None / Sub / Up, chosen per row by the
# minimum-sum-of-absolute-differences heuristic of the PNG spec). Plots
# with flat colour regions compress 10–50×; a block that would not
# shrink is written stored. Renders in every modern browser (Chrome,
# Firefox, Safari) as an infinite-loop animation.

# ---------------------------------------------------------------------------
# CRC-32 (PNG polynomial)
# ---------------------------------------------------------------------------

const _CRC_TABLE = let
    tbl = Vector{UInt32}(undef, 256)
    for n in 0:255
        c = UInt32(n)
        for _ in 1:8
            c = c & 1 == 1 ? UInt32(0xedb88320) ⊻ (c >> 1) : c >> 1
        end
        tbl[n + 1] = c
    end
    tbl
end

function crc32(data)
    c = UInt32(0xFFFFFFFF)
    @inbounds for b in data
        c = _CRC_TABLE[((c ⊻ UInt32(b)) & 0xFF) + 1] ⊻ (c >> 8)
    end
    return c ⊻ UInt32(0xFFFFFFFF)
end

# ---------------------------------------------------------------------------
# Adler-32 (zlib checksum)
# ---------------------------------------------------------------------------

function adler32(data)
    a = UInt32(1); b = UInt32(0)
    @inbounds for byte in data
        a = (a + UInt32(byte)) % 65521
        b = (b + a) % 65521
    end
    return (b << 16) | a
end

# ---------------------------------------------------------------------------
# Byte helpers
# ---------------------------------------------------------------------------

_u32be(x) = UInt8[(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF]
_u16be(x) = UInt8[(x >>  8) & 0xFF, x & 0xFF]
_u16le(x) = UInt8[x & 0xFF, (x >> 8) & 0xFF]

# ---------------------------------------------------------------------------
# Uncompressed DEFLATE stream + zlib wrapper
# ---------------------------------------------------------------------------

function _deflate_stored(data::Vector{UInt8})
    out = UInt8[]
    n = length(data)
    i = 1
    while i <= n || (n == 0 && i == 1)
        blk = min(65535, n - i + 1)
        blk == 0 && (blk = 0)   # empty final block for 0-length data
        is_final = (i - 1 + blk) == n
        push!(out, UInt8(is_final ? 1 : 0))          # BFINAL/BTYPE=00
        append!(out, _u16le(UInt16(blk)))            # LEN
        append!(out, _u16le(UInt16(~UInt16(blk))))   # NLEN (one's complement)
        if blk > 0
            append!(out, @view data[i:i + blk - 1])
        end
        i += blk
        blk == 0 && break
    end
    return out
end

# --- LZ77 + fixed-Huffman DEFLATE (RFC 1951, BTYPE = 01) -------------------

mutable struct _BitWriter
    out::Vector{UInt8}
    acc::UInt64
    n::Int
end
_BitWriter(sizehint::Int) = (w = _BitWriter(UInt8[], 0, 0); sizehint!(w.out, sizehint); w)

# Append the `nbits` low bits of `v`, least significant first (RFC 1951 §3.1.1).
@inline function _putbits!(w::_BitWriter, v::Integer, nbits::Int)
    w.acc |= UInt64(v) << w.n
    w.n += nbits
    while w.n >= 8
        push!(w.out, UInt8(w.acc & 0xff))
        w.acc >>= 8
        w.n -= 8
    end
    return w
end

_flushbits!(w::_BitWriter) = (w.n > 0 && push!(w.out, UInt8(w.acc & 0xff)); w.acc = 0; w.n = 0; w)

# Huffman codes are packed most significant bit first: reverse, then append.
@inline function _puthuff!(w::_BitWriter, code::Integer, len::Int)
    rev = 0
    for i in 0:len-1
        rev = (rev << 1) | ((code >> i) & 1)
    end
    return _putbits!(w, rev, len)
end

# The fixed literal/length code: 0–143 → 8 bits from 0x30, 144–255 → 9
# bits from 0x190, 256–279 → 7 bits from 0, 280–287 → 8 bits from 0xC0.
@inline function _fixed_litlen!(w::_BitWriter, v::Integer)
    v <= 143 && return _puthuff!(w, 0x30 + v, 8)
    v <= 255 && return _puthuff!(w, 0x190 + (v - 144), 9)
    v <= 279 && return _puthuff!(w, v - 256, 7)
    return _puthuff!(w, 0xc0 + (v - 280), 8)
end

const _LEN_BASE   = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                     35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258]
const _LEN_EXTRA  = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                     3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
const _DIST_BASE  = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
                     257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
                     8193, 12289, 16385, 24577]
const _DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                     7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]

@inline function _emit_match!(w::_BitWriter, len::Int, dist::Int)
    i = searchsortedlast(_LEN_BASE, len)
    _fixed_litlen!(w, 256 + i)
    _LEN_EXTRA[i] > 0 && _putbits!(w, len - _LEN_BASE[i], _LEN_EXTRA[i])
    d = searchsortedlast(_DIST_BASE, dist)
    _puthuff!(w, d - 1, 5)                       # fixed distance code: 5 bits
    _DIST_EXTRA[d] > 0 && _putbits!(w, dist - _DIST_BASE[d], _DIST_EXTRA[d])
    return w
end

"""
    _deflate_fixed(data; max_chain = 64) -> Vector{UInt8}

One final DEFLATE block with the fixed Huffman code: greedy LZ77 over a
32 KiB window (hash chains on 3-byte prefixes, at most `max_chain`
candidates per position, matches of 3–258 bytes).
"""
function _deflate_fixed(data::Vector{UInt8}; max_chain::Int = 64)
    n = length(data)
    w = _BitWriter(n ÷ 4 + 16)
    _putbits!(w, 1, 1)                           # BFINAL
    _putbits!(w, 1, 2)                           # BTYPE = 01, fixed Huffman
    HB = 15
    mask = (1 << HB) - 1
    head = zeros(Int, 1 << HB)
    prev = zeros(Int, n)
    @inline hash3(i) = ((Int(data[i]) << 10) ⊻ (Int(data[i + 1]) << 5) ⊻ Int(data[i + 2])) & mask
    i = 1
    @inbounds while i <= n
        bestlen = 0; bestdist = 0
        if i + 2 <= n
            h = hash3(i) + 1
            cand = head[h]
            maxlen = min(258, n - i + 1)
            chain = 0
            while cand > 0 && i - cand <= 32768 && chain < max_chain
                l = 0
                while l < maxlen && data[cand + l] == data[i + l]
                    l += 1
                end
                if l > bestlen
                    bestlen = l; bestdist = i - cand
                    l == maxlen && break
                end
                cand = prev[cand]
                chain += 1
            end
            prev[i] = head[h]; head[h] = i
        end
        if bestlen >= 3
            _emit_match!(w, bestlen, bestdist)
            for k in i + 1:min(i + bestlen - 1, n - 2)
                h = hash3(k) + 1
                prev[k] = head[h]; head[h] = k
            end
            i += bestlen
        else
            _fixed_litlen!(w, data[i])
            i += 1
        end
    end
    _fixed_litlen!(w, 256)                       # end of block
    return _flushbits!(w).out
end

function _zlib_wrap(data::Vector{UInt8})
    # CMF 0x78 (deflate, 32 KiB window); FLG 0x01 makes the header a
    # multiple of 31. Stored blocks if compression would not shrink it.
    body = _deflate_fixed(data)
    length(body) >= length(data) + 5 * cld(length(data) + 1, 65535) &&
        (body = _deflate_stored(data))
    return vcat(UInt8[0x78, 0x01], body, _u32be(adler32(data)))
end

# ---------------------------------------------------------------------------
# PNG filtering: per row the cheapest of None (0), Sub (1) and Up (2)
# by the sum of |filtered byte| as a signed value (PNG spec §12.8)
# ---------------------------------------------------------------------------

function _filter_rgb(rgb::Array{UInt8,3})
    h, w, _ = size(rgb)
    rowlen = 3 * w
    out = Vector{UInt8}(undef, h * (1 + rowlen))
    raw = Vector{UInt8}(undef, rowlen)
    prevraw = zeros(UInt8, rowlen)
    cands = [Vector{UInt8}(undef, rowlen) for _ in 1:3]
    cost(v) = sum(x -> x < 0x80 ? Int(x) : 256 - Int(x), v)
    idx = 1
    @inbounds for r in 1:h
        for c in 1:w, k in 1:3
            raw[3 * (c - 1) + k] = rgb[r, c, k]
        end
        for j in 1:rowlen
            a = j > 3 ? raw[j - 3] : 0x00
            cands[1][j] = raw[j]
            cands[2][j] = raw[j] - a
            cands[3][j] = raw[j] - prevraw[j]
        end
        best = argmin(cost.(cands))
        out[idx] = UInt8(best - 1); idx += 1
        copyto!(out, idx, cands[best], 1, rowlen); idx += rowlen
        copyto!(prevraw, raw)
    end
    return out
end

# ---------------------------------------------------------------------------
# Chunk writer
# ---------------------------------------------------------------------------

function _write_chunk(io, type::AbstractString, data::Vector{UInt8})
    write(io, _u32be(UInt32(length(data))))
    type_bytes = Vector{UInt8}(codeunits(type))
    write(io, type_bytes)
    write(io, data)
    write(io, _u32be(crc32(vcat(type_bytes, data))))
end

const _PNG_SIG = UInt8[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]

# ---------------------------------------------------------------------------
# APNG writer
# ---------------------------------------------------------------------------

"""
    write_apng(path, frames; delay_ms = 200, loop = 0)

Write an animated PNG at `path`. `frames` is a vector of `Array{UInt8, 3}`
RGB images (`size(frame) == (H, W, 3)`). `delay_ms` is the per-frame
delay; `loop = 0` loops forever.
"""

"""
    write_png(path, rgb) -> path

Write a single RGB frame as a plain PNG — no `acTL`/`fcTL`, so it is an
ordinary still image rather than a one-frame animation. Shares the chunk,
filter and zlib machinery with [`write_apng`](@ref).

Exists because a README hero image wants a still: a 48-frame APNG of a
diurnal cycle is ~75 MB, which no repository front page should carry.
"""
function write_png(path::AbstractString, rgb::Array{UInt8,3})
    h, w = size(rgb, 1), size(rgb, 2)
    size(rgb, 3) == 3 || error("write_png: expected an RGB array")
    open(path, "w") do io
        write(io, _PNG_SIG)
        _write_chunk(io, "IHDR", vcat(_u32be(UInt32(w)), _u32be(UInt32(h)),
                                      UInt8[8, 2, 0, 0, 0]))
        _write_chunk(io, "IDAT", _zlib_wrap(_filter_rgb(rgb)))
        _write_chunk(io, "IEND", UInt8[])
    end
    return path
end

function write_apng(path::AbstractString, frames::Vector{Array{UInt8,3}};
                    delay_ms::Int = 200, loop::Int = 0)
    isempty(frames) && error("write_apng: no frames given")
    h, w = size(frames[1], 1), size(frames[1], 2)
    for f in frames
        size(f) == (h, w, 3) || error("write_apng: frame size mismatch")
    end
    open(path, "w") do io
        write(io, _PNG_SIG)
        # IHDR
        ihdr = vcat(_u32be(UInt32(w)), _u32be(UInt32(h)),
                    UInt8[8, 2, 0, 0, 0])   # depth=8, color=RGB, all zero
        _write_chunk(io, "IHDR", ihdr)
        # acTL — animation control
        actl = vcat(_u32be(UInt32(length(frames))), _u32be(UInt32(loop)))
        _write_chunk(io, "acTL", actl)

        seq = UInt32(0)
        for (idx, frame) in enumerate(frames)
            # fcTL — frame control
            fctl = vcat(
                _u32be(seq),                        # sequence number
                _u32be(UInt32(w)),                  # width
                _u32be(UInt32(h)),                  # height
                _u32be(UInt32(0)),                  # x offset
                _u32be(UInt32(0)),                  # y offset
                _u16be(UInt16(delay_ms)),           # delay_num
                _u16be(UInt16(1000)),               # delay_den (=> ms)
                UInt8[0, 0],                        # dispose=NONE, blend=SOURCE
            )
            _write_chunk(io, "fcTL", fctl)
            seq += 1

            filtered   = _filter_rgb(frame)
            compressed = _zlib_wrap(filtered)

            if idx == 1
                _write_chunk(io, "IDAT", compressed)
            else
                # fdAT = 4-byte seq + compressed data
                fdat_data = vcat(_u32be(seq), compressed)
                _write_chunk(io, "fdAT", fdat_data)
                seq += 1
            end
        end

        _write_chunk(io, "IEND", UInt8[])
    end
    return path
end

# ---------------------------------------------------------------------------
# Helpers to build RGB frames from scalar fields with the viridis map
# ---------------------------------------------------------------------------

# Viridis anchor colors (same as plot_utils.jl)
const _VIRIDIS_APNG = (
    (68,   1,  84), (59,  82, 139), (33, 145, 140),
    (94, 201,  98), (253, 231, 37),
)

function _viridis_apng(t::Real)
    t = clamp(t, 0, 1)
    n = length(_VIRIDIS_APNG) - 1
    s = t * n
    i = clamp(Int(floor(s)) + 1, 1, n)
    frac = s - (i - 1)
    c0 = _VIRIDIS_APNG[i]; c1 = _VIRIDIS_APNG[i + 1]
    return (round(UInt8, c0[1] * (1 - frac) + c1[1] * frac),
            round(UInt8, c0[2] * (1 - frac) + c1[2] * frac),
            round(UInt8, c0[3] * (1 - frac) + c1[3] * frac))
end

"""
    scalar_to_rgb(Z, vmin, vmax; upscale = 4)

Turn a 2-D scalar matrix `Z` into an `Array{UInt8, 3}` RGB frame with the
viridis colormap. `upscale` replicates each cell for visibility. Non-
finite values render mid-gray.
"""
function scalar_to_rgb(Z::AbstractMatrix, vmin::Real, vmax::Real; upscale::Int = 4)
    ny, nx = size(Z)
    H = ny * upscale; W = nx * upscale
    rgb = Array{UInt8, 3}(undef, H, W, 3)
    span = vmax == vmin ? 1.0 : (vmax - vmin)
    @inbounds for j_out in 1:H
        j = min(cld(j_out, upscale), ny)
        for i_out in 1:W
            i = min(cld(i_out, upscale), nx)
            z = Z[j, i]
            t = isfinite(z) ? (z - vmin) / span : 0.5
            r, g, b = _viridis_apng(t)
            rgb[j_out, i_out, 1] = r
            rgb[j_out, i_out, 2] = g
            rgb[j_out, i_out, 3] = b
        end
    end
    return rgb
end

"""
    tile_frames(panels::Matrix{Array{UInt8,3}}) -> Array{UInt8,3}

Combine a matrix of RGB panels into one bigger RGB frame. Panels in each
row must have the same height; panels in each column must have the same
width.
"""
function tile_frames(panels::AbstractMatrix{<:Array{UInt8, 3}})
    n_rows, n_cols = size(panels)
    row_h = [size(panels[r, 1], 1) for r in 1:n_rows]
    col_w = [size(panels[1, c], 2) for c in 1:n_cols]
    H = sum(row_h); W = sum(col_w)
    out = fill(UInt8(30), H, W, 3)
    y = 1
    for r in 1:n_rows
        x = 1
        for c in 1:n_cols
            p = panels[r, c]
            out[y:y + row_h[r] - 1, x:x + col_w[c] - 1, :] .= p
            x += col_w[c]
        end
        y += row_h[r]
    end
    return out
end

"""
    add_label!(rgb, text; row = 4, col = 4, color = (255, 255, 255))

Very small bitmap-font label rendered on `rgb`. Handles ASCII 0-9, A-Z,
a-z, `.:_-` — enough for time stamps and short field titles.
"""
function add_label!(rgb::Array{UInt8, 3}, text::AbstractString;
                    row::Int = 4, col::Int = 4,
                    color::NTuple{3, Integer} = (255, 255, 255))
    r, g, b = color
    cx = col
    for ch in text
        _draw_char!(rgb, ch, row, cx, r, g, b)
        cx += 6   # 5-wide char + 1-space
    end
    return rgb
end

# Minimal 5×7 bitmap font — one bit per pixel, one Int per column
# packed as bits 0-6 = rows top→bottom. Only characters actually used.
const _FONT5x7 = Dict{Char, NTuple{5, UInt8}}(
    ' ' => (0x00,0x00,0x00,0x00,0x00),
    '.' => (0x00,0x00,0x60,0x60,0x00),
    ':' => (0x00,0x66,0x00,0x00,0x00),
    '_' => (0x00,0x00,0x40,0x40,0x40),
    '-' => (0x00,0x10,0x10,0x10,0x00),
    '/' => (0x00,0x40,0x30,0x08,0x04),
    '0' => (0x7C,0x82,0x82,0x82,0x7C),
    '1' => (0x00,0x84,0xFE,0x80,0x00),
    '2' => (0xC4,0xA2,0x92,0x8A,0x86),
    '3' => (0x44,0x82,0x92,0x92,0x6C),
    '4' => (0x30,0x28,0x24,0xFE,0x20),
    '5' => (0x4E,0x8A,0x8A,0x8A,0x72),
    '6' => (0x78,0x94,0x92,0x92,0x60),
    '7' => (0x02,0xE2,0x12,0x0A,0x06),
    '8' => (0x6C,0x92,0x92,0x92,0x6C),
    '9' => (0x0C,0x92,0x92,0x52,0x3C),
    'B' => (0xFE,0x92,0x92,0x92,0x6C),
    'G' => (0x7C,0x82,0x92,0x92,0xF4),
    'I' => (0x00,0x82,0xFE,0x82,0x00),
    'J' => (0x40,0x80,0x82,0x7E,0x02),
    'M' => (0xFE,0x04,0x18,0x04,0xFE),
    'Q' => (0x7C,0x82,0xA2,0x42,0xBC),
    'Z' => (0xC2,0xA2,0x92,0x8A,0x86),
    'b' => (0xFE,0x90,0x88,0x88,0x70),
    'c' => (0x70,0x88,0x88,0x88,0x40),
    'd' => (0x70,0x88,0x88,0x90,0xFE),
    'f' => (0x10,0xFC,0x12,0x02,0x04),
    'g' => (0x18,0xA4,0xA4,0xA4,0x7C),
    'j' => (0x40,0x80,0x88,0x7A,0x00),
    'k' => (0xFE,0x20,0x50,0x88,0x00),
    'm' => (0xF8,0x08,0x30,0x08,0xF0),
    'p' => (0xF8,0x28,0x28,0x28,0x10),
    'q' => (0x10,0x28,0x28,0x30,0xF8),
    'x' => (0x88,0x50,0x20,0x50,0x88),
    '?' => (0x04,0x02,0xA2,0x12,0x0C),
    '(' => (0x00,0x00,0x7C,0x82,0x00),
    ')' => (0x00,0x82,0x7C,0x00,0x00),
    '+' => (0x10,0x10,0x7C,0x10,0x10),
    '%' => (0x46,0x26,0x10,0xC8,0xC4),
    ',' => (0x00,0x00,0xA0,0x60,0x00),
    'A' => (0xF8,0x14,0x12,0x14,0xF8),
    'C' => (0x38,0x44,0x82,0x82,0x44),
    'D' => (0xFE,0x82,0x82,0x82,0x7C),
    'E' => (0xFE,0x92,0x92,0x92,0x82),
    'F' => (0xFE,0x12,0x12,0x02,0x02),
    'H' => (0xFE,0x10,0x10,0x10,0xFE),
    'K' => (0xFE,0x10,0x28,0x44,0x82),
    'L' => (0xFE,0x80,0x80,0x80,0x80),
    'N' => (0xFE,0x0C,0x30,0xC0,0xFE),
    'O' => (0x7C,0x82,0x82,0x82,0x7C),
    'P' => (0xFE,0x12,0x12,0x12,0x0C),
    'R' => (0xFE,0x12,0x32,0x52,0x8C),
    'S' => (0x4C,0x92,0x92,0x92,0x64),
    'T' => (0x02,0x02,0xFE,0x02,0x02),
    'U' => (0x7E,0x80,0x80,0x80,0x7E),
    'V' => (0x1E,0x60,0x80,0x60,0x1E),
    'W' => (0x7E,0x80,0x60,0x80,0x7E),
    'X' => (0x82,0x44,0x38,0x44,0x82),
    'Y' => (0x0E,0x10,0xE0,0x10,0x0E),
    'a' => (0x40,0xA8,0xA8,0xA8,0xF0),
    'e' => (0x38,0x54,0x54,0x54,0x18),
    'h' => (0xFE,0x08,0x04,0x04,0xF8),
    'i' => (0x00,0x00,0xFA,0x00,0x00),
    'l' => (0x00,0x00,0xFE,0x00,0x00),
    'n' => (0xFC,0x08,0x04,0x04,0xF8),
    'o' => (0x38,0x44,0x44,0x44,0x38),
    'r' => (0xF8,0x08,0x04,0x04,0x08),
    's' => (0x48,0x94,0x94,0x94,0x60),
    't' => (0x00,0x04,0x7E,0x84,0x00),
    'u' => (0x7C,0x80,0x80,0x40,0xFC),
    'v' => (0x1C,0x60,0x80,0x60,0x1C),
    'w' => (0x7C,0x80,0x60,0x80,0x7C),
    'y' => (0x0C,0x50,0xA0,0x50,0x0C),
    'z' => (0x88,0xC8,0xA8,0x98,0x88),
)

function _draw_char!(rgb::Array{UInt8,3}, ch::Char, row::Int, col::Int, r, g, b)
    # Render an unknown character as '?' rather than a blank. A missing
    # glyph used to vanish silently, so "surface temperature" came out as
    # "sur a e te erature" with nothing to indicate anything was wrong.
    ch = haskey(_FONT5x7, ch) ? ch : '?'
    cols = _FONT5x7[ch]
    H = size(rgb, 1); W = size(rgb, 2)
    for cx in 1:5
        colbits = cols[cx]
        for cy in 1:7
            if (colbits >> (cy - 1)) & 0x01 == 0x01
                y = row + cy - 1
                x = col + cx - 1
                if 1 <= y <= H && 1 <= x <= W
                    rgb[y, x, 1] = r
                    rgb[y, x, 2] = g
                    rgb[y, x, 3] = b
                end
            end
        end
    end
end
