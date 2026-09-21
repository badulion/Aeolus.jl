# Pure-Julia PNG + APNG encoder.
#
# Uses uncompressed DEFLATE stored blocks so we don't need CodecZlib.
# File sizes are ~1.5–2× a normally compressed PNG, but the whole
# encoder fits in this file with no external dependencies. Renders in
# every modern browser (Chrome, Firefox, Safari) as an infinite-loop
# animation.

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

function _zlib_wrap(data::Vector{UInt8})
    # 0x78 0x01 = deflate, no compression preset
    body = _deflate_stored(data)
    check = adler32(data)
    return vcat(UInt8[0x78, 0x01], body, _u32be(check))
end

# ---------------------------------------------------------------------------
# PNG filtering — we always use "None" filter (row byte 0x00)
# ---------------------------------------------------------------------------

function _filter_none_rgb(rgb::Array{UInt8,3})
    h, w, _ = size(rgb)
    out = Vector{UInt8}(undef, h * (1 + 3 * w))
    idx = 1
    @inbounds for r in 1:h
        out[idx] = 0x00; idx += 1
        for c in 1:w
            out[idx]     = rgb[r, c, 1]
            out[idx + 1] = rgb[r, c, 2]
            out[idx + 2] = rgb[r, c, 3]
            idx += 3
        end
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

            filtered   = _filter_none_rgb(frame)
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
    ch = haskey(_FONT5x7, ch) ? ch : ' '
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
