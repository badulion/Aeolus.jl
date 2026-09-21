# Turbulence physics add-ons for the urban microclimate flagship.
#
# This file is intentionally isolated from `src/*` — it patches three
# physics gaps of `examples/urban/urban_physics.jl` without touching the
# Aeolus core:
#
#   F1. `apply_surface_flux_most!`   — Monin-Obukhov Similarity Theory
#       (Businger 1971, Businger–Dyer stability functions; Louis 1979
#       compact form) replaces the constant-Cₕ bulk formula.
#
#   F2. `roughness_z0`               — momentum roughness length per
#       surface class (street/roof/park/water/wall), used by F1.
#
#   F3. `compute_dynamic_smagorinsky_cs!` +
#       `compute_smagorinsky_νt_dynamic!` — Germano–Lilly 1991 dynamic
#       procedure. A spatially varying `Cₛ(x, y, z)` is derived from a
#       test-filter step and fed into the same `|S|`-based eddy-viscosity
#       kernel used by Aeolus (`src/FiniteVolume.jl`).
#
# The file is a *drop-in include* for the urban example: it depends on
# `urban_physics.jl` (for `SURFACE_*` class enum) and `Aeolus` (only for
# the mesh accessor pattern).

include(joinpath(@__DIR__, "urban_physics.jl"))
using Aeolus

export roughness_z0,
       most_bulk_transfer,
       apply_surface_flux_most!,
       compute_dynamic_smagorinsky_cs!,
       compute_smagorinsky_νt_dynamic!

# ---------------------------------------------------------------------------
# F2. Per-class momentum roughness length
# ---------------------------------------------------------------------------

"""
    roughness_z0(cls::UInt8) -> Float64

Momentum roughness length `z₀` [m] for the given surface class. Values
are typical urban / natural-cover means (Grimmond & Oke 1999; Wieringa
1993):

  * `SURFACE_STREET` (1)  → 0.03   (asphalt with kerb-side clutter)
  * `SURFACE_ROOF`   (2)  → 0.10   (rough flat-roof surface)
  * `SURFACE_PARK`   (3)  → 0.15   (low vegetation, hedgerows)
  * `SURFACE_WATER`  (4)  → 1e-4   (calm open water)
  * `SURFACE_WALL`   (5)  → 0.05   (masonry façade tributary cell)

Scalar roughness `z₀ₕ` is taken as `z₀ / 10` (Brutsaert 1982) inside
[`most_bulk_transfer`](@ref).
"""
@inline function roughness_z0(cls::UInt8)::Float64
    return cls == SURFACE_STREET ? 0.03  :
           cls == SURFACE_ROOF   ? 0.10  :
           cls == SURFACE_PARK   ? 0.15  :
           cls == SURFACE_WATER  ? 1e-4  :
           cls == SURFACE_WALL   ? 0.05  :
                                   0.03
end

# ---------------------------------------------------------------------------
# F1. MOST bulk transfer with Businger–Dyer / Louis-1979 stability
# ---------------------------------------------------------------------------

const _KARMAN   = 0.4
const _G_ACCEL  = 9.81
# Louis-1979 empirical constants (unstable branch).
const _LOUIS_B  = 9.4
const _LOUIS_CH = 7.4      # scalar branch coefficient
const _LOUIS_CM = 7.4      # momentum branch coefficient

"""
    most_bulk_transfer(U, dz1, Ts, θ_air, z0) -> (Cd, Ch)

Bulk drag `Cd` and heat-transfer `Ch` coefficients between the first
atmospheric layer (at height `dz1/2`, characteristic wind `U`, potential
temperature `θ_air`) and a surface at temperature `Ts`, following
Monin-Obukhov Similarity Theory with the Businger (1971) / Businger–Dyer
stability functions, cast in the closed-form Louis (1979) approximation
so no iterative Obukhov-length solve is required.

Neutral limits (no stability correction):

    Cd₀ = κ² / log²(z / z₀)
    Ch₀ = κ² / [log(z / z₀) · log(z / z₀ₕ)]      with z₀ₕ = z₀ / 10

where `z = dz1` is the mid-layer height and `κ = 0.4`. The bulk
Richardson number is

    Ri = g · dz1 · (θ_air - Ts) / (θ_air · U²)

clamped to `[-1, 0.2]` (very-unstable / near-critical stable). The
Louis stability multiplier is

    unstable (Ri < 0):
        f_m = 1 - b·Ri / (1 + c_m √(|Ri|))
        f_h = 1 - b·Ri / (1 + c_h √(|Ri|))
    stable   (Ri ≥ 0):
        f_m = 1 / (1 + b·Ri / √(1 + d·Ri))²
        f_h = same form with the h-branch constants

with `b = 9.4`, `c_h = c_m = 7.4 · Cd₀ · b · √(z/z₀)`, and `d = 5`. The
returned coefficients are `Cd = Cd₀ · f_m` and `Ch = Ch₀ · f_h`.
"""
@inline function most_bulk_transfer(U::Real, dz1::Real, Ts::Real,
                                    θ_air::Real, z0::Real)::Tuple{Float64, Float64}
    z    = float(dz1)
    Uabs = max(float(U), 0.1)                       # floor wind speed
    z0m  = max(float(z0), 1e-5)
    z0h  = z0m * 0.1                                # Brutsaert scalar length
    logm = log(z / z0m)
    logh = log(z / z0h)
    Cd0  = (_KARMAN * _KARMAN) / (logm * logm)
    Ch0  = (_KARMAN * _KARMAN) / (logm * logh)

    Ri = _G_ACCEL * z * (float(θ_air) - float(Ts)) /
         (max(float(θ_air), 1.0) * Uabs * Uabs)
    Ri = clamp(Ri, -1.0, 0.2)

    fm::Float64 = 1.0
    fh::Float64 = 1.0
    if Ri < 0.0
        # Louis-1979 unstable: sqrt(z/z0) form for the denominator.
        sqz  = sqrt(z / z0m)
        cm   = _LOUIS_CM * Cd0 * _LOUIS_B * sqz
        ch   = _LOUIS_CH * Cd0 * _LOUIS_B * sqz
        rabs = sqrt(-Ri)
        fm   = 1.0 - _LOUIS_B * Ri / (1.0 + cm * rabs)
        fh   = 1.0 - _LOUIS_B * Ri / (1.0 + ch * rabs)
    elseif Ri > 0.0
        d    = 5.0
        denm = 1.0 + _LOUIS_B * Ri / sqrt(1.0 + d * Ri)
        denh = denm
        fm   = 1.0 / (denm * denm)
        fh   = 1.0 / (denh * denh)
    end
    # Guard against numerical wandering (should already be ≥ 0 by construction).
    fm = max(fm, 0.0)
    fh = max(fh, 0.0)
    return (Cd0 * fm, Ch0 * fh)
end

"""
    apply_surface_flux_most!(sim, Ts, sfc, params;
                              ρcp = 1005.0, U_min = 0.5,
                              dt = 1.0, H_cap = 400.0) -> Matrix{Float64}

Drop-in replacement for
[`apply_surface_flux!`](@ref) that uses MOST instead of a constant
`Cₕ`. For each surface cell:

  1. Read the wind speed at the first interior atmospheric layer
     `U = √(u² + v²)`, floored at `U_min`.
  2. Read `θ_air` at the same cell.
  3. Look up `z₀` via [`roughness_z0`](@ref).
  4. Get `(Cd, Ch)` from [`most_bulk_transfer`](@ref).
  5. Sensible heat flux

         H = ρcp · Ch · U · (Ts - θ_air),  clamped to ±H_cap.

The flux is deposited into `sim.θ.data[i, j, 1st interior k]` and the
2-D W/m² field is returned so `step_surfaces!` can subtract it from the
surface budget on the next substep.
"""
function apply_surface_flux_most!(sim, Ts::AbstractMatrix,
                                  sfc::AbstractMatrix{UInt8},
                                  params::SurfaceParams;
                                  ρcp::Real = 1005.0,
                                  U_min::Real = 0.5,
                                  dt::Real = 1.0,
                                  H_cap::Real = 400.0)
    Nx, Ny = size(Ts)
    Hx = sim.mesh.Hx; Hy = sim.mesh.Hy; Hz = sim.mesh.Hz
    dz = float(sim.mesh.dz)
    dz1 = 0.5 * dz                                  # first-layer mid-height
    sensible = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        ic = Hx + i; jc = Hy + j; kc = Hz + 1
        # u lives on the west face of cell (ic, jc, kc) — average two
        # neighboring faces to cell center.
        uc = 0.5 * (sim.u.data[ic, jc, kc] + sim.u.data[ic + 1, jc, kc])
        vc = 0.5 * (sim.v.data[ic, jc, kc] + sim.v.data[ic, jc + 1, kc])
        U  = sqrt(uc * uc + vc * vc)
        U  = max(U, float(U_min))

        θ_air = sim.θ.data[ic, jc, kc]
        z0    = roughness_z0(sfc[i, j])
        _, Ch = most_bulk_transfer(U, dz1, Ts[i, j], θ_air, z0)

        H_flx = float(ρcp) * Ch * U * (Ts[i, j] - θ_air)
        H_flx = clamp(H_flx, -float(H_cap), float(H_cap))
        Fθ    = H_flx / float(ρcp)
        sensible[i, j] = H_flx
        sim.θ.data[ic, jc, kc] += Fθ * float(dt) / dz
    end
    return sensible
end

# ---------------------------------------------------------------------------
# F3. Germano–Lilly dynamic Smagorinsky
# ---------------------------------------------------------------------------
#
# We use the classic Germano–Lilly 1991 identity between grid-filter
# and test-filter stresses:
#
#     L_ij ≡ ~(uᵢuⱼ) − ũᵢũⱼ
#     M_ij ≡ 2 Δ̂² |S̃| S̃_ij  −  2 Δ²  ~( |S| S_ij )
#     C_s² ≈ ⟨L_ij M_ij⟩ / ⟨M_ij M_ij⟩              (Lilly averaging)
#
# with `~(·)` a test filter of width Δ̂ = α Δ. In this
# implementation the test filter is a trivial 3×3×3 box average
# (one-cell stencil in each direction). For a top-hat over 3 cells the
# second-moment ratio Δ̂/Δ ≈ 2, so we take α² = 4 in the M_ij formula.
# Local (rather than horizontal-plane) averaging is used to keep
# `C_s²(x,y,z)` a 3-D field. Values are clipped to `[0, 0.5²]` so a
# single ill-conditioned cell cannot drive negative viscosity.

const _CS_MAX2 = 0.25       # (Cₛ ≤ 0.5)²
const _ALPHA2  = 4.0        # (Δ_test / Δ)² for a 3-cell box test filter

# In-place strain rate |S| plus the six components at cell center.
@inline function _strain_center(u::AbstractArray{T,3},
                                v::AbstractArray{T,3},
                                w::AbstractArray{T,3},
                                i::Int, j::Int, k::Int,
                                idx::Float64, idy::Float64,
                                idz::Float64) where {T}
    S11 = (u[i+1,j,k] - u[i,j,k]) * idx
    S22 = (v[i,j+1,k] - v[i,j,k]) * idy
    S33 = (w[i,j,k+1] - w[i,j,k]) * idz

    du_dy = ((u[i,j+1,k] + u[i+1,j+1,k]) -
             (u[i,j-1,k] + u[i+1,j-1,k])) * (0.25 * idy)
    dv_dx = ((v[i+1,j,k] + v[i+1,j+1,k]) -
             (v[i-1,j,k] + v[i-1,j+1,k])) * (0.25 * idx)
    S12   = 0.5 * (du_dy + dv_dx)

    du_dz = ((u[i,j,k+1] + u[i+1,j,k+1]) -
             (u[i,j,k-1] + u[i+1,j,k-1])) * (0.25 * idz)
    dw_dx = ((w[i+1,j,k] + w[i+1,j,k+1]) -
             (w[i-1,j,k] + w[i-1,j,k+1])) * (0.25 * idx)
    S13   = 0.5 * (du_dz + dw_dx)

    dv_dz = ((v[i,j,k+1] + v[i,j+1,k+1]) -
             (v[i,j,k-1] + v[i,j+1,k-1])) * (0.25 * idz)
    dw_dy = ((w[i,j+1,k] + w[i,j+1,k+1]) -
             (w[i,j-1,k] + w[i,j-1,k+1])) * (0.25 * idy)
    S23   = 0.5 * (dv_dz + dw_dy)

    S2   = 2 * (S11*S11 + S22*S22 + S33*S33) +
           4 * (S12*S12 + S13*S13 + S23*S23)
    magS = sqrt(S2)
    return (S11, S22, S33, S12, S13, S23, magS)
end

# 3×3×3 box average of a scalar field at (i, j, k).
@inline function _box27(a::AbstractArray{Float64,3}, i::Int, j::Int, k::Int)
    s = 0.0
    @inbounds for dk in -1:1, dj in -1:1, di in -1:1
        s += a[i + di, j + dj, k + dk]
    end
    return s * (1.0 / 27.0)
end

"""
    compute_dynamic_smagorinsky_cs!(cs_field, u, v, w, mesh) -> cs_field

Fill `cs_field::Array{Float64,3}` — same total shape as the mesh
(`totalsize(mesh)`) — with a per-cell **Smagorinsky coefficient squared**
`Cₛ²(x, y, z)` obtained by the Germano–Lilly 1991 dynamic procedure.

Steps at every interior cell:

  1. Interpolate `u, v, w` to cell centers (`Uᵢ`), and compute strain
     rate `S_ij` + magnitude `|S|` from the face-centered fields with
     the same stencil as Aeolus' built-in `_compute_smagorinsky_νt!`.
  2. Test-filter (`~·`) with a 3×3×3 box mean: obtain `Ũᵢ`, `S̃_ij`,
     `|S̃|`, and `~(|S| S_ij)`, `~(Uᵢ Uⱼ)`.
  3. Leonard stresses  `L_ij = ~(Uᵢ Uⱼ) − Ũᵢ Ũⱼ`.
  4. `M_ij = 2 (α Δ)² |S̃| S̃_ij − 2 Δ² ~( |S| S_ij )`,
     with `Δ = (dx dy dz)^(1/3)` and `α² = 4` for the 3-cell box
     test filter.
  5. Local Lilly ratio  `Cₛ² = max(0, L:M) / (M:M + ε)`.
  6. Clip to `[0, 0.5²]`.

Halo cells (and interior cells that would need a stencil crossing the
halo boundary — the last shell) fall back to `Cₛ² = 0.16²`.
"""
function compute_dynamic_smagorinsky_cs!(cs_field::Array{Float64,3},
                                         u, v, w, mesh)
    ud = u.data; vd = v.data; wd = w.data
    idx = 1.0 / float(mesh.dx)
    idy = 1.0 / float(mesh.dy)
    idz = 1.0 / float(mesh.dz)
    Δ2  = (float(mesh.dx) * float(mesh.dy) * float(mesh.dz))^(2/3)
    Δt2 = _ALPHA2 * Δ2      # (α Δ)²

    fill!(cs_field, 0.16 * 0.16)            # sensible default in halo

    ii, jj, kk = interior(mesh)
    imin, imax = first(ii), last(ii)
    jmin, jmax = first(jj), last(jj)
    kmin, kmax = first(kk), last(kk)

    # We need one extra cell of stencil for the 3×3×3 test filter on top
    # of the strain-rate stencil, which itself already uses (i±1, j±1,
    # k±1). Restrict the sweep by two cells on each side.
    ilo = imin + 1; ihi = imax - 1
    jlo = jmin + 1; jhi = jmax - 1
    klo = kmin + 1; khi = kmax - 1
    ilo > ihi && return cs_field
    jlo > jhi && return cs_field
    klo > khi && return cs_field

    # Scratch buffers over the extended stencil range [imin..imax, ...].
    Ni = imax - imin + 1
    Nj = jmax - jmin + 1
    Nk = kmax - kmin + 1
    Uc  = Array{Float64,3}(undef, Ni, Nj, Nk)
    Vc  = Array{Float64,3}(undef, Ni, Nj, Nk)
    Wc  = Array{Float64,3}(undef, Ni, Nj, Nk)
    S11 = Array{Float64,3}(undef, Ni, Nj, Nk)
    S22 = Array{Float64,3}(undef, Ni, Nj, Nk)
    S33 = Array{Float64,3}(undef, Ni, Nj, Nk)
    S12 = Array{Float64,3}(undef, Ni, Nj, Nk)
    S13 = Array{Float64,3}(undef, Ni, Nj, Nk)
    S23 = Array{Float64,3}(undef, Ni, Nj, Nk)
    magS = Array{Float64,3}(undef, Ni, Nj, Nk)

    # Pass 1: cell-centered velocities and strain rates on the interior.
    @inbounds for k in kmin:kmax, j in jmin:jmax, i in imin:imax
        li = i - imin + 1; lj = j - jmin + 1; lk = k - kmin + 1
        Uc[li, lj, lk] = 0.5 * (ud[i, j, k] + ud[i + 1, j, k])
        Vc[li, lj, lk] = 0.5 * (vd[i, j, k] + vd[i, j + 1, k])
        Wc[li, lj, lk] = 0.5 * (wd[i, j, k] + wd[i, j, k + 1])
        s11, s22, s33, s12, s13, s23, ms =
            _strain_center(ud, vd, wd, i, j, k, idx, idy, idz)
        S11[li, lj, lk] = s11
        S22[li, lj, lk] = s22
        S33[li, lj, lk] = s33
        S12[li, lj, lk] = s12
        S13[li, lj, lk] = s13
        S23[li, lj, lk] = s23
        magS[li, lj, lk] = ms
    end

    # Pass 2: dynamic Cs² at cells that have a 3×3×3 stencil available
    # inside the interior (one shell in from the interior boundary).
    @inbounds for k in klo:khi, j in jlo:jhi, i in ilo:ihi
        li = i - imin + 1; lj = j - jmin + 1; lk = k - kmin + 1

        # --- test-filter the cell-centered velocities.
        Ut = _box27(Uc, li, lj, lk)
        Vt = _box27(Vc, li, lj, lk)
        Wt = _box27(Wc, li, lj, lk)

        # --- test-filter the products u_i u_j (Leonard stresses).
        UUt = 0.0; VVt = 0.0; WWt = 0.0
        UVt = 0.0; UWt = 0.0; VWt = 0.0
        # --- test-filter |S| S_ij and S_ij themselves.
        MS_S11 = 0.0; MS_S22 = 0.0; MS_S33 = 0.0
        MS_S12 = 0.0; MS_S13 = 0.0; MS_S23 = 0.0
        St11 = 0.0; St22 = 0.0; St33 = 0.0
        St12 = 0.0; St13 = 0.0; St23 = 0.0
        inv27 = 1.0 / 27.0
        for dk in -1:1, dj in -1:1, di in -1:1
            mi = li + di; mj = lj + dj; mk = lk + dk
            uu = Uc[mi, mj, mk]; vv = Vc[mi, mj, mk]; ww = Wc[mi, mj, mk]
            UUt += uu * uu; VVt += vv * vv; WWt += ww * ww
            UVt += uu * vv; UWt += uu * ww; VWt += vv * ww
            ms = magS[mi, mj, mk]
            MS_S11 += ms * S11[mi, mj, mk]
            MS_S22 += ms * S22[mi, mj, mk]
            MS_S33 += ms * S33[mi, mj, mk]
            MS_S12 += ms * S12[mi, mj, mk]
            MS_S13 += ms * S13[mi, mj, mk]
            MS_S23 += ms * S23[mi, mj, mk]
            St11 += S11[mi, mj, mk]; St22 += S22[mi, mj, mk]; St33 += S33[mi, mj, mk]
            St12 += S12[mi, mj, mk]; St13 += S13[mi, mj, mk]; St23 += S23[mi, mj, mk]
        end
        UUt *= inv27; VVt *= inv27; WWt *= inv27
        UVt *= inv27; UWt *= inv27; VWt *= inv27
        MS_S11 *= inv27; MS_S22 *= inv27; MS_S33 *= inv27
        MS_S12 *= inv27; MS_S13 *= inv27; MS_S23 *= inv27
        St11 *= inv27; St22 *= inv27; St33 *= inv27
        St12 *= inv27; St13 *= inv27; St23 *= inv27

        # |S̃| from the test-filtered strain rate tensor.
        magSt2 = 2.0 * (St11*St11 + St22*St22 + St33*St33) +
                 4.0 * (St12*St12 + St13*St13 + St23*St23)
        magSt  = sqrt(magSt2)

        # Leonard stresses.
        L11 = UUt - Ut * Ut
        L22 = VVt - Vt * Vt
        L33 = WWt - Wt * Wt
        L12 = UVt - Ut * Vt
        L13 = UWt - Ut * Wt
        L23 = VWt - Vt * Wt

        # M_ij = 2 (αΔ)² |S̃| S̃_ij  −  2 Δ² ~( |S| S_ij )
        two = 2.0
        M11 = two * Δt2 * magSt * St11 - two * Δ2 * MS_S11
        M22 = two * Δt2 * magSt * St22 - two * Δ2 * MS_S22
        M33 = two * Δt2 * magSt * St33 - two * Δ2 * MS_S33
        M12 = two * Δt2 * magSt * St12 - two * Δ2 * MS_S12
        M13 = two * Δt2 * magSt * St13 - two * Δ2 * MS_S13
        M23 = two * Δt2 * magSt * St23 - two * Δ2 * MS_S23

        # Deviatoric L (subtract trace/3). Diagonal M is already
        # traceless in incompressible flow, so this is a small correction
        # keeping the ratio well-posed.
        Ltr = (L11 + L22 + L33) * (1.0 / 3.0)
        L11d = L11 - Ltr; L22d = L22 - Ltr; L33d = L33 - Ltr

        LM = L11d*M11 + L22d*M22 + L33d*M33 +
             2.0 * (L12*M12 + L13*M13 + L23*M23)
        MM = M11*M11 + M22*M22 + M33*M33 +
             2.0 * (M12*M12 + M13*M13 + M23*M23)

        cs2 = MM > 1e-30 ? LM / MM : 0.0
        cs2 = clamp(cs2, 0.0, _CS_MAX2)
        cs_field[i, j, k] = cs2
    end
    return cs_field
end

"""
    compute_smagorinsky_νt_dynamic!(νt, u, v, w, cs_field, mesh) -> νt

Same as Aeolus' `compute_smagorinsky_νt!` but the coefficient `C_s²` is
read per-cell from `cs_field` (as produced by
[`compute_dynamic_smagorinsky_cs!`](@ref)) instead of being a scalar.
The strain-rate stencil is the same 4-point-averaged 6-component `|S|`
form used in `src/FiniteVolume.jl`, and the resulting eddy viscosity is

    ν_t(i,j,k) = cs_field[i,j,k] · Δ² · |S|(i,j,k),
              Δ² = (dx dy dz)^{2/3}.

Halos of `u`, `v`, `w` must be filled. `νt` is a `Field{CellCenter}`;
`cs_field` is a plain `Array{Float64,3}` of the same total shape as the
mesh.
"""
function compute_smagorinsky_νt_dynamic!(νt, u, v, w,
                                         cs_field::AbstractArray{Float64,3},
                                         mesh)
    idx = 1.0 / float(mesh.dx)
    idy = 1.0 / float(mesh.dy)
    idz = 1.0 / float(mesh.dz)
    Δ2  = (float(mesh.dx) * float(mesh.dy) * float(mesh.dz))^(2/3)
    ii, jj, kk = interior(mesh)
    ud = u.data; vd = v.data; wd = w.data
    nd = νt.data
    @inbounds for k in kk, j in jj, i in ii
        _, _, _, _, _, _, ms =
            _strain_center(ud, vd, wd, i, j, k, idx, idy, idz)
        cs2 = cs_field[i, j, k]
        nd[i, j, k] = cs2 * Δ2 * ms
    end
    return νt
end
