# =========================================================================
# Dynamic roughness of a staggered cube array (P57-e, #190) — example 3
#
# A staggered array of cubes of side h at plan-area density λ_p = 0.25
# (cubes on a 2h × 2h lattice, alternate rows shifted by h; frontal area
# density λ_f = λ_p = 0.25), described as a height map on a 16-points-per-h
# fine grid, seen by LES grids of spacing Δ = h/2, h and 2h. The cubes are
# then partially resolved: the relief the grid keeps (h_Δ, the cell mean)
# exerts the surface-gradient drag of Anderson & Meneveau (2011), the rest
# enters z0 = α σ_Δ with the dynamic α of `DynamicRoughness` (spectral test
# filter at 2Δ, C_d = 1, z0_floor = 1e-4 h).
#
# The wind is a uniform log-law wind at z_ref (every drag term is ∝ U², so
# α does not depend on its strength). Reported per grid:
#   * α and the plane-mean modelled roughness ⟨z0⟩/h (sub-filter part only);
#   * the effective roughness of the whole surface, from the total drag
#     (modelled + resolved) τ = u*²: z0_eff = (z_ref − d) exp(−κU/u*), with
#     d = 0 (heights above the base, as the SGD model sees the surface) and
#     with d = d_Mac, the Macdonald displacement;
#   * the ratios to Macdonald et al. (1998) and Cheng & Castro (2002).
#
# References
#   Anderson, W. & Meneveau, C. (2011). J. Fluid Mech., 679, 288-314.
#   Macdonald, R. W., Griffiths, R. F. & Hall, D. J. (1998). Atmos.
#     Environ., 32, 1857-1864: d/h = 1 + A^-λp (λp − 1), A = 4.43
#     (staggered); z0/h = (1 − d/h) exp{−[0.5 β C_D/κ² (1 − d/h) λ_f]^-1/2},
#     β = 1.0 (staggered), C_D = 1.2, κ = 0.4.
#   Cheng, H. & Castro, I. P. (2002). Boundary-Layer Meteorol., 104,
#     229-259: staggered cubes, λp = 0.25, wind tunnel. The value used
#     here, z0/h = 0.053 with d/h = 0.83, is as quoted by secondary
#     sources; it was NOT checked against the paper's own table (no
#     access). Treat the Cheng & Castro ratio accordingly.
#
#   julia --project=. examples/validation/dynamic_roughness_cubes.jl   # ~10 s
# Output: a table on stdout (no files).
# =========================================================================

using Aeolus
using Printf
using Statistics

const κ = Aeolus.SurfaceExchange.von_karman()

# Macdonald et al. (1998), staggered array
function macdonald(λp, λf; A = 4.43, β = 1.0, C_D = 1.2, κM = 0.4)
    dh = 1 + A^(-λp) * (λp - 1)
    z0h = (1 - dh) * exp(-(0.5 * β * C_D / κM^2 * (1 - dh) * λf)^(-0.5))
    return dh, z0h
end
const CHENG_CASTRO_2002 = (d_h = 0.83, z0_h = 0.053)     # see the header: unverified

# Height map of the staggered array: ncube cubes per row, npc fine points
# per cube side, offset (in fine points) so that LES cells cut the cubes.
function cube_array(ncube, npc; h = 1.0, offset = npc ÷ 4)
    Nf = 2ncube * npc
    hf = zeros(Nf, Nf)
    for row in 0:ncube-1, col in 0:ncube-1
        x0 = col * 2npc + (isodd(row) ? npc : 0) + offset
        y0 = row * 2npc + offset
        for b in 1:npc, a in 1:npc
            hf[mod1(x0 + a, Nf), mod1(y0 + b, Nf)] = h
        end
    end
    return hf
end

h = 1.0
ncube, npc = 20, 16
hf = cube_array(ncube, npc; h = h)
λp = mean(hf .> 0)
Lf = 2ncube * h                      # side of the periodic tile
dMac, z0Mac = macdonald(λp, λp)
@printf("Staggered cubes: λp = λf = %.3f (measured from the map); Macdonald 1998: d/h = %.3f, z0/h = %.4f; Cheng & Castro 2002: d/h = %.2f, z0/h = %.3f\n",
        λp, dMac, z0Mac, CHENG_CASTRO_2002.d_h, CHENG_CASTRO_2002.z0_h)

z_ref = 2.0 * h                      # wind height above the base
U = 5.0
@printf("\n%-5s %-8s %8s %8s %10s %10s %10s %12s %12s %10s %10s\n", "C_d", "Δ/h", "α", "conv", "⟨z0⟩/h", "form/τ",
        "u*/U", "z0eff/h d=0", "z0eff/h dMac", "÷Mac", "÷C&C")
for (Δh, C_d) in ((0.5, 1.0), (1.0, 1.0), (2.0, 1.0), (0.5, 0.25), (0.5, 4.0))
    r = round(Int, Δh * npc)
    N = size(hf, 1) ÷ r
    dx = Lf / N
    st = subfilter_height(hf, r; dx = dx, dy = dx)
    dr = DynamicRoughness(st; z0_floor = 1e-4 * h, C_d = C_d)
    uc = fill(U, N, N); vc = zeros(N, N)
    update_dynamic_roughness!(dr, uc, vc, z_ref)
    D = roughness_surface_drag(dr, uc, vc, z_ref)
    us = sqrt(D.total)
    z0e0 = z_ref * exp(-κ * U / us)
    # with a displacement the same u* holds at the wind height z_ref − d
    z0ed = (z_ref - dMac * h) * exp(-κ * U / us)
    @printf("%-5.2f %-8.1f %8.4f %8s %10.4f %10.3f %10.4f %12.4f %12.4f %10.2f %10.2f\n",
            C_d, Δh, dr.α, string(dr.converged), mean(dr.z0) / h, D.form / D.total, us / U,
            z0e0 / h, z0ed / h, z0ed / (z0Mac * h), z0ed / (CHENG_CASTRO_2002.z0_h * h))
end
