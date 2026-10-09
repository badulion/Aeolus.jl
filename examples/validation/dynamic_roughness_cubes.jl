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
#   julia --project -m Aeolus run examples/validation/configs/dynamic_roughness_cubes.toml   # ~10 s
#   julia --project examples/validation/dynamic_roughness_cubes.jl [runs-root]
# The runs are the `dynamic_roughness` run type; this script reads their
# output/results.toml. Output: a table on stdout (no files).
# =========================================================================

using Printf
include(joinpath(@__DIR__, "read_runs.jl"))

const CHENG_CASTRO_2002 = (d_h = 0.83, z0_h = 0.053)     # see the header: unverified

runs = member_runs(runs_root(), validation_config("dynamic_roughness_cubes"))
r1 = runs[1][2]
h = r1.cfg["cubes"]["h"]
@printf("Staggered cubes: λp = λf = %.3f (measured from the map); Macdonald 1998: d/h = %.3f, z0/h = %.4f; Cheng & Castro 2002: d/h = %.2f, z0/h = %.3f\n",
        r1.res["lambda_p"], r1.res["d_mac"] / h, r1.res["z0_mac"] / h, CHENG_CASTRO_2002.d_h, CHENG_CASTRO_2002.z0_h)
@printf("\n%-5s %-8s %8s %8s %10s %10s %10s %12s %12s %10s %10s\n", "C_d", "Δ/h", "α", "conv", "⟨z0⟩/h", "form/τ",
        "u*/U", "z0eff/h d=0", "z0eff/h dMac", "÷Mac", "÷C&C")
for (_, r) in runs
    x = r.res; g = r.cfg["roughness"]
    @printf("%-5.2f %-8.1f %8.4f %8s %10.4f %10.3f %10.4f %12.4f %12.4f %10.2f %10.2f\n",
            g["c_d"], g["delta_over_h"], x["alpha"], string(x["converged"]), x["z0_mean"] / h, x["form_fraction"],
            x["ustar_over_u"], x["z0_eff_d0"] / h, x["z0_eff_dmac"] / h, x["ratio_mac"],
            x["z0_eff_dmac"] / (CHENG_CASTRO_2002.z0_h * h))
end
