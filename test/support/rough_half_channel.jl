# The rough half channel of the dynamic wall model (P57-e, #190): a random
# k⁻² surface and the half channel over it, partly resolved through the
# surface-gradient drag. Shared by test/era6_wall_models_dynamic.jl (the
# σ² statistics and the momentum budget over a few steps) and
# validation/era6_wall_models_dynamic.jl (grid invariance over 6 H/u*).
# The caller has `using .ChannelCases` (half_channel_case).

using Random: Xoshiro
using Statistics: mean, std
using FFTW: ifft
using Aeolus
using Aeolus.WallModels

# A periodic random surface on an Nf² map of side L whose radial spectrum
# is E(k) ∝ k^-2 for 1 ≤ k ≤ kmax (in units of 2π/L), rms σh, zero mean.
function _k2_surface(Nf, L, σh; seed = 1, kmax = Nf ÷ 2)
    rng = Xoshiro(seed)
    Hk = zeros(ComplexF64, Nf, Nf)
    for b in 1:Nf, a in 1:Nf
        kx = a - 1 <= Nf ÷ 2 ? a - 1 : a - 1 - Nf
        ky = b - 1 <= Nf ÷ 2 ? b - 1 : b - 1 - Nf
        k = hypot(kx, ky)
        (1 <= k <= kmax) || continue
        Hk[a, b] = k^-1.5 * cis(2π * rand(rng))      # |ĥ|² ∝ k⁻³ per mode → E(k) ∝ k⁻²
    end
    h = real(ifft(Hk))
    h .-= mean(h)
    return h .* (σh / std(h))
end

_blockmean(A, r) = [mean(@view A[(i-1)*r+1:i*r, (j-1)*r+1:j*r])
                    for i in 1:size(A, 1)÷r, j in 1:size(A, 2)÷r]

# A random k⁻² surface (256² map, 1 ≤ k ≤ 128, σ_h = 0.04 H), its
# sub-filter statistics at 32² and 16², and the half channel 2πH × 2πH × H,
# 32 × 32 × 8, StaticSmagorinsky(0.1), with the dynamic roughness of the
# 32² statistics (C_d = 4), not yet stepped.
function _rough_half_channel()
    Nf, L, H = 256, 2π, 1.0
    hf = _k2_surface(Nf, L, 0.04)
    C_d, z0f, Nz = 4.0, 1e-5, 8
    st32 = subfilter_height(hf, Nf ÷ 32; dx = L / 32, dy = L / 32)
    st16 = subfilter_height(hf, Nf ÷ 16; dx = L / 16, dy = L / 16)
    dr = DynamicRoughness(st32; z0_floor = z0f, C_d = C_d)
    case = half_channel_case(; N = (32, 32, Nz), roughness = dr, z0_over_H = 0.01)
    z = case.sim.mesh.dz / 2
    return (; Nf, L, H, hf, C_d, z0f, Nz, st32, st16, dr, case, z)
end
