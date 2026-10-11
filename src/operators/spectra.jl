# Isotropic energy spectrum and isotropic random-field initialiser
# (Phase 54, P54-c). Included into `Operators`; triply periodic fields on a
# `RegularGrid`, FFTW only.
#
# Both work with the same object: the DFT of each velocity component on its
# own staggered grid. A half-cell shift changes only the phase of a Fourier
# mode, so |û_j(k)|² is the energy the solver carries in that mode, and the
# shell sum is the solver's kinetic energy ½⟨u² + v² + w²⟩ (face values, as
# `kinetic_energy` and the energy-conserving C-grid advection count it).
# An earlier version transformed the two-point centre average instead,
# which weights every mode by Π_j cos²(k_jΔ_j/2) (→ 0 at the cut-off); the
# initialiser then divided by that factor, so a field "with the CBC
# spectrum" carried 22 % (32³) / 14 % (64³) more kinetic energy than CBC,
# ×2.1 in the cut-off shell (P54-c review, #205).

# Shell width: the fundamental wavenumber of the longest box side.
_shell_width(g::RegularGrid) = 2π / max(g.Lx, g.Ly, g.Lz)

"""
    isotropic_energy_spectrum(u, v, w; Δk = 2π / max(Lx, Ly, Lz)) -> (k, E)

Shell-summed kinetic-energy spectrum of a triply periodic velocity field.
Each component `u, v, w` (staggered `FaceX/FaceY/FaceZ` or all
`CellCenter`) is Fourier transformed on its own grid (FFTW `rfft` of the
interior; halos are not used). The staggering only shifts the phase of a
mode, so no interpolation is needed, and none is done: a two-point
average to cell centres would damp mode ``\\mathbf k`` by
``\\prod_j \\cos^2(k_jΔ_j/2)``, a filter that vanishes at the cut-off. The
energy
``\\tfrac12 |\\hat{\\mathbf u}(\\mathbf k)|^2`` of every mode is added to
the shell ``n = \\mathrm{round}(|\\mathbf k|/Δk)``, and

```math
E(k_n) = \\frac{1}{Δk} \\sum_{n - 1/2 \\,≤\\, |\\mathbf k|/Δk \\,<\\, n + 1/2}
         \\tfrac12 |\\hat{\\mathbf u}(\\mathbf k)|^2,
\\qquad k_n = nΔk,
```

normalised so that ``\\sum_n E(k_n)\\,Δk = \\tfrac12⟨u_i u_i⟩`` exactly
(Parseval), the average taken over the stored (face) values, i.e. the
kinetic energy of `kinetic_energy(sim)` divided by the volume. `k` runs from
0 (the mean flow, shell 0) to the shell of the corner mode, in rad m⁻¹;
`E` is in m³ s⁻². Unit-width (``Δk``) bins are the standard of
Rogallo (1981) and Pope (2000, §6.5).

References: Rogallo, R. S. (1981), *Numerical experiments in homogeneous
turbulence*, NASA TM 81315; Pope, S. B. (2000), *Turbulent Flows*, ch. 6.
"""
function isotropic_energy_spectrum(u::Field, v::Field, w::Field; Δk::Real = _shell_width(mesh(u)))
    g = mesh(u)
    g isa RegularGrid || throw(ArgumentError("isotropic_energy_spectrum: RegularGrid only"))
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    kx = _rfft_k(Nx, g.Lx); ky = _fft_k(Ny, g.Ly); kz = _fft_k(Nz, g.Lz)
    nmax = length(spectrum_wavenumbers(g; Δk = Δk)) - 1
    shell = zeros(Float64, nmax + 1)
    norm = 0.5 / (Nx * Ny * Nz)^2
    for f in (u, v, w)
        F = rfft(Array(interior_view(f)))      # native staggered values
        @inbounds for c in 1:size(F, 3), b in 1:size(F, 2), a in 1:size(F, 1)
            # rfft keeps kx ≥ 0 only: the modes with a conjugate partner
            # (0 < kx < Nyquist) count twice.
            wgt = (a == 1 || (iseven(Nx) && a == Nx ÷ 2 + 1)) ? 1 : 2
            n = round(Int, sqrt(kx[a]^2 + ky[b]^2 + kz[c]^2) / Δk)
            shell[n + 1] += wgt * norm * abs2(F[a,b,c])
        end
    end
    return spectrum_wavenumbers(g; Δk = Δk), shell ./ Δk
end

"""
    spectrum_wavenumbers(g::RegularGrid; Δk = 2π / max(Lx, Ly, Lz)) -> Vector

The shell wavenumbers ``k_n = nΔk`` (rad m⁻¹), ``n = 0`` to the shell of
the corner mode, at which [`isotropic_energy_spectrum`](@ref) of a field
on `g` is given.
"""
function spectrum_wavenumbers(g::RegularGrid; Δk::Real = _shell_width(g))
    kx = _rfft_k(g.Nx, g.Lx); ky = _fft_k(g.Ny, g.Ly); kz = _fft_k(g.Nz, g.Lz)
    kmax = sqrt(maximum(abs2, kx) + maximum(abs2, ky) + maximum(abs2, kz))
    return collect((0:round(Int, kmax / Δk)) .* float(Δk))
end

"""
    isotropic_random_field!(u, v, w, E; rng = Random.default_rng(), kmax) -> (u, v, w)
    isotropic_random_field!(sim, E; rng, kmax) -> sim

Fill the staggered velocity `u @ FaceX, v @ FaceY, w @ FaceZ` (interior)
with a solenoidal, random-phase, triply periodic field whose
[`isotropic_energy_spectrum`](@ref) is `E(k)` (``k`` in rad m⁻¹, ``E`` in
m³ s⁻²) in every shell ``1 ≤ n ≤ k_{max}/Δk``, and zero above. The `sim`
method writes `sim.u, sim.v, sim.w` and fills their halos with
`sim.bcs`.

Construction (Rogallo 1981, adapted to the C-grid). For every Fourier
mode a complex Gaussian vector ``\\hat b(\\mathbf k)`` is drawn and
projected so that the **discrete** staggered divergence vanishes: for
face values with Fourier amplitude ``a_j = \\hat b_j e^{-ik_jΔ_j/2}`` the
divergence ``(u_{i+1} - u_i)/Δx + …`` has amplitude
``2i\\sum_j \\hat b_j \\sin(k_jΔ_j/2)/Δ_j``, so ``\\hat b`` is projected
normal to ``S_j = \\sin(k_jΔ_j/2)/Δ_j`` (→ ``k_j/2`` as ``k_jΔ_j → 0``,
the continuous projection). Hermitian symmetry is imposed and every shell
is rescaled to hold exactly ``E(k_n)Δk`` of face energy
(``|a_j| = |\\hat b_j|``). Modes on a Nyquist plane are left empty. The
result is divergence-free to round-off under the C-grid divergence of the
pressure projection, so no projection step is needed before a run, and
its kinetic energy (`kinetic_energy(sim)`/volume) is ``\\sum_n E(k_n)Δk``.

`kmax` defaults to the largest sphere inside the grid,
``\\min_j π/Δ_j``.

References: Rogallo, R. S. (1981), NASA TM 81315; Kang, Chester &
Meneveau (2003), *J. Fluid Mech.* 480, 129 (the same construction on a
collocated grid for the Comte-Bellot–Corrsin initial condition).
"""
function isotropic_random_field!(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, E;
                                 rng::AbstractRNG = default_rng(),
                                 kmax::Real = _default_kmax(mesh(u)))
    g = mesh(u)
    g isa RegularGrid || throw(ArgumentError("isotropic_random_field!: RegularGrid only"))
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    Δ = (g.dx, g.dy, g.dz)
    Δk = _shell_width(g)
    nmax = floor(Int, kmax / Δk + 1e-9)
    mx = fftfreq(Nx, Nx); my = fftfreq(Ny, Ny); mz = fftfreq(Nz, Nz)
    kx = (2π / g.Lx) .* mx; ky = (2π / g.Ly) .* my; kz = (2π / g.Lz) .* mz
    C = (zeros(ComplexF64, Nx, Ny, Nz), zeros(ComplexF64, Nx, Ny, Nz), zeros(ComplexF64, Nx, Ny, Nz))
    shell_of = zeros(Int, Nx, Ny, Nz)          # 0 = mode left empty

    # 1. random vectors, discrete-solenoidal projection (normal to S_j)
    nyq(m, N) = iseven(N) && abs(m) == N ÷ 2
    for c in 1:Nz, b in 1:Ny, a in 1:Nx
        (nyq(mx[a], Nx) || nyq(my[b], Ny) || nyq(mz[c], Nz)) && continue
        kv = (kx[a], ky[b], kz[c])
        n = round(Int, sqrt(sum(abs2, kv)) / Δk)
        (n == 0 || n > nmax) && continue
        shell_of[a,b,c] = n
        S = ntuple(j -> sin(kv[j] * Δ[j] / 2) / Δ[j], 3)
        r = (randn(rng, ComplexF64), randn(rng, ComplexF64), randn(rng, ComplexF64))
        α = (S[1] * r[1] + S[2] * r[2] + S[3] * r[3]) / (S[1]^2 + S[2]^2 + S[3]^2)
        for j in 1:3
            C[j][a,b,c] = r[j] - α * S[j]
        end
    end

    # 2. Hermitian symmetry (the projection commutes with it: S(-k) = -S(k))
    neg(i, N) = mod(N - (i - 1), N) + 1
    for j in 1:3
        Cj = C[j]; S = similar(Cj)
        for c in 1:Nz, b in 1:Ny, a in 1:Nx
            S[a,b,c] = 0.5 * (Cj[a,b,c] + conj(Cj[neg(a, Nx), neg(b, Ny), neg(c, Nz)]))
        end
        Cj .= S
    end

    # 3. rescale every shell to E(k_n) Δk (face energy: |a_j| = |b_j|)
    norm = 0.5 / (Nx * Ny * Nz)^2
    have = zeros(Float64, nmax)
    for I in CartesianIndices(shell_of)
        n = shell_of[I]
        n > 0 && (have[n] += norm * (abs2(C[1][I]) + abs2(C[2][I]) + abs2(C[3][I])))
    end
    scale = [have[n] > 0 ? sqrt(max(float(E(n * Δk)), 0.0) * Δk / have[n]) : 0.0 for n in 1:nmax]
    for I in CartesianIndices(shell_of)
        n = shell_of[I]
        s = n > 0 ? scale[n] : 0.0
        for j in 1:3
            C[j][I] *= s
        end
    end

    # 4. phase shift to the faces, a_j = b_j e^{-ik_jΔ_j/2}, back to physical space
    for (j, f, kj) in ((1, u, kx), (2, v, ky), (3, w, kz))
        A = C[j]
        for c in 1:Nz, b in 1:Ny, a in 1:Nx
            shell_of[a,b,c] == 0 && continue
            k = j == 1 ? kj[a] : j == 2 ? kj[b] : kj[c]
            h = k * Δ[j] / 2
            A[a,b,c] *= cis(-h)
        end
        interior_view(f) .= real.(ifft(A))
    end
    return u, v, w
end

function isotropic_random_field!(sim, E; kw...)
    isotropic_random_field!(sim.u, sim.v, sim.w, E; kw...)
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    return sim
end

_default_kmax(g::RegularGrid) = min(π / g.dx, π / g.dy, π / g.dz)

"""
    rescale_to_spectrum!(u, v, w, E; kmax) -> (u, v, w)

Multiply every Fourier shell of the staggered, triply periodic velocity
`u @ FaceX, v @ FaceY, w @ FaceZ` by the factor that makes its
[`isotropic_energy_spectrum`](@ref) equal to `E(k)` (shells
``1 ≤ n ≤ k_{max}/Δk``; the mean and the shells above are zeroed).
Phases, and hence the triadic correlations a run has built, are kept;
the scaling is per Fourier mode, so a field that is discretely
divergence-free stays so. Halos are not filled.

This is the "evolve and rescale" preparation of an initial field with
realistic phases (Rogallo 1981; used for the Comte-Bellot–Corrsin case by
Kang, Chester & Meneveau 2003): a random-phase field has the right
spectrum but no energy transfer, and relaxes for about one eddy turnover
before the cascade is established.
"""
function rescale_to_spectrum!(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, E;
                              kmax::Real = _default_kmax(mesh(u)))
    g = mesh(u)
    Δk = _shell_width(g)
    nmax = floor(Int, kmax / Δk + 1e-9)
    k, Enow = isotropic_energy_spectrum(u, v, w)
    kx = _fft_k(g.Nx, g.Lx); ky = _fft_k(g.Ny, g.Ly); kz = _fft_k(g.Nz, g.Lz)
    scale = [1 <= n <= nmax && Enow[n+1] > 0 ? sqrt(max(float(E(n * Δk)), 0.0) / Enow[n+1]) : 0.0
             for n in 0:length(k)-1]
    for f in (u, v, w)
        F = fft(Array(interior_view(f)))
        @inbounds for c in 1:g.Nz, b in 1:g.Ny, a in 1:g.Nx
            n = round(Int, sqrt(kx[a]^2 + ky[b]^2 + kz[c]^2) / Δk)
            F[a,b,c] *= scale[n+1]
        end
        interior_view(f) .= real.(ifft(F))
    end
    return u, v, w
end

# =========================================================================
# A grid-independent isotropic field: the named analytic initial field
# `isotropic_turbulence` of the model (ONE-g, #456).
# =========================================================================

"""
    PassotPouquetSpectrum(u_rms, k_peak)

The energy spectrum ``E(k) = 16\\sqrt{2/π}\\, u_{rms}^2\\, k^4 / k_p^5\\,
\\exp(-2 (k/k_p)^2)`` (Passot & Pouquet 1987; the ``k^4`` low-wavenumber
range of Batchelor turbulence), normalised so that ``\\int_0^∞ E\\,dk =
\\tfrac32 u_{rms}^2`` (`u_rms` per component, m s⁻¹; `k_peak` in rad m⁻¹,
its maximum). A callable: `E(k)`.

Reference: Passot, T. & Pouquet, A. (1987), *J. Fluid Mech.* 181, 441–466.
"""
struct PassotPouquetSpectrum
    u_rms::Float64
    k_peak::Float64
end
(E::PassotPouquetSpectrum)(k::Real) =
    16 * sqrt(2 / π) * E.u_rms^2 * k^4 / E.k_peak^5 * exp(-2 * (k / E.k_peak)^2)

# SplitMix64 (Steele, Lea & Flood 2014): the counter-based hash behind the
# per-mode random numbers, so a mode's draw depends on (seed, mode) only.
@inline function _splitmix(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end
@inline _mode_key(seed, m1, m2, m3) =
    _splitmix(_splitmix(_splitmix(_splitmix(UInt64(seed)) ⊻ (m1 % UInt64)) ⊻ (m2 % UInt64)) ⊻ (m3 % UInt64))
# a uniform number in (0, 1] from a hash
@inline _unit(h::UInt64) = ((h >> 11) + 1) * 0x1.0p-53
# three complex standard normals (E|r_j|² = 1) of mode key h (Box-Muller)
@inline function _mode_normals(h::UInt64)
    out = ntuple(3) do j
        a = _splitmix(h + UInt64(2j - 1)); b = _splitmix(h + UInt64(2j))
        ρ = sqrt(-2 * log(_unit(a))); φ = 2π * _unit(b)
        complex(ρ * cos(φ), ρ * sin(φ)) / sqrt(2)
    end
    return out
end

"""
    isotropic_turbulence_field!(u, v, w, E; seed = 1, kmax = Inf) -> (u, v, w)

Fill the staggered velocity `u @ FaceX, v @ FaceY, w @ FaceZ` of a triply
periodic `RegularGrid` with the isotropic random field of spectrum `E(k)`
(rad m⁻¹ → m³ s⁻²) **defined in spectral space, independently of the
grid**: the field is

```math
\\mathbf u(\\mathbf x) = \\sum_{\\mathbf m} \\hat{\\mathbf u}_{\\mathbf m}\\, e^{i\\mathbf k\\cdot\\mathbf x},
\\qquad
\\hat{\\mathbf u}_{\\mathbf m} = \\sqrt{\\frac{E(k)\\,δV_k}{4πk^2}}\\;
\\mathsf P(\\mathbf k)\\,\\mathbf r_{\\mathbf m},
```

``\\mathbf k = 2π(m_1/L_x, m_2/L_y, m_3/L_z)``, ``δV_k = (2π)^3/(L_xL_yL_z)``,
``\\mathsf P = \\mathsf I - \\mathbf k\\mathbf k^T/k^2`` the continuous
solenoidal projection and ``\\mathbf r_{\\mathbf m}`` a complex standard
normal vector that depends only on `seed` and the integer mode ``\\mathbf
m`` (a counter-based hash, `_mode_key`; ``\\hat{\\mathbf u}_{-\\mathbf m} =
\\hat{\\mathbf u}_{\\mathbf m}^*``), so the expected shell energy is
``E(k)\\,Δk``. The grid evaluates the modes it carries, ``|m_i| < N_i/2``
(and ``|\\mathbf k| ≤`` `kmax`), each component at its own staggered
positions (the phase ``e^{i\\mathbf k\\cdot\\mathbf x_1}`` of its first
node). Two grids on the same box therefore hold the same Fourier
coefficients where both have the mode: the field of a fine grid,
truncated to the modes of a coarse one (a sharp spectral filter), *is* the
coarse grid's field, to round-off. That is what makes an LES and a DNS
start from the same large scales. The field is divergence-free in the
continuous sense (its discrete staggered divergence is ``O((kΔ)^2)``,
removed by the first projection of a run). Halos are not filled.

References: Rogallo, R. S. (1981), NASA TM 81315 (the random-phase
construction); Steele, Lea & Flood (2014), OOPSLA, 453–472 (SplitMix64).
"""
function isotropic_turbulence_field!(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, E;
                                     seed::Integer = 1, kmax::Real = Inf)
    g = mesh(u)
    g isa RegularGrid || throw(ArgumentError("isotropic_turbulence_field!: RegularGrid only"))
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    mx = fftfreq(Nx, Nx); my = fftfreq(Ny, Ny); mz = fftfreq(Nz, Nz)
    ck = (2π / g.Lx, 2π / g.Ly, 2π / g.Lz)
    δV = ck[1] * ck[2] * ck[3]
    keep(m, N) = 2 * abs(m) < N
    for (j, f, loc) in ((1, u, FaceX()), (2, v, FaceY()), (3, w, FaceZ()))
        x1 = (first(xnodes(g, loc)), first(ynodes(g, loc)), first(znodes(g, loc)))
        A = zeros(ComplexF64, Nx, Ny, Nz)
        @inbounds for c in 1:Nz, b in 1:Ny, a in 1:Nx
            m = (Int(mx[a]), Int(my[b]), Int(mz[c]))
            (keep(m[1], Nx) && keep(m[2], Ny) && keep(m[3], Nz)) || continue
            m == (0, 0, 0) && continue
            kv = (ck[1] * m[1], ck[2] * m[2], ck[3] * m[3])
            k2 = kv[1]^2 + kv[2]^2 + kv[3]^2
            k = sqrt(k2)
            k <= kmax || continue
            # the canonical half of the modes carries the draw; the other half its conjugate
            canon = m[3] > 0 || (m[3] == 0 && (m[2] > 0 || (m[2] == 0 && m[1] > 0)))
            mc = canon ? m : (-m[1], -m[2], -m[3])
            r = _mode_normals(_mode_key(seed, mc...))
            kc = canon ? kv : (-kv[1], -kv[2], -kv[3])
            kr = (kc[1] * r[1] + kc[2] * r[2] + kc[3] * r[3]) / k2
            pj = r[j] - kc[j] * kr
            amp = sqrt(max(float(E(k)), 0.0) * δV / (4π * k2))
            uh = amp * (canon ? pj : conj(pj))
            A[a, b, c] = uh * cis(kv[1] * x1[1] + kv[2] * x1[2] + kv[3] * x1[3]) * (Nx * Ny * Nz)
        end
        interior_view(f) .= real.(ifft(A))
    end
    return u, v, w
end

"""
    dissipation_rate(u, v, w, ν, νt) -> (resolved, sgs)

The domain-mean dissipation rate of the resolved kinetic energy (m² s⁻³)
at the cell centres of a `RegularGrid`, with the strain-rate stencil of
the SGS closures (`_strain_at`, ``|S| = (2S_{ij}S_{ij})^{1/2}``):

* `resolved` — the molecular ``2ν⟨S_{ij}S_{ij}⟩ = ν⟨|S|^2⟩`` (for a
  periodic incompressible flow ``ν⟨∂_ju_i∂_ju_i⟩``);
* `sgs` — the eddy-viscosity transfer to the subgrid scales
  ``⟨ν_t|S|^2⟩`` with the eddy viscosity `νt` (a cell-centred field: the
  closure's coefficient of the last evaluation; `nothing` for none).

Their sum is the dissipation of an LES, comparable with the ``ν⟨|S|^2⟩``
of a DNS. The halos of `u`, `v`, `w` must be filled.

Reference: Pope, S. B. (2000), *Turbulent Flows*, §§6.1, 13.4.
"""
function dissipation_rate(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, ν::Real, νt)
    g = mesh(u)
    g isa RegularGrid || throw(ArgumentError("dissipation_rate: RegularGrid only"))
    idx, idy, idz = 1 / g.dx, 1 / g.dy, 1 / g.dz
    ud, vd, wd = u.data, v.data, w.data
    res = 0.0; sgs = 0.0
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = g.Hx + i, g.Hy + j, g.Hz + k
        S = _strain_at(ud, vd, wd, I, J, K, idx, idy, idz)[7]
        res += S^2
        νt === nothing || (sgs += νt.data[I, J, K] * S^2)
    end
    n = g.Nx * g.Ny * g.Nz
    return (resolved = ν * res / n, sgs = sgs / n)
end
