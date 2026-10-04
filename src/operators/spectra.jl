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
    kmax = sqrt(maximum(abs2, kx) + maximum(abs2, ky) + maximum(abs2, kz))
    nmax = round(Int, kmax / Δk)
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
    k = collect((0:nmax) .* float(Δk))
    return k, shell ./ Δk
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
