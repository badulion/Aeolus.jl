# Isotropic energy spectrum and isotropic random-field initialiser
# (Phase 54, P54-c). Included into `Operators`; triply periodic fields on a
# `RegularGrid`, FFTW only.
#
# Both work with the same object: the DFT of the velocity interpolated to
# cell centres (the two-point average of `interpolate_to_center!`, taken
# with periodic wrap-around so halos are not needed). The initialiser
# builds face values whose centre interpolation has exactly the requested
# shell spectrum and whose staggered (C-grid) divergence vanishes, so the
# spectrum the diagnostic reports for a fresh field is the one asked for.

# Shell width: the fundamental wavenumber of the longest box side.
_shell_width(g::RegularGrid) = 2π / max(g.Lx, g.Ly, g.Lz)

# Cell-centred copy of a staggered component, periodic wrap-around.
function _to_center_periodic(f::Field{L}) where {L}
    a = interior_view(f)
    Nx, Ny, Nz = size(a)
    s = L === FaceX ? (1, 0, 0) : L === FaceY ? (0, 1, 0) : L === FaceZ ? (0, 0, 1) : (0, 0, 0)
    s == (0, 0, 0) && return Array(a)
    c = similar(a, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        c[i,j,k] = 0.5 * (a[i,j,k] + a[mod1(i + s[1], Nx), mod1(j + s[2], Ny), mod1(k + s[3], Nz)])
    end
    return c
end

"""
    isotropic_energy_spectrum(u, v, w; Δk = 2π / max(Lx, Ly, Lz)) -> (k, E)

Shell-summed kinetic-energy spectrum of a triply periodic velocity field.
`u, v, w` (staggered `FaceX/FaceY/FaceZ` or all `CellCenter`) are first
interpolated to cell centres (two-point average, periodic wrap; halos are
not used), then Fourier transformed (FFTW `rfft`). The energy
``\\tfrac12 |\\hat{\\mathbf u}(\\mathbf k)|^2`` of every mode is added to
the shell ``n = \\mathrm{round}(|\\mathbf k|/Δk)``, and

```math
E(k_n) = \\frac{1}{Δk} \\sum_{n - 1/2 \\,≤\\, |\\mathbf k|/Δk \\,<\\, n + 1/2}
         \\tfrac12 |\\hat{\\mathbf u}(\\mathbf k)|^2,
\\qquad k_n = nΔk,
```

normalised so that ``\\sum_n E(k_n)\\,Δk = \\tfrac12⟨u_i u_i⟩`` exactly
(Parseval), the average taken over the cell-centred field. `k` runs from
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
        F = rfft(_to_center_periodic(f))
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
mode a complex Gaussian vector ``\\hat c(\\mathbf k)`` is drawn and
projected so that the **discrete** staggered divergence vanishes:
for face values with Fourier amplitude ``a_j`` the cell-centred average
is ``\\hat c_j = a_j e^{ik_jΔ_j/2}\\cos(k_jΔ_j/2)`` and the divergence
``2i\\sum_j \\hat c_j \\tan(k_jΔ_j/2)/Δ_j``, so ``\\hat c`` is projected
normal to ``T_j = \\tan(k_jΔ_j/2)/Δ_j`` (→ ``k_j/2`` as ``k_jΔ_j → 0``,
the continuous projection). Hermitian symmetry is imposed, every shell is
rescaled to hold exactly ``E(k_n)Δk``, and the faces are recovered as
``a_j = \\hat c_j e^{-ik_jΔ_j/2}/\\cos(k_jΔ_j/2)``. Modes on a Nyquist
plane (where the centre average is blind) are left empty. The result is
divergence-free to round-off under the C-grid divergence of the pressure
projection, so no projection step is needed before a run.

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

    # 1. random vectors, discrete-solenoidal projection
    nyq(m, N) = iseven(N) && abs(m) == N ÷ 2
    for c in 1:Nz, b in 1:Ny, a in 1:Nx
        (nyq(mx[a], Nx) || nyq(my[b], Ny) || nyq(mz[c], Nz)) && continue
        kv = (kx[a], ky[b], kz[c])
        n = round(Int, sqrt(sum(abs2, kv)) / Δk)
        (n == 0 || n > nmax) && continue
        shell_of[a,b,c] = n
        T = ntuple(j -> tan(kv[j] * Δ[j] / 2) / Δ[j], 3)
        r = (randn(rng, ComplexF64), randn(rng, ComplexF64), randn(rng, ComplexF64))
        α = (T[1] * r[1] + T[2] * r[2] + T[3] * r[3]) / (T[1]^2 + T[2]^2 + T[3]^2)
        for j in 1:3
            C[j][a,b,c] = r[j] - α * T[j]
        end
    end

    # 2. Hermitian symmetry (the projection commutes with it: T(-k) = -T(k))
    neg(i, N) = mod(N - (i - 1), N) + 1
    for j in 1:3
        Cj = C[j]; S = similar(Cj)
        for c in 1:Nz, b in 1:Ny, a in 1:Nx
            S[a,b,c] = 0.5 * (Cj[a,b,c] + conj(Cj[neg(a, Nx), neg(b, Ny), neg(c, Nz)]))
        end
        Cj .= S
    end

    # 3. rescale every shell to E(k_n) Δk (energy of the centre field)
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

    # 4. centre amplitudes → face amplitudes, back to physical space
    for (j, f, kj) in ((1, u, kx), (2, v, ky), (3, w, kz))
        A = C[j]
        for c in 1:Nz, b in 1:Ny, a in 1:Nx
            shell_of[a,b,c] == 0 && continue
            k = j == 1 ? kj[a] : j == 2 ? kj[b] : kj[c]
            h = k * Δ[j] / 2
            A[a,b,c] *= cis(-h) / cos(h)
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
