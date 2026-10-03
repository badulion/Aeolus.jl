# P59-d (#185) — free-stream turbulence as a sum of random Fourier modes
# with a von Kármán spectrum, divergence-free on the staggered grid.
#
# Included inside `module Transition`, after flat_plate.jl.
#
# u′(x) = Σₙ Aₙ σₙ cos(kₙ·x + φₙ),  σₙ ⊥ kₙ,  is the curl of
# Ψ(x) = Σₙ Aₙ (σₙ × kₙ)/|kₙ|² sin(kₙ·x + φₙ).
# Ψ is sampled on the cell edges and u′ = ∇ₕ×Ψ is formed with the grid's
# own differences, so ∇ₕ·u′ = 0 to round-off whatever the modes (the
# discrete divergence of a discrete curl vanishes identically). A wall
# damping g(z) multiplies Ψ, not u′, for the same reason.

const _VK_A = 1.606
const _VK_B = 1.350

"""
    von_karman_spectrum(κ, L, q) -> E(κ)

The von Kármán energy spectrum in the normalisation of Jacobs & Durbin
(2001) and Brandt, Schlatter & Henningson (2004, eq. 2.6):

    E(κ) = (2/3) a (κL)^4 / (b + (κL)^2)^{17/6} · L q,   a = 1.606, b = 1.350,

with ``∫_0^∞ E\\,dκ = q`` (the turbulent kinetic energy, ``q = \\tfrac32
(Tu\\,U_∞)^2``) and the longitudinal integral scale
``L_{11} = \\tfrac{π}{2u'^2}∫E/κ\\,dκ = 0.643 L``.
"""
von_karman_spectrum(κ::Real, L::Real, q::Real) =
    (2 / 3) * _VK_A * (κ * L)^4 / (_VK_B + (κ * L)^2)^(17 / 6) * L * q

"""
    FreeStreamDisturbance(Tu, L, nmodes; seed = 1, U∞ = 1, κ_min = 0.25/L,
                          κ_max = 12/L, period_y = nothing, ndir = 10)

Homogeneous isotropic free-stream turbulence of intensity `Tu`
(``u'_{\\rm rms}/U_∞``, a fraction, e.g. `0.033`) and longitudinal
integral length `L` (``L_{11}``), represented by `nmodes` random Fourier
modes (Jacobs & Durbin 2001; Brandt, Schlatter & Henningson 2004).

* Wavenumbers lie on ``⌈nmodes/ndir⌉`` spherical shells, equispaced in
  ``\\ln κ`` between `κ_min` and `κ_max`; on each shell `ndir` directions —
  one of each antipodal pair of the 20 vertices of a dodecahedron, randomly
  rotated per shell (Brandt et al. 2004) — with uniformly random phases
  and polarisations ``σ ⊥ k``. `period_y` quantises ``k_y`` to multiples
  of ``2π/L_y`` so the field is periodic in a spanwise-periodic domain.
* Each shell carries the energy ``E(κ_s)Δκ_s`` of
  [`von_karman_spectrum`](@ref); the amplitudes are then scaled so the
  total kinetic energy is exactly ``q = \\tfrac32 (Tu\\,U_∞)^2``, and the
  spectrum length is iterated so that the integral scale of the
  *discrete* mode set, ``\\tfrac{π}{2u'^2}\\sum_s E_sΔκ_s/κ_s``, equals `L`
  (the finite ``κ`` range otherwise biases it). An `ArgumentError` is
  thrown if no spectrum length can give `L` with modes in
  ``[κ_{\\min}, κ_{\\max}]`` (e.g. an integral length close to the grid
  spacing when `κ_max` is set by the grid).
* Fields are sampled with [`disturbance_velocity!`](@ref) (divergence-free
  to round-off on the staggered grid) and fed into a spatially developing
  boundary layer through the fringe target with
  [`disturbance_fringe_update`](@ref), advected at ``U_∞`` (Taylor's frozen
  turbulence), as in Brandt et al. (2004).

References: Jacobs, R. G. & Durbin, P. A. (2001). Simulations of bypass
transition. *J. Fluid Mech.*, 428, 185-212. Brandt, L., Schlatter, P. &
Henningson, D. S. (2004). Transition in boundary layers subject to
free-stream turbulence. *J. Fluid Mech.*, 517, 167-198.
"""
struct FreeStreamDisturbance
    Tu::Float64
    L::Float64              # requested longitudinal integral length L11
    L_vk::Float64           # von Kármán spectrum length after calibration
    U∞::Float64
    k::Matrix{Float64}      # 3 × n wavevectors
    A::Vector{Float64}      # velocity amplitudes
    σ::Matrix{Float64}      # 3 × n unit polarisations (σ ⊥ k)
    φ::Vector{Float64}      # phases
    c::Matrix{Float64}      # 3 × n vector-potential coefficients A (σ × k)/|k|²
end

# The 20 vertices of a regular dodecahedron (unit sphere), one per antipodal pair.
function _dodecahedron_half()
    ϕ = (1 + sqrt(5)) / 2
    V = Vector{NTuple{3,Float64}}()
    for s1 in (1, -1), s2 in (1, -1), s3 in (1, -1)
        push!(V, (s1, s2, s3))
    end
    for s1 in (1, -1), s2 in (1, -1)
        push!(V, (0.0, s1 / ϕ, s2 * ϕ)); push!(V, (s1 / ϕ, s2 * ϕ, 0.0))
        push!(V, (s2 * ϕ, 0.0, s1 / ϕ))
    end
    H = NTuple{3,Float64}[]
    for v in V
        n = sqrt(sum(abs2, v)); u = v ./ n
        any(h -> all(isapprox.(h, .-u; atol = 1e-12)), H) || push!(H, u)
    end
    return H                                  # 10 directions
end

# A uniformly random rotation matrix (Arvo 1992 via a random unit quaternion).
function _random_rotation(rng)
    q = randn(rng, 4); q ./= sqrt(sum(abs2, q))
    a, b, c, d = q
    return [a^2+b^2-c^2-d^2  2(b*c-a*d)       2(b*d+a*c);
            2(b*c+a*d)       a^2-b^2+c^2-d^2  2(c*d-a*b);
            2(b*d-a*c)       2(c*d+a*b)       a^2-b^2-c^2+d^2]
end

_cross(a, b) = (a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1])

function FreeStreamDisturbance(Tu::Real, L::Real, nmodes::Integer; seed::Integer = 1,
                               U∞::Real = 1.0, κ_min::Real = 0.25 / L,
                               κ_max::Real = 12 / L, period_y = nothing,
                               ndir::Integer = 10)
    Tu >= 0 || throw(ArgumentError("FreeStreamDisturbance: Tu must be ≥ 0"))
    L > 0 && κ_max > κ_min > 0 || throw(ArgumentError("FreeStreamDisturbance: need L > 0, κ_max > κ_min > 0"))
    1 <= ndir <= 10 || throw(ArgumentError("FreeStreamDisturbance: 1 ≤ ndir ≤ 10"))
    rng = Random.MersenneTwister(seed)
    nsh = cld(nmodes, ndir)
    # shells equispaced in ln κ, Δκ the width of each log bin
    lk = range(log(κ_min), log(κ_max); length = nsh + 1)
    κs = [exp(0.5 * (lk[s] + lk[s+1])) for s in 1:nsh]
    Δκ = [exp(lk[s+1]) - exp(lk[s]) for s in 1:nsh]
    q = 1.5 * (Tu * U∞)^2
    # calibrate the spectrum length so the discrete integral scale is L
    L11_of(Lv) = (Es = [von_karman_spectrum(κs[s], Lv, 1.0) * Δκ[s] for s in 1:nsh];
                  (π / 2) * sum(Es ./ κs) / ((2 / 3) * sum(Es)))
    # the discrete set can only represent L11 between its all-at-κ_max and
    # all-at-κ_min limits; outside that range the κ window is too narrow
    lo, hi = L11_of(1e-6 / κ_max), L11_of(1e6 / κ_min)
    lo < L < hi ||
        throw(ArgumentError("FreeStreamDisturbance: L11 = $L is not representable by modes in " *
                            "κ ∈ [$κ_min, $κ_max] (attainable ≈ $(round(lo; sigdigits = 3)) to " *
                            "$(round(hi; sigdigits = 3))); widen the κ range or change L"))
    L_vk = L / 0.643
    for _ in 1:200
        L11 = L11_of(L_vk)
        abs(L11 / L - 1) < 1e-12 && break
        L_vk *= L / L11
    end
    abs(L11_of(L_vk) / L - 1) < 1e-8 ||
        error("FreeStreamDisturbance: integral-scale calibration did not converge")
    Es = [von_karman_spectrum(κs[s], L_vk, 1.0) * Δκ[s] for s in 1:nsh]
    Es .*= 1 / sum(Es)                         # shell fractions of q
    dirs = _dodecahedron_half()
    n = nsh * ndir
    k = zeros(3, n); σ = zeros(3, n); A = zeros(n); φ = zeros(n); c = zeros(3, n)
    m = 0
    for s in 1:nsh
        R = _random_rotation(rng)
        for d in 1:ndir
            m += 1
            kv = κs[s] .* (R * collect(dirs[d]))
            if period_y !== nothing
                κy = 2π / period_y
                kv[2] = κy * round(kv[2] / κy)
            end
            # random polarisation ⊥ k
            r = randn(rng, 3)
            kk = sum(abs2, kv)
            p = r .- (sum(r .* kv) / kk) .* kv
            p ./= sqrt(sum(abs2, p))
            k[:, m] .= kv; σ[:, m] .= p
            A[m] = sqrt(4 * q * Es[s] / ndir)   # ⟨½|A σ cos|²⟩ = A²/4 per mode
            φ[m] = 2π * rand(rng)
            c[:, m] .= A[m] .* collect(_cross(p, kv)) ./ kk
        end
    end
    return FreeStreamDisturbance(Float64(Tu), Float64(L), L_vk, Float64(U∞), k, A, σ, φ, c)
end

"""
    disturbance_velocity!(u, v, w, fst, t = 0; x_shift = fst.U∞ * t, i_range = :all,
                          damping = nothing, add = false) -> (u, v, w)

Sample the disturbance velocity of `fst` at time `t` (the frozen pattern
advected by ``x → x - x_{\\rm shift}``) on the staggered fields `u`, `v`,
`w`, over the storage `x` indices `i_range` (`:all` for every storage
point; all `y`, `z` storage points). The vector potential ``Ψ`` is
evaluated on the cell edges and ``u' = ∇_h × (gΨ)`` is formed with the
grid differences, so the discrete divergence of the result is zero to
round-off. `damping = g` (a function of the height above the bottom face
``z - z_0``) multiplies ``Ψ``, e.g. to keep the disturbances out of the
boundary layer. `add = true` adds to the fields instead of overwriting
the sampled points.
"""
function disturbance_velocity!(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                               fst::FreeStreamDisturbance, t::Real = 0.0;
                               x_shift::Real = fst.U∞ * t, i_range = :all,
                               damping = nothing, add::Bool = false)
    g = mesh(u)
    nx, ny, nz = size(u.data)
    irange = i_range === :all ? (1:nx) : i_range
    i1, i2 = first(irange), last(irange)
    # Ψ on edges over i1:i2+1, 1:ny+1, 1:nz+1 (one beyond for the differences)
    ni, nj, nk = i2 - i1 + 2, ny + 1, nz + 1
    Ψx = zeros(ni, nj, nk); Ψy = zeros(ni, nj, nk); Ψz = zeros(ni, nj, nk)
    xf(i) = g.x0 + (i - g.Hx - 1) * g.dx - x_shift
    yf(j) = g.y0 + (j - g.Hy - 1) * g.dy
    zf(k) = g.z0 + (k - g.Hz - 1) * g.dz
    ex_f = Vector{ComplexF64}(undef, ni); ex_c = similar(ex_f)
    ey_f = Vector{ComplexF64}(undef, nj); ey_c = similar(ey_f)
    ez_f = Vector{ComplexF64}(undef, nk); ez_c = similar(ez_f)
    @inbounds for m in axes(fst.k, 2)
        kx, ky, kz = fst.k[1, m], fst.k[2, m], fst.k[3, m]
        for a in 1:ni
            i = i1 + a - 1
            ex_f[a] = cis(kx * xf(i)); ex_c[a] = cis(kx * (xf(i) + 0.5g.dx))
        end
        for j in 1:nj
            ey_f[j] = cis(ky * yf(j)); ey_c[j] = cis(ky * (yf(j) + 0.5g.dy))
        end
        for k in 1:nk
            ez_f[k] = cis(kz * zf(k)); ez_c[k] = cis(kz * (zf(k) + 0.5g.dz))
        end
        eφ = cis(fst.φ[m])
        cx, cy, cz = fst.c[1, m], fst.c[2, m], fst.c[3, m]
        for k in 1:nk, j in 1:nj
            px = eφ * ey_f[j] * ez_f[k]          # Ψx at (x_c, y_f, z_f)
            py = eφ * ey_c[j] * ez_f[k]          # Ψy at (x_f, y_c, z_f)
            pz = eφ * ey_f[j] * ez_c[k]          # Ψz at (x_f, y_f, z_c)
            for a in 1:ni
                Ψx[a, j, k] += cx * imag(px * ex_c[a])
                Ψy[a, j, k] += cy * imag(py * ex_f[a])
                Ψz[a, j, k] += cz * imag(pz * ex_f[a])
            end
        end
    end
    if damping !== nothing
        @inbounds for k in 1:nk
            gf = damping(zf(k) - g.z0); gc = damping(zf(k) + 0.5g.dz - g.z0)
            @views Ψx[:, :, k] .*= gf; @views Ψy[:, :, k] .*= gf; @views Ψz[:, :, k] .*= gc
        end
    end
    idx, idy, idz = 1 / g.dx, 1 / g.dy, 1 / g.dz
    @inbounds for k in 1:nz, j in 1:ny, a in 1:(ni-1)
        i = i1 + a - 1
        uu = (Ψz[a, j+1, k] - Ψz[a, j, k]) * idy - (Ψy[a, j, k+1] - Ψy[a, j, k]) * idz
        vv = (Ψx[a, j, k+1] - Ψx[a, j, k]) * idz - (Ψz[a+1, j, k] - Ψz[a, j, k]) * idx
        ww = (Ψy[a+1, j, k] - Ψy[a, j, k]) * idx - (Ψx[a, j+1, k] - Ψx[a, j, k]) * idy
        if add
            u.data[i, j, k] += uu; v.data[i, j, k] += vv; w.data[i, j, k] += ww
        else
            u.data[i, j, k] = uu; v.data[i, j, k] = vv; w.data[i, j, k] = ww
        end
    end
    return u, v, w
end

"""
    disturbance_fringe_update(fst, fringe, base_target; damping = nothing,
                              interval = 0) -> update!(target, t)

The `update!` callback of [`fringe_forcing`](@ref) that makes the fringe
target ``\\mathbf u_{\\rm base} + \\mathbf u'(x - U_∞ t, y, z)``: the
free-stream disturbance advected through the fringe, superposed on the
(Blasius) base target. Only the storage columns where ``λ(x) > 0`` are
recomputed. `damping` as in [`disturbance_velocity!`](@ref) (e.g.
`z -> smooth_step(z / (3δ_{99}))` keeps the forcing out of the inflow
boundary layer). The target is refreshed when `t` differs by at least
`interval` from the time of the last refresh (`0`, the default: every
call). Note that `step_rk3!` calls `body_force!` with `sim.t` = ``t``,
``t + Δt``, ``t + 2Δt`` on its three sub-stages (it advances the clock
by one Euler step per sub-stage), so the disturbance pattern of the last
sub-stage is ``1.5Δt\\,U_∞`` ahead of the SSP-RK3 stage time
``t + Δt/2`` — a phase error of the frozen pattern, small against its
integral length when ``U_∞Δt ≪ L``.
"""
function disturbance_fringe_update(fst::FreeStreamDisturbance, fr::FringeRegion,
                                   base; damping = nothing, interval::Real = 0)
    ub, vb, wb = _target3(base)
    g = mesh(ub)
    xs = [g.x0 + (i - g.Hx - 1) * g.dx for i in 1:size(ub.data, 1)]
    xw(x) = g.x0 + mod(x - g.x0, g.Lx)
    cols = findall(i -> fringe_rate(fr, xw(xs[i])) > 0 ||
                        fringe_rate(fr, xw(xs[i] + 0.5g.dx)) > 0, eachindex(xs))
    isempty(cols) && throw(ArgumentError("disturbance_fringe_update: empty fringe"))
    irange = first(cols):last(cols)
    t_last = Ref(-Inf)
    return function (target, t)
        t - t_last[] < interval && return target
        t_last[] = t
        ut, vt, wt = _target3(target)
        for i in irange
            @views ut.data[i, :, :] .= ub.data[i, :, :]
            @views vt.data[i, :, :] .= vb.data[i, :, :]
            @views wt.data[i, :, :] .= wb.data[i, :, :]
        end
        disturbance_velocity!(ut, vt, wt, fst, t; i_range = irange, damping = damping,
                              add = true)
        return target
    end
end
