# P59-d (#185) — the Blasius flat-plate boundary layer, in-package.
#
# Included inside `module Transition`.
#
#   f‴ + ½ f f″ = 0,   f(0) = f′(0) = 0,   f′(∞) = 1,
#   η = z √(U∞/(ν x)),  u = U∞ f′(η),  w = ½ √(ν U∞/x) (η f′ − f).
#
# Solved by classical RK4 shooting on f″(0) with a secant iteration; the
# profile at arbitrary η is the cubic Hermite interpolant of the RK4 table
# (f, f′, f″ are tabulated together with their η-derivatives f′, f″, f‴,
# so every interpolant is C¹ and fourth-order accurate).

"""
    BlasiusSolution

Tabulated Blasius solution returned by [`blasius_solution`](@ref).

Fields: `η, f, fp, fpp` (the RK4 table, `fp = f′`, `fpp = f″`), `fpp0`
(the wall curvature ``f''(0)``), `delta_star` (``∫_0^∞ (1-f′)dη =
\\lim(η - f)``), `theta` (``∫_0^∞ f′(1-f′)dη``), `eta99` (``f′(η_{99}) =
0.99``), `H` (the shape factor `delta_star/theta`), `iterations` (secant
iterations used) and `residual` (``f′(η_{\\max}) - 1``).
"""
struct BlasiusSolution
    η::Vector{Float64}
    f::Vector{Float64}
    fp::Vector{Float64}
    fpp::Vector{Float64}
    fpp0::Float64
    delta_star::Float64
    theta::Float64
    eta99::Float64
    H::Float64
    iterations::Int
    residual::Float64
end

@inline _blasius_rhs(f, fp, fpp) = (fp, fpp, -0.5 * f * fpp)

# Classical RK4 from η = 0 with f″(0) = s; returns the table when `table`.
function _blasius_rk4(s::Float64, η_max::Float64, n::Int, table::Bool)
    h = η_max / n
    f, fp, fpp = 0.0, 0.0, s
    if table
        F = Vector{Float64}(undef, n + 1); FP = similar(F); FPP = similar(F)
        F[1] = f; FP[1] = fp; FPP[1] = fpp
    end
    for i in 1:n
        k1 = _blasius_rhs(f, fp, fpp)
        k2 = _blasius_rhs(f + 0.5h * k1[1], fp + 0.5h * k1[2], fpp + 0.5h * k1[3])
        k3 = _blasius_rhs(f + 0.5h * k2[1], fp + 0.5h * k2[2], fpp + 0.5h * k2[3])
        k4 = _blasius_rhs(f + h * k3[1], fp + h * k3[2], fpp + h * k3[3])
        f   += h / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
        fp  += h / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
        fpp += h / 6 * (k1[3] + 2k2[3] + 2k3[3] + k4[3])
        if table
            F[i+1] = f; FP[i+1] = fp; FPP[i+1] = fpp
        end
    end
    return table ? (F, FP, FPP) : (f, fp, fpp)
end

"""
    blasius_solution(; η_max = 12, n = 12_000, tol = 1e-14, maxiter = 60) -> BlasiusSolution

Solve the Blasius equation ``f''' + \\tfrac12 f f'' = 0``, ``f(0) = f'(0)
= 0``, ``f'(∞) = 1`` by RK4 shooting on ``s = f''(0)`` (``n`` steps over
``[0, η_{\\max}]``) with a secant iteration on ``f'(η_{\\max}; s) - 1``
until the change in ``s`` is below `tol`. The far-field condition is
imposed at ``η_{\\max} = 12``, where ``1 - f'`` of the true solution is
below ``10^{-15}``.

Measured with the defaults (5 secant iterations, ``f'(η_{\\max}) - 1 =
-3\\times10^{-16}``): ``f''(0) = 0.3320573362``, ``δ^*`` constant
1.7207876575, ``θ`` constant 0.6641146724 (``= 2f''(0)`` to 1e-15, the
momentum-integral identity), ``η_{99} = 4.90999``, ``H = 2.59110``.
Halving the step or moving ``η_{\\max}`` to 15 changes ``f''(0)`` by
< 1e-13 and ``δ^*`` by 3e-12.

References: Blasius, H. (1908). Grenzschichten in Flüssigkeiten mit
kleiner Reibung. *Z. Math. Phys.*, 56, 1-37. Howarth, L. (1938). On the
solution of the laminar boundary layer equations. *Proc. R. Soc. Lond.
A*, 164, 547-579. Schlichting, H. & Gersten, K. (2017). *Boundary-Layer
Theory*, 9th ed., Springer, §6.5.
"""
function blasius_solution(; η_max::Real = 12.0, n::Integer = 12_000,
                          tol::Real = 1e-14, maxiter::Integer = 60)
    ηm = Float64(η_max)
    resid(s) = _blasius_rk4(s, ηm, n, false)[2] - 1
    s0, s1 = 0.30, 0.36
    r0, r1 = resid(s0), resid(s1)
    it = 0
    while abs(s1 - s0) > tol && it < maxiter
        r1 == r0 && break
        s2 = s1 - r1 * (s1 - s0) / (r1 - r0)
        s0, r0 = s1, r1
        s1, r1 = s2, resid(s2)
        it += 1
    end
    it < maxiter || error("blasius_solution: secant iteration did not converge")
    F, FP, FPP = _blasius_rk4(s1, ηm, n, true)
    η = collect(range(0.0, ηm; length = n + 1))
    δs = ηm - F[end]
    # θ = ∫ f′(1 − f′) dη by composite Simpson on the table
    g = FP .* (1 .- FP)
    h = ηm / n
    θ = if iseven(n)
        h / 3 * (g[1] + g[end] + 4sum(@view g[2:2:end-1]) + 2sum(@view g[3:2:end-2]))
    else
        h * (sum(g) - 0.5 * (g[1] + g[end]))
    end
    sol0 = BlasiusSolution(η, F, FP, FPP, s1, δs, θ, NaN, δs / θ, it, r1)
    η99 = _blasius_eta_where(sol0, 0.99)
    return BlasiusSolution(η, F, FP, FPP, s1, δs, θ, η99, δs / θ, it, r1)
end

const _BLASIUS = Ref{Union{Nothing, BlasiusSolution}}(nothing)

"The default [`BlasiusSolution`](@ref), computed once on first use."
function default_blasius()
    s = _BLASIUS[]
    s === nothing || return s
    s = blasius_solution()
    _BLASIUS[] = s
    return s
end

@inline function _hermite(y0, y1, d0, d1, h, t)
    t2 = t * t; t3 = t2 * t
    return (2t3 - 3t2 + 1) * y0 + (t3 - 2t2 + t) * h * d0 +
           (-2t3 + 3t2) * y1 + (t3 - t2) * h * d1
end

"""
    blasius_profile(η; solution = default_blasius()) -> (f, f′, f″)

The Blasius similarity functions at ``η ≥ 0``: cubic Hermite interpolation
of the RK4 table of [`blasius_solution`](@ref) (fourth-order accurate and
C¹). Beyond the table, ``f = η - δ^*_c``, ``f' = 1``, ``f'' = 0``.
Negative ``η`` is not allowed.
"""
function blasius_profile(η::Real; solution::BlasiusSolution = default_blasius())
    η >= 0 || throw(DomainError(η, "blasius_profile: η must be ≥ 0"))
    s = solution
    ηt = s.η
    η >= ηt[end] && return (Float64(η) - s.delta_star, 1.0, 0.0)
    h = ηt[2] - ηt[1]
    i = min(floor(Int, η / h) + 1, length(ηt) - 1)
    t = (η - ηt[i]) / h
    f   = _hermite(s.f[i],   s.f[i+1],   s.fp[i],  s.fp[i+1],  h, t)
    fp  = _hermite(s.fp[i],  s.fp[i+1],  s.fpp[i], s.fpp[i+1], h, t)
    fppp0 = -0.5 * s.f[i] * s.fpp[i]; fppp1 = -0.5 * s.f[i+1] * s.fpp[i+1]
    fpp = _hermite(s.fpp[i], s.fpp[i+1], fppp0, fppp1, h, t)
    return (f, fp, fpp)
end

# η where f′ = target (0 < target < 1), by bisection on the interpolant.
function _blasius_eta_where(s::BlasiusSolution, target::Real)
    i = findfirst(>=(target), s.fp)
    i === nothing && error("f′ never reaches $target on the table")
    a, b = s.η[i-1], s.η[i]
    for _ in 1:100
        m = 0.5 * (a + b)
        blasius_profile(m; solution = s)[2] < target ? (a = m) : (b = m)
        b - a < 1e-15 && break
    end
    return 0.5 * (a + b)
end

"""
    blasius_velocity(x, z; U∞ = 1, ν, solution = default_blasius()) -> (u, w)

Blasius velocity at distance `x > 0` from the leading edge and height
`z ≥ 0` above the plate: ``u = U_∞ f'(η)``,
``w = \\tfrac12 \\sqrt{ν U_∞/x}\\,(η f' - f)`` with
``η = z\\sqrt{U_∞/(ν x)}``.
"""
function blasius_velocity(x::Real, z::Real; U∞::Real = 1.0, ν::Real,
                          solution::BlasiusSolution = default_blasius())
    x > 0 || throw(DomainError(x, "blasius_velocity: x must be > 0"))
    η = max(z, zero(z)) * sqrt(U∞ / (ν * x))
    f, fp, _ = blasius_profile(η; solution = solution)
    return (U∞ * fp, 0.5 * sqrt(ν * U∞ / x) * (η * fp - f))
end

"""
    blasius_skin_friction(Re_x) -> c_f

Laminar flat-plate skin friction ``c_f = 2 f''(0)/\\sqrt{Re_x} =
0.664/\\sqrt{Re_x}`` (with the computed ``f''(0)``).
"""
blasius_skin_friction(Re_x::Real) = 2 * default_blasius().fpp0 / sqrt(Re_x)

"""
    turbulent_skin_friction(Re_x) -> c_f

Turbulent flat-plate skin friction ``c_f = 0.0576\\,Re_x^{-1/5}``
(the one-seventh-power-law correlation for the smooth turbulent plate,
Schlichting, H. (1979), *Boundary-Layer Theory*, 7th ed., McGraw-Hill,
ch. XXI; the downstream reference of the ERCOFTAC T3A case).
"""
turbulent_skin_friction(Re_x::Real) = 0.0576 * Re_x^(-0.2)
