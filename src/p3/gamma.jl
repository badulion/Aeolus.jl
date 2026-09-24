# P3 part-file: gamma-function machinery and gamma-PSD moments (issue #1).
#
# Everything downstream in P3 — bulk mass, number-weighted fall speed,
# reflectivity, the regime-by-regime truncated integrals of m(D) — is a
# (possibly truncated) moment of a gamma size distribution, so it reduces
# to Γ and to the regularised incomplete gamma functions P(a,x), Q(a,x).
# `SpecialFunctions` does not build on every target this package supports,
# so these are implemented here following Press et al. (2007),
# *Numerical Recipes*, 3rd ed., §6.1-6.2.

# -------------------------------------------------------------------------
# Log-gamma and gamma
# -------------------------------------------------------------------------

# Lanczos coefficients, N = 14, shift g + 1/2 = 671/128 (Numerical Recipes 3rd ed.,
# `gammln`). Relative error of the Lanczos sum is below 1e-15 for x > 0.
const _LANCZOS_G = 5.24218750000000000     # = 671/128, the shifted g + 1/2
const _LANCZOS_C0 = 0.999999999999997092
const _LANCZOS_COF = (
    57.1562356658629235, -59.5979603554754912,
    14.1360979747417471, -0.491913816097620199,
    0.339946499848118887e-4, 0.465236289270485756e-4,
    -0.983744753048795646e-4, 0.158088703224912494e-3,
    -0.210264441724104883e-3, 0.217439618115212643e-3,
    -0.164318106536763890e-3, 0.844182239838527433e-4,
    -0.261908384015814087e-4, 0.368991826595316234e-5)

# Lanczos approximation of log Γ(x), valid for x > 0.
function _lgamma_lanczos_pos(x::T) where {T<:AbstractFloat}
    y = x
    tmp = x + T(_LANCZOS_G)
    tmp = (x + T(0.5)) * log(tmp) - tmp
    ser = T(_LANCZOS_C0)
    for c in _LANCZOS_COF
        y += one(T)
        ser += T(c) / y
    end
    return tmp + log(T(2.5066282746310005) * ser / x)   # √(2π)
end

"""
    lgamma_lanczos(x)

Natural logarithm of the absolute value of the gamma function,
`log |Γ(x)|`, for real `x`.

For `x ≥ 1/2` the 14-term Lanczos approximation of Press et al. (2007,
*Numerical Recipes* 3rd ed., §6.1, `gammln`) is used (relative error of
the Lanczos series < 1e-15). For `x < 1/2` Euler's reflection formula

    Γ(x) Γ(1-x) = π / sin(πx)

maps the argument into the Lanczos range. Non-positive integers are
poles and return `Inf`.

Named with a `_lanczos` suffix so that it does not clash with
`SpecialFunctions.loggamma` in user code that loads both packages.
"""
function lgamma_lanczos(x::Real)
    xf = float(x)
    T = typeof(xf)
    isnan(xf) && return xf
    xf == T(Inf) && return T(Inf)
    if xf <= 0 && isinteger(xf)
        return T(Inf)
    end
    if xf < T(0.5)
        # log|Γ(x)| = log(π/|sin πx|) - log Γ(1-x)
        return log(T(π) / abs(sinpi(xf))) - _lgamma_lanczos_pos(one(T) - xf)
    end
    return _lgamma_lanczos_pos(xf)
end

"""
    gamma_lanczos(x)

The gamma function `Γ(x)` for real `x`.

  * positive integers `n ≤ 171`: `(n-1)!` by direct product (exact up to
    `n = 23` in `Float64`);
  * other `x > 0`: `exp(lgamma_lanczos(x))`;
  * negative non-integers: reflection, `Γ(x) = π / (sin(πx) Γ(1-x))`;
  * non-positive integers are poles and raise a `DomainError`.

Overflows to `Inf` for `x > 171.62` in `Float64`.

Reference: Press et al. (2007), *Numerical Recipes* 3rd ed., §6.1.
"""
function gamma_lanczos(x::Real)
    xf = float(x)
    T = typeof(xf)
    isnan(xf) && return xf
    if isinteger(xf)
        xf <= 0 && throw(DomainError(x, "Γ has a pole at non-positive integers"))
        if xf <= 171
            r = one(T)
            for k in 2:(Int(xf) - 1)
                r *= k
            end
            return r
        end
        return T(Inf)
    end
    if xf < 0
        return T(π) / (sinpi(xf) * gamma_lanczos(one(T) - xf))
    end
    return exp(_lgamma_lanczos_pos(xf))
end
