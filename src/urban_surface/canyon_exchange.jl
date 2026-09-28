# Phase 49 (#30) — turbulent exchange inside a street canyon.
#
# Included into `module UrbanSurface`. `UrbanFacet` closes a roof with
# Monin–Obukhov similarity and a wall with a fixed convective coefficient
# `h_convective`; neither knows that the air next to a wall or a street is
# sheltered by the canyon. These two closures supply what the Town Energy
# Balance scheme (TEB; Masson 2000) uses for that: the wind speed inside the
# canyon, and the convective coefficient of a canyon facet in that wind. They
# are the closures the BUBBLE / CAPITOUL validation case (#30) drives the
# walls and the street with, so that the comparison with TEB's published
# skill is like for like.

"""
    canyon_wind_speed(U, z_ref, H, W; z0_town = H / 10) -> m s⁻¹

Mean wind speed inside a street canyon of height `H` and width `W` (m),
given the wind speed `U` measured `z_ref` metres **above roof level** —
the Town Energy Balance closure (Masson 2000, eq. 13; Lemonsu et al.
2004, appendix B, eq. B1):

```math
U_{can} = \\frac{2}{\\pi}\\,\\exp\\!\\left(-\\frac{H}{4W}\\right)
          \\frac{\\ln\\big((H/3)/z_{0,town}\\big)}
                {\\ln\\big((z_{ref} + H/3)/z_{0,town}\\big)}\\;U .
```

The logarithmic ratio carries the wind down from the forcing level to the
top of the canyon (a log profile with displacement ``2H/3`` and roughness
``z_{0,town}``); the exponential is the decay into the canyon (Rotach
1995); and ``2/π`` is the average over all canyon orientations of the
along-canyon component ``|\\cos(\\text{wind} - \\text{street})|`` — TEB
takes the flow across the canyon to be a recirculating skimming flow
that contributes nothing to the mean along-canyon speed.

The default ``z_{0,town} = H/10`` is the rule of thumb for the roughness
of a built-up area (Grimmond & Oke 1999, `z0 ≈ 0.1 z_H`); set it per site.

**Domain.** Lemonsu et al. (2004) keep this expression for skimming flow,
``H/W ≥ 1``, and replace it by wake-interference and isolated-roughness
forms for shallower canyons. Those extensions are not implemented, so
`H / W < 1` throws.

# References
  * Masson, V. (2000). A physically-based scheme for the urban energy
    budget in atmospheric models. *Boundary-Layer Meteorol.*, 94, 357-397.
  * Lemonsu, A., Grimmond, C. S. B. & Masson, V. (2004). Modeling the
    surface energy balance of the core of an old Mediterranean city:
    Marseille. *J. Appl. Meteorol.*, 43, 312-327, appendix B.
  * Rotach, M. W. (1995). Profiles of turbulence statistics in and above
    an urban street canyon. *Atmos. Environ.*, 29, 1473-1486.
  * Grimmond, C. S. B. & Oke, T. R. (1999). Aerodynamic properties of
    urban areas derived from analysis of surface form. *J. Appl.
    Meteorol.*, 38, 1262-1292.
"""
function canyon_wind_speed(U::Real, z_ref::Real, H::Real, W::Real; z0_town::Real = H / 10)
    (H > 0 && W > 0) || throw(ArgumentError("canyon_wind_speed: H and W must be positive"))
    H / W >= 1 ||
        throw(ArgumentError("canyon_wind_speed: H/W = $(H / W) < 1; the TEB skimming-flow " *
                            "closure is only used for H/W ≥ 1 (Lemonsu et al. 2004, app. B)"))
    z_ref > 0 || throw(ArgumentError("canyon_wind_speed: z_ref (above roof level) must be positive"))
    (0 < z0_town < H / 3) ||
        throw(ArgumentError("canyon_wind_speed: need 0 < z0_town < H/3"))
    U >= 0 || throw(ArgumentError("canyon_wind_speed: U must be non-negative"))
    return 2 / π * exp(-H / (4W)) * log(H / 3 / z0_town) / log((z_ref + H / 3) / z0_town) * U
end

"""
    rowley_convection_coefficient(U) -> W m⁻² K⁻¹

Convective heat-transfer coefficient of a building wall (or canyon floor)
in a wind of speed `U` (m s⁻¹) along it,

```math
h = 11.8 + 4.2\\,U ,
```

the wind-tunnel correlation of Rowley, Algren & Blackshaw (1930) that the
Town Energy Balance scheme uses for its wall resistance
``R_{wall} = ρc_p / h`` with ``U = U_{eff}`` the canyon wind (Masson 2000;
Lemonsu et al. 2004, eq. 2). Use it with [`canyon_wind_speed`](@ref) as an
[`UrbanFacet`](@ref)'s `h_convective`. TEB adds the turbulent velocity
scales ``u_* + w_*`` to `U` in quadrature; pass that combination if you
have it.

# References
  * Rowley, F. B., Algren, A. B. & Blackshaw, J. L. (1930). Surface
    conductances as affected by air velocity, temperature and character
    of surface. *ASHVE Trans.*, 36, 429-446.
  * Masson, V. (2000), *Boundary-Layer Meteorol.*, 94, 357-397.
"""
function rowley_convection_coefficient(U::Real)
    U >= 0 || throw(ArgumentError("rowley_convection_coefficient: U must be non-negative"))
    return 11.8 + 4.2 * U
end
