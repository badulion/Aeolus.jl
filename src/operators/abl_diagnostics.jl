# =========================================================================
# Era VI, Phase 54 (P54-d, #174) — wall-bounded flow diagnostics.
# =========================================================================
#
# Included into `module Operators`. Three plane-averaged diagnostics of a
# horizontally periodic, wall-bounded LES on the staggered C-grid, shared by
# every Era VI benchmark (the half-channel ABL and the two-wall channel of
# `WallModels`, and the issues that reuse them):
#
#   phi_m_profile          (κz/u*) ∂⟨U⟩/∂z at the interior w-faces
#   total_stress_profile   resolved + SGS + viscous + wall momentum flux
#   streamwise_spectrum    one-sided E11(k1) at chosen levels (Parseval-exact)
#
# They read fields only (no closure, no wall law), so they live with the
# operators; κ, u*, the wall stress and the eddy viscosity are arguments.

export phi_m_profile, total_stress_profile, streamwise_spectrum
# P58-e (#193): the resolved TKE profile of the collapse runs
export resolved_tke_profile

# plane means of the interior of a field, one per level k = 1..Nz
function _plane_means(f::Field)
    g = mesh(f)
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    m = zeros(eltype(f.data), Nz)
    @inbounds for k in 1:Nz
        s = zero(eltype(f.data))
        for j in 1:Ny, i in 1:Nx
            s += f.data[Hx+i, Hy+j, Hz+k]
        end
        m[k] = s / (Nx * Ny)
    end
    return m
end

"""
    phi_m_profile(u::Field{FaceX}, v::Field{FaceY}, u_star; κ) -> (z, Φ)
    phi_m_profile(U::AbstractVector, dz, u_star; κ)              -> (z, Φ)

The non-dimensional mean shear of the surface layer,

```math
Φ_m(z) = \\frac{κ z}{u_*}\\,\\frac{∂⟨U⟩}{∂z},
```

in the **staggered-consistent** form: the plane-mean wind speed
``⟨U⟩_k = |(⟨u⟩_k, ⟨v⟩_k)|`` lives at the cell centres
``z_k = (k - ½)Δz``, so its derivative is the centred difference
``(⟨U⟩_{k+1} - ⟨U⟩_k)/Δz`` at the w-face between them,
``z = kΔz`` (``k = 1, …, N_z - 1``), which is where Φ_m is returned. No
interpolation of the derivative is involved, so for the log law
``⟨U⟩ = (u_*/κ)\\ln(z/z_0)`` the result is

```math
Φ_m(z) = \\frac{z}{Δz}\\ln\\frac{z + Δz/2}{z - Δz/2} = 1 + \\frac{1}{12}\\Big(\\frac{Δz}{z}\\Big)^2 + O\\Big(\\frac{Δz}{z}\\Big)^4,
```

independent of ``z_0``: 1 to second order, ``\\ln 3 = 1.0986`` at the first
face. Heights are above the bottom of the mesh (`mesh.z0`). In the
neutral surface layer Φ_m = 1; Porté-Agel, Meneveau & Parlange (2000,
Fig. 7) and Mason & Thomson (1992) show the Smagorinsky model's overshoot
to Φ_m ≈ 1.5-2 near the wall.

`κ` (the von Kármán constant) has no default here: the Era V contract
`SurfaceExchange.von_karman()` owns it, and the case builders of
`WallModels` pass it.

# References

  * Porté-Agel, F., Meneveau, C. & Parlange, M. B. (2000). A scale-dependent
    dynamic model for large-eddy simulation: application to a neutral
    atmospheric boundary layer. *J. Fluid Mech.*, 415, 261-284.
  * Mason, P. J. & Thomson, D. J. (1992). Stochastic backscatter in
    large-eddy simulations of boundary layers. *J. Fluid Mech.*, 242, 51-78.
"""
function phi_m_profile(U::AbstractVector, dz::Real, u_star::Real; κ::Real)
    n = length(U)
    n >= 2 || throw(ArgumentError("phi_m_profile: need at least two levels"))
    u_star > 0 || throw(ArgumentError("phi_m_profile: u_star must be positive, got $u_star"))
    T = float(promote_type(eltype(U), typeof(dz)))
    z = T[k * dz for k in 1:n-1]
    Φ = T[κ * z[k] / u_star * (U[k+1] - U[k]) / dz for k in 1:n-1]
    return (z = z, Φ = Φ)
end
function phi_m_profile(u::Field{FaceX}, v::Field{FaceY}, u_star::Real; κ::Real)
    um = _plane_means(u); vm = _plane_means(v)
    return phi_m_profile(hypot.(um, vm), mesh(u).dz, u_star; κ = κ)
end

"""
    total_stress_profile(u, v, w, νt; ν = 0, τ_wall_bottom = 0,
                         τ_wall_top = 0, component = :x, irange = nothing)
        -> (z, resolved, sgs, viscous, wall, total)

The plane-mean vertical flux of horizontal momentum (kinematic,
m² s⁻²), counted **positive downward** (the wall-ward flux of a flow in
`+x`), at the `Nz + 1` w-faces ``z = (k-1)Δz``, ``k = 1, …, N_z + 1``:

```math
τ(z) = \\underbrace{-⟨u\\,w⟩}_{\\text{resolved}}
     + \\underbrace{⟨ν_t\\,∂u/∂z⟩}_{\\text{SGS}}
     + \\underbrace{ν\\,∂⟨u⟩/∂z}_{\\text{viscous}}
     + τ_{\\text{wall}} .
```

Each term is formed with **the stencil the solver integrates**, so the
discrete momentum budget of a horizontally periodic run closes exactly:
the resolved flux is the centred flux-form product of `momentum_advect!`
(``w`` averaged in x to the x-z edge, ``u`` averaged in z), and the SGS
flux is the z-flux of `add_eddy_viscosity_u!` (``ν_t`` averaged over the
four cells around the edge), i.e. the default eddy-viscosity stress of the
SGS contract (`add_sgs_momentum!`). Pass a closure's vertical viscosity
(`f.νv`) as `νt` for an anisotropic closure; a closure that overrides
`add_sgs_momentum!` with another stress needs its own term. The halos of
`u, v, w, νt` must be filled (with the velocity boundary conditions).

At a free-slip (`ImpermeableWall`) z-wall the resolved, SGS and viscous
fluxes are zero, and the modelled wall stress (from `WallModels`) is the
whole flux: `τ_wall_bottom` is added at ``k = 1`` and `-τ_wall_top` at
``k = N_z + 1`` (momentum leaves through the top wall upward). With
`component = :y` the same is returned for ``v`` (on y-z edges).

For the pressure-driven half channel of height ``H`` and forcing
``u_*^2/H`` in statistically steady state the total is
``u_*^2(1 - z/H)`` (Porté-Agel et al. 2000, Fig. 3); for the two-wall
channel it is ``u_τ^2(1 - z/δ)``.

`irange` (default: all of `1:Nx`) restricts the mean to the columns
`i ∈ irange` (the x-faces / y-z edges of those columns), for the
streamwise-developing flow of an inflow–outflow domain, which is
homogeneous only in y (N-inflow-d, #266: the total stress a few
boundary-layer depths behind a synthetic inflow). The budget statement
above is then local to the x band.

# References

  * Porté-Agel, F., Meneveau, C. & Parlange, M. B. (2000). *J. Fluid Mech.*,
    415, 261-284.
  * Bou-Zeid, E., Meneveau, C. & Parlange, M. B. (2005). A scale-dependent
    Lagrangian dynamic model for large eddy simulation of complex turbulent
    flows. *Phys. Fluids*, 17, 025105.
"""
function total_stress_profile(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                              νt::Field{CellCenter}; ν::Real = 0,
                              τ_wall_bottom::Real = 0, τ_wall_top::Real = 0,
                              component::Symbol = :x, irange = nothing)
    component in (:x, :y) ||
        throw(ArgumentError("total_stress_profile: component must be :x or :y, got :$component"))
    g = mesh(u)
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    is = irange === nothing ? (1:Nx) : UnitRange{Int}(irange)
    (first(is) >= 1 && last(is) <= Nx && !isempty(is)) ||
        throw(ArgumentError("total_stress_profile: irange must lie in 1:$Nx"))
    nc = length(is) * Ny
    T = eltype(u.data)
    idz = 1 / g.dz
    a = component === :x ? u.data : v.data
    wd = w.data; nd = νt.data
    # neighbour offset of the edge average: x for u (x-z edge), y for v (y-z edge)
    di, dj = component === :x ? (1, 0) : (0, 1)
    res = zeros(T, Nz + 1); sgs = zeros(T, Nz + 1); vis = zeros(T, Nz + 1)
    @inbounds for k in 1:Nz+1
        sr = zero(T); ss = zero(T); sv = zero(T)
        K = Hz + k
        for j in 1:Ny, i in is
            I, J = Hx + i, Hy + j
            wa = (wd[I-di, J-dj, K] + wd[I, J, K]) / 2
            ua = (a[I, J, K-1] + a[I, J, K]) / 2
            sr -= wa * ua
            ne = (nd[I-di, J-dj, K-1] + nd[I, J, K-1] + nd[I-di, J-dj, K] + nd[I, J, K]) / 4
            du = (a[I, J, K] - a[I, J, K-1]) * idz
            ss += ne * du
            sv += du
        end
        res[k] = sr / nc; sgs[k] = ss / nc; vis[k] = ν * sv / nc
    end
    # the wall faces: the solver's interior stencils see zero flux through a
    # free-slip wall (w = 0, mirrored u/v halos); the wall model is the flux
    wall = zeros(T, Nz + 1)
    wall[1] = τ_wall_bottom
    wall[Nz+1] = -τ_wall_top
    z = T[(k - 1) * g.dz for k in 1:Nz+1]
    return (z = z, resolved = res, sgs = sgs, viscous = vis, wall = wall,
            total = res .+ sgs .+ vis .+ wall)
end

"""
    streamwise_spectrum(u::Field, levels; remove_mean = true) -> (k1, E11)

The one-sided streamwise (x) energy spectrum of the field `u` (normally
`sim.u`) at the levels `levels` (indices `k` of the interior), averaged
over the `Ny` lines of each level:

```math
E_{11}(k_{1,n}) = \\frac{c_n\\,L_x}{2π\\,N_x^2}\\,\\big⟨|\\hat u_n|^2\\big⟩_y,
\\qquad k_{1,n} = \\frac{2π n}{L_x},\\; n = 0, …, N_x/2,
```

with ``\\hat u_n = \\sum_i u_i e^{-2π\\mathrm{i}\\,n i/N_x}`` (FFTW's `rfft`),
``c_n = 2`` for the modes that stand for a ± pair and ``c_n = 1`` for
``n = 0`` and the Nyquist mode of an even ``N_x``. The normalisation is
Parseval's theorem: ``\\sum_n E_{11}(k_{1,n})\\,Δk_1 = ⟨u'^2⟩`` (the
plane variance; ``Δk_1 = 2π/L_x``), exactly up to round-off. With
`remove_mean = true` the line mean is subtracted first (``E_{11}(0) = 0``);
with `false`, the sum is ``⟨u^2⟩``. Returns `k1` (length ``N_x/2 + 1``)
and `E11`, a matrix with one column per level.

Used for the inertial-range ``k_1^{-5/3}`` check of the neutral-ABL runs
(Porté-Agel et al. 2000, Fig. 9; Bou-Zeid et al. 2005, Fig. 6).
"""
function streamwise_spectrum(u::Field, levels; remove_mean::Bool = true)
    g = mesh(u)
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    T = float(eltype(u.data))
    lv = collect(levels)
    all(k -> 1 <= k <= Nz, lv) ||
        throw(ArgumentError("streamwise_spectrum: levels must lie in 1:$Nz"))
    nk = Nx ÷ 2 + 1
    E = zeros(T, nk, length(lv))
    line = zeros(T, Nx)
    plan = plan_rfft(line)
    c = fill(T(2), nk); c[1] = 1
    iseven(Nx) && (c[nk] = 1)
    norm = g.Lx / (2π * Nx^2)
    for (m, k) in enumerate(lv)
        for j in 1:Ny
            @inbounds for i in 1:Nx
                line[i] = u.data[Hx+i, Hy+j, Hz+k]
            end
            remove_mean && (line .-= sum(line) / Nx)
            û = plan * line
            @inbounds for n in 1:nk
                E[n, m] += c[n] * abs2(û[n])
            end
        end
        E[:, m] .*= norm / Ny
    end
    k1 = T[2π * (n - 1) / g.Lx for n in 1:nk]
    return (k1 = k1, E11 = E)
end

"""
    resolved_tke_profile(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}) -> e

The plane-averaged **resolved turbulent kinetic energy** per level,

```math
e_k = \\tfrac12\\left(⟨u'^2⟩_k + ⟨v'^2⟩_k + ⟨w'^2⟩_k\\right),
\\qquad ϕ' = ϕ - ⟨ϕ⟩_k ,
```

(m² s⁻²), the fluctuations taken about the plane mean of each staggered
component on its own level: `u`, `v` at the cell-centre height of level
`k`, `w` on the bottom face of cell `k` (zero on a wall face). Mean flow
and its shear do not count, so a laminar parallel flow has ``e = 0`` to
round-off. The volume mean of a uniform grid is `sum(e)/Nz`.

Used by the turbulence-collapse runs of P58-e (#193): the resolved TKE of
a flux-cooled half channel relative to the neutral run (Donda et al.
2015, Fig. 3).
"""
function resolved_tke_profile(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ})
    g = mesh(u)
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    T = float(eltype(u.data))
    e = zeros(T, Nz)
    n = Nx * Ny
    @inbounds for k in 1:Nz
        for f in (u.data, v.data, w.data)
            s = zero(T); s2 = zero(T)
            for j in 1:Ny, i in 1:Nx
                x = f[Hx+i, Hy+j, Hz+k]
                s += x
            end
            m = s / n
            for j in 1:Ny, i in 1:Nx
                s2 += (f[Hx+i, Hy+j, Hz+k] - m)^2
            end
            e[k] += s2 / (2n)
        end
    end
    return e
end
