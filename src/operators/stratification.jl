# Pointwise stratification fields — Era VI, Phase 58 (P58-a, #170).
#
# N² and the gradient Richardson number at cell centres. The physics (the
# N² of a θ difference, the Ri_g definition with its S² floor) is the
# TurbulenceStability contract; this file only supplies the stencils.

"""
    buoyancy_frequency_squared!(N2, θ; θ₀, g)
    buoyancy_frequency_squared!(N2, θ, qv; θ₀, g)

Squared buoyancy frequency at cell centres,

```math
N^2_k = \\frac{g}{θ_0}\\,\\frac{θ_{v,k+1/2} - θ_{v,k-1/2}}{Δz},
```

the vertical difference of the buoyancy that [`add_buoyancy!`](@ref)
applies at the two w faces of the cell: θ (and `qv`) are interpolated to
the faces exactly as there, and with `qv` the virtual potential
temperature ``θ_v = θ(1 + 0.61 q_v)`` of the moist buoyancy is used
(`qv = nothing` is the dry form). On a uniform grid this is the centred
difference ``(g/θ_0)(θ_{k+1} - θ_{k-1})/(2Δz)``. The difference itself is
`TurbulenceStability.buoyancy_frequency_squared`.

`N2`, `θ` (and `qv`) live at `CellCenter`; the halos of `θ` and `qv`
must be filled (at a wall the mirror halo halves the boundary gradient).
Interior cells only. Returns `N2`.
"""
buoyancy_frequency_squared!(N2::Field, θ::Field; θ₀::Real, g::Real) =
    _buoyancy_frequency_squared!(N2, θ, nothing, θ₀, g, mesh(θ))
buoyancy_frequency_squared!(N2::Field, θ::Field, qv; θ₀::Real, g::Real) =
    _buoyancy_frequency_squared!(N2, θ, qv, θ₀, g, mesh(θ))

@inline _θv_face(θ, ::Nothing, i, j, k) = 0.5 * (θ[i, j, k-1] + θ[i, j, k])
@inline function _θv_face(θ, qv, i, j, k)
    θface  = 0.5 * (θ[i, j, k-1]  + θ[i, j, k])
    qvface = 0.5 * (qv[i, j, k-1] + qv[i, j, k])
    return θface * (1 + 0.61 * qvface)
end

function _buoyancy_frequency_squared!(N2::Field{CellCenter}, θ::Field{CellCenter}, qv,
                                      θ₀, g, m::RegularGrid)
    qd = qv === nothing ? nothing : qv.data
    θd, nd = θ.data, N2.data
    dz = m.dz
    ii, jj, kk = interior(m)
    @inbounds for k in kk, j in jj, i in ii
        lo = _θv_face(θd, qd, i, j, k)
        hi = _θv_face(θd, qd, i, j, k + 1)
        nd[i, j, k] = buoyancy_frequency_squared(lo, hi, dz, g, θ₀)
    end
    return N2
end

"""
    gradient_richardson!(Ri, N2, u, v, w; S2_min = RI_SHEAR2_FLOOR)

Gradient Richardson number at cell centres,
``Ri = N^2/\\max(|S|^2, S^2_{\\min})`` (`TurbulenceStability.gradient_richardson`),
with ``|S|^2 = 2 S_{ij} S_{ij}`` from the strain-rate stencil of
[`compute_smagorinsky_νt!`](@ref) — the shear the Smagorinsky closure
sees, so ``K = (C_sΔ)^2 |S| f(Ri)`` is consistent. For a pure vertical
shear ``u(z)`` it is ``(∂u/∂z)^2`` (centred over ``2Δz``).

`N2` is a `CellCenter` field from [`buoyancy_frequency_squared!`](@ref);
`u, v, w` live on the faces and their halos must be filled. Interior
cells only. Returns `Ri`.
"""
gradient_richardson!(Ri::Field, N2::Field, u::Field, v::Field, w::Field;
                     S2_min::Real = RI_SHEAR2_FLOOR) =
    _gradient_richardson!(Ri, N2, u, v, w, S2_min, mesh(u))

function _gradient_richardson!(Ri::Field{CellCenter}, N2::Field{CellCenter},
                               u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                               S2_min, m::RegularGrid)
    idx = 1 / m.dx; idy = 1 / m.dy; idz = 1 / m.dz
    ud, vd, wd, nd, rd = u.data, v.data, w.data, N2.data, Ri.data
    ii, jj, kk = interior(m)
    @inbounds for k in kk, j in jj, i in ii
        magS = _strain_at(ud, vd, wd, i, j, k, idx, idy, idz)[7]
        rd[i, j, k] = gradient_richardson(nd[i, j, k], magS^2; S2_min = S2_min)
    end
    return Ri
end

"""
    strain_rate_squared!(S2, u, v, w)

``|S|^2 = 2 S_{ij} S_{ij}`` at cell centres, the strain-rate stencil of
[`compute_smagorinsky_νt!`](@ref) and the ``S^2`` of
[`gradient_richardson!`](@ref). Halos of `u, v, w` must be filled.
Interior cells only. Returns `S2`.
"""
strain_rate_squared!(S2::Field, u::Field, v::Field, w::Field) =
    _strain_rate_squared!(S2, u, v, w, mesh(u))

function _strain_rate_squared!(S2::Field{CellCenter}, u::Field{FaceX}, v::Field{FaceY},
                               w::Field{FaceZ}, m::RegularGrid)
    idx = 1 / m.dx; idy = 1 / m.dy; idz = 1 / m.dz
    ud, vd, wd, sd = u.data, v.data, w.data, S2.data
    ii, jj, kk = interior(m)
    @inbounds for k in kk, j in jj, i in ii
        sd[i, j, k] = _strain_at(ud, vd, wd, i, j, k, idx, idy, idz)[7]^2
    end
    return S2
end
