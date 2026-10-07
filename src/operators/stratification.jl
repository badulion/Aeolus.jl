# Pointwise stratification fields — Era VI, Phase 58 (P58-a, #170).
#
# N² and the gradient Richardson number at cell centres. The physics (the
# N² of a θ difference, the Ri_g definition with its S² floor) is the
# TurbulenceStability contract; this file only supplies the stencils.

"""
    buoyancy_frequency_squared!(N2, θ; θ₀, g)
    buoyancy_frequency_squared!(N2, θ, qv; θ₀, g)
    buoyancy_frequency_squared!(N2, θ, qv, qc; θ₀, g, p)

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

**Cloudy air** (`qc`, #239). With a condensate field `qc` (cloud or fog
water, kg kg⁻¹) the buoyancy variable is the density potential
temperature with condensate loading, ``θ_ρ = θ(1 + 0.61 q_v - q_c)``
(interpolated to the faces like θ_v), and in a **saturated** cell —
one with ``q_c > 0``, which after a saturation adjustment means
``q_v = q_s`` — ``N^2`` is the saturated buoyancy frequency,
`MoistThermodynamics.saturated_buoyancy_frequency_squared` (Durran &
Klemp 1982 in a saturated environment, with the virtual effect of the
humidity jump kept at a cloud edge), from the same face differences of
θ, ``q_v`` and ``q_w = q_v + q_c`` over Δz and the cell's own θ,
``q_v``, ``q_c`` and pressure `p`. Moist-adiabatic (moist-neutral)
saturated air then has ``N^2 ≈ 0`` (``|N^2| < 10^{-10}`` s⁻² at
Δz = 12.5 m) instead of the ``1.4 × 10^{-4}`` s⁻² of the θ_v form
(a 1 km column from 285 K). Where `qc` is zero in a cell and both neighbours the result
is the `qv` form bit for bit.

`p` (Pa) is the pressure of the saturated adjustment: a number or a
vector indexed by the storage level `k` (length `Nz + 2Hz`), like `θ₀`.

`θ₀` is a number (the constant reference of the Boussinesq core) or a
vector indexed by the storage level `k` of the cell (length `Nz + 2Hz`):
the height-dependent reference ``\\bar θ(z)`` of the sound-proof cores,
whose buoyancy is ``g(θ - \\bar θ)/\\bar θ`` (`ReferenceState.θ̄c_pad`; used
by `StabilityCorrected`, P55-e).

`N2`, `θ` (and `qv`, `qc`) live at `CellCenter`; the halos of `θ`, `qv`
and `qc` must be filled (at a wall the mirror halo halves the boundary
gradient). Interior cells only. Returns `N2`.
"""
buoyancy_frequency_squared!(N2::Field, θ::Field; θ₀::Union{Real,AbstractVector}, g::Real) =
    _buoyancy_frequency_squared!(N2, θ, nothing, θ₀, g, mesh(θ))
buoyancy_frequency_squared!(N2::Field, θ::Field, qv; θ₀::Union{Real,AbstractVector}, g::Real) =
    _buoyancy_frequency_squared!(N2, θ, qv, θ₀, g, mesh(θ))
buoyancy_frequency_squared!(N2::Field, θ::Field, qv::Field, ::Nothing;
                            θ₀::Union{Real,AbstractVector}, g::Real, p = nothing) =
    _buoyancy_frequency_squared!(N2, θ, qv, θ₀, g, mesh(θ))
function buoyancy_frequency_squared!(N2::Field, θ::Field, qv::Field, qc::Field;
                                     θ₀::Union{Real,AbstractVector}, g::Real,
                                     p::Union{Real,AbstractVector})
    p isa AbstractVector && length(p) != size(θ.data, 3) &&
        throw(DimensionMismatch("p must be indexed by the storage level: length $(size(θ.data, 3)), got $(length(p))"))
    return _buoyancy_frequency_squared!(N2, θ, _Cloudy(qv.data, qc.data, p), θ₀, g, mesh(θ))
end

# the reference θ (or pressure) of storage level k: a constant, or a profile
@inline _θref(θ₀::Real, k) = θ₀
@inline _θref(θ₀::AbstractVector, k) = @inbounds θ₀[k]

# θ_v on the face below cell (i, j, k) in direction (di, dj, dk) — the one
# definition of the moist buoyancy variable at a face (also the horizontal
# θ_v gradient of the Reynolds-stress closure, P56-d)
@inline _θv_face(θ, qv, i, j, k) = _θv_face_dir(θ, qv, i, j, k, 0, 0, 1)
@inline _θv_face_dir(θ, ::Nothing, i, j, k, di, dj, dk) = 0.5 * (θ[i-di, j-dj, k-dk] + θ[i, j, k])
@inline function _θv_face_dir(θ, qv, i, j, k, di, dj, dk)
    θface  = 0.5 * (θ[i-di, j-dj, k-dk]  + θ[i, j, k])
    qvface = 0.5 * (qv[i-di, j-dj, k-dk] + qv[i, j, k])
    return θface * (1 + 0.61 * qvface)
end

# The humidity of cloudy air (#239): vapour, condensate and the pressure of
# the saturated N² (a number or a storage-indexed vector).
struct _Cloudy{A,P}
    qv::A
    qc::A
    p::P
end

# N² of cell (i, j, k) from the faces below and above, over Δz = dz: the one
# per-cell stencil of the uniform and the stretched grids.
@inline _n2_cell(θ, qv, i, j, k, dz, g, θr) =
    buoyancy_frequency_squared(_θv_face(θ, qv, i, j, k), _θv_face(θ, qv, i, j, k + 1), dz, g, θr)
@inline function _n2_cell(θ, w::_Cloudy, i, j, k, dz, g, θr)
    qv, qc = w.qv, w.qc
    if qc[i, j, k] > 0
        # saturated: the moist-adiabatic parcel (Durran & Klemp 1982) against
        # the face differences of θ, q_v and q_w
        dθ  = (0.5 * (θ[i, j, k] + θ[i, j, k+1]) - 0.5 * (θ[i, j, k-1] + θ[i, j, k])) / dz
        dqv = (0.5 * (qv[i, j, k] + qv[i, j, k+1]) - 0.5 * (qv[i, j, k-1] + qv[i, j, k])) / dz
        dqc = (0.5 * (qc[i, j, k] + qc[i, j, k+1]) - 0.5 * (qc[i, j, k-1] + qc[i, j, k])) / dz
        return saturated_buoyancy_frequency_squared(θ[i, j, k], _θref(w.p, k), qv[i, j, k],
                                                    qc[i, j, k], dθ, dqv, dqv + dqc,
                                                    DEFAULT_CONSTANTS; θ_ref = θr, g = g)
    end
    return buoyancy_frequency_squared(_θρ_face(θ, qv, qc, i, j, k),
                                      _θρ_face(θ, qv, qc, i, j, k + 1), dz, g, θr)
end

# θ_ρ = θ (1 + 0.61 q_v − q_c) on the face below cell (i, j, k): θ_v with
# condensate loading (the linearised density potential temperature)
@inline function _θρ_face(θ, qv, qc, i, j, k)
    θface  = 0.5 * (θ[i, j, k-1]  + θ[i, j, k])
    qvface = 0.5 * (qv[i, j, k-1] + qv[i, j, k])
    qcface = 0.5 * (qc[i, j, k-1] + qc[i, j, k])
    return θface * (1 + 0.61 * qvface - qcface)
end

"""
    cloud_humidity(qc, p, mesh) -> nothing or (qc, qch, p)

The condensate a closure's ``N^2`` uses (#239): `nothing` for `qc =
nothing`, else the field `qc`, a scratch copy `qch` (its halos are filled
there, the tracer is not touched) and the pressure `p` of the saturated
``N^2`` (a number, or a vector indexed by the storage level). Shared by
[`StabilityCorrected`](@ref) and the prognostic closures.
"""
cloud_humidity(::Nothing, p, g) = nothing
function cloud_humidity(qc::Field, p, g)
    p === nothing && throw(ArgumentError("a condensate field qc needs the pressure p of the saturated N²"))
    p isa AbstractVector && length(p) != size(qc.data, 3) &&
        throw(DimensionMismatch("p must be indexed by the storage level: length $(size(qc.data, 3)), got $(length(p))"))
    return (qc = qc, qch = Field(CellCenter(), g), p = p isa Real ? Float64(p) : collect(Float64, p))
end

"Copy the condensate of [`cloud_humidity`](@ref) into its scratch and fill the halos with `bcs`."
function filled_condensate!(cl, bcs)
    copyto!(cl.qch.data, cl.qc.data)
    fill_halos!(cl.qch, bcs)
    return cl.qch
end

_humidity_data(::Nothing) = nothing
_humidity_data(q) = q.data
_humidity_data(q::_Cloudy) = q

function _buoyancy_frequency_squared!(N2::Field{CellCenter}, θ::Field{CellCenter}, qv,
                                      θ₀, g, m::RegularGrid)
    launch!(Backends.backend(N2.data), _n2_body!, interior(m), N2.data, θ.data, _humidity_data(qv),
            m.dz, g, θ₀)
    return N2
end

# KA-l (#304): the N², Ri_g and S² kernels are point bodies run by `launch!`
# over the interior (the loops' expressions, bit for bit).
@inline function _n2_body!(i, j, k, nd, θd, qd, dz, g, θ₀)
    @inbounds nd[i, j, k] = _n2_cell(θd, qd, i, j, k, dz, g, _θref(θ₀, k))
    return nothing
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
    launch!(Backends.backend(Ri.data), _ri_body!, interior(m), Ri.data, N2.data,
            u.data, v.data, w.data, idx, idy, idz, S2_min)
    return Ri
end

@inline function _ri_body!(i, j, k, rd, nd, ud, vd, wd, idx, idy, idz, S2_min)
    magS = _strain_at(ud, vd, wd, i, j, k, idx, idy, idz)[7]
    @inbounds rd[i, j, k] = gradient_richardson(nd[i, j, k], magS^2; S2_min = S2_min)
    return nothing
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
    launch!(Backends.backend(S2.data), _s2_body!, interior(m), S2.data, u.data, v.data, w.data,
            idx, idy, idz)
    return S2
end

@inline function _s2_body!(i, j, k, sd, ud, vd, wd, idx, idy, idz)
    @inbounds sd[i, j, k] = _strain_at(ud, vd, wd, i, j, k, idx, idy, idz)[7]^2
    return nothing
end
