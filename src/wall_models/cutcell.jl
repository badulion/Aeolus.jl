# The wall model on cut-cell walls: the modelled stress at the image
# point of a ghost-cell stencil, and the slip velocity that makes the
# ghost-cell reconstruction carry it. `CutCell.wall_shear_stress` is
# extended by method (no new name).

export wall_slip_velocity, slip_wall_value

"""
    wall_shear_stress(u, v, w, stencils, geom, law::AbstractWallLaw; ν) -> Vector

The **modelled** wall shear stress (kinematic, m² s⁻²) at every
`CellCenter` ghost-cell stencil of a cut-cell geometry: the method of
`CutCell.wall_shear_stress` for a wall law. As the ν-linear method, the
cell-centred velocity is interpolated to the stencil's image point and
its wall-normal component removed; the tangential speed ``u_t`` there,
at the image-point probe distance ``h`` = `s.probe` from the true
surface, gives

```math
τ_w = u_τ^2, \\qquad u_τ = \\texttt{friction\\_velocity}(law, u_t, h; ν).
```

`law = nothing` is the ν-linear stress ``ν u_t/h``
(`wall_shear_stress(u, v, w, stencils, geom, ν)`). With
[`SpaldingLaw`](@ref) the two agree for ``h^+ < 1``; in the log layer
the linear one underpredicts by ``1 - u^+/y^+`` (at ``y^+ = 50``,
about 70 %). The vector is indexed like `stencils` (0 at an invalid
stencil); with `geom.area` it integrates to a drag, with `s.n` it has a
direction (along ``-u_t``).

**Probe distance.** Build the stencils with `probe_factor ≥ 2`
(`build_ghost_stencils(geom, mesh; probe_factor = 2)`): the image value
is a trilinear interpolation, which at one cell from the wall
underestimates a logarithmic profile (on a 30° channel at
``Re_τ = 1000``, ``Δ = δ/15``: ``⟨τ_w⟩/(δ\\,dp/dx)`` = 0.931 at
`probe_factor = 1`, 0.990 at 2; `validation/era6_wall_models_immersed.jl`).

This is the immersed-boundary wall model of Tessicini et al. (2002):
the law is applied at a probe point off the true surface along its
normal, not at a Cartesian face.

# References

  * Tessicini, F., Iaccarino, G., Fatica, M., Wang, M. & Verzicco, R.
    (2002). Wall modeling for large-eddy simulation using an immersed
    boundary method. *Annual Research Briefs 2002*, CTR Stanford,
    181-187.
  * Chester, S., Meneveau, C. & Parlange, M. B. (2007). *J. Comput.
    Phys.*, 225, 427-448.
"""
function wall_shear_stress(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                           stencils::AbstractVector{<:GhostCellStencil{CellCenter}},
                           geom::CutCellGeometry{T}, law::AbstractWallLaw;
                           ν = nothing) where {T}
    τ = Vector{T}(undef, length(stencils))
    _each_tangential(u, v, w, stencils, geom) do m, s, ut, t
        τ[m] = s.valid ? T(wall_stress(law, ut, s.probe; ν = ν)) : zero(T)
    end
    return τ
end

# Calls f(m, s, u_t, t⃗) for every stencil, with the tangential speed and
# velocity at its image point (zeros for an invalid stencil).
function _each_tangential(f, u, v, w, stencils, geom::CutCellGeometry{T}) where {T}
    uc, vc, wc = _center_velocity(u, v, w, geom, geom.mesh)
    @inbounds for (m, s) in enumerate(stencils)
        if !s.valid
            f(m, s, zero(T), (zero(T), zero(T), zero(T)))
            continue
        end
        ui = _image_value(uc, s); vi = _image_value(vc, s); wi = _image_value(wc, s)
        un = ui * s.n[1] + vi * s.n[2] + wi * s.n[3]
        t = (ui - un * s.n[1], vi - un * s.n[2], wi - un * s.n[3])
        f(m, s, sqrt(t[1]^2 + t[2]^2 + t[3]^2), t)
    end
    return nothing
end

"""
    wall_slip_velocity(u, v, w, stencils, geom, law; ν, ν_eff = ν) -> Vector{NTuple{3}}

The slip velocity on the true surface that makes the ghost-cell
reconstruction carry the modelled stress, per `CellCenter` stencil. The
reconstruction ([`CutCell.reconstruct_wall_velocity!`](@ref Aeolus.CutCell.reconstruct_wall_velocity!)) makes the
velocity linear along the normal between the wall value ``u_s`` and the
image value ``u_t``, so the resolved stress there is
``ν_{eff}(u_t - u_s)/h``. Setting it equal to the modelled ``τ_w``
([`wall_shear_stress`](@ref) with `law`) gives

```math
u_s = u_t - \\frac{τ_w\\,h}{ν_{eff}},
```

along the tangential direction of the image velocity. `ν_eff` is the
viscosity the resolved stencil applies at the wall (``ν + ν_t``). In the
viscous sublayer (``τ_w = ν u_t/h``, `ν_eff = ν`) the slip is zero:
no-slip is recovered. With `ν_eff = ν` in the log layer it is negative
(``u_t - u_τ y^+``, a wall moving against the flow), which is why a
wall-modelled LES passes its eddy viscosity. `law = nothing` gives zero
slip (the ν-linear stress is the stencil's own).

Use [`slip_wall_value`](@ref) to hand it to `reconstruct_wall_velocity!`
as the `wall_value` of one component.
"""
function wall_slip_velocity(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                            stencils::AbstractVector{<:GhostCellStencil{CellCenter}},
                            geom::CutCellGeometry{T}, law; ν = nothing,
                            ν_eff = ν) where {T}
    out = Vector{NTuple{3,T}}(undef, length(stencils))
    law === nothing || ν_eff !== nothing ||
        throw(ArgumentError("wall_slip_velocity: ν_eff (or ν) is required"))
    _each_tangential(u, v, w, stencils, geom) do m, s, ut, t
        if !s.valid || ut == 0 || law === nothing
            out[m] = (zero(T), zero(T), zero(T))
            return
        end
        τ = T(wall_stress(law, ut, s.probe; ν = ν))
        us = ut - τ * s.probe / T(ν_eff)
        r = us / ut
        out[m] = (r * t[1], r * t[2], r * t[3])
    end
    return out
end

"""
    slip_wall_value(slip, stencils, geom, component) -> (x, y, z) -> value

A `wall_value` function for `CutCell.reconstruct_wall_velocity!`: the
`component` (1 = u, 2 = v, 3 = w) of the [`wall_slip_velocity`](@ref)
`slip` of the `CellCenter` stencil whose foot point is nearest to the
query point (searched in the cell of the query and its 26 neighbours;
0 when none is there). The face stencils of `u`, `v`, `w` have foot
points of their own on the same surface; this maps each to the slip
computed at the nearest cell-centre probe.

```julia
slip = wall_slip_velocity(u, v, w, sc, geom, SpaldingLaw(); ν = ν, ν_eff = ν + νt_wall)
reconstruct_wall_velocity!(u, su, geom; wall_value = slip_wall_value(slip, sc, geom, 1))
```
"""
function slip_wall_value(slip::AbstractVector{<:NTuple{3}}, stencils, geom::CutCellGeometry{T},
                         component::Integer) where {T}
    1 <= component <= 3 || throw(ArgumentError("component must be 1, 2 or 3"))
    mesh = geom.mesh
    cell(x, y, z) = (floor(Int, (x - mesh.x0) / mesh.dx), floor(Int, (y - mesh.y0) / mesh.dy),
                     floor(Int, (z - mesh.z0) / mesh.dz))
    bins = Dict{NTuple{3,Int},Vector{Int}}()
    for (m, s) in enumerate(stencils)
        s.valid || continue
        push!(get!(bins, cell(s.xw...), Int[]), m)
    end
    return function (x, y, z)
        c = cell(x, y, z)
        best, bd = 0, T(Inf)
        for a in -1:1, b in -1:1, d in -1:1
            for m in get(bins, (c[1] + a, c[2] + b, c[3] + d), ())
                xw = stencils[m].xw
                d2 = (xw[1] - x)^2 + (xw[2] - y)^2 + (xw[3] - z)^2
                d2 < bd && (best = m; bd = d2)
            end
        end
        return best == 0 ? zero(T) : slip[best][component]
    end
end
