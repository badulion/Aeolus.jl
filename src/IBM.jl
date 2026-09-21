module IBM

# Phase 13 — Immersed Boundary Method (direct forcing).
#
# We represent solid obstacles inside the fluid domain by a Boolean mask
# defined at cell centers. The direct-forcing approach sets velocity in
# solid cells to zero every time step; the pressure-projection solve
# then makes the surrounding flow divergence-free naturally, giving a
# no-penetration wall around each solid cell.
#
# Extensions (later phases): ghost-cell IBM for higher-order wall
# treatment; moving obstacles; sub-cell reconstruction of the interface.

using ..Grids: RegularGrid, interior
using ..Fields: Field, interior_view
using ..Simulations: BoussinesqSimulation

export SolidMask, mark_solid!, apply_solid_mask!, add_ibm_penalty!,
       implicit_ibm_correction!

"""
    SolidMask(mesh; init = zeros)

A Boolean mask on `mesh` (cell centers) marking solid cells. `true` =
solid, `false` = fluid. Storage matches a `Field(CellCenter, mesh)`
(interior + halos) so it can be inspected with `interior_view` on the
underlying `mask` array.
"""
struct SolidMask{M<:RegularGrid, A<:AbstractArray{Bool,3}}
    mesh::M
    mask::A
end

function SolidMask(mesh::RegularGrid; init = nothing)
    N = (mesh.Nx + 2mesh.Hx, mesh.Ny + 2mesh.Hy, mesh.Nz + 2mesh.Hz)
    m = falses(N...)
    init === nothing || (m .= init)
    return SolidMask(mesh, m)
end

"""
    mark_solid!(sm, predicate)

Set `sm.mask[i,j,k] = true` for every interior cell whose center
satisfies `predicate(x, y, z)`.
"""
function mark_solid!(sm::SolidMask, predicate::Function)
    m  = sm.mesh
    ii, jj, kk = interior(m)
    xs = m.x0 .+ ((1:m.Nx) .- 0.5) .* m.dx
    ys = m.y0 .+ ((1:m.Ny) .- 0.5) .* m.dy
    zs = m.z0 .+ ((1:m.Nz) .- 0.5) .* m.dz
    @inbounds for (ki, kv) in enumerate(kk),
                   (ji, jv) in enumerate(jj),
                   (ii_, iv) in enumerate(ii)
        sm.mask[iv, jv, kv] = predicate(xs[ii_], ys[ji], zs[ki])
    end
    return sm
end

"""
    apply_solid_mask!(sim, sm)

Zero the face-centered velocity components on any face that touches a
solid cell (either side of the face is masked). Applied after the
pressure projection every time step.
"""
function apply_solid_mask!(sim::BoussinesqSimulation, sm::SolidMask)
    m  = sm.mesh
    ii, jj, kk = interior(m)
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    mk = sm.mask
    @inbounds for k in kk, j in jj, i in ii
        if mk[i, j, k]
            # zero all six faces of the solid cell
            ud[i,   j,   k] = 0
            ud[i+1, j,   k] = 0
            vd[i,   j,   k] = 0
            vd[i,   j+1, k] = 0
            wd[i,   j,   k] = 0
            wd[i,   j,   k+1] = 0
        end
    end
    return sim
end

"""
    add_ibm_penalty!(sim, sm; τ = 1e-3)

Angot-style volume penalization: add `-(mask / τ) · u` to each momentum
tendency (`sim.tu`, `sim.tv`, `sim.tw`). With small `τ` the tendency
strongly drives `u → 0` in solid cells while leaving fluid cells alone;
the pressure projection then resolves the flow around the obstacle in
one consistent solve.

Call inside a custom step just after computing tendencies and *before*
the predictor step, so the penalty is included in the flow the
projection sees.
"""
function add_ibm_penalty!(sim, sm::SolidMask; τ::Real = 1e-3)
    m  = sm.mesh
    ii, jj, kk = interior(m)
    mk = sm.mask
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    tud = sim.tu.data; tvd = sim.tv.data; twd = sim.tw.data
    invτ = 1 / τ
    @inbounds for k in kk, j in jj, i in ii
        # A face is "in solid" if either bordering cell is solid.
        if mk[i, j, k]
            tud[i,   j, k] -= invτ * ud[i,   j, k]
            tud[i+1, j, k] -= invτ * ud[i+1, j, k]
            tvd[i, j,   k] -= invτ * vd[i, j,   k]
            tvd[i, j+1, k] -= invτ * vd[i, j+1, k]
            twd[i, j, k]   -= invτ * wd[i, j, k]
            twd[i, j, k+1] -= invτ * wd[i, j, k+1]
        end
    end
    return sim
end

"""
    implicit_ibm_correction!(sim, sm; τ = 0.0, dt = 0.0)

Volume-penalization applied *implicitly* to the intermediate velocity
after the predictor step. For each face touching a solid cell:

    u* ← u* / (1 + Δt/τ)

Unlike [`add_ibm_penalty!`](@ref) — which is *explicit* and requires
`Δt / τ < 1` for stability — this form is unconditionally stable for
any positive `τ`. `τ = 0` treats the penalty as infinite and just zeros
the face velocity (`u* ← 0`). Best combined with `step!(sim, dt;
post_predictor! = (s, Δt) -> implicit_ibm_correction!(s, sm; dt = Δt))`.
"""
function implicit_ibm_correction!(sim, sm::SolidMask; τ::Real = 0.0, dt::Real = 0.0)
    m  = sm.mesh
    ii, jj, kk = interior(m)
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    mk = sm.mask
    factor = τ > 0 ? 1 / (1 + dt / τ) : 0.0
    @inbounds for k in kk, j in jj, i in ii
        if mk[i, j, k]
            ud[i,   j, k] *= factor
            ud[i+1, j, k] *= factor
            vd[i, j,   k] *= factor
            vd[i, j+1, k] *= factor
            wd[i, j, k]   *= factor
            wd[i, j, k+1] *= factor
        end
    end
    return sim
end

end # module
