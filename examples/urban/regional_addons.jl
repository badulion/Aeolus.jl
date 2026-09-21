# Batch G — regional / large-scale coupling scaffolds.
#
# Same pragmatic-scaffold spirit as `numerical_addons.jl`:
#
#   G1  Mesoscale forcing profile — a `MesoForcing` container that holds
#       vertical profiles of θ_geo(z, t), q_geo(z, t), U_geo(z, t),
#       V_geo(z, t). A helper `apply_mesoscale_nudging!` relaxes the
#       horizontal-mean state toward these profiles.
#   G2  Sea / lake coupling — a `WaterMask` matrix and a `LakeExchange`
#       helper that pins water-surface θ to a lake temperature and
#       injects a corresponding surface flux.
#   G3  Terrain — a 2-D height map that the rasterizer respects: any
#       cell with z below the terrain height is marked solid, giving
#       simple flat-topped hills.

using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.IBM: SolidMask

export MesoForcing, apply_mesoscale_nudging!,
       WaterMask, apply_lake_exchange!,
       add_terrain_to_mask!

# --- G1. Mesoscale forcing profile ---------------------------------------

"""
    MesoForcing(z_nodes; θ, qv, U, V)

Bundle of 1-D profiles the outer mesoscale model provides at Aeolus's
lateral / top boundary. `θ`, `qv`, `U`, `V` are all functions of
z-level `k` (accepted as vectors of length `Nz` or callable). All
static in time here; a Phase-16-like time-interpolation layer can wrap
this.
"""
struct MesoForcing
    θ_prof::Vector{Float64}
    qv_prof::Vector{Float64}
    U_prof::Vector{Float64}
    V_prof::Vector{Float64}
end

MesoForcing(; θ::AbstractVector, qv::AbstractVector,
              U::AbstractVector, V::AbstractVector) =
    MesoForcing(collect(θ), collect(qv), collect(U), collect(V))

"""
    apply_mesoscale_nudging!(sim, meso; τ = 3600.0, qv_field = nothing)

Nudge the horizontal-mean of `θ`, `u`, `v` (and optionally `qv`) toward
the profiles in `meso` on a time scale `τ`. Interior tendencies get
`-α·(⟨φ⟩ - φ_meso)` added; only the mean is nudged, so turbulence
survives.
"""
function apply_mesoscale_nudging!(sim, meso::MesoForcing; τ::Real = 3600.0,
                                    qv_field = nothing)
    m = sim.mesh
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    length(meso.θ_prof) == Nz || error("MesoForcing: θ_prof length ≠ Nz")
    invτ = 1 / τ
    @inbounds for k in 1:Nz
        # Compute horizontal means
        Θ̄ = 0.0; ū = 0.0; v̄ = 0.0; q̄ = 0.0
        for j in 1:Ny, i in 1:Nx
            Θ̄ += sim.θ.data[Hx+i, Hy+j, Hz+k]
            ū += sim.u.data[Hx+i, Hy+j, Hz+k]
            v̄ += sim.v.data[Hx+i, Hy+j, Hz+k]
            qv_field === nothing || (q̄ += interior_view(qv_field)[i, j, k])
        end
        cnt = Nx * Ny
        Θ̄ /= cnt; ū /= cnt; v̄ /= cnt; q̄ /= cnt
        Δθ = Θ̄ - meso.θ_prof[k]
        Δu = ū - meso.U_prof[k]
        Δv = v̄ - meso.V_prof[k]
        Δqv = q̄ - meso.qv_prof[k]
        for j in 1:Ny, i in 1:Nx
            sim.tθ.data[Hx+i, Hy+j, Hz+k] -= invτ * Δθ
            sim.tu.data[Hx+i, Hy+j, Hz+k] -= invτ * Δu
            sim.tv.data[Hx+i, Hy+j, Hz+k] -= invτ * Δv
            if qv_field !== nothing
                interior_view(qv_field)[i, j, k] -= invτ * Δqv
            end
        end
    end
    return sim
end

# --- G2. Water / lake surface --------------------------------------------

"""
    WaterMask

A 2-D `Matrix{Bool}` marking which ground cells are lake / river / sea.
The `apply_lake_exchange!` helper pins their skin temperature to a
constant water temperature and returns the corresponding sensible flux.
"""
struct WaterMask
    mask::Matrix{Bool}
end
WaterMask(mesh) = WaterMask(falses(mesh.Nx, mesh.Ny))

"""
    apply_lake_exchange!(Ts, wm, T_water, sensible; C_h_water = 3e-3, U = 3.0,
                          ρcp = 1005.0)

For each water-mask cell:
  1. Force `Ts[i, j] = T_water` (huge thermal inertia proxy).
  2. Compute a bulk sensible-heat flux to the atmosphere at that cell,
     `H = ρcp · C_h_water · U · (T_water − θ_air_at_that_cell)`, and
     ADD it to `sensible[i, j]` so the atmosphere feels the sea/land
     breeze.

Simple, but captures the essential sea-breeze forcing.
"""
function apply_lake_exchange!(Ts::AbstractMatrix, wm::WaterMask,
                              T_water::Real, sensible::AbstractMatrix,
                              θ_at_ground::AbstractMatrix;
                              C_h_water::Real = 3e-3, U::Real = 3.0,
                              ρcp::Real = 1005.0)
    Nx, Ny = size(Ts)
    @inbounds for j in 1:Ny, i in 1:Nx
        if wm.mask[i, j]
            Ts[i, j] = T_water
            H = ρcp * C_h_water * U * (T_water - θ_at_ground[i, j])
            sensible[i, j] += H
        end
    end
    return sensible
end

# --- G3. Terrain -----------------------------------------------------------

"""
    add_terrain_to_mask!(sm::SolidMask, mesh, terrain_map::AbstractMatrix)

Given a 2-D height map `terrain_map[i, j]` in meters, mark every solid
cell whose z-centre is at or below the terrain height at that (i, j).
This gives simple flat-topped Cartesian hills — deficient for real
terrain (staircase-shape corners), but a scaffold that plugs cleanly
into the existing IBM. Combine with the OSM building mask via a union.
"""
function add_terrain_to_mask!(sm::SolidMask, mesh, terrain::AbstractMatrix)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    dz = mesh.dz
    @inbounds for j in 1:Ny, i in 1:Nx
        h = terrain[i, j]
        for k in 1:Nz
            z_centre = (k - 0.5) * dz
            if z_centre <= h
                sm.mask[Hx+i, Hy+j, Hz+k] = true
            else
                break
            end
        end
    end
    return sm
end
