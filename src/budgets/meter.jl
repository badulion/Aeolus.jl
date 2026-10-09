# The level meter of the budgets: the per-level, area-weighted content of
# a field in the air,
#
#   M_k = w_k Σ_ij∈fluid x_ijk,   w_k = ρ̄_k c Π̄_k Δz_k / (Nx Ny)
#
# so Σ_k M_k is the amount per unit ground area of the domain. With c the
# specific heat c_p and Π̄ the Exner function of the reference state, and
# x = θ, it is the heat of the air H_air = Σ ρ̄ c_p Π̄ θ ΔV / A (J m⁻², the
# dry enthalpy c_p T with T = θ Π̄): the meter of the energy budget. With
# c = Π̄ = 1 and x a mixing ratio it is the mass of water or of a tracer
# (kg m⁻²); with x a velocity component, the momentum (kg m⁻¹ s⁻¹).
# Solid cells (buildings) are not air and are left out. A = Nx·Ny·dx·dy
# is the area of the domain, so ΔV/A = Δz_k / (Nx Ny).
#
# Everything here is a function of (mesh, reference profiles, mask):
# nothing refers to a case, so every domain, the urban case and its nest
# parent measure their air with the same code. A `StageBudget`
# (`budgets/stage_budget.jl`) uses one meter per quantity to attribute the
# change of the quantity to the stage entries of a step (moved from the
# urban case's air-heat ledger, ONE-c #452).
#
# The meter works per level: `measure_levels!` returns the content of each
# level, so a layer (the canopy, 0–100 m) is a weighted sum of the same
# numbers, with a partial weight for the level the layer top cuts.

"""
    LevelMeter(mesh, ρ̄, π̄, cp, solid = nothing; W = nothing)
    LevelMeter(sim, ref, solid = nothing)

Measures the per-level content ``M_k = ρ̄_k c\\,Π̄_k Σ_{ij} x\\,Δz_k /
(N_x N_y)`` of a field `x` on `mesh` (per unit ground area of the
domain). For the heat of the air, ``H = Σ ρ̄_k c_p Π̄_k θ\\,ΔV / A``
(J m⁻²), `x = θ` and `cp`, `π̄` are the specific heat and the Exner
function; for the mass of water or of a tracer (kg m⁻²) `π̄ = 1` and
`cp = 1`.

* `ρ̄`, `π̄` — reference density (kg m⁻³) and Exner function at the
  `Nz` cell centres; `cp` — the specific heat (J kg⁻¹ K⁻¹) the model
  converts heat fluxes into θ increments with (1 for a mass);
* `solid` — a halo-padded `Bool` array (an `IBM.SolidMask`'s `mask`):
  solid cells are not air and are left out; `nothing` = all fluid;
* `W` — the per-level weight of the quantity the dynamical core's
  transport conserves, ``Σ W_k θ ΔV`` (1 for the Boussinesq core, ρ̄ for
  the anelastic core, ρ̄θ̄ for the pseudo-incompressible one). The ledger
  uses it to split the change of `H` by the dynamics into what the
  transport conserves and the redistribution between levels of different
  ρ̄Π̄ (see [`split_transport`](@ref)). Default 1.

The second form, the meter of the heat of the air, takes ρ̄, π̄ and cp
from the fields `ρ̄`, `π̄`, `cp` of a reference state (an urban
`CaseReference`, or [`reference_profiles`](@ref)) and `W` from the core
(`Aeolus.Anelastic.constraint_weights` for a sound-proof core).

Level thickness: `mesh.dz` on a `RegularGrid`, the cell thickness
`dzc` on a `StretchedGrid`.
"""
struct LevelMeter
    Nx::Int
    Ny::Int
    Nz::Int
    Hx::Int
    Hy::Int
    Hz::Int
    w::Vector{Float64}        # J m⁻² K⁻¹ per K of level sum Σ_ij θ: ρ̄ c_p Π̄ Δz / (Nx Ny)
    v::Vector{Float64}        # conserved-weight analogue: W Δz / (Nx Ny)
    zf::Vector{Float64}       # level faces (m above the domain bottom), length Nz + 1
    fluid::Array{Float64,3}   # interior-sized: 1.0 = air, 0.0 = solid (a Float64 mask: the sum vectorizes)
    S::Vector{Float64}        # scratch: per-level fluid sums of θ
    Ssolid::Vector{Float64}   # scratch: per-level sums of θ over the solid cells
end

"Cell thicknesses (length `Nz`) of `mesh`: `dz` (RegularGrid) or `dzc` (StretchedGrid)."
function level_thickness(mesh)
    if hasproperty(mesh, :dzc)
        return Float64[mesh.dzc[mesh.Hz+k] for k in 1:mesh.Nz]
    end
    return fill(Float64(mesh.dz), mesh.Nz)
end

function LevelMeter(mesh, ρ̄::AbstractVector, π̄::AbstractVector, cp::Real, solid = nothing;
                      W = nothing)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    length(ρ̄) == Nz && length(π̄) == Nz ||
        throw(DimensionMismatch("LevelMeter: ρ̄ and π̄ need Nz = $Nz levels"))
    Δz = level_thickness(mesh)
    n = Nx * Ny
    w = [ρ̄[k] * cp * π̄[k] * Δz[k] / n for k in 1:Nz]
    Wk = W === nothing ? ones(Nz) : collect(Float64, W)
    length(Wk) == Nz || throw(DimensionMismatch("LevelMeter: W needs Nz = $Nz levels"))
    v = [Wk[k] * Δz[k] / n for k in 1:Nz]
    zf = vcat(0.0, cumsum(Δz))
    fluid = trues(Nx, Ny, Nz)
    if solid !== nothing
        fluid = .!collect(Bool, view(solid, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz))
    end
    return LevelMeter(Nx, Ny, Nz, Hx, Hy, Hz, w, v, zf, collect(Float64, fluid), zeros(Nz), zeros(Nz))
end

"The per-level weight of the conserved transport quantity of `sim` (`nothing`: 1)."
transport_conserved_weight(sim) = nothing
function transport_conserved_weight(sim::MassWeightedSimulation)
    Wc, _ = constraint_weights(sim)
    Hz, Nz = sim.mesh.Hz, sim.mesh.Nz
    return Float64[Wc[Hz+k] for k in 1:Nz]
end

LevelMeter(sim, ref, solid = nothing) =
    LevelMeter(sim.mesh, ref.ρ̄, ref.π̄, ref.cp, solid; W = transport_conserved_weight(sim))

_θdata(θ) = hasproperty(θ, :data) ? θ.data : θ

"""
    measure_levels!(out, meter, x) -> out

The content of each level, ``out_k = w_k Σ_{ij∈fluid} x_{ijk}`` (per
unit ground area of the domain: J m⁻² for the heat meter and `x = θ`),
for a `Field` `x` (or its halo-padded data array). One pass over the
interior; the per-level sums are left in `meter.S`, and those over the
solid cells (which the dynamics may exchange heat with, see
`budgets/stage_budget.jl`) in `meter.Ssolid`.
"""
function measure_levels!(out::AbstractVector, m::LevelMeter, θ)
    d = _θdata(θ)
    # the fluid and solid sums of every level in one pass, each in the
    # canonical, thread-invariant plane order (KA-b, #296)
    plane_reduce!((m.S, m.Ssolid), backend(d), _air_heat_at,
                  (1:m.Nx, 1:m.Ny, 1:m.Nz), d, m.fluid, m.Hx, m.Hy, m.Hz)
    @inbounds for k in 1:m.Nz
        out[k] = m.w[k] * m.S[k]
    end
    return out
end

"""
    measure_levels!(out, meter, xs::Tuple) -> out

The content of several fields together (the water of the air: vapour,
cloud and rain), level by level: the sum of [`measure_levels!`](@ref) of
each, with `meter.S` and `meter.Ssolid` the summed level sums.
"""
function measure_levels!(out::AbstractVector, m::LevelMeter, xs::Tuple)
    S = zeros(m.Nz); Ss = zeros(m.Nz)
    for x in xs
        measure_levels!(out, m, x)
        S .+= m.S; Ss .+= m.Ssolid
    end
    copyto!(m.S, S); copyto!(m.Ssolid, Ss)
    @inbounds for k in 1:m.Nz
        out[k] = m.w[k] * m.S[k]
    end
    return out
end

@inline function _air_heat_at(i, j, k, d, fl, Hx, Hy, Hz)
    @inbounds x = d[Hx+i, Hy+j, Hz+k]
    @inbounds y = fl[i, j, k] * x          # x or 0, exactly
    return (y, x - y)                      # (fluid, solid): x - y is 0 or x, exactly
end

"The content ``w_k Σ_{solid} x`` per level of the solid cells at the last [`measure_levels!`](@ref)."
solid_levels!(out::AbstractVector, m::LevelMeter) = (out .= m.w .* m.Ssolid; out)

"""
    layer_fraction(meter, z0, z1) -> Vector{Float64}

The fraction of each level that lies between heights `z0` and `z1` (m
above the domain bottom): 1 inside, 0 outside, the overlap fraction for
the levels the bounds cut. `sum(f .* levels)` is then the heat of that
layer from per-level heats.
"""
function layer_fraction(m::LevelMeter, z0::Real, z1::Real)
    f = zeros(m.Nz)
    for k in 1:m.Nz
        a, b = m.zf[k], m.zf[k+1]
        f[k] = clamp((min(b, z1) - max(a, z0)) / (b - a), 0.0, 1.0)
    end
    return f
end

"""
    measure(meter, x; layer = (0, Inf)) -> amount per unit ground area
    air_heat(sim, ref, solid = nothing; layer = (0, Inf)) -> J m⁻²

The content ``Σ_k w_k Σ_{ij} x`` of the field `x` between the heights
`layer = (z0, z1)` (m above the domain bottom; a level the bounds cut
counts with its overlap fraction). `air_heat` is the heat of the air
``Σ ρ̄_k c_p Π̄_k θ\\,ΔV / A`` of the simulation `sim` on the reference
state `ref` (fields `ρ̄`, `π̄`, `cp`), with the halo-padded solid mask
`solid`.
"""
function measure(m::LevelMeter, θ; layer = (0.0, Inf))
    lv = measure_levels!(zeros(m.Nz), m, θ)
    return sum(layer_fraction(m, layer[1], layer[2]) .* lv)
end
air_heat(sim, ref, solid = nothing; layer = (0.0, Inf)) =
    measure(LevelMeter(sim, ref, solid), sim.θ; layer = layer)

"""
    split_transport(meter, dH) -> (conserved, redistribution)

Split a per-level change `dH` of the measured quantity (J m⁻² for the
heat of the air, e.g. what one dynamics step did) into two per-level parts that sum to it, according to
what the transport conserves. With the per-level θ-sum change
``ΔS_k = dH_k / w_k`` and ``c = Σ w_k / Σ v_k`` (`w` the ledger weight
ρ̄c_pΠ̄Δz/(NxNy), `v` the conserved weight WΔz/(NxNy)):

* `conserved_k = c v_k ΔS_k` — its column sum is ``c Σ_k v_k ΔS_k``, the
  change of what the transport conserves (in heat units), so it is zero
  to round-off for a flux-form transport with closed boundaries. Over the
  whole column it is the numerics residual; over a sub-layer it is the
  heat the dynamics carries across the layer's top;
* `redistribution_k = (w_k − c v_k) ΔS_k` — the heat the ledger sees
  created or destroyed because the transport conserves ``Σ W θ`` and not
  ``Σ ρ̄ Π̄ θ``: moving warm air up into levels of lower ρ̄Π̄ lowers H.

The same split as [`water_split`](@ref) makes for water (#131).
"""
function split_transport(m::LevelMeter, dH::AbstractVector)
    c = sum(m.w) / sum(m.v)
    cons = similar(m.w); red = similar(m.w)
    @inbounds for k in 1:m.Nz
        ΔS = dH[k] / m.w[k]
        cons[k] = c * m.v[k] * ΔS
        red[k] = dH[k] - cons[k]
    end
    return cons, red
end
