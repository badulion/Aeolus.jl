# Nesting: the vertical measure of the conservative operators (N-nest-b, #256).
#
# The conservative prolongation is a primitive-function reconstruction: the
# parent cell averages define the primitive F exactly at the parent faces,
# a polynomial interpolates F there, and a child value is the difference
# quotient of that polynomial across the child cell. N-nest-a did this in
# "parent cell numbers", which is exact only when the parent is uniform and
# the average is a volume average. Here the construction is done in an
# explicit vertical *coordinate* (the measure):
#
#   * :volume — the coordinate is the height z. On a uniform RegularGrid
#     parent this is exactly the N-nest-a integer construction (and the
#     operators use that code path, bit for bit). On a z-stretched parent
#     the Lagrange nodes are the parent's actual face heights.
#   * :mass   — the coordinate is the cumulative reference mass
#     μ(z) = ∫ W dz, with W the constraint weight of a sound-proof
#     simulation (ρ̄ anelastic, ρ̄θ̄ pseudo-incompressible). A cell average
#     in μ is the W-weighted average, so Σ W φ ΔV over the child cells of a
#     parent cell is the parent's W φ ΔV exactly (the primitive is matched
#     at the parent faces), and a constant stays constant (F is linear in
#     μ, reproduced by any interpolating polynomial of degree ≥ 1).
#
# Horizontal directions are always uniform volume measure (W depends on z
# only) and keep the N-nest-a integer plans.

Base.show(io::IO, m::NestMeasure) =
    print(io, "NestMeasure(:", m.kind, ", ", length(m.pc) - 1, " parent / ",
          length(m.cc) - 1, " child storage levels)")

# Child face heights, storage-indexed (length Nc + 2Hc + 1). Faces inside the
# overlap are generated from the parent faces, so a child face lying on a
# parent face *is* that parent face (bitwise), which is what makes the
# Lagrange basis exactly 0/1 there and the reconstruction exactly
# conservative. Faces in the child halos continue at the child spacing.
function _child_zfaces(nest::NestedGrid, ph::Vector{Float64})
    # a stretched, non-uniform or non-aligned child: its own faces, snapped
    # onto the coincident parent faces (#328)
    (nest.rz > 0 && nest.child_mesh isa RegularGrid) || return copy(nest.axes[3].cf)
    pm = nest.parent_mesh; cm = nest.child_mesh
    Hp = pm.Hz; Hc = cm.Hz; Nc = cm.Nz; rz = nest.rz
    ch = Vector{Float64}(undef, Nc + 2Hc + 1)
    Pb = Hp + nest.iz                       # parent storage face at the child bottom
    for f in eachindex(ch)
        n = f - Hc - 1                      # child face number, 0 = child bottom
        if 0 <= n <= Nc
            P = Pb + fld(n, rz); m = mod(n, rz)
            ch[f] = m == 0 ? ph[P] : ph[P] + (m / rz) * (ph[P+1] - ph[P])
        end
    end
    zb = ph[Pb]; zt = ph[Pb + nest.nz]
    for f in eachindex(ch)
        n = f - Hc - 1
        n < 0  && (ch[f] = zb + n * cm.dz)
        n > Nc && (ch[f] = zt + (n - Nc) * cm.dz)
    end
    return ch
end

"""
    nest_measure(nest)                          -> NestMeasure (:volume)
    nest_measure(nest, Wp, Wc)                  -> NestMeasure (:mass)
    nest_measure(nest, parent_sim, child_sim; rtol=1e-4)

The vertical measure of `nest`'s conservative operators (see
[`NestMeasure`](@ref)).

  * `nest_measure(nest)` — the volume measure (the default of every
    operator; on a stretched parent its coordinates are the parent's
    `zfaces`).
  * `nest_measure(nest, Wp, Wc)` — the mass measure for weights `Wp`
    (parent) and `Wc` (child), storage-indexed cell vectors
    (`length = Nz + 2Hz`). When `W` is the same constant on both grids
    the mass measure *is* the volume measure and that is returned (the
    analytic limit: the operators are then the N-nest-a operators bit for
    bit).
  * `nest_measure(nest, parent_sim, child_sim)` for two
    [`MassWeightedSimulation`](@ref)s — the mass measure from their
    `constraint_weights` (`ρ̄` or `ρ̄θ̄` of the sims' `ReferenceState`s),
    after asserting that parent and child share one reference profile
    ([`nest_reference_mismatch`](@ref) `≤ rtol`). The default
    `rtol = 1e-4` admits the midpoint-rule difference of one smooth
    profile sampled on both grids (`≲ 1e-4` for parent layers up to
    ~400 m in an 8 km scale height) and rejects a different profile (a
    1 K warmer surface is `3.5e-3`).
"""
function nest_measure(nest::NestedGrid)
    return get!(nest.measures, _VOLUME_KEY) do
        ph = _zfaces_storage(nest.parent_mesh)
        ch = _child_zfaces(nest, ph)
        NestMeasure(:volume, ph, ch, ph, ch, diff(ch), diff(ph),
                    _overlap_lengths(nest.axes[3], ph, ch), Dict{Tuple{Symbol,Int},_Plan1D}())
    end
end

function nest_measure(nest::NestedGrid, Wp::AbstractVector, Wc::AbstractVector)
    pm = nest.parent_mesh; cm = nest.child_mesh
    length(Wp) == pm.Nz + 2pm.Hz ||
        error("nest_measure: parent weight has length $(length(Wp)), expected " *
              "Nz + 2Hz = $(pm.Nz + 2pm.Hz) (storage-indexed cell values)")
    length(Wc) == cm.Nz + 2cm.Hz ||
        error("nest_measure: child weight has length $(length(Wc)), expected " *
              "Nz + 2Hz = $(cm.Nz + 2cm.Hz) (storage-indexed cell values)")
    all(>(0), Wp) && all(>(0), Wc) || error("nest_measure: weights must be positive")
    # the analytic limit: a constant W is a rescaled volume, and the
    # operators are then the volume operators, bit for bit
    w0 = Wp[1]
    (all(==(w0), Wp) && all(==(w0), Wc)) && return nest_measure(nest)
    ph = _zfaces_storage(pm)
    ch = _child_zfaces(nest, ph)
    Hp = pm.Hz; Hc = cm.Hz
    # child: μ = 0 at the child bottom face, cumulative child masses W Δz
    cc = zeros(Float64, length(ch))
    for f in Hc+1:length(ch)-1
        cc[f+1] = cc[f] + Wc[f] * (ch[f+1] - ch[f])
    end
    for f in Hc:-1:1
        cc[f] = cc[f+1] - Wc[f] * (ch[f+1] - ch[f])
    end
    # parent: covered faces are copied from the coincident child faces, so
    # a covered parent cell's mass is exactly the sum of its children's
    pc = zeros(Float64, length(ph))
    Pb = Hp + nest.iz; Pt = Pb + nest.nz
    if nest.rz > 0 && cm isa RegularGrid
        for K in 0:nest.nz
            pc[Pb + K] = cc[Hc + 1 + K * nest.rz]
        end
    else
        # general (#328): a box face is the child's μ at that height — the
        # coincident child face's on a common face, else inside the child
        # cell below it (W is constant in a cell, μ linear)
        az = nest.axes[3]
        for K in 0:nest.nz
            f = az.cface[K+1]
            if f != 0
                pc[Pb + K] = cc[f]
            else
                g = az.fbr[K+1]
                pc[Pb + K] = cc[g] + Wc[g] * (ph[Pb + K] - ch[g])
            end
        end
    end
    for P in Pt:length(ph)-1
        pc[P+1] = pc[P] + Wp[P] * (ph[P+1] - ph[P])
    end
    for P in Pb-1:-1:1
        pc[P] = pc[P+1] - Wp[P] * (ph[P+1] - ph[P])
    end
    pw = [Wp[P] * (ph[P+1] - ph[P]) for P in 1:length(ph)-1]
    return NestMeasure(:mass, pc, cc, ph, ch, diff(cc), pw, _overlap_lengths(nest.axes[3], pc, cc),
                       Dict{Tuple{Symbol,Int},_Plan1D}())
end

"""
    nest_reference_mismatch(nest, parent_ref, child_ref) -> Float64

How far the parent and child [`ReferenceState`](@ref)s are from being one
reference profile, as the largest relative difference, over the parent
layers the child covers, between the parent's `ρ̄ Δz` and the sum of the
child's `ρ̄ Δz` over the `rz` child layers inside it (and the same for
`θ̄`, as a `Δz`-weighted mean). A shared, smooth profile sampled at cell
centres on the two grids differs only by the midpoint-rule error of the
parent layer, `≈ (Δz_p² − Δz_c²)/24 · |ρ̄''/ρ̄|`, i.e. `~4e-7` for a
25 m parent layer in a scale height of 8 km. A genuinely different
profile (another surface temperature, a surface pressure set at the
child's own bottom) shows up at `1e-3` and above. The thermodynamic
constants must agree exactly (an error is raised otherwise).
"""
function nest_reference_mismatch(nest::NestedGrid, pr::ReferenceState, cr::ReferenceState)
    (pr.R_d == cr.R_d && pr.cp == cr.cp && pr.p₀ == cr.p₀ && pr.g == cr.g) ||
        error("nest_reference_mismatch: parent and child ReferenceStates use different " *
              "thermodynamic constants (R_d, cp, p₀, g)")
    pm = nest.parent_mesh; cm = nest.child_mesh
    pr.mesh.Nz == pm.Nz || error("nest_reference_mismatch: parent_ref is not on the nest's parent mesh")
    cr.mesh.Nz == cm.Nz || error("nest_reference_mismatch: child_ref is not on the nest's child mesh")
    worst = 0.0
    az = nest.axes[3]
    for K in 1:nest.nz
        Kp = nest.iz + K - 1
        dzp = _dzp(pm, Kp)
        mρ = 0.0; mθ = 0.0
        if nest.rz > 0 && cm isa RegularGrid
            for kc in 1:nest.rz
                k = (K - 1) * nest.rz + kc
                mρ += cr.ρ̄[k] * cm.dz
                mθ += cr.θ̄[k] * cm.dz
            end
        else                                   # the overlaps of the box layer (#328)
            for q in az.ptr[K]:az.ptr[K+1]-1
                k = az.cidx[q] - az.Hc
                mρ += cr.ρ̄[k] * az.len[q]
                mθ += cr.θ̄[k] * az.len[q]
            end
        end
        worst = max(worst, abs(mρ - pr.ρ̄[Kp] * dzp) / (pr.ρ̄[Kp] * dzp),
                           abs(mθ - pr.θ̄[Kp] * dzp) / (pr.θ̄[Kp] * dzp))
    end
    return worst
end

_weighting_key(s::MassWeightedSimulation) = nameof(typeof(s.weighting))

function nest_measure(nest::NestedGrid, psim::MassWeightedSimulation,
                      csim::MassWeightedSimulation; rtol::Real=1e-4)
    typeof(psim.weighting) === typeof(csim.weighting) ||
        error("nest_measure: parent is $(_weighting_key(psim)) but child is " *
              "$(_weighting_key(csim)); a nest must use one sound-proof system")
    psim.mesh === nest.parent_mesh || size(psim.θ.data) == totalsize(nest.parent_mesh) ||
        error("nest_measure: parent_sim is not on the nest's parent mesh")
    csim.mesh === nest.child_mesh || size(csim.θ.data) == totalsize(nest.child_mesh) ||
        error("nest_measure: child_sim is not on the nest's child mesh")
    key = (:mass, objectid(psim.ref), objectid(csim.ref), Symbol(_weighting_key(psim)))
    haskey(nest.measures, key) && return nest.measures[key]
    mis = nest_reference_mismatch(nest, psim.ref, csim.ref)
    mis <= rtol ||
        error("nest_measure: the parent and child ReferenceStates are not one " *
              "reference profile (relative mismatch " * @sprintf("%.3e", mis) *
              " > rtol = " * @sprintf("%.1e", rtol) * " over the overlap). Build both " *
              "from the same profile and surface pressure, with the child's surface " *
              "at the parent's.")
    meas = nest_measure(nest, constraint_weights(psim)[1], constraint_weights(csim)[1])
    nest.measures[key] = meas
    return meas
end

# The measure an operator uses: the explicit one, or the nest's volume
# measure. `nothing` on a uniform parent short-circuits to the N-nest-a
# integer plans (identical numbers, no measure allocation).
_uses_integer_z(nest::NestedGrid, ::Nothing) =
    nest.parent_mesh isa RegularGrid && nest.axes[3].integer !== nothing
_uses_integer_z(nest::NestedGrid, m::NestMeasure) =
    m.kind === :volume && _uses_integer_z(nest, nothing)
_resolve(nest::NestedGrid, m::NestMeasure) = m
_resolve(nest::NestedGrid, ::Nothing) = nest_measure(nest)

# =========================================================================
# Fluid mask (solid-aware restriction)
# =========================================================================

"""
    nest_fluid_mask(nest, solid) -> BitArray{3}

The child fluid mask (`true` = fluid) used by solid-aware restriction,
from the IBM [`SolidMask`](@ref) of the child or a storage-sized Boolean
`solid` array (`true` = solid, e.g. the Berlin case's `geom.sm.mask`).
A child face is fluid when both cells it separates are fluid, the
convention of `apply_solid_mask!`.
"""
nest_fluid_mask(nest::NestedGrid, sm::SolidMask) = nest_fluid_mask(nest, sm.mask)

function nest_fluid_mask(nest::NestedGrid, solid::AbstractArray{Bool,3})
    size(solid) == totalsize(nest.child_mesh) ||
        error("nest_fluid_mask: solid mask has size $(size(solid)), expected the child " *
              "storage size $(totalsize(nest.child_mesh))")
    return .!BitArray(solid)
end

# A fluid mask already computed by `_fluid` (a caller restricting a whole
# field set computes it once): passed through unchanged. Before N-nest-e
# (#263) restrict_state_to_parent! handed the BitArray fluid mask back as
# `solid`, which `_fluid` inverted, so the field set was averaged over the
# *solid* children.
struct _FluidMask
    χ::BitArray{3}
end
_fluid(nest, f::_FluidMask) = f.χ
_fluid(nest, ::Nothing) = nothing
_fluid(nest, sm::SolidMask) = nest_fluid_mask(nest, sm)
_fluid(nest, a::AbstractArray{Bool,3}) = nest_fluid_mask(nest, a)

# The fluid mask once, wrapped, for a caller that restricts repeatedly (a
# NestedRun computes it at construction, #329).
_fluid_mask(nest, ::Nothing) = nothing
_fluid_mask(nest, f::_FluidMask) = f
_fluid_mask(nest, solid) = _FluidMask(_fluid(nest, solid))
