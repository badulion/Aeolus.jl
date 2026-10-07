# Nesting: composite diagnostics, restriction and refluxing of the field
# set (moved from AdaptiveMesh.jl and generalised to named field sets,
# N-nest-a #250). The time coupling, step_nested!, is in nesting/nested_run.jl.

# =========================================================================
# Composite diagnostics
# =========================================================================

"""
    composite_integral(parent_field, child_field, nest; measure=nothing) -> Float64

Volume integral of a cell-centred scalar over the *composite* grid:
parent cells not covered by the nest plus all child cells. This is the
quantity a conservative two-way nest must leave invariant (up to the
genuine sources and the fluxes through the parent's own boundary).
Parent cell volumes use the actual layer thickness (stretched parents).
With the mass `measure` of a sound-proof pair ([`nest_measure`](@ref))
it is the reference-mass integral `∫ W φ dV` (`W = ρ̄` or `ρ̄θ̄`).
"""
function composite_integral(pf::Field{CellCenter}, cf::Field{CellCenter},
                            nest::NestedGrid; measure=nothing)
    size(pf.data) == totalsize(nest.parent_mesh) ||
        error("composite_integral: parent field is not on the nest's parent mesh")
    size(cf.data) == totalsize(nest.child_mesh) ||
        error("composite_integral: child field is not on the nest's child mesh")
    return _composite_sum(_CellValue(pf.data), _CellValue(cf.data), nest, measure)
end

# The cell value a composite integral sums: a cell-centred field, `u`
# averaged to the cell centre, or one. Callable structs (no closures, no
# temporary fields: #329, item 6 — `:momentum_x` used to allocate two
# child- and parent-sized fields per call, twice per two-way step).
struct _CellValue{A}; d::A; end
@inline (c::_CellValue)(i, j, k) = @inbounds c.d[i, j, k]
struct _UCellValue{A}; d::A; end
@inline (c::_UCellValue)(i, j, k) = @inbounds 0.5 * (c.d[i, j, k] + c.d[i+1, j, k])
struct _OneValue; end
@inline (::_OneValue)(i, j, k) = 1.0

function _composite_sum(fp, fc, nest::NestedGrid, measure)
    pm = nest.parent_mesh; cm = nest.child_mesh
    cx, cy, cz = nest_covered_range(nest)
    total = 0.0
    @inbounds for k in 1:pm.Nz
        Vp = pm.dx * pm.dy * _pcell(nest, measure, k)
        for j in 1:pm.Ny, i in 1:pm.Nx
            (i in cx) && (j in cy) && (k in cz) && continue
            total += fp(pm.Hx+i, pm.Hy+j, pm.Hz+k) * Vp
        end
    end
    ωz = _zweights(measure)
    @inbounds for k in 1:cm.Nz
        Vc = ωz === nothing ? cm.dx * cm.dy * cm.dz : cm.dx * cm.dy * ωz[cm.Hz+k]
        for j in 1:cm.Ny, i in 1:cm.Nx
            total += fc(cm.Hx+i, cm.Hy+j, cm.Hz+k) * Vc
        end
    end
    return total
end

"""
    composite_integral(parent_sim, child_sim, nest, quantity; state=nothing) -> Float64

Volume integral of `quantity` over the *composite* grid: parent cells not
covered by the nest plus all child cells.

  * `:theta` (or `:θ`), `:qv` — the simulations' own cell-centred fields;
  * any extra scalar of `state`'s field set (e.g. `:co2`), integrated
    from its bound parent and child fields;
  * `:momentum_x` — `u` averaged to cell centres;
  * `:mass` — for the Boussinesq system `ρ₀ ∫dV`, a constant; mass
    conservation across a nest interface is therefore diagnosed by
    [`nest_conservation_error`](@ref) from the interface flux imbalance
    instead.

For a [`MassWeightedSimulation`](@ref) pair every integral is weighted
by the reference mass, `∫ W φ dV` (and `:mass` is `∫ W dV`).
"""
function composite_integral(psim::_NestSim, csim::_NestSim,
                            nest::NestedGrid, quantity::Symbol;
                            state::Union{Nothing,NestInterpState}=nothing)
    meas = state === nothing ? _sim_measure(nest, psim, csim) : state.measure
    quantity === :mass && return _composite_sum(_OneValue(), _OneValue(), nest, meas)
    (quantity === :theta || quantity === :θ) &&
        return composite_integral(psim.θ, csim.θ, nest; measure=meas)
    if state !== nothing
        i = findfirst(==(quantity), state.names)
        i === nothing ||
            return composite_integral(state.parent[i], state.child[i], nest; measure=meas)
    end
    quantity === :qv && return composite_integral(psim.qv, csim.qv, nest; measure=meas)
    quantity === :momentum_x || error("composite_integral: unknown quantity $quantity " *
        "(:mass, :theta, :qv, :momentum_x, or a scalar of `state`'s field set)")
    return _composite_sum(_UCellValue(psim.u.data), _UCellValue(csim.u.data), nest, meas)
end

"Ledger key of a field-set scalar: `:theta` for θ (historical), else its name."
_ledger_key(name::Symbol) = name === :θ ? :theta : name

"Composite integral of every scalar of the set (θ first, then the bound scalars)."
function _composite_scalars!(out::Vector{Float64}, psim, csim, nest, s::NestInterpState)
    resize!(out, 1 + length(s.names))
    out[1] = composite_integral(psim.θ, csim.θ, nest; measure=s.measure)
    for i in eachindex(s.names)
        out[1 + i] = composite_integral(s.parent[i], s.child[i], nest; measure=s.measure)
    end
    return out
end
_composite_scalars(psim, csim, nest, s::NestInterpState) =
    _composite_scalars!(Float64[], psim, csim, nest, s)

"Ledger keys of the scalars, in the order of `_composite_scalars!`."
_scalar_ledger_keys(s::NestInterpState) = Symbol[:theta; [_ledger_key(n) for n in s.names]]

# =========================================================================
# Restriction and refluxing of the field set
# =========================================================================

"""
    restrict_state_to_parent!(parent_sim, child_sim, nest, state=nothing; solid=nothing)

Conservative restriction of the whole field set, child → parent:
`θ` and every bound scalar of `state` (`qv`, tracers) by volume-weighted
averaging, and `u, v, w` by area-weighted averaging on the faces strictly
inside the nest (the interface planes belong to
[`apply_flux_correction!`](@ref)). Without a `state` only `u, v, w, θ`
are restricted.

Selected by dispatch: for a [`MassWeightedSimulation`](@ref) pair the
averages are weighted by the shared reference mass (`ρ̄` or `ρ̄θ̄`, see
[`restrict_to_parent!`](@ref)). `solid` (the child's IBM
[`SolidMask`](@ref) or a Boolean `solid` array) makes every member
solid-aware: building cells and faces are left out of the averages.
A mixed Boussinesq / sound-proof pair throws an `ArgumentError` (no
consistent measure, #327).
"""
function restrict_state_to_parent!(psim::_NestSim, csim::_NestSim, nest::NestedGrid,
                                   s::Union{Nothing,NestInterpState}=nothing;
                                   solid=nothing)
    _check_two_way_pair(psim, csim, "restrict_state_to_parent!", s)
    meas = s === nothing ? _sim_measure(nest, psim, csim) : s.measure
    χ = _fluid_mask(nest, solid)              # computed once for the whole set
    restrict_to_parent!(psim.θ, csim.θ, nest; measure=meas, solid=χ)
    if s !== nothing
        for i in eachindex(s.names)
            restrict_to_parent!(s.parent[i], s.child[i], nest; measure=meas, solid=χ)
        end
    end
    restrict_to_parent!(psim.u, csim.u, nest; measure=meas, solid=χ)
    restrict_to_parent!(psim.v, csim.v, nest; measure=meas, solid=χ)
    restrict_to_parent!(psim.w, csim.w, nest; measure=meas, solid=χ)
    return psim
end

"""
    reflux!(parent_field, parent_registers, child_registers, nest; periodic,
            measure=nothing)

Berger & Colella (1989) refluxing of one cell-centred scalar.
`parent_registers` and `child_registers` are `(Φx, Φy, Φz)`: the
time-integrated advective+diffusive fluxes `Φ = ∫ F dt` (field value ×
length) on the parent and child faces over one parent step. The
coarse-minus-fine mismatch on the six interface planes is booked into the
parent cells just outside the nest (`periodic`, required: the parent's
periodic axes, e.g. `nest_periodic_axes(parent_sim)`, see
[`apply_flux_correction!`](@ref)), so the composite integral
([`composite_integral`](@ref)) of the scalar is unchanged by the nest to
round-off.

With the mass `measure` of a sound-proof pair the registers are mass
fluxes (`Φ = ∫ W F dt`) and the correction is divided by the parent
cell's reference mass, so the composite `∫ W φ dV` is unchanged.

Reference: Berger, M. J. & Colella, P. (1989), *J. Comput. Phys.* 82,
64–84, §3 (the flux-register correction).
"""
function reflux!(pf::Field{CellCenter}, preg::NTuple{3,Field}, creg::NTuple{3,Field},
                 nest::NestedGrid; measure=nothing,
                 periodic::NTuple{3,Bool})
    location(preg[1]) isa FaceX && location(preg[2]) isa FaceY &&
        location(preg[3]) isa FaceZ ||
        error("reflux!: registers must be (FaceX, FaceY, FaceZ) fields")
    mm = compute_nest_flux_mismatch(preg[1], creg[1], nest) +
         compute_nest_flux_mismatch(preg[2], creg[2], nest) +
         compute_nest_flux_mismatch(preg[3], creg[3], nest)
    return apply_flux_correction!(pf, mm, nest; measure, periodic)
end

"""
    reflux_scalars!(parent_sim, nest, state, registers)

[`reflux!`](@ref) every cell-centred scalar of `state`'s field set that
has an entry in `registers`, a mapping
`name => (parent_registers, child_registers)` with each register a
`(Φx, Φy, Φz)` tuple of face fields (`:θ` refers to `parent_sim.θ`, the
other names to their bound parent fields). Scalars without registers are
left untouched. Uses the state's measure (mass-flux registers for a
sound-proof pair) and the periodic axes of `parent_sim`'s boundary
conditions ([`nest_periodic_axes`](@ref)). Returns the names that were refluxed.
"""
function reflux_scalars!(psim::_NestSim, nest::NestedGrid,
                         s::NestInterpState, registers)
    done = Symbol[]
    per = nest_periodic_axes(psim)
    _foreach_scalar(s, psim, nothing) do name, _, _, pf
        key = haskey(registers, name) ? name :
              (name === :θ && haskey(registers, :theta)) ? :theta : nothing
        key === nothing && return
        preg, creg = registers[key]
        reflux!(pf, Tuple(preg), Tuple(creg), nest; measure=s.measure, periodic=per)
        push!(done, name)
    end
    return done
end

"Interface mass-flux imbalance, normalized by (velocity scale × interface area)."
function _interface_imbalance(psim::_NestSim, csim::_NestSim,
                              nest::NestedGrid, meas=nothing)
    nn = (nest.nx, nest.ny, nest.nz)
    l1 = 0.0                         # Σ |coarse − fine| over the six planes, unstored
    for (ax, pf, cf) in ((1, psim.u, csim.u), (2, psim.v, csim.v), (3, psim.w, csim.w))
        l1 += _plane_mismatch!(nothing, pf, cf, nest, ax, 0, meas)
        l1 += _plane_mismatch!(nothing, pf, cf, nest, ax, nn[ax], meas)
    end
    U = max(maximum(abs, interior_view(psim.u)),
            maximum(abs, interior_view(psim.v)),
            maximum(abs, interior_view(psim.w)),
            maximum(abs, interior_view(csim.u)))
    U = U > 0 ? U : 1.0
    return l1 / (U * mismatch_area(nest))
end

"""
    nest_conservation_error(parent_sim, child_sim, nest, quantity) -> Float64

Relative error in the globally integrated `quantity` across a nested
step, as recorded by the most recent [`step_nested!`](@ref).

  * `:mass` — the L1 coarse-fine **mass-flux imbalance** across the six
    interface planes, normalized by (velocity scale × interface area).
    This is the discrete statement that the parent and the child agree on
    the volume flux through the interface; `apply_flux_correction!` drives
    it to machine zero, and skipping the correction leaves it at the size
    of the genuine coarse/fine disagreement. `:mass_before_correction` is
    the value before the correction and `:mass_after_projection` the value
    after the two-way re-projection of the parent ([`TwoWayFeedback`](@ref)).
  * `:theta`, `:momentum_x`, and every other scalar of the step's field
    set (`:qv`, tracers) — relative change of
    [`composite_integral`](@ref) over the step.

If no step has been recorded, `:mass` is evaluated instantaneously and the
other quantities raise an error.
"""
function nest_conservation_error(psim::_NestSim, csim::_NestSim,
                                 nest::NestedGrid, quantity::Symbol)
    quantity === :θ && (quantity = :theta)
    haskey(nest.ledger, quantity) && return nest.ledger[quantity]
    quantity === :mass &&
        return _interface_imbalance(psim, csim, nest, _sim_measure(nest, psim, csim))
    error("nest_conservation_error: no recorded value for :$quantity — " *
          "run step_nested! first.")
end
