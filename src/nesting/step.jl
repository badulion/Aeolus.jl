# Nesting: composite diagnostics, restriction and refluxing of the field
# set (moved from AdaptiveMesh.jl and generalised to named field sets,
# N-nest-a #250). The time coupling, step_nested!, is in nesting/nested_run.jl.

# =========================================================================
# Composite diagnostics
# =========================================================================

"""
    composite_integral(parent_field, child_field, nest) -> Float64

Volume integral of a cell-centred scalar over the *composite* grid:
parent cells not covered by the nest plus all child cells. This is the
quantity a conservative two-way nest must leave invariant (up to the
genuine sources and the fluxes through the parent's own boundary).
"""
function composite_integral(pf::Field{CellCenter}, cf::Field{CellCenter},
                            nest::NestedGrid)
    pm = nest.parent_mesh; cm = nest.child_mesh
    size(pf.data) == totalsize(pm) ||
        error("composite_integral: parent field is not on the nest's parent mesh")
    size(cf.data) == totalsize(cm) ||
        error("composite_integral: child field is not on the nest's child mesh")
    Vp = pm.dx * pm.dy * pm.dz; Vc = cm.dx * cm.dy * cm.dz
    cx, cy, cz = nest_covered_range(nest)
    total = 0.0
    pd = pf.data
    @inbounds for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
        (i in cx) && (j in cy) && (k in cz) && continue
        total += pd[pm.Hx+i, pm.Hy+j, pm.Hz+k] * Vp
    end
    cd = cf.data
    @inbounds for k in 1:cm.Nz, j in 1:cm.Ny, i in 1:cm.Nx
        total += cd[cm.Hx+i, cm.Hy+j, cm.Hz+k] * Vc
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
"""
function composite_integral(psim::_NestSim, csim::_NestSim,
                            nest::NestedGrid, quantity::Symbol;
                            state::Union{Nothing,NestInterpState}=nothing)
    pm = nest.parent_mesh; cm = nest.child_mesh
    Vp = pm.dx * pm.dy * pm.dz; Vc = cm.dx * cm.dy * cm.dz
    quantity === :mass && return (pm.Nx*pm.Ny*pm.Nz - nest.nx*nest.ny*nest.nz) * Vp +
                                 cm.Nx*cm.Ny*cm.Nz * Vc
    (quantity === :theta || quantity === :θ) &&
        return composite_integral(psim.θ, csim.θ, nest)
    if state !== nothing
        i = findfirst(==(quantity), state.names)
        i === nothing || return composite_integral(state.parent[i], state.child[i], nest)
    end
    quantity === :qv && return composite_integral(psim.qv, csim.qv, nest)
    quantity === :momentum_x || error("composite_integral: unknown quantity $quantity " *
        "(:mass, :theta, :qv, :momentum_x, or a scalar of `state`'s field set)")
    cx, cy, cz = nest_covered_range(nest)
    total = 0.0
    pd = psim.u.data
    @inbounds for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
        (i in cx) && (j in cy) && (k in cz) && continue
        total += 0.5 * (pd[pm.Hx+i, pm.Hy+j, pm.Hz+k] +
                        pd[pm.Hx+i+1, pm.Hy+j, pm.Hz+k]) * Vp
    end
    cd = csim.u.data
    @inbounds for k in 1:cm.Nz, j in 1:cm.Ny, i in 1:cm.Nx
        total += 0.5 * (cd[cm.Hx+i, cm.Hy+j, cm.Hz+k] +
                        cd[cm.Hx+i+1, cm.Hy+j, cm.Hz+k]) * Vc
    end
    return total
end

"Ledger key of a field-set scalar: `:theta` for θ (historical), else its name."
_ledger_key(name::Symbol) = name === :θ ? :theta : name

"Composite integral of every scalar of the set, keyed like `nest.ledger`."
_composite_scalars(psim, csim, nest, s::NestInterpState) =
    Dict(_ledger_key(n) => composite_integral(pf, cf, nest)
         for (n, _, cf, pf) in _scalar_entries(s, psim, csim))

# =========================================================================
# Restriction and refluxing of the field set
# =========================================================================

"""
    restrict_state_to_parent!(parent_sim, child_sim, nest, state=nothing)

Conservative restriction of the whole field set, child → parent:
`θ` and every bound scalar of `state` (`qv`, tracers) by volume-weighted
averaging, and `u, v, w` by area-weighted averaging on the faces strictly
inside the nest (the interface planes belong to
[`apply_flux_correction!`](@ref)). Without a `state` only `u, v, w, θ`
are restricted.
"""
function restrict_state_to_parent!(psim::_NestSim,
                                   csim::_NestSim, nest::NestedGrid,
                                   s::Union{Nothing,NestInterpState}=nothing)
    restrict_to_parent!(psim.θ, csim.θ, nest)
    if s !== nothing
        for i in eachindex(s.names)
            restrict_to_parent!(s.parent[i], s.child[i], nest)
        end
    end
    restrict_to_parent!(psim.u, csim.u, nest)
    restrict_to_parent!(psim.v, csim.v, nest)
    restrict_to_parent!(psim.w, csim.w, nest)
    return psim
end

"""
    reflux!(parent_field, parent_registers, child_registers, nest)

Berger & Colella (1989) refluxing of one cell-centred scalar.
`parent_registers` and `child_registers` are `(Φx, Φy, Φz)`: the
time-integrated advective+diffusive fluxes `Φ = ∫ F dt` (field value ×
length) on the parent and child faces over one parent step. The
coarse-minus-fine mismatch on the six interface planes is booked into the
parent cells just outside the nest, so the composite integral
([`composite_integral`](@ref)) of the scalar is unchanged by the nest to
round-off.

Reference: Berger, M. J. & Colella, P. (1989), *J. Comput. Phys.* 82,
64–84, §3 (the flux-register correction).
"""
function reflux!(pf::Field{CellCenter}, preg::NTuple{3,Field}, creg::NTuple{3,Field},
                 nest::NestedGrid)
    location(preg[1]) isa FaceX && location(preg[2]) isa FaceY &&
        location(preg[3]) isa FaceZ ||
        error("reflux!: registers must be (FaceX, FaceY, FaceZ) fields")
    mm = compute_nest_flux_mismatch(preg[1], creg[1], nest) +
         compute_nest_flux_mismatch(preg[2], creg[2], nest) +
         compute_nest_flux_mismatch(preg[3], creg[3], nest)
    return apply_flux_correction!(pf, mm, nest)
end

"""
    reflux_scalars!(parent_sim, nest, state, registers)

[`reflux!`](@ref) every cell-centred scalar of `state`'s field set that
has an entry in `registers`, a mapping
`name => (parent_registers, child_registers)` with each register a
`(Φx, Φy, Φz)` tuple of face fields (`:θ` refers to `parent_sim.θ`, the
other names to their bound parent fields). Scalars without registers are
left untouched. Returns the names that were refluxed.
"""
function reflux_scalars!(psim::_NestSim, nest::NestedGrid,
                         s::NestInterpState, registers)
    done = Symbol[]
    for (name, _, _, pf) in _scalar_entries(s, psim, nothing)
        key = haskey(registers, name) ? name :
              (name === :θ && haskey(registers, :theta)) ? :theta : nothing
        key === nothing && continue
        preg, creg = registers[key]
        reflux!(pf, Tuple(preg), Tuple(creg), nest)
        push!(done, name)
    end
    return done
end

"Interface mass-flux imbalance, normalized by (velocity scale × interface area)."
function _interface_imbalance(psim::_NestSim, csim::_NestSim,
                              nest::NestedGrid)
    m = compute_nest_flux_mismatch(psim.u, csim.u, nest) +
        compute_nest_flux_mismatch(psim.v, csim.v, nest) +
        compute_nest_flux_mismatch(psim.w, csim.w, nest)
    U = max(maximum(abs, interior_view(psim.u)),
            maximum(abs, interior_view(psim.v)),
            maximum(abs, interior_view(psim.w)),
            maximum(abs, interior_view(csim.u)))
    U = U > 0 ? U : 1.0
    return mismatch_l1(m) / (U * mismatch_area(nest))
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
    of the genuine coarse/fine disagreement.
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
    quantity === :mass && return _interface_imbalance(psim, csim, nest)
    error("nest_conservation_error: no recorded value for :$quantity — " *
          "run step_nested! first.")
end
