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
    pm = nest.parent_mesh; cm = nest.child_mesh
    size(pf.data) == totalsize(pm) ||
        error("composite_integral: parent field is not on the nest's parent mesh")
    size(cf.data) == totalsize(cm) ||
        error("composite_integral: child field is not on the nest's child mesh")
    cx, cy, cz = nest_covered_range(nest)
    total = 0.0
    pd = pf.data
    @inbounds for k in 1:pm.Nz
        Vp = pm.dx * pm.dy * _pcell(nest, measure, k)
        for j in 1:pm.Ny, i in 1:pm.Nx
            (i in cx) && (j in cy) && (k in cz) && continue
            total += pd[pm.Hx+i, pm.Hy+j, pm.Hz+k] * Vp
        end
    end
    cd = cf.data
    ωz = _zweights(measure)
    @inbounds for k in 1:cm.Nz
        Vc = ωz === nothing ? cm.dx * cm.dy * cm.dz : cm.dx * cm.dy * ωz[cm.Hz+k]
        for j in 1:cm.Ny, i in 1:cm.Nx
            total += cd[cm.Hx+i, cm.Hy+j, cm.Hz+k] * Vc
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
    pm = nest.parent_mesh; cm = nest.child_mesh
    meas = state === nothing ? _sim_measure(nest, psim, csim) : state.measure
    if quantity === :mass
        one_p = Field(CellCenter(), pm); fill!(one_p.data, 1.0)
        one_c = Field(CellCenter(), cm); fill!(one_c.data, 1.0)
        return composite_integral(one_p, one_c, nest; measure=meas)
    end
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
    up = Field(CellCenter(), pm); uc = Field(CellCenter(), cm)
    pd = psim.u.data; cd = csim.u.data
    @inbounds for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
        up.data[pm.Hx+i, pm.Hy+j, pm.Hz+k] = 0.5 * (pd[pm.Hx+i, pm.Hy+j, pm.Hz+k] +
                                                    pd[pm.Hx+i+1, pm.Hy+j, pm.Hz+k])
    end
    @inbounds for k in 1:cm.Nz, j in 1:cm.Ny, i in 1:cm.Nx
        uc.data[cm.Hx+i, cm.Hy+j, cm.Hz+k] = 0.5 * (cd[cm.Hx+i, cm.Hy+j, cm.Hz+k] +
                                                    cd[cm.Hx+i+1, cm.Hy+j, cm.Hz+k])
    end
    return composite_integral(up, uc, nest; measure=meas)
end

"Ledger key of a field-set scalar: `:theta` for θ (historical), else its name."
_ledger_key(name::Symbol) = name === :θ ? :theta : name

"Composite integral of every scalar of the set, keyed like `nest.ledger`."
_composite_scalars(psim, csim, nest, s::NestInterpState) =
    Dict(_ledger_key(n) => composite_integral(pf, cf, nest; measure=s.measure)
         for (n, _, cf, pf) in _scalar_entries(s, psim, csim))

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
"""
function restrict_state_to_parent!(psim::_NestSim, csim::_NestSim, nest::NestedGrid,
                                   s::Union{Nothing,NestInterpState}=nothing;
                                   solid=nothing)
    meas = s === nothing ? _sim_measure(nest, psim, csim) : s.measure
    χ = _fluid(nest, solid)                   # computed once for the whole set
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
    reflux!(parent_field, parent_registers, child_registers, nest)

Berger & Colella (1989) refluxing of one cell-centred scalar.
`parent_registers` and `child_registers` are `(Φx, Φy, Φz)`: the
time-integrated advective+diffusive fluxes `Φ = ∫ F dt` (field value ×
length) on the parent and child faces over one parent step. The
coarse-minus-fine mismatch on the six interface planes is booked into the
parent cells just outside the nest, so the composite integral
([`composite_integral`](@ref)) of the scalar is unchanged by the nest to
round-off.

With the mass `measure` of a sound-proof pair the registers are mass
fluxes (`Φ = ∫ W F dt`) and the correction is divided by the parent
cell's reference mass, so the composite `∫ W φ dV` is unchanged.

Reference: Berger, M. J. & Colella, P. (1989), *J. Comput. Phys.* 82,
64–84, §3 (the flux-register correction).
"""
function reflux!(pf::Field{CellCenter}, preg::NTuple{3,Field}, creg::NTuple{3,Field},
                 nest::NestedGrid; measure=nothing)
    location(preg[1]) isa FaceX && location(preg[2]) isa FaceY &&
        location(preg[3]) isa FaceZ ||
        error("reflux!: registers must be (FaceX, FaceY, FaceZ) fields")
    mm = compute_nest_flux_mismatch(preg[1], creg[1], nest) +
         compute_nest_flux_mismatch(preg[2], creg[2], nest) +
         compute_nest_flux_mismatch(preg[3], creg[3], nest)
    return apply_flux_correction!(pf, mm, nest; measure)
end

"""
    reflux_scalars!(parent_sim, nest, state, registers)

[`reflux!`](@ref) every cell-centred scalar of `state`'s field set that
has an entry in `registers`, a mapping
`name => (parent_registers, child_registers)` with each register a
`(Φx, Φy, Φz)` tuple of face fields (`:θ` refers to `parent_sim.θ`, the
other names to their bound parent fields). Scalars without registers are
left untouched. Uses the state's measure (mass-flux registers for a
sound-proof pair). Returns the names that were refluxed.
"""
function reflux_scalars!(psim::_NestSim, nest::NestedGrid,
                         s::NestInterpState, registers)
    done = Symbol[]
    for (name, _, _, pf) in _scalar_entries(s, psim, nothing)
        key = haskey(registers, name) ? name :
              (name === :θ && haskey(registers, :theta)) ? :theta : nothing
        key === nothing && continue
        preg, creg = registers[key]
        reflux!(pf, Tuple(preg), Tuple(creg), nest; measure=s.measure)
        push!(done, name)
    end
    return done
end

"Interface mass-flux imbalance, normalized by (velocity scale × interface area)."
function _interface_imbalance(psim::_NestSim, csim::_NestSim,
                              nest::NestedGrid, meas=nothing)
    m = compute_nest_flux_mismatch(psim.u, csim.u, nest; measure=meas) +
        compute_nest_flux_mismatch(psim.v, csim.v, nest; measure=meas) +
        compute_nest_flux_mismatch(psim.w, csim.w, nest; measure=meas)
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
    quantity === :mass &&
        return _interface_imbalance(psim, csim, nest, _sim_measure(nest, psim, csim))
    error("nest_conservation_error: no recorded value for :$quantity — " *
          "run step_nested! first.")
end
# Time-stepping coordination
# =========================================================================

"""
    step_nested!(parent_sim, child_sim, nest, dt; substeps=nothing, order=3,
                 flux_correction=true, sponge=nothing, impose_interface=false,
                 scalars=(), state=nothing)

One parent step of size `dt` plus `substeps` child steps of size
`dt/substeps` (default `max(rx,ry,rz)`, which keeps the child CFL equal to
the parent's).

The transferred field set is `u, v, w, θ`, `qv` when both simulations
are moist, and the extra cell-centred `scalars`
(`:co2 => (child_field, parent_field)`, …); see [`NestInterpState`](@ref).
`step!` advances only the simulations' own prognostics, so an extra
scalar is carried across the interface (boundary data, sponge,
restriction) but advanced by whoever owns it.

Sequence:

 1. interpolate the parent field set at time `t` onto the child mesh;
 2. `step!(parent_sim, dt)`; interpolate the new parent field set;
 3. for each child substep, **linearly interpolate the parent state in
    time** to the substep midpoint, write it into the child halos,
    `step!` the child, then apply the sponge (if any);
 4. conservatively restrict the child field set back onto the parent
    ([`restrict_state_to_parent!`](@ref));
 5. compute the interface flux mismatch and apply the flux correction.

Conservation diagnostics for `:mass`, `:momentum_x` and every scalar of
the set (`:theta`, `:qv`, tracers) are recorded in `nest.ledger`; read
them with [`nest_conservation_error`](@ref). Returns
`(parent_sim, child_sim)`.

`state` may be a pre-allocated `NTuple{3,NestInterpState}` (built with
`nest_interp_state(nest, parent_sim, child_sim; scalars)`) to avoid
re-allocating the boundary buffers every step; it then defines the field
set and `scalars` is ignored.

The parent may be z-stretched (a `BoussinesqSimulation` on a
`StretchedGrid`). Both sims may instead be
[`MassWeightedSimulation`](@ref)s sharing one `ReferenceState` profile:
every transfer then uses the reference-mass measure
([`nest_measure`](@ref)) and the ledger records `∫ W φ dV`. `solid`
(the child's IBM mask) makes the restriction solid-aware.
"""
function step_nested!(psim::_NestSim, csim::_NestSim,
                      nest::NestedGrid, dt::Real;
                      substeps::Union{Nothing,Integer}=nothing,
                      order::Int=3,
                      flux_correction::Bool=true,
                      sponge::Union{Nothing,NestSponge}=nothing,
                      impose_interface::Bool=false,
                      scalars=(),
                      state=nothing,
                      solid=nothing,
                      record::Bool=true)
    ns = substeps === nothing ? max(nest.rx, nest.ry, nest.rz) : Int(substeps)
    ns >= 1 || error("step_nested!: substeps must be >= 1")

    s_old, s_new, s_bc = state === nothing ?
        ntuple(_ -> nest_interp_state(nest, psim, csim; scalars), 3) :
        state

    I0 = _composite_scalars(psim, csim, nest, s_old)
    m0 = composite_integral(psim, csim, nest, :momentum_x)

    interpolate_parent_state!(s_old, psim, nest; order, region=:all)
    step!(psim, dt)
    interpolate_parent_state!(s_new, psim, nest; order, region=:all)

    dtc = dt / ns
    for m in 1:ns
        _blend_state!(s_bc, s_old, s_new, (m - 0.5) / ns)
        apply_nest_boundary!(csim, s_bc; impose_interface)
        step!(csim, dtc)
        sponge === nothing || apply_nest_sponge!(csim, s_bc, sponge; dt=dtc)
    end

    # --- conservative restriction (interior faces / covered cells) --------
    restrict_state_to_parent!(psim, csim, nest, s_bc; solid)

    meas = s_bc.measure
    e_before = _interface_imbalance(psim, csim, nest, meas)
    if flux_correction
        for (pu, cu) in ((psim.u, csim.u), (psim.v, csim.v), (psim.w, csim.w))
            apply_flux_correction!(pu, compute_nest_flux_mismatch(pu, cu, nest; measure=meas),
                                   nest; measure=meas)
        end
    end
    e_after = _interface_imbalance(psim, csim, nest, meas)

    if record
        nest.ledger[:mass] = e_after
        nest.ledger[:mass_before_correction] = e_before
        I1 = _composite_scalars(psim, csim, nest, s_bc)
        for (k, v0) in I0
            nest.ledger[k] = abs(I1[k] - v0) / max(abs(v0), 1e-300)
        end
        m1 = composite_integral(psim, csim, nest, :momentum_x)
        nest.ledger[:momentum_x] = abs(m1 - m0) / max(abs(m0), 1e-300)
    end
    return (psim, csim)
end
=======
