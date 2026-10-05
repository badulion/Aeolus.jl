# Nesting: composite diagnostics and step_nested! (moved verbatim from
# AdaptiveMesh.jl, N-nest-a #250).

# =========================================================================
# Composite diagnostics
# =========================================================================

"""
    composite_integral(parent_sim, child_sim, nest, quantity) -> Float64

Volume integral of `quantity` over the *composite* grid: parent cells not
covered by the nest plus all child cells. `quantity ∈ (:mass, :theta,
:momentum_x)`. For the Boussinesq system `:mass` is `ρ₀ ∫dV`, a constant —
mass conservation across a nest interface is therefore diagnosed by
[`nest_conservation_error`](@ref) from the interface flux imbalance
instead.
"""
function composite_integral(psim::BoussinesqSimulation, csim::BoussinesqSimulation,
                            nest::NestedGrid, quantity::Symbol)
    pm = nest.parent_mesh; cm = nest.child_mesh
    Vp = pm.dx * pm.dy * pm.dz; Vc = cm.dx * cm.dy * cm.dz
    quantity === :mass && return (pm.Nx*pm.Ny*pm.Nz - nest.nx*nest.ny*nest.nz) * Vp +
                                 cm.Nx*cm.Ny*cm.Nz * Vc
    cx, cy, cz = nest_covered_range(nest)
    total = 0.0
    if quantity === :theta
        pd = psim.θ.data
        @inbounds for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
            (i in cx) && (j in cy) && (k in cz) && continue
            total += pd[pm.Hx+i, pm.Hy+j, pm.Hz+k] * Vp
        end
        cd = csim.θ.data
        @inbounds for k in 1:cm.Nz, j in 1:cm.Ny, i in 1:cm.Nx
            total += cd[cm.Hx+i, cm.Hy+j, cm.Hz+k] * Vc
        end
    elseif quantity === :momentum_x
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
    else
        error("composite_integral: unknown quantity $quantity " *
              "(:mass, :theta, :momentum_x)")
    end
    return total
end

"Interface mass-flux imbalance, normalized by (velocity scale × interface area)."
function _interface_imbalance(psim::BoussinesqSimulation, csim::BoussinesqSimulation,
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
  * `:theta`, `:momentum_x` — relative change of
    [`composite_integral`](@ref) over the step.

If no step has been recorded, `:mass` is evaluated instantaneously and the
other quantities raise an error.
"""
function nest_conservation_error(psim::BoussinesqSimulation, csim::BoussinesqSimulation,
                                 nest::NestedGrid, quantity::Symbol)
    haskey(nest.ledger, quantity) && return nest.ledger[quantity]
    quantity === :mass && return _interface_imbalance(psim, csim, nest)
    error("nest_conservation_error: no recorded value for :$quantity — " *
          "run step_nested! first.")
end

# =========================================================================
# Time-stepping coordination
# =========================================================================

"""
    step_nested!(parent_sim, child_sim, nest, dt; substeps=nothing, order=3,
                 flux_correction=true, sponge=nothing, impose_interface=false,
                 state=nothing)

One parent step of size `dt` plus `substeps` child steps of size
`dt/substeps` (default `max(rx,ry,rz)`, which keeps the child CFL equal to
the parent's).

Sequence:

 1. interpolate the parent state at time `t` onto the child mesh;
 2. `step!(parent_sim, dt)`; interpolate the new parent state;
 3. for each child substep, **linearly interpolate the parent state in
    time** to the substep midpoint, write it into the child halos,
    `step!` the child, then apply the sponge (if any);
 4. conservatively restrict the child interior back onto the parent;
 5. compute the interface flux mismatch and apply the flux correction.

Conservation diagnostics for `:mass`, `:theta` and `:momentum_x` are
recorded in `nest.ledger`; read them with
[`nest_conservation_error`](@ref). Returns `(parent_sim, child_sim)`.

`state` may be a pre-allocated `NTuple{3,NestInterpState}` to avoid
re-allocating the boundary buffers every step.
"""
function step_nested!(psim::BoussinesqSimulation, csim::BoussinesqSimulation,
                      nest::NestedGrid, dt::Real;
                      substeps::Union{Nothing,Integer}=nothing,
                      order::Int=3,
                      flux_correction::Bool=true,
                      sponge::Union{Nothing,NestSponge}=nothing,
                      impose_interface::Bool=false,
                      state=nothing,
                      record::Bool=true)
    ns = substeps === nothing ? max(nest.rx, nest.ry, nest.rz) : Int(substeps)
    ns >= 1 || error("step_nested!: substeps must be >= 1")

    θ0 = composite_integral(psim, csim, nest, :theta)
    m0 = composite_integral(psim, csim, nest, :momentum_x)

    s_old, s_new, s_bc = state === nothing ?
        (nest_interp_state(nest), nest_interp_state(nest), nest_interp_state(nest)) :
        state

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
    restrict_to_parent!(psim.θ, csim.θ, nest)
    restrict_to_parent!(psim.u, csim.u, nest)
    restrict_to_parent!(psim.v, csim.v, nest)
    restrict_to_parent!(psim.w, csim.w, nest)

    e_before = _interface_imbalance(psim, csim, nest)
    if flux_correction
        apply_flux_correction!(psim.u, compute_nest_flux_mismatch(psim.u, csim.u, nest), nest)
        apply_flux_correction!(psim.v, compute_nest_flux_mismatch(psim.v, csim.v, nest), nest)
        apply_flux_correction!(psim.w, compute_nest_flux_mismatch(psim.w, csim.w, nest), nest)
    end
    e_after = _interface_imbalance(psim, csim, nest)

    if record
        nest.ledger[:mass] = e_after
        nest.ledger[:mass_before_correction] = e_before
        θ1 = composite_integral(psim, csim, nest, :theta)
        m1 = composite_integral(psim, csim, nest, :momentum_x)
        nest.ledger[:theta] = abs(θ1 - θ0) / max(abs(θ0), 1e-300)
        nest.ledger[:momentum_x] = abs(m1 - m0) / max(abs(m0), 1e-300)
    end
    return (psim, csim)
end
