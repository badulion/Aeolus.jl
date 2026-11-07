# The output catalogue of a domain (ONE-c, #452): the core's variables
# and every component's `outputs`, from which `[output] variables`
# selects; `Aeolus.Output` writes them.

"""
    CORE_OUTPUTS

The names in the core's output catalogue ([`core_outputs`](@ref)):

| name | dims | |
| :-- | :-- | :-- |
| `u`, `v`, `w`, `p`, `theta`, `qv` | 3-D | the prognostic fields (`qv` with moisture), at their own staggered positions |
| `u_profile`, `v_profile`, `theta_profile`, `qv_profile` | z | horizontal means per level |
| `heat_flux_profile` | z | the total (resolved + SGS) upward θ flux, at the cell centres (the mean of its two faces; the bottom face holds the wall flux the components apply, [`wall_theta_flux`](@ref)), `Operators.total_heat_flux_profile` |
| `w_variance_profile`, `resolved_tke_profile` | z | the plane variance of w at the cell centres (`Operators.w_variance_profile`) and the resolved TKE per level (`Operators.resolved_tke_profile`) |
| `u_mean`, `v_mean`, `w_mean`, `theta_mean`, `qv_mean` | scalar | domain means (time series) |
| `temperature`, `temperature_profile`, `temperature_mean` | 3-D, z, scalar | the absolute air temperature ``T = θ\\,Π̄(z)`` (K, CF `air_temperature`) with the Exner function of the domain's thermodynamic reference ([`thermodynamic_reference`](@ref), #536): its 3-D field, horizontal means per level and domain mean |
| `budget_energy`, `budget_water`, `budget_momentum`, `budget_tracer` | scalar group | the measured budgets ([`with_budgets`](@ref)): cumulative storage, every attributed term and the closure (time series) |
| `u_squared` | 3-D | ``u^2`` at the x faces: its time mean (`averaging = "mean"`) less the square of the mean `u` is the variance of u |
| `w_rms` | scalar | the root-mean-square of w over the interior faces (m s⁻¹): the amplitude of a growing mode |
| `turbulent_fraction_profile` | z | the plane fraction γ of turbulent cells (`Transition.turbulent_fraction`, the vorticity-fluctuation indicator) |
| `shear_production_profile`, `buoyancy_production_profile` | z | the resolved (+ SGS) shear production ``P = −⟨u'w'⟩∂_zU − ⟨v'w'⟩∂_zV`` and buoyancy flux ``B = (g/θ_0)⟨w'θ'⟩`` per level (`Transition.flux_richardson`; ``Ri_f = −B/P``); `buoyancy_production_profile` on the Boussinesq core needs `θ₀ > 0` |
| `vorticity_y` | 3-D | the y component of the vorticity at the cell centres (s⁻¹) |
| `kinetic_energy` | scalar | the resolved kinetic energy ``½⟨u^2 + v^2 + w^2⟩`` of the face values (m² s⁻²) |
| `dissipation`, `dissipation_sgs` | scalar | the dissipation of the resolved energy, molecular plus SGS, and its SGS part (`Operators.dissipation_rate`, m² s⁻³) |
| `energy_spectrum` | k | the shell-summed energy spectrum ``E(k)`` (`Operators.isotropic_energy_spectrum`, m³ s⁻²; a triply periodic regular grid) |

`qv*` need moisture; `budget_water` needs moisture, `budget_tracer` a
component that carries a tracer, `budget_energy` on the Boussinesq core
`θ₀ > 0`; `temperature*` a thermodynamic reference (every domain built from
a configuration has one; a bare Boussinesq core made in Julia does not,
and its catalogue leaves them out).
"""
const CORE_OUTPUTS = ("u", "v", "w", "p", "theta", "qv",
                      "u_profile", "v_profile", "theta_profile", "qv_profile",
                      "heat_flux_profile", "w_variance_profile", "resolved_tke_profile",
                      "u_mean", "v_mean", "w_mean", "theta_mean", "qv_mean",
                      "budget_energy", "budget_water", "budget_momentum", "budget_tracer",
                      "kinetic_energy", "dissipation", "dissipation_sgs", "energy_spectrum",
                      "u_squared", "w_rms", "turbulent_fraction_profile", "shear_production_profile", "buoyancy_production_profile",
                      "vorticity_y", "temperature", "temperature_profile", "temperature_mean")

"""
    DEFAULT_OUTPUTS

The variables a run writes when `[output] variables` is not given: `u`,
`v`, `w`, `theta` and the time series of the domain means of u, v, w and
θ (the design's default set).
"""
const DEFAULT_OUTPUTS = ("u", "v", "w", "theta", "u_mean", "v_mean", "w_mean", "theta_mean")

"""
    OUTPUT_RENAMES

Hints for output names whose meaning changed, appended to the problem of
a selection no enabled component publishes: since #536
`surface_temperature` is an absolute (skin) temperature, published by
`[surface] model = "most"` and `"tiles"`; the surface potential
temperature is `surface_theta`.
"""
const OUTPUT_RENAMES = Dict(
    "surface_temperature" => "; since #536 surface_temperature is the absolute surface (skin) temperature of " *
                             "[surface] model = \"most\" or \"tiles\", and the prescribed surface potential " *
                             "temperature (its old meaning) is surface_theta")

"The budget quantities each `budget_*` output variable turns on."
const BUDGET_OUTPUTS = Dict("budget_energy" => (:energy,), "budget_water" => (:water,),
                            "budget_momentum" => (:momentum_x, :momentum_y), "budget_tracer" => (:tracer,))

"""
    output_names(::Type{C}, opts) -> Tuple{Vararg{String}}

The names in the output catalogue that a component of type `C` with the
resolved options `opts` publishes: [`outputs`](@ref) of the built
component has exactly these (the contract tests check it). The
configuration is validated against them before anything is built, so
selecting a variable whose component is off is an error. Default: none.
"""
output_names(::Type{<:AbstractComponent}, opts) = ()

# plane sums per level of a field's interior, in the canonical order
_level_sums(f) = plane_partials!(zeros(f.mesh.Nz), identity, interior(f.mesh), f.data)

function _profile(f)
    m = f.mesh
    return _level_sums(f) ./ (m.Nx * m.Ny)
end

# the resolved kinetic energy per unit volume of the face values (the energy
# the solver carries; Parseval with `isotropic_energy_spectrum`)
function _kinetic_energy(sim)
    n = sim.mesh.Nx * sim.mesh.Ny * sim.mesh.Nz
    return 0.5 * (sum(abs2, interior_view(sim.u)) + sum(abs2, interior_view(sim.v)) + sum(abs2, interior_view(sim.w))) / n
end

# the dissipation of the resolved energy: the strain of the current state with
# the molecular ν and the closure's last eddy viscosity (none without an :sgs component)
function _dissipation(d)
    s = d.state
    for f in (s.u, s.v, s.w)
        fill_halos!(f, s.bcs)
    end
    return dissipation_rate(s.u, s.v, s.w, s.ν, sgs_closure(d) === nothing ? nothing : s.νt)
end

# The total (resolved + SGS) upward θ flux at the w faces: the closure's last
# vertical diffusivity (the core's κt without a closure), the bottom face the
# wall flux the domain's components apply (`Operators.total_heat_flux_profile`)
function _heat_flux_faces(d)
    s = d.state
    fill_halos!(s.w, s.bcs); fill_halos!(s.θ, s.θ_bcs)
    clo = sgs_closure(d)
    κv = clo === nothing ? s.κt : sgs_fields(clo, s).κv
    fill_halos!(κv, s.bcs)
    return total_heat_flux_profile(s.w, s.θ, κv; surface = first(_sum_wall_flux(d.components, d))).total
end
# a face profile (Nz + 1) at the cell centres: the mean of the two faces
_centred(f) = [(f[k] + f[k+1]) / 2 for k in 1:length(f)-1]

# the velocity halos, filled (the transition diagnostics read neighbours)
_fill_velocity!(s) = (fill_halos!(s.u, s.bcs); fill_halos!(s.v, s.bcs); fill_halos!(s.w, s.bcs); s)

# the resolved (+ SGS) production terms of `Transition.flux_richardson`; the
# buoyancy reference: θ₀ of the Boussinesq core, the base state's θ̄ at the
# first level on the anelastic one
function _flux_richardson(d)
    s = _fill_velocity!(d.state)
    fill_halos!(s.θ, s.θ_bcs)
    θ0 = s isa BoussinesqSimulation ? s.θ₀ : s.ref.θ̄[1]
    sgs = sgs_closure(d) !== nothing
    sgs && fill_halos!(s.νt, s.bcs)
    return flux_richardson(s.u, s.v, s.w, s.θ; θ₀ = θ0, g = s.g_accel, νt = sgs ? s.νt : nothing, Pr_t = s.Pr_t)
end

# ω_y at the cell centres (`Operators._vorticity_at`; on a StretchedGrid with
# the per-level vertical metrics of the Smagorinsky kernel, #505)
function _vorticity_y(d)
    s = _fill_velocity!(d.state); m = s.mesh
    gm = FiniteVolume.grid_metrics(m)
    ω = Array{Float64}(undef, m.Nx, m.Ny, m.Nz)
    for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
        K = m.Hz + k
        ω[i, j, k] = Operators._vorticity_at(s.u.data, s.v.data, s.w.data, m.Hx + i, m.Hy + j, K,
                                             1 / m.dx, 1 / m.dy, FiniteVolume._idz(gm, K),
                                             FiniteVolume._idz_off(gm, K))[2]
    end
    return ω
end

# the domain mean: level means weighted by the level thickness (of the
# field times a per-level factor `w`, e.g. the reference Exner function)
function _domain_mean(f, w = nothing)
    m = f.mesh
    dz = hasproperty(m, :dzc) ? [m.dzc[m.Hz+k] for k in 1:m.Nz] : fill(Float64(m.dz), m.Nz)
    S = _level_sums(f)
    acc = 0.0
    @inbounds for k in eachindex(S)
        acc += S[k] * dz[k] * (w === nothing ? 1.0 : w[k])
    end
    return acc / (m.Nx * m.Ny * sum(dz))
end

# the absolute air temperature T = θ Π̄_k of the interior cells (#536)
function _temperature(d)
    π̄ = required_reference(d).π̄
    θ = interior_view(d.state.θ)
    return θ .* reshape(π̄, 1, 1, length(π̄))
end

# the cumulative terms of a measured budget, by "<stage>__<name>", with the
# storage and the closure (one StageBudget, or the sum of two: momentum x, y
# are written apart)
function _budget_group(sb::StageBudget)
    b = stage_budget(sb)
    g = Dict{String,Float64}("storage" => b.storage, "closure" => b.closure)
    for ((st, n), v) in b.terms
        g[string(st, "__", n)] = v
    end
    return g
end

"""
    core_outputs(d::Domain) -> Vector{OutputVariable}

The core's output catalogue of domain `d` ([`CORE_OUTPUTS`](@ref)): the
variables its state has (the moisture ones only with moisture, the budget
ones only for the quantities its budgets measure).
"""
function core_outputs(d::Domain)
    sim = d.state
    moist = hasproperty(sim, :has_moisture) && sim.has_moisture
    iv(f) = interior_view(f)
    vs = OutputVariable[
        OutputVariable("u", "m s-1", "x wind (at the x faces)", :xyz, d -> iv(d.state.u)),
        OutputVariable("v", "m s-1", "y wind (at the y faces)", :xyz, d -> iv(d.state.v)),
        OutputVariable("w", "m s-1", "upward wind (at the z faces)", :xyz, d -> iv(d.state.w)),
        OutputVariable("p", "m2 s-2", "kinematic pressure", :xyz, d -> iv(d.state.p)),
        OutputVariable("theta", "K", "potential temperature", :xyz, d -> iv(d.state.θ)),
        OutputVariable("u_profile", "m s-1", "horizontal mean of u", :z, d -> _profile(d.state.u)),
        OutputVariable("v_profile", "m s-1", "horizontal mean of v", :z, d -> _profile(d.state.v)),
        OutputVariable("theta_profile", "K", "horizontal mean of theta", :z, d -> _profile(d.state.θ)),
        OutputVariable("heat_flux_profile", "K m s-1", "total (resolved + SGS) upward heat flux, at the cell centres",
                       :z, d -> _centred(_heat_flux_faces(d))),
        OutputVariable("w_variance_profile", "m2 s-2", "horizontal variance of w at the cell centres", :z,
                       d -> w_variance_profile(d.state.w)),
        OutputVariable("resolved_tke_profile", "m2 s-2", "resolved turbulent kinetic energy per level (fluctuations about the plane means)",
                       :z, d -> resolved_tke_profile(d.state.u, d.state.v, d.state.w)),
        OutputVariable("u_mean", "m s-1", "domain mean of u", :scalar, d -> _domain_mean(d.state.u)),
        OutputVariable("v_mean", "m s-1", "domain mean of v", :scalar, d -> _domain_mean(d.state.v)),
        OutputVariable("w_mean", "m s-1", "domain mean of w", :scalar, d -> _domain_mean(d.state.w)),
        OutputVariable("theta_mean", "K", "domain mean of theta", :scalar, d -> _domain_mean(d.state.θ)),
        OutputVariable("u_squared", "m2 s-2", "square of the x wind (at the x faces)", :xyz, d -> iv(d.state.u) .^ 2),
        OutputVariable("w_rms", "m s-1", "root-mean-square of w over the interior faces", :scalar,
                       d -> (wi = iv(d.state.w); sqrt(sum(abs2, wi) / length(wi)))),
        OutputVariable("turbulent_fraction_profile", "1", "plane fraction of turbulent cells (vorticity-fluctuation indicator)",
                       :z, d -> (s = _fill_velocity!(d.state); turbulent_fraction(s.u, s.v, s.w))),
        OutputVariable("shear_production_profile", "m2 s-3", "shear production of the resolved TKE, -<u'w'> dU/dz - <v'w'> dV/dz",
                       :z, d -> _flux_richardson(d).P),
        OutputVariable("buoyancy_production_profile", "m2 s-3", "buoyancy flux of the resolved TKE, (g/theta0) <w'theta'>",
                       :z, d -> _flux_richardson(d).B),
        OutputVariable("vorticity_y", "s-1", "y component of the vorticity at the cell centres", :xyz, _vorticity_y),
        OutputVariable("kinetic_energy", "m2 s-2", "resolved kinetic energy, domain mean of (u^2 + v^2 + w^2)/2",
                       :scalar, d -> _kinetic_energy(d.state)),
        OutputVariable("dissipation", "m2 s-3", "dissipation of the resolved kinetic energy, molecular + SGS",
                       :scalar, d -> (r = _dissipation(d); r.resolved + r.sgs)),
        OutputVariable("dissipation_sgs", "m2 s-3", "SGS part of the dissipation (the eddy viscosity's transfer)",
                       :scalar, d -> _dissipation(d).sgs),
        OutputVariable("energy_spectrum", "m3 s-2", "shell-summed kinetic-energy spectrum E(k)", :k,
                       d -> (s = d.state; last(isotropic_energy_spectrum(s.u, s.v, s.w))))]
    if thermodynamic_reference(d) !== nothing
        append!(vs, OutputVariable[
            OutputVariable("temperature", "K", "air temperature, theta times the reference Exner function", :xyz,
                           _temperature; standard_name = "air_temperature"),
            OutputVariable("temperature_profile", "K", "horizontal mean of the air temperature", :z,
                           d -> _profile(d.state.θ) .* required_reference(d).π̄; standard_name = "air_temperature"),
            OutputVariable("temperature_mean", "K", "domain mean of the air temperature", :scalar,
                           d -> _domain_mean(d.state.θ, required_reference(d).π̄); standard_name = "air_temperature")])
    end
    if moist
        append!(vs, OutputVariable[
            OutputVariable("qv", "kg kg-1", "water vapour mixing ratio", :xyz, d -> iv(d.state.qv)),
            OutputVariable("qv_profile", "kg kg-1", "horizontal mean of qv", :z, d -> _profile(d.state.qv)),
            OutputVariable("qv_mean", "kg kg-1", "domain mean of qv", :scalar, d -> _domain_mean(d.state.qv))])
    end
    b = d.budgets
    if b !== nothing
        units = Dict(:energy => "J m-2", :water => "kg m-2", :tracer => "kg m-2", :momentum_x => "kg m-1 s-1",
                     :momentum_y => "kg m-1 s-1")
        for (name, qs) in BUDGET_OUTPUTS
            all(q -> q in b.quantities, qs) || continue
            if length(qs) == 1
                q = only(qs)
                push!(vs, OutputVariable(name, units[q], "$(q) budget of the air, cumulative since the start",
                                         :scalar, d -> _budget_group(d.budgets.budgets.measured[q]); group = true))
            else
                push!(vs, OutputVariable(name, units[first(qs)], "momentum budget of the air (x, y), cumulative since the start",
                                         :scalar, d -> merge(
                                             Dict("x__" * k => v for (k, v) in _budget_group(d.budgets.budgets.measured[:momentum_x])),
                                             Dict("y__" * k => v for (k, v) in _budget_group(d.budgets.budgets.measured[:momentum_y])));
                                         group = true))
            end
        end
    end
    return vs
end

"""
    domain_catalogue(d::Domain) -> Vector{OutputVariable}

The output catalogue of domain `d`: the core's ([`core_outputs`](@ref))
and every component's ([`outputs`](@ref)), checked for duplicate names.
"""
domain_catalogue(d::Domain) =
    check_catalogue(vcat(core_outputs(d), OutputVariable[v for c in d.components for v in outputs(c)]))

"""
    output_problems(names, types_opts; moisture, theta0, core, published = ()) -> Vector{String}

The problems of the `[output] variables` selection `names` against the
catalogue of a domain whose enabled components are the pairs
`types_opts` (`C => opts`): a name no catalogue has, a variable whose
component is off (or that needs moisture), and the requirements of the
budget variables (`budget_tracer`: a component publishing `tracer`).
`published` names what components built outside `types_opts` publish
(the model's couplings).
"""
function output_problems(names, types_opts; moisture::Bool, theta0::Real, core::Symbol, published = ())
    probs = String[]
    comp = Dict{String,Any}()
    for (C, opts) in types_opts, n in output_names(C, opts)
        comp[n] = C
    end
    for n in published
        comp[n] = nothing
    end
    for n in names
        if n in CORE_OUTPUTS
            startswith(n, "qv") && !moisture && push!(probs, "$n is written only with dynamics.moisture = true")
            n == "budget_water" && !moisture && push!(probs, "budget_water needs dynamics.moisture = true")
            n == "budget_tracer" && !haskey(comp, "tracer") &&
                push!(probs, "budget_tracer needs a component that carries a tracer (one that publishes \"tracer\")")
            n == "buoyancy_production_profile" && core === :boussinesq && theta0 <= 0 &&
                push!(probs, "buoyancy_production_profile needs dynamics.theta0 > 0 (the buoyancy g/theta0)")
            n == "budget_energy" && core === :boussinesq && theta0 <= 0 &&
                push!(probs, "budget_energy needs dynamics.theta0 > 0 (the Boussinesq reference density p_ref / (R_d theta0))")
        elseif !haskey(comp, n)
            push!(probs, "$n: no enabled component publishes it (the core's: $(join(CORE_OUTPUTS, ", ")))" *
                         get(OUTPUT_RENAMES, n, ""))
        end
    end
    return probs
end
