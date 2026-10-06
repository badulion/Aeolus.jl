# The θ top boundary of the case (N-parent-b, #254).
#
# The 200 m child domain ends inside the boundary layer's capping
# stratification, not at a lid: above it the free atmosphere is warmer
# (θ_lapse = 3 K/km in the flagship). `cfg.θ_top` chooses the θ condition
# of the top face:
#
#   :initial_gradient  (default) FluxBC(Γ) with Γ = ∂θ/∂z of the initial
#                      profile at the top face — the fixed-gradient top of
#                      PALM and DALES (Heus et al. 2010, Eq. 39). Diffusion
#                      carries κΓ through the face (warm air above: into
#                      the domain);
#   :noflux            the closed lid of the case before #254;
#   a number           FluxBC(value), K/m.
#
# The flux the top face carries is heat the ledger must see. The dynamics
# books it as (:dynamics, :top_flux) through `add_heat!`: the θ diffusivity
# at the top face (molecular ν plus the SGS closure's vertical κ, face
# averaged as the kernels average it) is captured in the body-force hook of
# every dynamics stage, where the closure has just filled it, and the flux
# of the accepted (sub)step is booked once it is known to be accepted. In
# the nested configuration the open top (N-open-b, N-berlin-nest #265)
# replaces it.

"""
    θ_top_gradient(cfg; H = cfg.extent[3], dz = H / cfg.size[3]) -> K/m

∂θ/∂z of the initial profile [`initial_θ`](@ref) of `cfg` at the top face,
as the centred difference across it, `(θ(z_top + Δz/2) − θ(z_top − Δz/2))/Δz`,
the difference the ghost cell of a `FluxBC` top reproduces (exact for the
piecewise-linear profile of the case when the inversion is below the top
level). `H` and `dz` (the top and the thickness of the top cell) default
to the child's grid; the nesting parent (#259) passes its own 3 km top
and top-cell thickness.
"""
function θ_top_gradient(cfg::CaseConfig; H::Real = cfg.extent[3], dz::Real = H / cfg.size[3])
    θi = initial_θ(cfg)
    return (θi(0.0, 0.0, H + dz / 2) - θi(0.0, 0.0, H - dz / 2)) / dz
end

"""
    θ_top_bc(cfg) -> AbstractBC

The θ boundary condition of the top face for `cfg.θ_top`: `FluxBC(Γ)` with
`Γ =` [`θ_top_gradient`](@ref)`(cfg)` for `:initial_gradient`, `NoFlux()`
for `:noflux`, `FluxBC(x)` for a number `x` (K/m).
"""
function θ_top_bc(cfg::CaseConfig)
    t = cfg.θ_top
    t isa Real && return FluxBC(Float64(t))
    t === :initial_gradient && return FluxBC(θ_top_gradient(cfg))
    t === :noflux && return NoFlux()
    error("CaseConfig.θ_top must be :initial_gradient, :noflux or a gradient in K/m, got $(repr(t))")
end

"""
    TopHeatFlux

The heat flux through a fixed-gradient θ top, for the air-heat ledger:
`Γ` (K/m), `w` = ρ̄ c_p Π̄ of the top level (J m⁻³ K⁻¹), `rate` — the
flux (W m⁻² of domain) with the diffusivity of the last dynamics stage,
`Γ w ⟨ν + ½(κ_N + κ_{N+1})⟩` — and `total` (J m⁻², the run's sum, for
reports).
"""
mutable struct TopHeatFlux
    Γ::Float64
    w::Float64
    rate::Float64
    total::Float64
end

"The top-flux meter of `case` (`nothing` unless the θ top is a `FluxBC`)."
function top_heat_flux(cfg::CaseConfig, sim, ref)
    bc = sim.θ_bcs.top
    bc isa FluxBC || return nothing
    Nz = length(ref.ρ̄)
    return TopHeatFlux(Float64(bc.f), ref.ρ̄[Nz] * ref.cp * ref.π̄[Nz], 0.0, 0.0)
end

# The vertical SGS diffusivity of θ the core used in this stage: the
# closure's fields (`sgs_fields`), or the built-in Smagorinsky's `sim.κt`
# (zero when the closure is off).
_θ_vertical_κ(sim, closure) = closure === nothing ? sim.κt : sgs_fields(closure, sim).κv

"""
    capture_top_flux!(tf::TopHeatFlux, sim, closure) -> tf

Record the heat flux (W m⁻²) the top face carries in the current dynamics
stage: `Γ ρ̄c_pΠ̄ ⟨ν + ½(κ_v[N] + κ_v[N+1])⟩` over the columns. Called from
the body-force hook, after the closure filled its coefficients.
"""
function capture_top_flux!(tf::TopHeatFlux, sim, closure)
    m = sim.mesh
    κ = _θ_vertical_κ(sim, closure).data
    kN = m.Hz + m.Nz
    s = 0.0
    @inbounds for j in m.Hy+1:m.Hy+m.Ny, i in m.Hx+1:m.Hx+m.Nx
        s += 0.5 * (κ[i, j, kN] + κ[i, j, kN+1])
    end
    ν = sim.has_viscosity ? Float64(sim.ν) : 0.0
    tf.rate = tf.Γ * tf.w * (ν + s / (m.Nx * m.Ny))
    return tf
end

"""
    book_top_flux!(case, h) -> case

Book the heat the top face carried over an accepted dynamics (sub)step of
`h` seconds as `(:dynamics, :top_flux)` in the air-heat ledger, at the top
level. No-op without a fixed-gradient top.
"""
function book_top_flux!(case, h::Real)
    tf = case.top
    tf === nothing && return case
    J = tf.rate * h
    tf.total += J
    hl = case.ledger.heat
    hl.active && add_heat!(hl, :top_flux, J; k = case.mesh.Nz)
    return case
end
