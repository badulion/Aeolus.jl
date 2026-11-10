# The fringe region of a spatially developing boundary layer on the
# x-periodic grid, with free-stream turbulence carried in through it
# (ONE-l, #475; the `flat_plate` run type of EX-d #441, P59-d #185): the
# adapter of `Transition.FringeRegion`, `blasius_target`,
# `blasius_freestream` and `FreeStreamDisturbance`.

"""
    Fringe(fringe, target, update!, U∞)

A fringe region over the last `[fringe] length` of an x-periodic domain
(Nordström, Nordin & Henningson 1999; `Transition.FringeRegion`, ramps
`rise` and `fall`, rate `lambda_max`) that relaxes the outflow to the
Blasius boundary layer of a flat plate whose virtual leading edge lies
upstream of the inflow (`Transition.blasius_target`), so the flow that
re-enters at `x = 0` is that layer at ``Re_{δ^*_0}`` = `inflow_reynolds`.
Units are those of the inflow displacement thickness ``δ^*_0`` and the
free stream ``U_∞`` = `velocity` (``δ^*_0 = 1`` m), so
`dynamics.nu` = ``U_∞/Re_{δ^*_0}`` (a cross check).

  * **Initial state**: the Blasius layer at its distance from the leading
    edge, at least the inflow's (`Transition.set_blasius!`), set when the
    component is built.
  * **Top**: `top = "freestream"` (the default) is the free stream above a
    growing layer, the displacement velocity ``U_∞ dδ^*/dx`` with its
    return in the fringe rise (`Transition.blasius_freestream`, a
    `FreeStreamBoundary`; [`core_boundaries`](@ref)); `"wall"` keeps the
    `[boundaries]` lid. The plate is the `[boundaries] z = "noslip"` floor.
  * **Free-stream turbulence**: with `tu > 0` the target carries the
    random-mode von Kármán field of `Transition.FreeStreamDisturbance`
    (`modes` modes, integral scale `length_scale`, `seed`; the largest
    wavenumber ``2π/(\\text{kappa\\_cells}\\,\\max Δ)``) swept at ``U_∞``,
    faded in above `damping_height` × the inflow δ₉₉
    (`Transition.disturbance_fringe_update`; Brandt, Schlatter &
    Henningson 2004).

The relaxation ``−λ(x)(\\mathbf u − \\mathbf u_{\\rm target})`` is added
in stage 7 (`:body_force`) of every substage, the target refreshed to the
substage's time first. Its checkpoint state is the target and the time
of its last refresh (`Transition.DisturbanceFringeUpdate`: the first
substage of a step keeps the previous step's last target, so a restart
that recomputed it would not continue the run bit for bit). Its output `skin_friction`
is the plate's ``c_f = 2τ_w/U_∞^2`` per column (`Transition.skin_friction`).

References: Nordström, J., Nordin, N. & Henningson, D. (1999), SIAM J.
Sci. Comput. 20, 1365–1393; Brandt, L., Schlatter, P. & Henningson, D. S.
(2004), J. Fluid Mech. 517, 167–198.
"""
struct Fringe{FR, T, U} <: AbstractComponent
    fringe::FR
    target::T
    update!::U
    U∞::Float64
end

component_name(::Type{<:Fringe}) = :fringe
section(::Type{<:Fringe}) = "fringe"
schema(::Type{<:Fringe}) =
    Config.section("fringe"; doc = "A fringe region relaxing the outflow of an x-periodic flat-plate boundary layer " *
                                   "to its Blasius inflow, with free-stream turbulence (Transition.FringeRegion).") do s
        Config.option!(s, "profile", Symbol; default = :none, allowed = [:none, :blasius],
                       doc = "none (off) or blasius: the Blasius layer of a flat plate (initial state, fringe target, free-stream top)")
        Config.option!(s, "inflow_reynolds", Float64; default = 300.0, range = (0, Inf), open_lower = true, units = "1",
                       doc = "Re_δ*₀ = U∞ δ*₀/ν of the inflow layer (δ*₀ = 1 m: the length unit); dynamics.nu = velocity / inflow_reynolds")
        Config.option!(s, "velocity", Float64; default = 1.0, range = (0, Inf), open_lower = true, units = "m s^-1",
                       doc = "the free-stream velocity U∞")
        Config.option!(s, "length", Float64; default = 80.0, range = (0, Inf), open_lower = true, units = "m",
                       doc = "the fringe: the last `length` of the domain")
        Config.option!(s, "rise", Float64; default = 24.0, range = (0, Inf), open_lower = true, units = "m",
                       doc = "the ramp up of λ at the fringe start")
        Config.option!(s, "fall", Float64; default = 16.0, range = (0, Inf), open_lower = true, units = "m",
                       doc = "the ramp down of λ at the domain end")
        Config.option!(s, "lambda_max", Float64; default = 0.4, range = (0, Inf), units = "s^-1",
                       doc = "the relaxation rate on the plateau")
        Config.option!(s, "top", Symbol; default = :freestream, allowed = [:freestream, :wall],
                       doc = "freestream (the displacement velocity of the growing layer, Transition.blasius_freestream) or wall (the [boundaries] lid)")
        Config.option!(s, "tu", Float64; default = 0.0, range = (0, Inf), units = "1",
                       doc = "free-stream turbulence intensity Tu; 0: none")
        Config.option!(s, "length_scale", Float64; default = 10.0, range = (0, Inf), open_lower = true, units = "m",
                       doc = "integral scale L11 of the von Kármán spectrum")
        Config.option!(s, "modes", Int; default = 200, range = (1, typemax(Int)), doc = "random Fourier modes")
        Config.option!(s, "seed", Int; default = 1, range = (0, typemax(Int)), doc = "seed of the modes")
        Config.option!(s, "kappa_cells", Float64; default = 3.0, range = (0, Inf), open_lower = true, units = "1",
                       doc = "the largest wavenumber is 2π/(kappa_cells × max(Δx, Δy, Δz))")
        Config.option!(s, "damping_height", Float64; default = 2.0, range = (0, Inf), open_lower = true, units = "1",
                       doc = "the modes are faded in above the inflow layer: smooth_step(z / (damping_height × δ₉₉))")
    end
enabled(::Type{<:Fringe}, opts) = opts.profile !== :none

# the fringe region of the options on `mesh`
_fringe_region(opts, m) = FringeRegion(x_start = m.Lx - opts.length, x_end = m.Lx, rise = opts.rise, fall = opts.fall,
                                       λ_max = opts.lambda_max)
# the inflow's distance from the virtual leading edge: (Re_δ*₀/δ*_c)² ν/U∞ with δ*₀ = 1
_x_in(opts, ν) = (opts.inflow_reynolds / default_blasius().delta_star)^2 * ν

"""
    core_boundaries(::Type{C}, opts, mesh, ν) -> Union{Nothing,NamedTuple}

The boundary conditions of the core that an enabled component of type `C`
with the resolved options `opts` sets, `(; top = bc)`, or `nothing` (the
default: the `[boundaries]` faces). Asked before the core is built (`ν`
its viscosity); [`Fringe`](@ref)'s free-stream top is one.
"""
core_boundaries(::Type{<:AbstractComponent}, opts, mesh, ν) = nothing
function core_boundaries(::Type{<:Fringe}, opts, mesh, ν)
    opts.top === :freestream || return nothing
    x_in = _x_in(opts, ν)
    return (; top = blasius_freestream(mesh, _fringe_region(opts, mesh); x_le = -x_in, U∞ = opts.velocity, ν = ν))
end

function build(::Type{<:Fringe}, opts, domain)
    sim = domain.state; m = grid(domain); ν = sim.ν
    x_in = _x_in(opts, ν); x_le = -x_in; U∞ = Float64(opts.velocity)
    fr = _fringe_region(opts, m)
    base = blasius_target(m, fr; x_le = x_le, U∞ = U∞, ν = ν)
    update! = nothing
    target = base
    if opts.tu > 0
        fst = FreeStreamDisturbance(opts.tu, opts.length_scale, opts.modes; seed = opts.seed, U∞ = U∞, period_y = m.Ly,
                                    κ_max = 2π / (opts.kappa_cells * max(m.dx, m.dy, m.dz)))
        δ99 = blasius_solution().eta99 * sqrt(ν * x_in / U∞)
        target = deepcopy(base)
        h = opts.damping_height
        update! = disturbance_fringe_update(fst, fr, base; damping = z -> smooth_step(z / (h * δ99)))
    end
    set_blasius!(sim; x_le = x_le, U∞ = U∞, ν = ν, x_min = x_in)
    return Fringe(fr, target, update!, U∞)
end
stages(::Fringe) = (Stage(:body_force),)

# with free-stream turbulence: the target and the time of its last refresh
checkpoint_state(c::Fringe) = c.update! === nothing ? nothing :
    (t_last = c.update!.t_last[], u = copy(c.target.u.data), v = copy(c.target.v.data), w = copy(c.target.w.data))
function restore_checkpoint_state!(c::Fringe, s)
    s === nothing && return c
    c.update!.t_last[] = s.t_last
    c.target.u.data .= s.u; c.target.v.data .= s.v; c.target.w.data .= s.w
    return c
end

function run_stage!(c::Fringe, d, ::Val{:body_force}, dt)
    s = d.state
    c.update! === nothing || c.update!(c.target, s.t)
    add_fringe!(s.tu, s.tv, s.tw, s.u, s.v, s.w, c.fringe, c.target)
    return nothing
end

output_names(::Type{<:Fringe}, opts) = ("skin_friction",)
function outputs(c::Fringe)
    function cf(d)
        s = d.state; m = s.mesh
        uw = s.bcs.bottom isa NoSlipWall ? s.bcs.bottom.u_wall : 0.0
        ui = interior_view(s.u)
        out = Array{Float64}(undef, m.Nx, m.Ny)
        for j in 1:m.Ny
            out[:, j] .= skin_friction(ui[:, j, :], m.dz, s.ν; U_e = fill(c.U∞, m.Nx), u_wall = uw)
        end
        return out
    end
    return OutputVariable[OutputVariable("skin_friction", "1", "skin friction coefficient 2 tau_w / U_inf^2 of the plate",
                                         :xy, cf)]
end

# --- the fused sweep (#533) ---------------------------------------------------
# The relaxation of `add_fringe!` (the damping profile along x) as a sweep
# term; the target is refreshed to the substage's time before the sweep, as
# `run_stage!` does before its pass.
struct FringeSweep{F<:Fringe, S, X<:FiniteVolume.SweepTerm} <: FiniteVolume.SweepTerm
    c::F
    sim::S
    term::X
end
function FiniteVolume.sweep_term(c::Fringe, d)
    g = d.state.mesh
    coords = (xnodes(g, FaceX()), xnodes(g, FaceY()), xnodes(g, FaceZ()))
    return FringeSweep(c, d.state, FiniteVolume.DampingTerm(c.fringe.profile, Val(1), coords, c.target, g))
end
FiniteVolume.footprint(t::FringeSweep) = FiniteVolume.footprint(t.term)
FiniteVolume.sweep_inputs(t::FringeSweep) = FiniteVolume.sweep_inputs(t.term)
FiniteVolume.prepare!(t::FringeSweep, sim, Δt) =
    (t.c.update! === nothing || t.c.update!(t.c.target, t.sim.t); nothing)
@inline FiniteVolume.add_row!(t::FringeSweep, buf, a, b, s, j, k, ii, st) =
    FiniteVolume.add_row!(t.term, buf, a, b, s, j, k, ii, st)
@inline FiniteVolume.add_tendency!(t::FringeSweep, tb, i, j, k, st) =    # the per-point form (#533 step 8)
    FiniteVolume.add_tendency!(t.term, tb, i, j, k, st)
