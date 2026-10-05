# A passive pollutant from three stacks: centred advection, linear decay
# (Aeolus.Chemistry, e-folding 1 h) and a day-boosted point source.
#
# KNOWN: the centred forward-Euler update is not positivity preserving;
# the result is clipped to [0, 100]. The scalar-advection issue can move
# this tracer onto the `:transport` stage.

"""
    PollutantCoupling

Additive coupling: registers the tracer `:pollutant` (output, not on the
`:transport` stage) and advances it every step in `:tracers` with
`advect_centered!`, `apply_linear_chemistry!` (τ = `tau`, 3600 s) and
the stack sources (`0.5 × dayboost` per second, dayboost 1 between 07
and 20 h, else 0.3), then the SGS diffusion with `κt` of the
`:scalar_advection` coupling ([`sgs_diffuse!`](@ref), #177). `stacks` are `(x, y, z)` in metres relative to the
OSM anchor (`cfg.lat0/lon0`); stacks outside the domain are dropped.
"""
struct PollutantCoupling{F} <: AbstractCoupling
    tend::F
    stacks::Vector{NTuple{3,Int}}
    tau::Float64
end

coupling_name(::Type{<:PollutantCoupling}) = :pollutant
modules_used(::Type{<:PollutantCoupling}) = (:Chemistry, :Operators)
default_options(::Type{<:PollutantCoupling}) =
    (stacks = [(0.0, 0.0, 60.0), (-150.0, 150.0, 25.0), (200.0, -100.0, 40.0)], tau = 3600.0)
stages(::Type{<:PollutantCoupling}) = ((:tracers, 10, EVERY_STEP),)

function build(::Type{<:PollutantCoupling}, case, o)
    mesh = case.mesh
    f = ensure_tracer!(case, :pollutant; units = "1", long_name = "passive pollutant tracer")
    set!(f, (x, y, z) -> 0.0)
    ox, oy = case.cfg.offset
    ijk = NTuple{3,Int}[]
    for (sx, sy, sz) in o.stacks
        cx = sx - ox + mesh.Lx / 2        # domain coordinates
        cy = sy - oy + mesh.Ly / 2
        (0 <= cx < mesh.Lx && 0 <= cy < mesh.Ly && 0 <= sz < mesh.Lz) || continue
        push!(ijk, (clamp(round(Int, cx / mesh.dx), 1, mesh.Nx),
                    clamp(round(Int, cy / mesh.dy), 1, mesh.Ny),
                    clamp(round(Int, sz / mesh.dz), 1, mesh.Nz)))
    end
    return PollutantCoupling(Field(CellCenter(), mesh), ijk, Float64(o.tau))
end

function run_stage!(c::PollutantCoupling, case, ::Val{:tracers}, dt)
    sim, mesh = case.sim, case.mesh
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    pollutant, tp = tracer(case, :pollutant), c.tend
    hour = case.clock.hour
    fill_halos!(pollutant, FieldBoundaryConditions(:periodic))
    fill!(tp.data, 0)
    advect_centered!(tp, sim.u, sim.v, sim.w, pollutant)
    apply_linear_chemistry!(tp, pollutant, c.tau)
    dayboost = (hour > 7 && hour < 20) ? 1.0 : 0.3
    for (ci, cj, ck) in c.stacks
        tp.data[Hx+ci, Hy+cj, Hz+ck] += 0.5 * dayboost
    end
    interior_view(pollutant) .+= dt .* view(tp.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    # SGS diffusion with κt, the helper of the :transport stage (#177)
    d = tracer_sgs_diffusion(case)
    d === nothing || sgs_diffuse!(d, case, pollutant, dt; κv = sgs_vertical_diffusivity(case))
    clamp!(interior_view(pollutant), 0.0, 100.0)
    return nothing
end

function smoke_check(c::PollutantCoupling, case)
    p = interior_view(tracer(case, :pollutant))
    return ["pollutant within the [0, 100] clip" => all(x -> 0 <= x <= 100, p),
            "emits iff a stack is in the window" => (sum(p) > 0) == !isempty(c.stacks)]
end

register!(PollutantCoupling)
