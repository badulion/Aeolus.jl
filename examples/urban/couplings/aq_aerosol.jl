# Toy two-moment aerosol (mass + number): coagulation, sedimentation and
# dry deposition.
#
# KNOWN (the activation issue unifies it with AerosolCloud.AerosolReservoir):
# the state starts at zero, has no source, is not advected and not written
# out, so it stays zero. Its deposition velocities come from
# `deposition_velocity` in aq_chem.jl.

# ---------------------------------------------------------------------------
# A4 — Two-moment aerosol prognostics
# ---------------------------------------------------------------------------

"""
    AerosolState(mesh) -> NamedTuple(q_a, N_a)

Two-moment aerosol state:

  * `q_a::Field{CellCenter}` — aerosol mass mixing ratio [kg/kg-like].
  * `N_a::Field{CellCenter}` — aerosol number concentration [#/kg-like].

Both are initialized to zero; the caller populates them (e.g. with an
initial urban background) and advects them alongside the chemistry
species.
"""
function AerosolState(mesh)
    q_a = Field(CellCenter(), mesh)
    N_a = Field(CellCenter(), mesh)
    return (q_a = q_a, N_a = N_a)
end

"""
    apply_aerosol_step!(state, dt; V_sed = 0.01, sfc = nothing, dz = 1.0,
                        k_coag = nothing)

Advance the aerosol state by `dt`:

  * First-order coagulation on number:  `dN/dt = -k · N²`, with
    `k = k_coag` if given, otherwise `1e-13 · V_sed`. Mass `q_a` is
    preserved (particles merge, mass conserved).
  * Mass-conserving sedimentation of both `q_a` and `N_a` using a
    1st-order upwind vertical flux at settling velocity `V_sed`
    (m/s, positive downward). Requires `dz`.
  * Dry deposition at the surface via [`deposition_velocity(:PM, cls)`](@ref)
    when a surface-class map `sfc` is supplied.

Numerical constraints: the sedimentation Courant number `V_sed·dt/dz`
is clamped to `≤ 1` per substep for stability.
"""
function apply_aerosol_step!(state::NamedTuple, dt::Real;
                             V_sed::Real = 0.01,
                             sfc::Union{Nothing, AbstractMatrix{UInt8}} = nothing,
                             dz::Real = 1.0,
                             k_coag::Union{Nothing, Real} = nothing)
    q_v = interior_view(state.q_a)
    N_v = interior_view(state.N_a)
    Nx, Ny, Nz = size(q_v)

    # ---- Coagulation (number only; mass preserved) ----
    k = k_coag === nothing ? 1.0e-13 * V_sed : Float64(k_coag)
    @inbounds for kk in 1:Nz, j in 1:Ny, i in 1:Nx
        Nc = N_v[i, j, kk]
        # Semi-implicit for stability at large N: dN/dt = -k N²
        # N_new = N / (1 + k · N · dt)
        N_v[i, j, kk] = Nc / (1.0 + k * Nc * dt)
    end

    # ---- Sedimentation (1st-order upwind, mass conserving) ----
    C = clamp(V_sed * dt / dz, 0.0, 1.0)
    if C > 0
        # Sweep top → bottom so we can update in place.
        @inbounds for j in 1:Ny, i in 1:Nx
            # top face flux = 0 (no aerosol source from above the domain)
            for kk in Nz:-1:2
                # flux from cell kk into kk-1
                dq = C * q_v[i, j, kk]
                dN = C * N_v[i, j, kk]
                q_v[i, j, kk]     -= dq
                N_v[i, j, kk]     -= dN
                q_v[i, j, kk - 1] += dq
                N_v[i, j, kk - 1] += dN
            end
        end
    end

    # ---- Dry deposition at the surface ----
    if sfc !== nothing
        invdz = 1.0 / dz
        @inbounds for j in 1:Ny, i in 1:Nx
            Vd = deposition_velocity(:PM, sfc[i, j])
            frac = clamp(Vd * dt * invdz, 0.0, 1.0)
            q_v[i, j, 1] *= (1.0 - frac)
            N_v[i, j, 1] *= (1.0 - frac)
        end
    end

    return state
end

"""
    AQAerosol

Slot `:aerosol`, provider `:toy_two_moment`: every step, after the
chemistry, PM dry deposition then [`apply_aerosol_step!`](@ref)
(`V_sed` 5 mm/s) on the [`AerosolState`](@ref).
"""
struct AQAerosol{S} <: AbstractCoupling
    aerosol::S
    V_sed::Float64
end

coupling_name(::Type{<:AQAerosol}) = :aq_aerosol
slot(::Type{<:AQAerosol}) = :aerosol
provider(::Type{<:AQAerosol}) = :toy_two_moment
default_options(::Type{<:AQAerosol}) = (V_sed = 0.005,)
stages(::Type{<:AQAerosol}) = ((:chemistry, 20, EVERY_STEP),)
build(::Type{<:AQAerosol}, case, o) = AQAerosol(AerosolState(case.mesh), o.V_sed)

function run_stage!(c::AQAerosol, case, ::Val{:chemistry}, dt)
    sfc, dz = case.geom.sfc, case.mesh.dz
    apply_dry_deposition!(c.aerosol.q_a, sfc, :PM, dt; dz = dz)
    apply_aerosol_step!(c.aerosol, dt; V_sed = c.V_sed, sfc = sfc, dz = dz)
    return nothing
end

register!(AQAerosol)
