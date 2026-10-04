# =========================================================================
# Era VI, Phase 54 (P54-b, #173) — Lagrangian-averaged dynamic Smagorinsky.
# =========================================================================
#
# Included into `module Operators`, after the closure contract
# (sgs_closures.jl). The Germano identity is the shared kernel
# `_germano_identity!` of Operators.jl (also behind the plane / local
# averaging of `compute_dynamic_smagorinsky_νt!`); this file adds only the
# averaging along pathlines of Meneveau, Lund & Cabot (1996) and the
# closure that carries its memory through the contract.

"""
    LagrangianDynamicSmagorinsky(mesh; filter_ratio = 3, cs_max = 0.5, C0 = 0.16,
                                 θ = 1.5, memory_time = nothing, Pr_t = nothing,
                                 mask = nothing)

The Lagrangian-averaged dynamic Smagorinsky closure of Meneveau, Lund &
Cabot (1996): the Germano identity is averaged backwards along fluid
pathlines instead of over homogeneous planes, so the coefficient is local
and needs no direction of statistical homogeneity (an urban canopy has
none). With ``L^d_{ij} M_{ij}`` and ``M_{ij} M_{ij}`` from the dynamic
procedure ([`compute_dynamic_smagorinsky_νt!`](@ref): 3 × 3 × 3 box test
filter, test-to-grid width ratio `filter_ratio`), the two memories obey
(MLC96 eqs. 2.10–2.13, their time discretisation 3.1–3.3)

```math
\\mathcal I_{LM}^{n+1}(\\mathbf x) = ε\\,[L_{ij}M_{ij}]^{n+1}(\\mathbf x)
    + (1-ε)\\,\\mathcal I_{LM}^{n}(\\mathbf x - \\mathbf u^n Δt), \\qquad
\\mathcal I_{MM}^{n+1}(\\mathbf x) = ε\\,[M_{ij}M_{ij}]^{n+1}(\\mathbf x)
    + (1-ε)\\,\\mathcal I_{MM}^{n}(\\mathbf x - \\mathbf u^n Δt),
```

```math
ε = \\frac{Δt/T}{1 + Δt/T}, \\qquad
T = θ\\,Δ\\,(\\mathcal I_{LM}\\mathcal I_{MM})^{-1/8}, \\qquad
C_s^2 = \\frac{\\mathcal I_{LM}}{\\mathcal I_{MM}},
```

with ``θ = 1.5``, ``Δ = (Δx Δy Δz)^{1/3}``, the upstream values found by
trilinear interpolation (periodic directions wrap; across a non-periodic
boundary the departure point is clamped to the boundary cell), and
``\\mathcal I_{LM}`` clipped at ``≥ 0`` after every update. ``T`` is
evaluated with the upstream memories; where ``\\mathcal I_{LM}`` has been
clipped to 0 it is evaluated with ``\\mathcal I_{LM} ≥ 10^{-8}\\mathcal
I_{MM}`` so the point can recover (otherwise ``T = ∞`` freezes it), and
where ``\\mathcal I_{MM} = 0`` (no strain upstream) ``ε = 1``.
`memory_time` (s) replaces ``T`` by a fixed relaxation time (`Inf`: pure
transport). ``C_s^2`` is clipped to ``[0, c_{s,\\max}^2]`` and
``ν_t = C_s^2 Δ^2 |S|``, ``κ_t = ν_t/\\mathrm{Pr}_t``. No damping function
is applied anywhere.

**Contract** ([`AbstractSGSClosure`](@ref)): the memory advances only in
[`advance_sgs!`](@ref) (once per accepted step, from the state after the
step; the departure point uses the cell-centred velocity committed with
the previous memory, ``\\mathbf u^n``), and [`sgs_coefficients!`](@ref) is
pure: it takes ``C_s^2`` from the committed memory and ``|S|`` from the
current state, so the three RK3 substages and the CFL estimate see one
coefficient. [`sgs_state`](@ref) / [`restore_sgs_state!`](@ref) copy the
memory, so a step [`step_adaptive!`](@ref
Aeolus.TimeIntegration.step_adaptive!) rejects is rolled back bit for bit.
**Initialisation** (MLC96 §3): the first commit sets
``\\mathcal I_{MM} = M_{ij}M_{ij}`` and ``\\mathcal I_{LM} = C_0^2
\\mathcal I_{MM}`` with ``C_0 = 0.16``; until then the coefficient is
``C_0^2``.

**Solids** (`mask`, a storage-sized cell mask, `true` = solid, e.g.
`IBM.SolidMask.mask`; halos filled from the simulation's boundary
conditions on first use): ``ν_t = 0`` and the memories are 0 in solid
cells; the dynamic procedure reads the velocity with the no-slip value 0
on every face that touches a solid and test-filters over fluid
neighbours only (renormalised, the masked [`box_filter!`](@ref)); in the
upstream interpolation a departure point whose nearest cell is solid is
replaced by the cell itself, and solid corners of the trilinear stencil
are dropped with the remaining weights renormalised. Nothing stored
inside an obstacle can reach a fluid coefficient.

The coefficient of the last call is in `closure.workspace.cs2`, the
memories in `closure.I_LM`, `closure.I_MM` (interior-sized arrays).

References: Meneveau, C., Lund, T. S. & Cabot, W. H. (1996), *A Lagrangian
dynamic subgrid-scale model of turbulence*, J. Fluid Mech. 319, 353–385;
Germano, Piomelli, Moin & Cabot (1991), *Phys. Fluids A* 3, 1760; Lilly
(1992), *Phys. Fluids A* 4, 633.
"""
struct LagrangianDynamicSmagorinsky{T<:AbstractFloat, W<:DynamicSmagorinskyWorkspace{T}, P, M} <: AbstractEddyViscosityClosure
    workspace::W
    I_LM::Array{T,3}
    I_MM::Array{T,3}
    Up::Array{T,3}; Vp::Array{T,3}; Wp::Array{T,3}   # cell-centred uⁿ of the memory
    new_LM::Array{T,3}; new_MM::Array{T,3}            # scratch of the update
    initialised::Base.RefValue{Bool}
    filter_ratio::T
    cs_max::T
    C0::T
    θ::T
    memory_time::Union{Nothing, T}
    Pr_t::P
    mask::M
end

function LagrangianDynamicSmagorinsky(g::RegularGrid{T}; filter_ratio::Real = 3,
                                      cs_max::Real = 0.5, C0::Real = 0.16, θ::Real = 1.5,
                                      memory_time = nothing, Pr_t = nothing,
                                      mask = nothing) where {T}
    min(g.Hx, g.Hy, g.Hz) >= 2 ||
        throw(ArgumentError("LagrangianDynamicSmagorinsky: needs halo ≥ 2, mesh has $((g.Hx, g.Hy, g.Hz))"))
    memory_time === nothing || memory_time > 0 ||
        throw(ArgumentError("memory_time must be positive (or nothing for MLC96's T), got $memory_time"))
    i() = zeros(T, g.Nx, g.Ny, g.Nz)
    return LagrangianDynamicSmagorinsky(DynamicSmagorinskyWorkspace(g), i(), i(), i(), i(), i(),
                                        i(), i(), Ref(false), T(filter_ratio), T(cs_max), T(C0),
                                        T(θ), memory_time === nothing ? nothing : T(memory_time),
                                        Pr_t, _LazyMask(mask, T))
end

turbulent_prandtl(c::LagrangianDynamicSmagorinsky, sim) = _prandtl(c.Pr_t, sim)

_periodic_flags(bcs) = (bcs.west isa Periodic, bcs.south isa Periodic, bcs.bottom isa Periodic)

# Pure: C_s² from the committed memory, |S| from the current state.
function sgs_viscosity!(νt, c::LagrangianDynamicSmagorinsky{T}, sim) where {T}
    g = mesh(sim.u)
    ws = c.workspace
    m = _sgs_mask(c.mask, sim)
    ub, vb, wb = _masked_velocities!(m, sim.u.data, sim.v.data, sim.w.data)
    _strain_magnitude!(ws, ub, vb, wb, g)
    cs2max = c.cs_max^2
    cell = _cellmask(m)
    if c.initialised[]
        @inbounds for I in eachindex(ws.cs2)
            mm = c.I_MM[I]
            ws.cs2[I] = mm > 0 ? clamp(c.I_LM[I] / mm, zero(T), cs2max) : zero(T)
        end
    else
        fill!(ws.cs2, min(c.C0^2, cs2max))       # I_LM = C₀² I_MM (MLC96 §3)
    end
    if cell !== nothing
        @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            cell[g.Hx+i, g.Hy+j, g.Hz+k] && (ws.cs2[i,j,k] = zero(T))
        end
    end
    _dynamic_νt!(νt.data, ws, g, cell)
    return νt
end

"""
    advance_sgs!(c::LagrangianDynamicSmagorinsky, sim, dt)

Commit the Lagrangian memories over an accepted step of length `dt`:
the Germano identity at the new state (the shared kernel of the dynamic
procedure; the velocity halos are filled first), then one relaxation
step along the pathlines (MLC96 eqs. 3.1–3.3), and the new cell-centred
velocity is kept for the next departure point. The first commit
initialises the memories (``\\mathcal I_{LM} = C_0^2 \\mathcal I_{MM}``,
``\\mathcal I_{MM} = M_{ij}M_{ij}``).
"""
function advance_sgs!(c::LagrangianDynamicSmagorinsky{T}, sim, dt) where {T}
    g = mesh(sim.u)
    ws = c.workspace
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    m = _sgs_mask(c.mask, sim)
    _germano_identity!(ws, sim.u.data, sim.v.data, sim.w.data, g, c.filter_ratio^2, m)
    if c.initialised[]
        _lagrangian_relax!(c.new_LM, c.new_MM, c.I_LM, c.I_MM, ws.LM, ws.MM,
                           c.Up, c.Vp, c.Wp, T(dt), g, c.θ, c.memory_time,
                           _cellmask(m), _periodic_flags(sim.bcs))
        copyto!(c.I_LM, c.new_LM); copyto!(c.I_MM, c.new_MM)
    else
        c.I_MM .= ws.MM
        c.I_LM .= c.C0^2 .* ws.MM
        c.initialised[] = true
    end
    inner = (2:g.Nx+1, 2:g.Ny+1, 2:g.Nz+1)
    copyto!(c.Up, view(ws.Uc, inner...))
    copyto!(c.Vp, view(ws.Vc, inner...))
    copyto!(c.Wp, view(ws.Wc, inner...))
    return c
end

sgs_state(c::LagrangianDynamicSmagorinsky) =
    (copy(c.I_LM), copy(c.I_MM), copy(c.Up), copy(c.Vp), copy(c.Wp), c.initialised[])

function restore_sgs_state!(c::LagrangianDynamicSmagorinsky, s)
    copyto!(c.I_LM, s[1]); copyto!(c.I_MM, s[2])
    copyto!(c.Up, s[3]); copyto!(c.Vp, s[4]); copyto!(c.Wp, s[5])
    c.initialised[] = s[6]
    return c
end

# Index of coordinate q (in cells, 1 … N) along a direction: wrapped when
# periodic, clamped otherwise.
@inline _wrapidx(n, N, per) = per ? mod1(n, N) : clamp(n, 1, N)

"""
    _upstream(ILM, IMM, x, y, z, i, j, k, g, per, cell) -> (lm, mm)

Trilinear interpolation of both memories at the departure point
`(x, y, z)` (fractional cell indices of the interior arrays; cell `i` has
its centre at `x = i`). Periodic directions wrap; in a non-periodic one
the point is clamped to `[1, N]`. With a cell mask: a point whose nearest
cell is solid returns the values of cell `(i, j, k)`; solid corners are
dropped and the remaining weights renormalised.
"""
@inline function _upstream(ILM, IMM, x, y, z, i, j, k, g, per, cell)
    N = (g.Nx, g.Ny, g.Nz)
    q = (per[1] ? x : clamp(x, 1, N[1]), per[2] ? y : clamp(y, 1, N[2]),
         per[3] ? z : clamp(z, 1, N[3]))
    q0 = floor.(Int, q)
    f = q .- q0
    if cell !== nothing
        ir = _wrapidx(round(Int, q[1]), N[1], per[1])
        jr = _wrapidx(round(Int, q[2]), N[2], per[2])
        kr = _wrapidx(round(Int, q[3]), N[3], per[3])
        @inbounds cell[g.Hx+ir, g.Hy+jr, g.Hz+kr] && return ILM[i,j,k], IMM[i,j,k]
    end
    lm = zero(eltype(ILM)); mm = zero(eltype(IMM)); wsum = zero(eltype(ILM))
    @inbounds for dk in 0:1, dj in 0:1, di in 0:1
        a = _wrapidx(q0[1] + di, N[1], per[1])
        b = _wrapidx(q0[2] + dj, N[2], per[2])
        c = _wrapidx(q0[3] + dk, N[3], per[3])
        w = (di == 0 ? 1 - f[1] : f[1]) * (dj == 0 ? 1 - f[2] : f[2]) * (dk == 0 ? 1 - f[3] : f[3])
        cell !== nothing && cell[g.Hx+a, g.Hy+b, g.Hz+c] && continue
        lm += w * ILM[a,b,c]; mm += w * IMM[a,b,c]; wsum += w
    end
    cell === nothing && return lm, mm
    wsum > 0 || return ILM[i,j,k], IMM[i,j,k]
    return lm / wsum, mm / wsum
end

"""
    _lagrangian_relax!(LMn, MMn, ILM, IMM, LM, MM, U, V, W, dt, g, θ, memory_time, cell, per)

One step of the Lagrangian averaging (MLC96 eqs. 3.1–3.3) on interior-sized
arrays: `LMn = max(ε LM + (1-ε) ILM(x - UΔt), 0)`,
`MMn = ε MM + (1-ε) IMM(x - UΔt)`, with the cell-centred velocity `U, V, W`
of the old memory, [`_upstream`](@ref) interpolation, and
`ε = (Δt/T)/(1 + Δt/T)`; `T = θΔ(I_LM I_MM)^{-1/8}` from the upstream
memories (with `I_LM ≥ 10⁻⁸ I_MM` in the time scale; `ε = 1` where
`I_MM = 0`), or the fixed `memory_time`. Solid cells get 0.
"""
function _lagrangian_relax!(LMn, MMn, ILM, IMM, LM, MM, U, V, W, dt, g, θ, memory_time,
                            cell, per)
    T = eltype(LMn)
    Δ = (g.dx * g.dy * g.dz)^(1/3)
    cx = dt / g.dx; cy = dt / g.dy; cz = dt / g.dz
    εfix = memory_time === nothing ? zero(T) : (dt / memory_time) / (1 + dt / memory_time)
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        if cell !== nothing && cell[g.Hx+i, g.Hy+j, g.Hz+k]
            LMn[i,j,k] = zero(T); MMn[i,j,k] = zero(T)
            continue
        end
        lm, mm = _upstream(ILM, IMM, i - U[i,j,k] * cx, j - V[i,j,k] * cy, k - W[i,j,k] * cz,
                           i, j, k, g, per, cell)
        ε = if memory_time !== nothing
            εfix
        elseif mm > 0
            Tl = θ * Δ * (max(lm, T(1e-8) * mm) * mm)^(-T(1) / 8)
            (dt / Tl) / (1 + dt / Tl)
        else
            one(T)
        end
        LMn[i,j,k] = max(ε * LM[i,j,k] + (1 - ε) * lm, zero(T))
        MMn[i,j,k] = ε * MM[i,j,k] + (1 - ε) * mm
    end
    return nothing
end
