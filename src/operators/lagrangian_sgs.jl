# =========================================================================
# Era VI, Phase 54 (P54-b, #173) — Lagrangian-averaged dynamic Smagorinsky,
# and its scale-dependent form (P54-e, #186; Bou-Zeid et al. 2005).
# =========================================================================
#
# Included into `module Operators`, after the closure contract
# (sgs_closures.jl). The Germano identity is the shared kernel
# `_germano_identity!` of Operators.jl (also behind the plane / local
# averaging of `compute_dynamic_smagorinsky_νt!`); this file adds only the
# averaging along pathlines of Meneveau, Lund & Cabot (1996), the second
# test filter and power law of the scale-dependent model, and the closure
# that carries the memories through the contract.

"""
    LagrangianDynamicSmagorinsky(mesh; filter_ratio = 3, cs_max = 0.5, C0 = 0.16,
                                 θ = 1.5, memory_time = nothing, Pr_t = nothing,
                                 mask = nothing, scale_dependent = false,
                                 filter_ratio2 = 5, beta_min = 0.125, beta_max = 8.0,
                                 beta = nothing, cadence = 1)

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

**Scale dependence** (`scale_dependent = true`; LASD, Bou-Zeid,
Meneveau & Parlange 2005; P54-e, #186). The scale-invariant model above
(LASI) assumes ``C_s`` is the same at the grid and the test-filter
scales; near a wall, where the filter widths reach the distance to it,
the coefficient falls toward the grid scale and LASI under-predicts it.
LASD adds a second test filter (the 5 × 5 × 5 box, width
`filter_ratio2` ``= α_2 = 5`` from its second moment; see
[`_second_test_filter!`](@ref)) and two more memories

```math
\\mathcal I_{QN}^{n+1} = ε\\,[Q^d_{ij}N_{ij}]^{n+1}
    + (1-ε)\\,\\mathcal I_{QN}^{n}(\\mathbf x - \\mathbf u^n Δt), \\qquad
\\mathcal I_{NN}^{n+1} = ε\\,[N_{ij}N_{ij}]^{n+1}
    + (1-ε)\\,\\mathcal I_{NN}^{n}(\\mathbf x - \\mathbf u^n Δt),
```

with the same ``ε`` and ``T`` (from ``\\mathcal I_{QN}\\mathcal I_{NN}``)
and the same pathlines. ``\\mathcal I_{LM}/\\mathcal I_{MM}`` and
``\\mathcal I_{QN}/\\mathcal I_{NN}`` are the coefficients at ``αΔ`` and
``α_2Δ``; a power law between them gives the grid-scale coefficient
([`_scale_dependent_cs2!`](@ref)):
``φ = \\ln(C^2_{α_2Δ}/C^2_{αΔ})/\\ln(α_2/α)``, ``β = 2^φ`` clipped to
`[beta_min, beta_max]` (Bou-Zeid's ``β ≥ 0.125``; the upper clip 8 is
ours, symmetric in ``φ``), ``C^2_Δ = C^2_{αΔ} α^{-φ}``. For Bou-Zeid's
filters (``α = 2``, ``α_2 = 4``) this is their ``β = C^2_{4Δ}/C^2_{2Δ}``,
``C^2_Δ = C^2_{2Δ}/β``; here ``β`` is the coefficient ratio per doubling
of the filter width. `closure.beta` holds the ``β`` of the last call.
`beta` (a number) fixes ``β``; `beta = 1` is LASI bit for bit. The
memories initialise with ``\\mathcal I_{QN} = C_0^2 \\mathcal I_{NN}``
(``β = 1``). No wall damping is applied: the near-wall coefficient comes
from the scale dependence, the wall stress from the wall model.

**Cadence** (`cadence = N`, default 1): the Germano identities (the
expensive part) are evaluated every `N` commits; at the commits between,
the memories are only transported along the pathlines (``ε = 0``), and at
the next evaluation ``ε`` uses the time since the last one. This is the
usual practice of Lagrangian dynamic codes; `N = 1` is the exact model.

The coefficient of the last call is in `closure.workspace.cs2`, the
memories in `closure.I_LM`, `closure.I_MM` (and `closure.I_QN`,
`closure.I_NN`; interior-sized arrays).

References: Meneveau, C., Lund, T. S. & Cabot, W. H. (1996), *A Lagrangian
dynamic subgrid-scale model of turbulence*, J. Fluid Mech. 319, 353–385;
Bou-Zeid, E., Meneveau, C. & Parlange, M. B. (2005), *A scale-dependent
Lagrangian dynamic model for large eddy simulation of complex turbulent
flows*, Phys. Fluids 17, 025105; Porté-Agel, F., Meneveau, C. & Parlange,
M. B. (2000), *A scale-dependent dynamic model for large-eddy simulation:
application to a neutral atmospheric boundary layer*, J. Fluid Mech. 415,
261–284;
Germano, Piomelli, Moin & Cabot (1991), *Phys. Fluids A* 3, 1760; Lilly
(1992), *Phys. Fluids A* 4, 633.
"""
struct LagrangianDynamicSmagorinsky{T<:AbstractFloat, W<:DynamicSmagorinskyWorkspace{T}, P, M, S} <: AbstractEddyViscosityClosure
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
    # scale dependence (Bou-Zeid, Meneveau & Parlange 2005; P54-e, #186)
    scale_dependent::Bool
    second::S                                         # _SecondTestFilter, or nothing
    I_QN::Array{T,3}; I_NN::Array{T,3}
    new_QN::Array{T,3}; new_NN::Array{T,3}
    beta::Array{T,3}                                  # β of the last coefficient call
    filter_ratio2::T
    beta_min::T
    beta_max::T
    beta_fixed::Union{Nothing, T}
    # coefficient cadence: the Germano identities every `cadence` commits,
    # the pathline transport of the memories at every one
    cadence::Int
    since::Base.RefValue{Int}                         # commits since the last identity
    acc_dt::Base.RefValue{T}                          # time since the last identity
end

function LagrangianDynamicSmagorinsky(g::RegularGrid{T}; filter_ratio::Real = 3,
                                      cs_max::Real = 0.5, C0::Real = 0.16, θ::Real = 1.5,
                                      memory_time = nothing, Pr_t = nothing,
                                      mask = nothing, scale_dependent::Bool = false,
                                      filter_ratio2::Real = 5, beta_min::Real = 0.125,
                                      beta_max::Real = 8.0, beta = nothing,
                                      cadence::Integer = 1) where {T}
    min(g.Hx, g.Hy, g.Hz) >= 2 ||
        throw(ArgumentError("LagrangianDynamicSmagorinsky: needs halo ≥ 2, mesh has $((g.Hx, g.Hy, g.Hz))"))
    memory_time === nothing || memory_time > 0 ||
        throw(ArgumentError("memory_time must be positive (or nothing for MLC96's T), got $memory_time"))
    filter_ratio2 > filter_ratio ||
        throw(ArgumentError("filter_ratio2 ($filter_ratio2) must exceed filter_ratio ($filter_ratio)"))
    0 < beta_min <= 1 <= beta_max ||
        throw(ArgumentError("need 0 < beta_min ≤ 1 ≤ beta_max, got ($beta_min, $beta_max)"))
    beta === nothing || beta > 0 ||
        throw(ArgumentError("a fixed beta must be positive, got $beta"))
    beta === nothing || scale_dependent ||
        throw(ArgumentError("a fixed beta needs scale_dependent = true"))
    cadence >= 1 || throw(ArgumentError("cadence must be ≥ 1, got $cadence"))
    i() = zeros(T, g.Nx, g.Ny, g.Nz)
    sd() = scale_dependent ? i() : zeros(T, 0, 0, 0)
    second = scale_dependent ? _SecondTestFilter(g) : nothing
    return LagrangianDynamicSmagorinsky(DynamicSmagorinskyWorkspace(g), i(), i(), i(), i(), i(),
                                        i(), i(), Ref(false), T(filter_ratio), T(cs_max), T(C0),
                                        T(θ), memory_time === nothing ? nothing : T(memory_time),
                                        Pr_t, _LazyMask(mask, T), scale_dependent, second,
                                        sd(), sd(), sd(), sd(), sd(), T(filter_ratio2),
                                        T(beta_min), T(beta_max),
                                        beta === nothing ? nothing : T(beta),
                                        Int(cadence), Ref(0), Ref(zero(T)))
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
    if !c.initialised[]
        fill!(ws.cs2, min(c.C0^2, cs2max))       # I_LM = C₀² I_MM (MLC96 §3)
        c.scale_dependent && fill!(c.beta, one(T))
    elseif c.scale_dependent
        _scale_dependent_cs2!(ws.cs2, c.beta, c, cs2max)
    else
        @inbounds for I in eachindex(ws.cs2)
            mm = c.I_MM[I]
            ws.cs2[I] = mm > 0 ? clamp(c.I_LM[I] / mm, zero(T), cs2max) : zero(T)
        end
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
    _scale_dependent_cs2!(cs2, beta, c, cs2max)

The coefficient of the scale-dependent model (Bou-Zeid, Meneveau &
Parlange 2005) from the committed memories: ``C^2_{αΔ} = \\mathcal
I_{LM}/\\mathcal I_{MM}`` and ``C^2_{α_2Δ} = \\mathcal I_{QN}/\\mathcal
I_{NN}`` are the Lagrangian coefficients at the two test-filter scales;
assuming the power law ``C^2(aΔ) = C^2(Δ)\\,a^φ`` between them,

```math
φ = \\frac{\\ln(C^2_{α_2Δ}/C^2_{αΔ})}{\\ln(α_2/α)}, \\qquad
β = 2^φ \\in [β_{\\min}, β_{\\max}], \\qquad
C^2_Δ = C^2_{αΔ}\\,α^{-φ} = C^2_{αΔ}\\,β^{-\\log_2 α},
```

with ``φ = \\log_2 β`` after the clip, so for ``α = 2``, ``α_2 = 4``
(Bou-Zeid's filters) ``β = C^2_{4Δ}/C^2_{2Δ}`` and ``C^2_Δ =
C^2_{2Δ}/β`` as in Bou-Zeid et al. ``β`` is the coefficient ratio per doubling
of the filter width at any ``α, α_2``. Where ``C^2_{αΔ} = 0`` the
coefficient is 0 and ``β`` is stored as 1 (no scale information); where
only ``C^2_{α_2Δ} = 0``, ``β = β_{\\min}``. With a fixed `beta` the same
formula runs with that β (``β = 1`` is the scale-invariant model, bit
for bit). ``C^2_Δ`` is clipped to ``[0, c_{s,\\max}^2]``.
"""
function _scale_dependent_cs2!(cs2, beta, c::LagrangianDynamicSmagorinsky{T}, cs2max) where {T}
    lnα = log(c.filter_ratio)
    inv_lr = one(T) / log2(c.filter_ratio2 / c.filter_ratio)     # φ per log₂ of the ratio
    φmin, φmax = log2(c.beta_min), log2(c.beta_max)
    φfix = c.beta_fixed === nothing ? zero(T) : log2(c.beta_fixed)
    @inbounds for I in eachindex(cs2)
        mm = c.I_MM[I]
        ca = mm > 0 ? c.I_LM[I] / mm : zero(T)              # C²(αΔ)
        if !(ca > 0)
            cs2[I] = zero(T); beta[I] = one(T)
            continue
        end
        φ = if c.beta_fixed !== nothing
            φfix
        else
            nn = c.I_NN[I]
            cb = nn > 0 ? c.I_QN[I] / nn : zero(T)          # C²(α₂Δ)
            cb > 0 ? clamp(log2(cb / ca) * inv_lr, φmin, φmax) : φmin
        end
        beta[I] = clamp(exp2(φ), c.beta_min, c.beta_max)    # β = 2^φ
        cs2[I] = clamp(ca * exp(-φ * lnα), zero(T), cs2max) # C²(αΔ) α^(-φ)
    end
    return nothing
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
    per = _periodic_flags(sim.bcs)
    cell = _cellmask(m)
    sf = c.second
    c.since[] += 1
    c.acc_dt[] += T(dt)
    full = !c.initialised[] || c.since[] >= c.cadence
    if full                                         # the Germano identities at the new state
        _germano_identity!(ws, sim.u.data, sim.v.data, sim.w.data, g, c.filter_ratio^2, m)
        sf === nothing || _second_test_filter!(sf, ws, g, c.filter_ratio2^2, cell, per)
    end
    if !c.initialised[]
        c.I_MM .= ws.MM
        c.I_LM .= c.C0^2 .* ws.MM
        if sf !== nothing                           # β = 1 at the start
            c.I_NN .= sf.NN
            c.I_QN .= c.C0^2 .* sf.NN
        end
        c.initialised[] = true
    else
        # with the sources: relax over the time since they were last
        # computed; between them (cadence > 1): transport only (ε = 0)
        mt = full ? c.memory_time : T(Inf)
        dtε = full ? c.acc_dt[] : T(dt)
        lmp = (c.new_LM, c.new_MM, c.I_LM, c.I_MM, ws.LM, ws.MM)
        pairs = sf === nothing ? (lmp,) :
                (lmp, (c.new_QN, c.new_NN, c.I_QN, c.I_NN, sf.QN, sf.NN))
        _lagrangian_relax!(pairs, c.Up, c.Vp, c.Wp, T(dt), g, c.θ, mt, cell, per, dtε)
        copyto!(c.I_LM, c.new_LM); copyto!(c.I_MM, c.new_MM)
        if sf !== nothing
            copyto!(c.I_QN, c.new_QN); copyto!(c.I_NN, c.new_NN)
        end
    end
    if full
        c.since[] = 0; c.acc_dt[] = zero(T)
        inner = (2:g.Nx+1, 2:g.Ny+1, 2:g.Nz+1)
        copyto!(c.Up, view(ws.Uc, inner...))
        copyto!(c.Vp, view(ws.Vc, inner...))
        copyto!(c.Wp, view(ws.Wc, inner...))
    else
        _cell_velocity!(c.Up, c.Vp, c.Wp, m, sim, g)
    end
    return c
end

# The cell-centred velocity of the state (solid faces read as 0), as the
# Germano pass leaves it in the workspace: the departure velocity of the
# next commit when the identity is skipped (cadence > 1).
function _cell_velocity!(U, V, W, m, sim, g)
    ud, vd, wd = _masked_velocities!(m, sim.u.data, sim.v.data, sim.w.data)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        U[i,j,k] = 0.5 * (ud[I,J,K] + ud[I+1,J,K])
        V[i,j,k] = 0.5 * (vd[I,J,K] + vd[I,J+1,K])
        W[i,j,k] = 0.5 * (wd[I,J,K] + wd[I,J,K+1])
    end
    return nothing
end

sgs_state(c::LagrangianDynamicSmagorinsky) =
    (copy(c.I_LM), copy(c.I_MM), copy(c.Up), copy(c.Vp), copy(c.Wp), c.initialised[],
     copy(c.I_QN), copy(c.I_NN), c.since[], c.acc_dt[])

function restore_sgs_state!(c::LagrangianDynamicSmagorinsky, s)
    copyto!(c.I_LM, s[1]); copyto!(c.I_MM, s[2])
    copyto!(c.Up, s[3]); copyto!(c.Vp, s[4]); copyto!(c.Wp, s[5])
    c.initialised[] = s[6]
    copyto!(c.I_QN, s[7]); copyto!(c.I_NN, s[8])
    c.since[] = s[9]; c.acc_dt[] = s[10]
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
@inline _upstream(ILM, IMM, x, y, z, i, j, k, g, per, cell) =
    _upstream_n((ILM, IMM), x, y, z, i, j, k, g, per, cell)

# The same for a tuple of memories (one stencil, one set of weights).
@inline function _upstream_n(mems::NTuple{M}, x, y, z, i, j, k, g, per, cell) where {M}
    N = (g.Nx, g.Ny, g.Nz)
    q = (per[1] ? x : clamp(x, 1, N[1]), per[2] ? y : clamp(y, 1, N[2]),
         per[3] ? z : clamp(z, 1, N[3]))
    q0 = floor.(Int, q)
    f = q .- q0
    own() = @inbounds map(A -> A[i,j,k], mems)
    if cell !== nothing
        ir = _wrapidx(round(Int, q[1]), N[1], per[1])
        jr = _wrapidx(round(Int, q[2]), N[2], per[2])
        kr = _wrapidx(round(Int, q[3]), N[3], per[3])
        @inbounds cell[g.Hx+ir, g.Hy+jr, g.Hz+kr] && return own()
    end
    Tm = eltype(first(mems))
    acc = ntuple(_ -> zero(Tm), Val(M)); wsum = zero(Tm)
    @inbounds for dk in 0:1, dj in 0:1, di in 0:1
        a = _wrapidx(q0[1] + di, N[1], per[1])
        b = _wrapidx(q0[2] + dj, N[2], per[2])
        c = _wrapidx(q0[3] + dk, N[3], per[3])
        w = (di == 0 ? 1 - f[1] : f[1]) * (dj == 0 ? 1 - f[2] : f[2]) * (dk == 0 ? 1 - f[3] : f[3])
        cell !== nothing && cell[g.Hx+a, g.Hy+b, g.Hz+c] && continue
        acc = map((s, A) -> s + w * A[a,b,c], acc, mems); wsum += w
    end
    cell === nothing && return acc
    wsum > 0 || return own()
    wt = wsum                                       # (a fresh binding: no Core.Box)
    return map(s -> s / wt, acc)
end

"""
    _lagrangian_relax!(LMn, MMn, ILM, IMM, LM, MM, U, V, W, dt, g, θ, memory_time, cell, per,
                       dtε = dt)

One step of the Lagrangian averaging (MLC96 eqs. 3.1–3.3) on interior-sized
arrays: `LMn = max(ε LM + (1-ε) ILM(x - UΔt), 0)`,
`MMn = ε MM + (1-ε) IMM(x - UΔt)`, with the cell-centred velocity `U, V, W`
of the old memory, [`_upstream`](@ref) interpolation, and
`ε = (Δt/T)/(1 + Δt/T)`; `T = θΔ(I_LM I_MM)^{-1/8}` from the upstream
memories (with `I_LM ≥ 10⁻⁸ I_MM` in the time scale; `ε = 1` where
`I_MM = 0`), or the fixed `memory_time`. Solid cells get 0. The departure
point is `x - U dt`; `ε` uses `dtε` (the time since the sources were last
computed, under a coefficient cadence; `dt` otherwise).
"""
_lagrangian_relax!(LMn, MMn, ILM, IMM, LM, MM, U, V, W, dt, g, θ, memory_time,
                   cell, per, dtε = dt) =
    _lagrangian_relax!(((LMn, MMn, ILM, IMM, LM, MM),), U, V, W, dt, g, θ, memory_time,
                       cell, per, dtε)

# Several memory pairs (LM, MM) and (QN, NN) along the same pathlines in one
# sweep: one departure point and one set of trilinear weights per cell,
# each pair with its own time scale T.
function _lagrangian_relax!(pairs::Tuple, U, V, W, dt, g, θ, memory_time, cell, per, dtε)
    T = eltype(first(first(pairs)))
    Δ = (g.dx * g.dy * g.dz)^(1/3)
    cx = dt / g.dx; cy = dt / g.dy; cz = dt / g.dz
    εfix = memory_time === nothing ? zero(T) : (dtε / memory_time) / (1 + dtε / memory_time)
    mems = _relax_memories(pairs)
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        if cell !== nothing && cell[g.Hx+i, g.Hy+j, g.Hz+k]
            foreach(p -> (p[1][i,j,k] = zero(T); p[2][i,j,k] = zero(T)), pairs)
            continue
        end
        up = _upstream_n(mems, i - U[i,j,k] * cx, j - V[i,j,k] * cy, k - W[i,j,k] * cz,
                         i, j, k, g, per, cell)
        _relax_pairs!(pairs, up, 1, i, j, k, θ, Δ, εfix, memory_time, dtε)
    end
    return nothing
end

_relax_memories(pairs::Tuple{}) = ()
_relax_memories(pairs::Tuple) = (first(pairs)[3], first(pairs)[4], _relax_memories(Base.tail(pairs))...)

@inline _relax_pairs!(::Tuple{}, up, n, i, j, k, θ, Δ, εfix, memory_time, dtε) = nothing
@inline function _relax_pairs!(pairs::Tuple, up, n, i, j, k, θ, Δ, εfix, memory_time, dtε)
    LMn, MMn, _, _, LM, MM = first(pairs)
    T = eltype(LMn)
    lm = up[n]; mm = up[n+1]
    ε = if memory_time !== nothing
        εfix
    elseif mm > 0
        Tl = θ * Δ * (max(lm, T(1e-8) * mm) * mm)^(-T(1) / 8)
        (dtε / Tl) / (1 + dtε / Tl)
    else
        one(T)
    end
    @inbounds LMn[i,j,k] = max(ε * LM[i,j,k] + (1 - ε) * lm, zero(T))
    @inbounds MMn[i,j,k] = ε * MM[i,j,k] + (1 - ε) * mm
    _relax_pairs!(Base.tail(pairs), up, n + 2, i, j, k, θ, Δ, εfix, memory_time, dtε)
end

# -------------------------------------------------------------------------
# The second test filter of the scale-dependent model (P54-e, #186)
# -------------------------------------------------------------------------

"""
    _SecondTestFilter(mesh)

Scratch storage of the second (wider) test filter of the scale-dependent
Lagrangian model. The 21 Germano quantities of [`_germano_terms`](@ref)
plus the fluid indicator (22 components, component first) are box-summed
plane by plane: `Ap` (one shell plane), `Bp` (after the x pass), `C`
(every plane after the y pass), `Ep` (one plane after the z pass); the
interior-sized results are `QN` ``= Q^d_{ij}N_{ij}`` and `NN` ``=
N_{ij}N_{ij}``.
"""
struct _SecondTestFilter{T}
    Ap::Array{T,3}
    Bp::Array{T,3}
    C::Array{T,4}
    Ep::Array{T,3}
    QN::Array{T,3}
    NN::Array{T,3}
end

function _SecondTestFilter(g::RegularGrid{T}) where {T}
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    return _SecondTestFilter{T}(zeros(T, 22, Nx + 2, Ny + 2), zeros(T, 22, Nx, Ny + 2),
                                zeros(T, 22, Nx, Ny, Nz + 2), zeros(T, 22, Nx, Ny),
                                zeros(T, Nx, Ny, Nz), zeros(T, Nx, Ny, Nz))
end

# The shell indices of the 5-cell box around interior index i (shell index
# l = interior index + 1; 0 = dropped). Periodic: the stencil wraps on the
# interior. Bounded: it stays in the interior and the points beyond are
# dropped (their weight and their fluid count both, so the result stays a
# renormalised average); the halo cell of a free-slip wall carries the
# mirrored, reversed shear, which the 5-cell box would otherwise reach two
# cells from the wall.
function _box5_stencil(N, per)
    return [ntuple(m -> begin
                       s = i + m - 3
                       per ? mod1(s, N) + 1 : (1 <= s <= N ? s + 1 : 0)
                   end, 5) for i in 1:N]
end

# dst[:, i, …] = Σ_s src[:, s, …] over the stencil of i, along the first
# spatial dimension (`dim = 2`) or the second (`dim = 3`) of a plane.
function _box5_plane_pass!(dst::AbstractArray{T,3}, src::AbstractArray{T,3}, st, dim) where {T}
    D = size(dst, 1)
    @inbounds if dim == 2
        for c2 in axes(dst, 3), i in axes(dst, 2)
            first = true
            for s in st[i]
                s == 0 && continue
                if first
                    @simd for c in 1:D; dst[c, i, c2] = src[c, s, c2]; end
                    first = false
                else
                    @simd for c in 1:D; dst[c, i, c2] += src[c, s, c2]; end
                end
            end
        end
    else
        for j in axes(dst, 3)
            first = true
            for s in st[j]
                s == 0 && continue
                for c1 in axes(dst, 2)
                    if first
                        @simd for c in 1:D; dst[c, c1, j] = src[c, c1, s]; end
                    else
                        @simd for c in 1:D; dst[c, c1, j] += src[c, c1, s]; end
                    end
                end
                first = false
            end
        end
    end
    return dst
end

"""
    _second_test_filter!(sf, ws, g, α2sq, cell, per)

The Germano identity at the second test-filter scale of the
scale-dependent model (Porté-Agel, Meneveau & Parlange 2000; Bou-Zeid,
Meneveau & Parlange 2005): with ``\\hat{·}`` the 5 × 5 × 5 box average
of cell-centred quantities,

```math
Q_{ij} = \\widehat{u_i u_j} - \\hat u_i \\hat u_j, \\qquad
N_{ij} = 2Δ^2\\left(\\widehat{|S| S_{ij}} - α_2^2 |\\hat S| \\hat S_{ij}\\right),
```

and `sf.QN` ``= Q^d_{ij}N_{ij}``, `sf.NN` ``= N_{ij}N_{ij}`` at every
interior cell, from the shell-indexed velocity and strain that
[`_germano_identity!`](@ref) left in `ws` (call it first). The box is
separable and summed in three passes; a 5-point box has second moment
``2Δ^2``, so with the implicit grid filter its width is ``\\sqrt{Δ^2 +
12·2Δ^2} = 5Δ`` (the convention that gives the first filter's ``α = 3``),
`α2sq` ``= α_2^2 = 25``. In a periodic direction the box wraps; in a
bounded one it stays in the interior and the points beyond are dropped
(renormalised; unlike the 3-cell first filter, which reads the halo
cell, the 5-cell box would reach the mirrored halo two cells from a
free-slip wall). With a cell mask the box
runs over fluid cells only and `QN = NN = 0` in solid cells.
"""
function _second_test_filter!(sf::_SecondTestFilter{T}, ws::DynamicSmagorinskyWorkspace{T},
                              g, α2sq, cell, per) where {T}
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    oH = (g.Hx - 1, g.Hy - 1, g.Hz - 1)             # shell → data index
    stx = _box5_stencil(Nx, per[1]); sty = _box5_stencil(Ny, per[2])
    stz = _box5_stencil(Nz, per[3])
    Ap, Bp, C, Ep = sf.Ap, sf.Bp, sf.C, sf.Ep
    # x and y passes, one shell plane at a time (the plane stays in cache)
    @inbounds for c in 1:Nz+2
        for b in 1:Ny+2, a in 1:Nx+2
            fluid = cell === nothing || !cell[oH[1] + a, oH[2] + b, oH[3] + c]
            t = _germano_terms(ws, a, b, c)
            for q in 1:21
                Ap[q, a, b] = fluid ? t[q] : zero(T)
            end
            Ap[22, a, b] = fluid ? one(T) : zero(T)
        end
        _box5_plane_pass!(Bp, Ap, stx, 2)
        _box5_plane_pass!(view(C, :, :, :, c), Bp, sty, 3)
    end
    # z pass and the identity, one interior plane at a time
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    D = 22
    @inbounds for k in 1:Nz
        first = true
        for s in stz[k]
            s == 0 && continue
            for j in 1:Ny, i in 1:Nx
                if first
                    @simd for q in 1:D; Ep[q, i, j] = C[q, i, j, s]; end
                else
                    @simd for q in 1:D; Ep[q, i, j] += C[q, i, j, s]; end
                end
            end
            first = false
        end
        for j in 1:Ny, i in 1:Nx
            n = Ep[22, i, j]
            if (cell !== nothing && cell[g.Hx + i, g.Hy + j, g.Hz + k]) || !(n > 0)
                sf.QN[i,j,k] = zero(T); sf.NN[i,j,k] = zero(T)
                continue
            end
            t = ntuple(q -> Ep[q, i, j], Val(21))
            sf.QN[i,j,k], sf.NN[i,j,k] = _germano_LM_MM(t, one(T) / n, Δ2, α2sq)
        end
    end
    return nothing
end
