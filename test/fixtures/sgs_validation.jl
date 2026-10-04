# A-posteriori-in-time validation of the SGS closures against filtered
# DNS (Era VI, P56-e, #192). Included by test/era6_sgs_validation.jl and
# examples/sgs/rotating_stratified_decay.jl.
#
# The method ("a priori in time"). One DNS of decaying isotropic
# turbulence (the P56-c fixture, test/fixtures/dns_decay.jl: 2π box,
# ν ∝ N^{-4/3}, spun up from a fixed seed to t₀ = 2) is continued from t₀
# in three ways:
#
#   :isotropic   unchanged;
#   :rotating    f-plane rotation f = u'/(Ro L), Ro = 0.2;
#   :stratified  a uniform background stratification N = u'/(Fr L),
#                Fr = 0.3 (Riley, Metcalfe & Weissman 1981),
#
# with u' and the integral scale L measured at t₀. At t₀ and at the save
# times the DNS velocity is filtered with the Gaussian of width Δ = rh
# (r = 4) and resampled onto the r-times-coarser grid by Fourier
# truncation (`coarse_grain`), which is how an LES on that grid sees the
# filtered DNS; the exact subfilter stress τᵢⱼ = (uᵢuⱼ)‾ − ūᵢūⱼ
# (`subfilter_stress!`, P56-c) and the subfilter heat flux are sampled at
# the coarse centres by the 2×2×2 mean (`centre_sample`), which keeps τ
# positive semi-definite. Each LES starts from the filtered DNS at t₀ — e and τᵢⱼ
# from the exact subfilter stress (`initialize_tke!`,
# `initialize_reynolds_stress!`) — runs with the same ν, f and N, and is
# compared with the filtered DNS later on.
#
# Stratification in a periodic box. θ = θ₀ + Γz + θ′ with Γ = N²θ₀/g;
# the simulations carry θ₀ + θ′ (periodic), and the background enters as
# the body force −Γw on θ′ (`background_advection`). The buoyancy is
# then g θ′/θ₀ and the discrete exchange with −Γw is exact (the face
# average of θ in the w equation and the centre average of w in the θ
# equation are adjoint). The SGS closures need the full θ (N², the
# heat flux −K∇θ): `BackgroundGradient` hands them a view of the
# simulation whose θ is θ₀ + Γ(z − z_mid) + θ′, with halos continued
# across the periodic boundary with the jump ΓL_z.

using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Grids: Location
import Aeolus.BoundaryConditions
import Aeolus.Operators: sgs_coefficients!, sgs_fields, add_sgs_momentum!, add_sgs_scalar!,
                         advance_sgs!, sgs_state, restore_sgs_state!, sgs_explicit_stress,
                         sgs_scalar_tensor, _fill_sgs_halos!
using Aeolus.Operators: AbstractSGSClosure, SGSFields
vmean(x) = sum(x) / length(x)
vmean(f, x) = sum(f, x) / length(x)

isdefined(@__MODULE__, :dns_decay) || include(joinpath(@__DIR__, "dns_decay.jl"))

const VAL_RO = 0.2
const VAL_FR = 0.3
const VAL_T0 = 2.0
const VAL_θ0 = 300.0
const VAL_G = 9.81

# -------------------------------------------------------------------------
# Periodic background gradient
# -------------------------------------------------------------------------

"""
    LinearJumpPeriodic(jump, target)

Periodic in z, except for the one array `target` (the data of the full θ
field): there the halo above the top is the bottom interior plus `jump`
and the halo below the bottom the top interior minus `jump` — the
continuation of θ₀ + Γz + θ′ with θ′ periodic (jump = ΓL_z). Every other
field filled with these boundary conditions (the closures fill e, τᵢⱼ
and q_v halos with `sim.θ_bcs` too) is plain periodic.
"""
struct LinearJumpPeriodic{T, A} <: BoundaryConditions.AbstractBC
    jump::T
    target::A
end

function BoundaryConditions._fill_z!(d, b::LinearJumpPeriodic, ::LinearJumpPeriodic, Hz, Nz, ::Location)
    j = d === b.target ? b.jump : zero(b.jump)
    @inbounds for h in 1:Hz
        d[:, :, h]           .= d[:, :, Nz + h] .- j
        d[:, :, Hz + Nz + h] .= d[:, :, Hz + h] .+ j
    end
    return d
end

# Periodic for the operators that ask (the tensor diffusion's wall faces).
Aeolus.Operators.bounded_directions(::FieldBoundaryConditions{<:Any, <:Any, <:Any, <:Any, <:LinearJumpPeriodic}) =
    (false, false, false)

# The simulation as the closures see it: every property of `sim`, except
# θ (the full field) and θ_bcs.
struct SimView{S, F, B}
    sim::S
    θ::F
    θ_bcs::B
end
Base.getproperty(p::SimView, s::Symbol) =
    s === :θ ? getfield(p, :θ) : s === :θ_bcs ? getfield(p, :θ_bcs) :
    s === :sim ? getfield(p, :sim) : getproperty(getfield(p, :sim), s)
Base.propertynames(p::SimView) = propertynames(getfield(p, :sim))

"""
    BackgroundGradient(inner, sim, Γ)

Wrap an SGS closure so that it sees θ = θ_sim + Γ(z − z_mid) with the
jump-periodic halos of [`LinearJumpPeriodic`](@ref); its θ flux is
applied to that full field. Every contract generic delegates to `inner`.
"""
struct BackgroundGradient{C, F, B, T} <: AbstractSGSClosure
    inner::C
    θfull::F
    bcs::B
    Γ::T
end

function BackgroundGradient(inner, sim, Γ)
    g = sim.mesh
    θfull = Field(CellCenter(), g)
    bc = LinearJumpPeriodic(float(Γ * g.Lz), θfull.data)
    bcs = FieldBoundaryConditions(BoundaryConditions.Periodic(), BoundaryConditions.Periodic(),
                                  BoundaryConditions.Periodic(), BoundaryConditions.Periodic(), bc, bc)
    return BackgroundGradient(inner, θfull, bcs, float(Γ))
end

function _view(c::BackgroundGradient, sim)
    g = sim.mesh
    θi = interior_view(c.θfull); si = interior_view(sim.θ)
    zmid = g.Lz / 2
    @inbounds for k in 1:g.Nz
        z = (k - 0.5) * g.dz
        θi[:, :, k] .= si[:, :, k] .+ c.Γ * (z - zmid)
    end
    fill_halos!(c.θfull, c.bcs)
    return SimView(sim, c.θfull, c.bcs)
end
_unwrap(sim::SimView) = sim
_unwrap(sim) = sim

sgs_fields(c::BackgroundGradient, sim) = sgs_fields(c.inner, sim)
sgs_coefficients!(f::SGSFields, c::BackgroundGradient, sim) = sgs_coefficients!(f, c.inner, _view(c, sim))
_fill_sgs_halos!(f::SGSFields, c::BackgroundGradient, sim) = _fill_sgs_halos!(f, c.inner, _view(c, sim))
add_sgs_momentum!(tu, tv, tw, c::BackgroundGradient, f::SGSFields, sim; stress::Symbol = :laplacian) =
    add_sgs_momentum!(tu, tv, tw, c.inner, f, _view(c, sim); stress = stress)
function add_sgs_scalar!(tϕ, ϕ, c::BackgroundGradient, f::SGSFields, sim)
    if ϕ === sim.θ
        v = _view(c, sim)
        return add_sgs_scalar!(tϕ, v.θ, c.inner, f, v)
    end
    return add_sgs_scalar!(tϕ, ϕ, c.inner, f, sim)
end
sgs_scalar_tensor(c::BackgroundGradient, f::SGSFields, sim) = sgs_scalar_tensor(c.inner, f, _view(c, sim))
advance_sgs!(c::BackgroundGradient, sim, dt) = (advance_sgs!(c.inner, _view(c, sim), dt); c)
sgs_state(c::BackgroundGradient) = sgs_state(c.inner)
restore_sgs_state!(c::BackgroundGradient, s) = (restore_sgs_state!(c.inner, s); c)
sgs_explicit_stress(c::BackgroundGradient) = sgs_explicit_stress(c.inner)

"""
    background_advection(Γ) -> body_force!(sim, Δt)

The advection of the background gradient, ``∂_tθ′ \\mathrel{+}= −Γ\\,\\bar w``
with ``\\bar w`` the two-face average at the cell centre.
"""
function background_advection(Γ)
    return function (sim, Δt)
        g = sim.mesh
        Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
        tθ = sim.tθ.data; w = sim.w.data
        @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            I, J, K = Hx + i, Hy + j, Hz + k
            tθ[I, J, K] -= Γ * (w[I, J, K] + w[I, J, K + 1]) / 2
        end
        return nothing
    end
end

# -------------------------------------------------------------------------
# DNS: spin-up and the three continuations
# -------------------------------------------------------------------------

# u' and the integral scale L = (π/2u'²) ∫ E(k)/k dk (Pope 2000, eq. 6.260).
function integral_scale(sim)
    k, E = isotropic_energy_spectrum(sim.u, sim.v, sim.w)
    up2 = 2 / 3 * sum(E) * (k[2] - k[1])
    L = π / (2up2) * sum(E[i] / k[i] for i in eachindex(k) if k[i] > 0) * (k[2] - k[1])
    return sqrt(up2), L
end

_copy_state!(dst, src) = (for s in (:u, :v, :w, :θ); getproperty(dst, s).data .= getproperty(src, s).data; end; dst)

"""
    validation_base(; N = 48, t0 = 2.0, seed = 1) -> NamedTuple

Spin up the P56-c DNS to `t0` and derive the continuation parameters:
`u′`, `L`, the turnover `T = L/u′`, `f = u′/(Ro L)`, `N_bv = u′/(Fr L)`
and `Γ = N²θ₀/g`.
"""
function validation_base(; N::Integer = 48, t0::Real = VAL_T0, seed::Integer = 1)
    d = dns_decay(; N = N, t_end = t0, seed = seed)
    up, L = integral_scale(d.sim)
    T = L / up
    f = up / (VAL_RO * L)
    Nbv = up / (VAL_FR * L)
    return (; dns = d, sim0 = d.sim, N, ν = d.ν, up, L, T, f, Nbv,
              Γ = Nbv^2 * VAL_θ0 / VAL_G, Re_λ = d.Re_λ, wall = d.wall)
end

case_params(base, case::Symbol) =
    case === :isotropic ? (f = 0.0, Γ = 0.0) :
    case === :rotating ? (f = base.f, Γ = 0.0) :
    case === :stratified ? (f = 0.0, Γ = base.Γ) :
    case === :rotating_stratified ? (f = base.f, Γ = base.Γ) :
    throw(ArgumentError("case must be :isotropic, :rotating, :stratified or :rotating_stratified"))

_new_sim(mesh, ν, f, Γ) =
    BoussinesqSimulation(mesh; ν = ν, f_coriolis = f, θ₀ = Γ > 0 ? VAL_θ0 : 0.0, g_accel = VAL_G)

_field_like(f::Field{L}, g) where {L} = Field(L(), g)

"""
    filtered_snapshot(sim, r, Γ) -> NamedTuple

The DNS state as the LES grid (r times coarser) sees it: the
Gaussian-filtered (Δ = rh) velocity and θ′ resampled by `coarse_grain`
(`u, v, w, θ` arrays at the staggered locations), the exact subfilter
stress τᵢⱼ (`τ::SymmetricTensorField`, coarse) and vertical heat flux
`hw = (wθ′)‾ − w̄θ̄′` (coarse, cell centres), and the volume means `K`
(resolved KE per unit volume of the coarse field), `e_sgs = ½⟨τₖₖ⟩`,
`hw_mean`, `II_mean` and `III_mean` (the Lumley invariants of the exact
stress, volume means),
`r33` (⟨τ₃₃⟩/⟨τ₁₁⟩).
"""
function filtered_snapshot(sim, r::Integer, Γ)
    g = sim.mesh
    Δ = r * g.dx
    filt = (o, f) -> gaussian_filter!(o, f, Δ)
    out = map((sim.u, sim.v, sim.w, sim.θ)) do f
        ff = _field_like(f, g)
        gaussian_filter!(ff, f, Δ)
        coarse_grain(ff, r)
    end
    τf = SymmetricTensorField(g)
    subfilter_stress!(τf, sim.u, sim.v, sim.w, filt)
    cg = a -> centre_sample(a, r)
    Nc = size(out[4])
    τ = SymmetricTensorField{Float64}(Nc...)
    for (dst, src) in zip(Aeolus.Operators.components(τ), Aeolus.Operators.components(τf))
        dst .= cg(src)
    end
    # subfilter heat flux of θ′ (the background part has zero mean, file header)
    hw = zeros(Nc)
    if Γ > 0
        wc = Field(CellCenter(), g); θc = Field(CellCenter(), g); p = Field(CellCenter(), g)
        a = interior_view(sim.w)
        interior_view(wc) .= (a .+ circshift(a, (0, 0, -1))) ./ 2
        interior_view(θc) .= interior_view(sim.θ) .- VAL_θ0
        interior_view(p) .= interior_view(wc) .* interior_view(θc)
        fw = Field(CellCenter(), g); fθ = Field(CellCenter(), g); fp = Field(CellCenter(), g)
        gaussian_filter!(fw, wc, Δ); gaussian_filter!(fθ, θc, Δ); gaussian_filter!(fp, p, Δ)
        hw = cg(interior_view(fp) .- interior_view(fw) .* interior_view(fθ))
    end
    K = 0.5 * (vmean(abs2, out[1]) + vmean(abs2, out[2]) + vmean(abs2, out[3]))
    II, III = lumley_invariants(τ)
    return (; u = out[1], v = out[2], w = out[3], θ = out[4], τ, hw, K,
              e_sgs = 0.5 * (vmean(τ.xx) + vmean(τ.yy) + vmean(τ.zz)),
              hw_mean = vmean(hw), II_mean = vmean(filter(isfinite, II)),
              III_mean = vmean(filter(isfinite, III)),
              r33 = vmean(τ.zz) / vmean(τ.xx))
end

"""
    centre_sample(a, r) -> Array

A cell-centred fine-grid array at the centres of the r-times-coarser grid
(r even): the mean of the 2 × 2 × 2 fine centres around each coarse
centre. A convex combination, so a positive semi-definite stress stays
so (a Fourier resample of τᵢⱼ does not).
"""
function centre_sample(a::AbstractArray{T,3}, r::Integer) where {T}
    iseven(r) || throw(ArgumentError("centre_sample: r must be even"))
    Nc = size(a) .÷ r
    out = zeros(T, Nc)
    h = r ÷ 2
    @inbounds for k in 1:Nc[3], j in 1:Nc[2], i in 1:Nc[1]
        s = zero(T)
        for c in 0:1, b in 0:1, a_ in 0:1
            s += a[r * (i - 1) + h + a_, r * (j - 1) + h + b, r * (k - 1) + h + c]
        end
        out[i, j, k] = s / 8
    end
    return out
end

"""
    run_dns_case(base, case; nturn = 2, nsave = 8, r = 4) -> NamedTuple

Continue the DNS of `base` from t₀ for `nturn` turnovers as `case`,
recording the full-resolution kinetic energy `K` and the potential
energy of the stratification `PE` (per unit volume) at every step, and a
[`filtered_snapshot`](@ref) at `nsave + 1` equally spaced times
(t₀ included).
"""
function run_dns_case(base, case::Symbol; nturn::Real = 2, nsave::Integer = 8, r::Integer = 4,
                      cfl::Real = 0.5)
    p = case_params(base, case)
    mesh = base.sim0.mesh
    sim = _new_sim(mesh, base.ν, p.f, p.Γ)
    _copy_state!(sim, base.sim0)
    p.Γ > 0 && (interior_view(sim.θ) .= VAL_θ0)
    sim.t = 0.0
    bf = p.Γ > 0 ? background_advection(p.Γ) : nothing
    tsave = range(0, nturn * base.T; length = nsave + 1)
    snaps = [filtered_snapshot(sim, r, p.Γ)]
    # potential energy of the background stratification, (g/θ₀) ⟨θ′²⟩/(2Γ)
    pe() = p.Γ > 0 ? VAL_G / VAL_θ0 * vmean(x -> (x - VAL_θ0)^2, interior_view(sim.θ)) / (2p.Γ) : 0.0
    t = [0.0]; K = [kinetic_energy(sim) / (2π)^3]; PE = [pe()]
    wall = @elapsed for n in 2:length(tsave)
        while sim.t < tsave[n] - 1e-12
            umax = maximum(f -> maximum(abs, interior_view(f)), (sim.u, sim.v, sim.w))
            dt = min(cfl * mesh.dx / umax, tsave[n] - sim.t)
            p.f != 0 && (dt = min(dt, 0.1 / p.f))
            p.Γ > 0 && (dt = min(dt, 0.1 / base.Nbv))
            step_rk3!(sim, dt; body_force! = bf)
            push!(t, sim.t); push!(K, kinetic_energy(sim) / (2π)^3); push!(PE, pe())
        end
        push!(snaps, filtered_snapshot(sim, r, p.Γ))
    end
    return (; case, sim, tsave = collect(tsave), snaps, t, K, PE, wall, p...)
end

# -------------------------------------------------------------------------
# LES from the filtered DNS
# -------------------------------------------------------------------------

# Pressure projection of the LES initial field (the coarse resample is
# divergence-free in the DNS's discrete sense, not in the LES grid's).
function _project!(sim)
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    divergence(sim.div_buf, sim.u, sim.v, sim.w)
    Aeolus.PressureSolvers.solve!(sim.φ, sim.div_buf, sim.solver)
    fill_halos!(sim.φ, sim.bcs)
    ∂x(sim.dφdx, sim.φ); ∂y(sim.dφdy, sim.φ); ∂z(sim.dφdz, sim.φ)
    interior_view(sim.u) .-= interior_view(sim.dφdx)
    interior_view(sim.v) .-= interior_view(sim.dφdy)
    interior_view(sim.w) .-= interior_view(sim.dφdz)
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    return sim
end

"""
    les_closure(model, sim, snap0, Γ) -> closure

`model` ∈ (:smagorinsky, :dynamic, :tke, :rsm). Smagorinsky with the
default ``C_s = 0.17`` (Lilly 1967); dynamic Smagorinsky with plane
averaging (Germano et al. 1991, Lilly 1992); the TKE closure (P56-b)
with ``e = ½τₖₖ`` and the Reynolds-stress closure (P56-d) with ``τ``,
both from the exact subfilter stress of `snap0`. Wrapped in
[`BackgroundGradient`](@ref) when Γ > 0. Every closure gets the same
turbulent Prandtl number as the code's default.
"""
function les_closure(model::Symbol, sim, snap0, Γ)
    g = sim.mesh
    c = if model === :smagorinsky
        StaticSmagorinsky(0.17)
    elseif model === :dynamic
        DynamicSmagorinsky(g)
    elseif model === :tke
        initialize_tke!(TKEClosure(g), snap0.τ)
    elseif model === :rsm
        first(initialize_reynolds_stress!(ReynoldsStressClosure(g), snap0.τ))
    else
        throw(ArgumentError("unknown model $(repr(model))"))
    end
    return Γ > 0 ? BackgroundGradient(c, sim, Γ) : c
end

_inner(c::BackgroundGradient) = c.inner
_inner(c) = c

"""
    les_stress(c, sim) -> SymmetricTensorField

The SGS stress a closure carries or implies on the LES grid, at cell
centres: the Reynolds-stress closure's prognostic τ; for an eddy
viscosity, ``τ = \\tfrac23 e δ − 2ν_t\\bar S`` with the TKE closure's
``e`` (zero trace for the Smagorinsky closures, whose ``e`` is unknown).
"""
function les_stress(c, sim)
    ci = _inner(c)
    g = sim.mesh
    ii = (g.Hx+1:g.Hx+g.Nx, g.Hy+1:g.Hy+g.Ny, g.Hz+1:g.Hz+g.Nz)
    if ci isa ReynoldsStressClosure
        return SymmetricTensorField((Array(ci.τ[m].data[ii...]) for m in 1:6)...)
    end
    f = Aeolus.Operators.compute_sgs!(c, sim)
    uc = map(Aeolus.Operators._to_center_periodic, (sim.u, sim.v, sim.w))
    U = ntuple(m -> (F = Field(CellCenter(), g); interior_view(F) .= uc[m]; F), 3)
    S = filtered_strain(U...)
    τ = SymmetricTensorField(g)
    eddy_viscosity_stress!(τ, S, Array(interior_view(f.νh)))
    if ci isa TKEClosure
        e = Array(interior_view(ci.e))
        τ.xx .+= 2e ./ 3; τ.yy .+= 2e ./ 3; τ.zz .+= 2e ./ 3
    end
    return τ
end

"""
    les_heat_flux(c, sim) -> Array

The vertical SGS heat flux the closure applies, at cell centres:
``−κ_t ∂_zθ`` (eddy diffusivity, `sim.κt`) or ``−c_θ τ_{3j}∂_jθ`` (the
Daly–Harlow tensor of the Reynolds-stress closure), with the full θ.
"""
function les_heat_flux(c::BackgroundGradient, sim)
    v = _view(c, sim)
    f = Aeolus.Operators.compute_sgs!(c, sim)
    g = sim.mesh
    θ = v.θ.data
    H = (g.Hx, g.Hy, g.Hz)
    out = zeros(g.Nx, g.Ny, g.Nz)
    ci = c.inner
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = H[1] + i, H[2] + j, H[3] + k
        dθ = ((θ[I+1, J, K] - θ[I-1, J, K]) / (2g.dx), (θ[I, J+1, K] - θ[I, J-1, K]) / (2g.dy),
              (θ[I, J, K+1] - θ[I, J, K-1]) / (2g.dz))
        if ci isa ReynoldsStressClosure
            Kt = ci.K
            out[i, j, k] = -(Kt.K13.data[I, J, K] * dθ[1] + Kt.K23.data[I, J, K] * dθ[2] +
                             Kt.K33.data[I, J, K] * dθ[3])
        else
            out[i, j, k] = -f.κv.data[I, J, K] * dθ[3]
        end
    end
    return out
end

"""
    run_les(base, dnsrun, model; r = 4, cfl = 0.4) -> NamedTuple

The LES of `model` on the r-times-coarser grid from the filtered DNS at
t₀ of `dnsrun`, advanced to each save time of the DNS run. Returns per
save time: resolved KE `K`, SGS energy `e` (½⟨tr τ⟩ of the carried or
implied stress; NaN for the Smagorinsky closures), mean ν_t, the volume
mean Lumley invariants II, III and ⟨τ₃₃⟩/⟨τ₁₁⟩ of the model stress, the mean vertical
heat flux `hw`, and the model stresses `τs`. `wall` is the LES time.
"""
function run_les(base, dnsrun, model::Symbol; r::Integer = 4, cfl::Real = 0.4)
    g0 = base.sim0.mesh
    N = g0.Nx ÷ r
    mesh = RegularGrid(; size = (N, N, N), extent = (g0.Lx, g0.Ly, g0.Lz), halo = (3, 3, 3))
    sim = _new_sim(mesh, base.ν, dnsrun.f, dnsrun.Γ)
    s0 = dnsrun.snaps[1]
    interior_view(sim.u) .= s0.u; interior_view(sim.v) .= s0.v; interior_view(sim.w) .= s0.w
    interior_view(sim.θ) .= dnsrun.Γ > 0 ? VAL_θ0 : 0.0
    _project!(sim)
    c = les_closure(model, sim, s0, dnsrun.Γ)
    bf = dnsrun.Γ > 0 ? background_advection(dnsrun.Γ) : nothing
    K = Float64[]; e = Float64[]; νt = Float64[]; II = Float64[]; III = Float64[]
    r33 = Float64[]; hw = Float64[]
    τs = SymmetricTensorField{Float64}[]
    function record!()
        τ = les_stress(c, sim)
        push!(τs, τ)
        push!(K, kinetic_energy(sim) / (2π)^3)
        ci = _inner(c)
        push!(e, ci isa TKEClosure ? vmean(interior_view(ci.e)) :
                 ci isa ReynoldsStressClosure ? 0.5 * (vmean(τ.xx) + vmean(τ.yy) + vmean(τ.zz)) : NaN)
        push!(νt, vmean(interior_view(sim.νt)))
        I2, I3 = lumley_invariants(τ)
        push!(II, vmean(filter(isfinite, I2))); push!(III, vmean(filter(isfinite, I3)))
        push!(r33, vmean(τ.zz) / vmean(τ.xx))
        push!(hw, dnsrun.Γ > 0 ? vmean(les_heat_flux(c, sim)) : 0.0)
    end
    record!()
    wall = @elapsed for n in 2:length(dnsrun.tsave)
        while sim.t < dnsrun.tsave[n] - 1e-12
            umax = maximum(f -> maximum(abs, interior_view(f)), (sim.u, sim.v, sim.w))
            dt = min(cfl * mesh.dx / umax, dnsrun.tsave[n] - sim.t)
            dnsrun.f != 0 && (dt = min(dt, 0.1 / dnsrun.f))
            dnsrun.Γ > 0 && (dt = min(dt, 0.1 / base.Nbv))
            step_rk3!(sim, dt; sgs = c, body_force! = bf)
        end
        record!()
    end
    return (; model, sim, closure = c, K, e, νt, II, III, r33, hw, τs, wall)
end
