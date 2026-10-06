# Era VI, Phase 58 (#170) — the TurbulenceStability contract:
# gradient-Richardson stability functions, N² and Ri_g.
#
#   1. limits of every implementer: f(0) = 1, f ≥ 0, non-increasing in
#      Ri > 0; the Lilly cutoff is exactly 0 above Ri_c and continuous there;
#   2. MOST round trip: K_m, K_h of MOSTConsistent reproduce the surface-
#      layer fluxes of the analytic MOST profiles;
#   3. critical Ri of the log-linear and BDBH similarity functions;
#   4. the pointwise N² and Ri_g fields on analytic flows (exact for linear
#      profiles; second order for a tanh shear layer);
#   5. the fog column delegates to ShortTail bit for bit;
#   6. the mean-profile N² of TimeIntegration and Anelastic bit for bit;
#   7. cloudy-air N² (#239): condensate loading, the saturated N² (Durran &
#      Klemp 1982 in a saturated environment, a lifted parcel at a cloud
#      top) — ≈ 0 on a moist adiabat, within 4 % of the fog column's parcel
#      N² — and the dry / qv forms bit for bit.

using Aeolus.TurbulenceStability
using Aeolus.TurbulenceStability: AbstractRiStability
import Aeolus.SurfaceExchange: phi_m, phi_h, psi_m, psi_h

# The log-linear similarity functions, φ = 1 + αζ in stable air (Webb 1970;
# Businger et al. 1971 with α ≈ 5), Businger–Dyer (γ = 16) in unstable air.
struct LogLinearMOST <: AbstractStabilityFunctions
    α::Float64
end
const _BDBH = BusingerDyerBeljaarsHoltslag()
phi_m(s::LogLinearMOST, ζ::Real) = ζ < 0 ? phi_m(_BDBH, ζ) : 1 + s.α * ζ
phi_h(s::LogLinearMOST, ζ::Real) = ζ < 0 ? phi_h(_BDBH, ζ) : 1 + s.α * ζ
psi_m(s::LogLinearMOST, ζ::Real) = ζ < 0 ? psi_m(_BDBH, ζ) : -s.α * ζ
psi_h(s::LogLinearMOST, ζ::Real) = ζ < 0 ? psi_h(_BDBH, ζ) : -s.α * ζ

# 4th-order centred first derivative
_d4(f, z, h) = (8 * (f(z + h) - f(z - h)) - (f(z + 2h) - f(z - 2h))) / (12h)

@testset "Era VI — TurbulenceStability (P58-a, #170)" begin

@testset "1. limits of every implementer" begin
    impls = [LillyCutoff(), BrownEtAl1994(), ShortTail(), LongTail(),
             MOSTConsistent(), MOSTConsistent(LogLinearMOST(5.0))]
    Ris = vcat(range(-5, 0; length = 201), range(0, 10; length = 2001), [1e2, 1e4])
    for sf in impls
        for fac in (stability_factor_m, stability_factor_h)
            @test abs(fac(sf, 0.0) - 1) <= eps()
            fs = [fac(sf, Ri) for Ri in Ris]
            @test all(isfinite, fs)
            @test all(>=(0), fs)
            pos = [fac(sf, Ri) for Ri in Ris if Ri > 0]
            @test all(diff(pos) .<= 0)           # non-increasing in stable air
            # unstable air mixes at least as much as neutral air
            @test all(fac(sf, Ri) >= 1 for Ri in Ris if Ri < 0)
        end
        @test neutral_prandtl(sf) > 0
    end

    # Lilly / Mason: exactly 0 at and above Ri_c, continuous at Ri_c
    for Ri_c in (0.25, 1 / 3)
        sf = LillyCutoff(Ri_c = Ri_c)
        @test stability_factor_m(sf, Ri_c) == 0
        @test all(stability_factor_m(sf, Ri) == 0 for Ri in Ri_c .* (1:0.5:10))
        @test all(stability_factor_h(sf, Ri) == 0 for Ri in Ri_c .* (1:0.5:10))
        # the left limit: f(Ri_c − δ) = √(δ/Ri_c) → 0. In Float64 the closest
        # point below Ri_c gives √(eps) ≈ 1e-8, so the 1e-12 continuity is
        # measured in BigFloat at δ = 1e-26
        δ = big"1e-26"
        f_left = stability_factor_m(LillyCutoff(Ri_c = BigFloat(Ri_c)), BigFloat(Ri_c) - δ)
        @test f_left < 1e-12
        @test abs(f_left - sqrt(δ / BigFloat(Ri_c))) < 1e-30
    end
    # the TKE-equilibrium form K = l²(S² − N²/Ri_c)^{1/2}
    let sf = LillyCutoff(), S2 = 4e-4, N2 = 3e-5
        @test sqrt(S2) * stability_factor_m(sf, N2 / S2) ≈ sqrt(S2 - N2 / 0.25) rtol = 1e-14
    end

    # Brown et al. (1994) / MONC constants, quoted values
    let sf = BrownEtAl1994()
        @test stability_factor_m(sf, 0.1) ≈ (1 - 0.4)^4 rtol = 1e-15
        @test stability_factor_h(sf, 0.1) ≈ 0.6^4 * (1 - 0.12) rtol = 1e-15
        @test stability_factor_m(sf, -0.5) ≈ 3.0 rtol = 1e-15     # √(1 + 8)
        @test stability_factor_h(sf, -0.5) ≈ sqrt(21.0) rtol = 1e-15
        @test stability_factor_m(sf, 0.25) == 0 && stability_factor_h(sf, 0.3) == 0
        @test stability_factor_m(sf, nextfloat(0.25, -1)) < 1e-60
        @test neutral_prandtl(sf) == 0.7
    end
    # Louis (1979), b = d = 5: f_m(Ri = 3) = 1/(1 + 30/4) = 2/17
    let sf = LongTail()
        @test stability_factor_m(sf, 3.0) ≈ 2 / 17 rtol = 1e-15
        @test stability_factor_h(sf, 3.0) ≈ 1 / (1 + 45 * 4) rtol = 1e-15
        # long tail: no critical Ri, f_m → √5/(10 √Ri), with a relative
        # correction 1/√(20 Ri) (2e-5 at Ri = 1e8)
        @test stability_factor_m(sf, 1e8) * sqrt(1e8) ≈ sqrt(5) / 10 rtol = 1e-4
        @test stability_factor_m(sf, 1e4) > 1e-3
        @test neutral_prandtl(sf) == 0.74
    end
    # short tail: (1 + 5 Ri)⁻² — it falls below the long tail for Ri > 0
    @test stability_factor_m(ShortTail(), 1.0) == 1 / 36
    @test all(stability_factor_m(ShortTail(), Ri) < stability_factor_m(LongTail(), Ri)
              for Ri in 0.01:0.01:10)

    @test gradient_richardson(1e-4, 4e-4) == 0.25
    @test gradient_richardson(1e-4, 0.0) == 1e-4 / RI_SHEAR2_FLOOR
    @test isfinite(gradient_richardson(-1e-4, 0.0))
    # N²: the Boussinesq and logarithmic forms agree to O(Δθ/θ)
    @test buoyancy_frequency_squared(300.0, 301.0, 100.0, 9.81, 300.0) ≈ 9.81 / 300 / 100
    @test buoyancy_frequency_squared(300.0, 301.0, 100.0, 9.81) ≈
          buoyancy_frequency_squared(300.0, 301.0, 100.0, 9.81, 300.5) rtol = 1e-5
end

@testset "2. MOST round trip (MOSTConsistent)" begin
    κ, g, θ₀, u_star, z0, z = 0.4, 9.81, 290.0, 0.3, 0.1, 10.0
    worst_m = 0.0; worst_h = 0.0
    for sf in (BusingerDyerBeljaarsHoltslag(), LogLinearMOST(5.0))
        most = MOSTConsistent(sf)
        @test stability_factor_m(most, 0.0) == 1 && stability_factor_h(most, 0.0) == 1
        for ζ in range(-2, 10; length = 49)
            L = ζ == 0 ? Inf : z / ζ
            θ_star = ζ == 0 ? 0.0 : u_star^2 * θ₀ / (κ * g * L)
            u(zz) = u_star / κ * (log(zz / z0) - psi_m(sf, zz / L) + psi_m(sf, z0 / L))
            θ(zz) = θ₀ + θ_star / κ * (log(zz / z0) - psi_h(sf, zz / L) + psi_h(sf, z0 / L))
            S  = _d4(u, z, 1e-2)
            θz = _d4(θ, z, 1e-2)
            Ri = gradient_richardson(g / θ₀ * θz, S^2)
            # the inversion recovers ζ
            @test most_stability_parameter(most, Ri) ≈ ζ atol = 1e-7
            Km = (κ * z)^2 * S * stability_factor_m(most, Ri)
            Kh = (κ * z)^2 * S * stability_factor_h(most, Ri)
            em = abs(Km - u_star^2 / S) / (u_star^2 / S)
            eh = ζ == 0 ? abs(Kh * θz) : abs(Kh * θz - u_star * θ_star) / abs(u_star * θ_star)
            worst_m = max(worst_m, em); worst_h = max(worst_h, eh)
            @test em < 1e-6
            @test eh < 1e-6
        end
    end
    @info "P58-a MOST round trip, ζ ∈ [−2, 10], BDBH and log-linear α = 5" max_rel_err_Km = worst_m max_rel_err_KhΘz = worst_h
end

@testset "3. critical Richardson number" begin
    ll = LogLinearMOST(5.0)
    Ri_ll = most_gradient_richardson(ll, 1e4)
    @test abs(Ri_ll - 0.2) < 1e-3
    @test Ri_ll < 0.2                                   # approached from below
    # beyond the critical value the MOST-consistent factors vanish
    @test stability_factor_m(MOSTConsistent(ll), 0.2) == 0
    @test stability_factor_h(MOSTConsistent(ll), 0.25) == 0
    @test stability_factor_m(MOSTConsistent(ll), 0.19) > 0
    # Beljaars–Holtslag has no critical value
    Ri_bh = most_gradient_richardson(BusingerDyerBeljaarsHoltslag(), 100.0)
    @test Ri_bh > 1
    @test stability_factor_m(MOSTConsistent(), 5.0) > 0
    # unstable Businger–Dyer with γ_m = γ_h: Ri = ζ exactly
    @test most_gradient_richardson(BusingerDyerBeljaarsHoltslag(), -1.5) ≈ -1.5 rtol = 1e-14
    @info "P58-a critical Ri" log_linear_Ri_at_ζ_1e4 = Ri_ll BDBH_Ri_at_ζ_100 = Ri_bh
end

# Fill every data point (halos included) of a field from f(z) of its
# cell-centre height (u on FaceX and v on FaceY share the centre z).
function _fill_z!(fld, f)
    m = fld.mesh
    for k in axes(fld.data, 3)
        z = (k - m.Hz - 0.5) * m.dz
        fld.data[:, :, k] .= f(z)
    end
    return fld
end

function _ri_field(Nz, Lz, ufun, θfun; qv = nothing, θ₀ = 300.0, g = 9.81, N = 4)
    m = RegularGrid(; size = (N, N, Nz), extent = (Lz, Lz, Lz), halo = (1, 1, 1))
    u = _fill_z!(Field(FaceX(), m), ufun)
    v = Field(FaceY(), m); w = Field(FaceZ(), m)
    θ = _fill_z!(Field(CellCenter(), m), θfun)
    N2 = Field(CellCenter(), m); Ri = Field(CellCenter(), m)
    if qv === nothing
        buoyancy_frequency_squared!(N2, θ; θ₀ = θ₀, g = g)
    else
        q = _fill_z!(Field(CellCenter(), m), z -> qv)
        buoyancy_frequency_squared!(N2, θ, q; θ₀ = θ₀, g = g)
    end
    gradient_richardson!(Ri, N2, u, v, w)
    S2 = strain_rate_squared!(Field(CellCenter(), m), u, v, w)
    @test interior_view(Ri) == gradient_richardson.(interior_view(N2), interior_view(S2))
    return m, N2, Ri, S2
end

@testset "4. pointwise N² and Ri_g fields" begin
    θ₀, g, S, Γ = 300.0, 9.81, 0.02, 0.03
    # linear profiles on 16³: Ri_g = gΓ/(θ₀S²) everywhere
    m, N2, Ri, S2f = _ri_field(16, 160.0, z -> S * z, z -> θ₀ + Γ * z)
    Ri_exact = g * Γ / (θ₀ * S^2)
    err = maximum(abs, interior_view(Ri) .- Ri_exact) / Ri_exact
    @test err < 1e-12
    @test maximum(abs, interior_view(S2f) .- S^2) < 1e-12 * S^2   # |S|² = (∂u/∂z)²
    @test maximum(abs, interior_view(N2) .- g * Γ / θ₀) < 1e-12 * g * Γ / θ₀
    # moist: θ_v = θ (1 + 0.61 q_v), as in the moist buoyancy
    q = 0.01
    _, N2q, _, _ = _ri_field(16, 160.0, z -> S * z, z -> θ₀ + Γ * z; qv = q)
    @test maximum(abs, interior_view(N2q) .- g * Γ * (1 + 0.61q) / θ₀) <
          1e-12 * g * Γ / θ₀
    # no shear: the floor keeps Ri finite
    _, _, Ri0, _ = _ri_field(8, 80.0, z -> 1.0, z -> θ₀ + Γ * z)
    @test all(isfinite, interior_view(Ri0))
    @test interior_view(Ri0)[1] ≈ g * Γ / θ₀ / RI_SHEAR2_FLOOR

    # tanh shear layer, constant N²: Ri = J cosh⁴((z − z_c)/δ), J = N²δ²/U²
    U, δ, N2c = 2.0, 10.0, 1e-4
    Lz = 8δ; zc = Lz / 2
    J = N2c * δ^2 / U^2
    errs = Float64[]
    for Nz in (32, 64, 128)
        mm, _, Rt, _ = _ri_field(Nz, Lz, z -> U * tanh((z - zc) / δ),
                              z -> θ₀ * (1 + N2c * z / g))
        zs = [(k - 0.5) * mm.dz for k in 1:Nz]
        R = interior_view(Rt)[1, 1, :]
        sel = abs.(zs .- zc) .<= 2δ
        push!(errs, maximum(abs.(R[sel] .- J .* cosh.((zs[sel] .- zc) ./ δ) .^ 4) ./
                            (J .* cosh.((zs[sel] .- zc) ./ δ) .^ 4)))
    end
    orders = log2.(errs[1:end-1] ./ errs[2:end])
    @test all(o -> abs(o - 2) <= 0.15, orders)
    @info "P58-a Ri_g fields" linear_rel_err = err tanh_rel_errs = errs tanh_orders = orders
end

# The pre-#170 implementations, verbatim, as references for "bit for bit".
_fstab_old(Ri) = Ri >= 0 ? 1 / (1 + 5Ri)^2 : sqrt(1 - 16Ri)

function _fog_K_old(col; c = Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS)
    κ = Aeolus.RadiationFog._KARMAN
    s = col.set; nz = length(col.T)
    K = zeros(nz + 1)
    for k in 1:nz - 1
        zi = col.zf[k + 1]
        Δz = col.z[k + 1] - col.z[k]
        l  = κ * zi / (1 + κ * zi / s.λ)
        S2 = ((col.u[k + 1] - col.u[k])^2 + (col.v[k + 1] - col.v[k])^2) / Δz^2
        S2 = max(S2, 1e-8)
        N2 = moist_stability(col, k; c = c)
        Ri = N2 / S2
        K[k + 1] = l^2 * sqrt(S2) * _fstab_old(Ri) + s.K_min
    end
    return K
end

function _brunt_vaisala_old(sim::BoussinesqSimulation{T}) where {T}
    sim.has_buoyancy || return zero(T)
    m = sim.mesh
    ii, jj, kk = interior(m)
    n_h = length(ii) * length(jj)
    invdz = 1 / m.dz
    coef  = sim.g_accel / sim.θ₀
    N²max = zero(T)
    prev  = zero(T)
    d = sim.θ.data
    @inbounds for (kc, k) in enumerate(kk)
        s = zero(T)
        for j in jj, i in ii
            s += d[i, j, k]
        end
        mean_k = s / n_h
        if kc > 1
            N²max = max(N²max, coef * (mean_k - prev) * invdz)
        end
        prev = mean_k
    end
    return sqrt(max(N²max, zero(T)))
end

function _brunt_vaisala_profile_old(θ̄::Vector{T}, dz::T, g::T) where {T}
    Nz = length(θ̄)
    N² = zeros(T, Nz)
    Nz == 1 && return N²
    lθ = log.(θ̄)
    N²[1] = g * (lθ[2] - lθ[1]) / dz
    for k in 2:Nz-1
        N²[k] = g * (lθ[k+1] - lθ[k-1]) / (2dz)
    end
    N²[Nz] = g * (lθ[Nz] - lθ[Nz-1]) / dz
    return N²
end

@testset "5. the fog column delegates to ShortTail bit for bit" begin
    @test !isdefined(Aeolus.RadiationFog, :_fstab)
    Ris = vcat(.-(10.0 .^ range(-8, 2; length = 300)), 0.0, 10.0 .^ range(-8, 4; length = 300))
    @test all(stability_factor_m(ShortTail(), Ri) === _fstab_old(Ri) for Ri in Ris)
    @test all(stability_factor_h(ShortTail(), Ri) === _fstab_old(Ri) for Ri in Ris)
    col = parisfog_2011_column()
    run!(col, 3600)                                      # 1 h of the ParisFog case
    K = eddy_diffusivity(col)
    @test K == _fog_K_old(col)
    ex = surface_exchange(col)
    lm = log(col.z[1] / col.set.z0m)
    @test ex.C_D == Aeolus.RadiationFog._KARMAN^2 / lm^2 * _fstab_old(ex.Ri_b)
    @info "P58-a ParisFog 1 h: K profile (m² s⁻¹), first interfaces" K = K[1:6] Ri_b = ex.Ri_b
end

@testset "6. mean-profile N² of TimeIntegration and Anelastic, bit for bit" begin
    rng = Xoshiro(170)
    g = RegularGrid(; size = (6, 5, 24), extent = (60.0, 50.0, 240.0), halo = (2, 2, 2))
    sim = BoussinesqSimulation(g; θ₀ = 300.0)
    set!(sim.θ, (x, y, z) -> 300.0 + 0.004z + 0.3 * sin(z / 17))
    interior_view(sim.θ) .+= 0.05 .* randn(rng, 6, 5, 24)
    N_new = Aeolus.TimeIntegration._brunt_vaisala(sim)
    @test N_new > 0
    @test N_new === _brunt_vaisala_old(sim)
    @test cfl_diagnostics(sim, CFLConstraints()).N === _brunt_vaisala_old(sim)
    for θ̄ in (300.0 .* exp.(1e-4 .* (0:39) .* 25 ./ 9.81),     # constant N² = 1e-4
              290.0 .+ 0.01 .* (0:39) .* 25 .+ 0.5 .* rand(rng, 40))
        @test Aeolus.Anelastic._brunt_vaisala(θ̄, 25.0, 9.81) ==
              _brunt_vaisala_profile_old(θ̄, 25.0, 9.81)
    end
    @test Aeolus.Anelastic._brunt_vaisala([300.0], 25.0, 9.81) == [0.0]
    # the exactness that motivated the log form survives
    N2 = Aeolus.Anelastic._brunt_vaisala(300.0 .* exp.(1e-4 .* (0:39) .* 25 ./ 9.81), 25.0, 9.81)
    @test maximum(abs, N2 .- 1e-4) < 1e-15
end

# The pre-#239 pointwise N² kernel, verbatim, as the reference for "bit for bit".
function _n2_field_old!(N2, θ, qv, θ₀, g)
    m = θ.mesh
    qd = qv === nothing ? nothing : qv.data
    θd, nd = θ.data, N2.data
    face(i, j, k) = qd === nothing ? 0.5 * (θd[i, j, k-1] + θd[i, j, k]) :
        0.5 * (θd[i, j, k-1] + θd[i, j, k]) * (1 + 0.61 * (0.5 * (qd[i, j, k-1] + qd[i, j, k])))
    ii, jj, kk = interior(m)
    for k in kk, j in jj, i in ii
        nd[i, j, k] = buoyancy_frequency_squared(face(i, j, k), face(i, j, k + 1), m.dz, g, θ₀)
    end
    return N2
end

isdefined(@__MODULE__, :moist_adiabat_column!) ||
    include(joinpath(@__DIR__, "support", "moist_adiabat.jl"))

@testset "7. cloudy-air N² (#239)" begin
    MT = Aeolus.MoistThermodynamics; c = MT.DEFAULT_CONSTANTS
    θ₀, g = 290.0, 9.81

    # (a) dry and qv-only forms unchanged bit for bit; an all-zero condensate
    #     field gives the qv form bit for bit
    rng = Xoshiro(239)
    m = RegularGrid(; size = (5, 4, 12), extent = (50.0, 40.0, 120.0), halo = (1, 1, 1))
    θ = Field(CellCenter(), m); qv = Field(CellCenter(), m)
    θ.data .= 290.0 .+ 0.5 .* randn(rng, size(θ.data))
    qv.data .= 0.008 .+ 1e-3 .* rand(rng, size(qv.data))
    N2 = Field(CellCenter(), m); N2o = Field(CellCenter(), m)
    buoyancy_frequency_squared!(N2, θ; θ₀ = θ₀, g = g)
    @test N2.data == _n2_field_old!(N2o, θ, nothing, θ₀, g).data
    buoyancy_frequency_squared!(N2, θ, qv; θ₀ = θ₀, g = g)
    @test N2.data == _n2_field_old!(N2o, θ, qv, θ₀, g).data
    N2c = Field(CellCenter(), m)
    buoyancy_frequency_squared!(N2c, θ, qv, Field(CellCenter(), m); θ₀ = θ₀, g = g, p = 1e5)
    @test N2c.data == N2.data
    buoyancy_frequency_squared!(N2c, θ, qv, nothing; θ₀ = θ₀, g = g)
    @test N2c.data == N2.data
    @test_throws DimensionMismatch buoyancy_frequency_squared!(N2c, θ, qv, Field(CellCenter(), m);
                                                               θ₀ = θ₀, g = g, p = zeros(3))

    # (b) condensate loading in an unsaturated cell: uniform θ and qv, cloud
    #     water only in the cell above → N² = (g/θ₀) θ (−q_c/2) / Δz exactly
    θ.data .= 288.0; qv.data .= 0.006
    qc = Field(CellCenter(), m)
    k = m.Hz + 6
    qc.data[:, :, k + 1] .= 4e-4
    buoyancy_frequency_squared!(N2c, θ, qv, qc; θ₀ = θ₀, g = g, p = 9.5e4)
    @test interior_view(N2c)[2, 2, 6] ≈ (g / θ₀) * 288.0 * (-0.5 * 4e-4) / m.dz rtol = 1e-12
    @test interior_view(N2c)[2, 2, 6] < 0          # loaded air above: unstable
    @test interior_view(N2c)[2, 2, 3] == 0          # far from the cloud: neutral as before

    # (c) moist adiabat, saturated, constant total water: N² ≈ 0, second order
    errs = Float64[]; N2_qv = 0.0
    for Nz in (20, 40, 80)
        mc = RegularGrid(; size = (2, 2, Nz), extent = (100.0, 100.0, 1000.0), halo = (1, 1, 1))
        θc = Field(CellCenter(), mc); qvc = Field(CellCenter(), mc); qcc = Field(CellCenter(), mc)
        pv = moist_adiabat_column!(θc, qvc, qcc)
        @test all(>(0), qcc.data)
        Nf = Field(CellCenter(), mc)
        buoyancy_frequency_squared!(Nf, θc, qvc, qcc; θ₀ = θ₀, g = g, p = pv)
        push!(errs, maximum(abs, interior_view(Nf)))
        buoyancy_frequency_squared!(Nf, θc, qvc; θ₀ = θ₀, g = g)
        N2_qv = minimum(interior_view(Nf))
    end
    @test N2_qv > 1.4e-4                            # the θ_v form calls it stable
    @test errs[end] < 1e-6 * N2_qv                  # the saturated form: neutral
    @test all(o -> abs(o - 2) < 0.1, log2.(errs[1:end-1] ./ errs[2:end]))

    # (d) the contract function
    sbf = MT.saturated_buoyancy_frequency_squared
    Γd = MT.dry_adiabatic_lapse_rate()
    @test sbf(280.0, 2e4, 0.0, 0.0, 4e-3, 0.0, 0.0) ≈ (c.grav / 280.0) * 4e-3 rtol = 1e-3   # 177 K: q → 0, dry N²
    θs, ps = 290.0, 9e4
    Π = MT.exner_function(ps); T = θs * Π; Γm = MT.moist_adiabatic_lapse_rate(T, ps)
    qs = MT.saturation_specific_humidity(T, ps); qcs = 1e-3
    rs = MT.saturation_mixing_ratio(T, ps); Lv = MT.latent_heat_vapor(T); Rd = MT.R_d(c)
    # the environment's saturated q_v gradient for a θ gradient dθ (hydrostatic, R_d T)
    dqs_env(dθ) = (h = 1.0; dTdz = Π * dθ - Γd; dp = exp(c.grav * h / (Rd * T));
                   (MT.saturation_specific_humidity(T + dTdz * h, ps / dp) -
                    MT.saturation_specific_humidity(T - dTdz * h, ps * dp)) / (2h))
    # Durran & Klemp (1982) eq. 36, by hand: (g/T)(∂T/∂z + Γ_m)(1 + L r_s/(R_d T)) − g ∂q_w/∂z/(1 + q_w)
    dk82(dθ, dqc) = (c.grav / T) * (Π * dθ - Γd + Γm) * (1 + Lv * rs / (Rd * T)) -
                    c.grav * (dqs_env(dθ) + dqc) / (1 + qs + qcs)
    dk_rel = Float64[]
    for dθ in (-2e-3, 3e-3, 6e-3, 1e-2), dqc in (0.0, -2e-6)
        mine = sbf(θs, ps, qs, qcs, dθ, dqs_env(dθ), dqs_env(dθ) + dqc)
        push!(dk_rel, abs(mine / dk82(dθ, dqc) - 1))
    end
    @test all(<(0.02), dk_rel)                    # saturated environment: DK82 to O(q)
    # conditional instability: a θ gradient between the moist and the dry
    # adiabat is dry-stable and saturated-unstable
    dθc = 0.5 * (Γd - Γm) / Π
    @test sbf(θs, ps, qs, qcs, dθc, dqs_env(dθc), dqs_env(dθc)) < 0
    @test sbf(θs, ps, qs, qcs, 3e-3, dqs_env(3e-3), dqs_env(3e-3)) < (c.grav / θs) * 3e-3
    # against a lifted parcel (reversible moist adiabat over 1 m, θ_ρ of the
    # dynamics), in saturated environments and at a cloud top under drier air
    function parcel_n2(dθ, dqv, dqc; h = 1.0)
        dp = exp(c.grav * h / (Rd * T))
        Tp = T - Γm * h; pp = ps / dp
        qvp = MT.saturation_specific_humidity(Tp, pp); qcp = qs + qcs - qvp
        θρp = Tp / MT.exner_function(pp) * (1 + 0.61qvp - qcp)
        θρe = (θs + dθ * h) * (1 + 0.61 * (qs + dqv * h) - (qcs + dqc * h))
        return -(c.grav / θs) * (θρp - θρe) / h
    end
    pc_abs = Float64[]
    for (dθ, dqv, dqc) in ((3e-3, dqs_env(3e-3), 0.0), (-1e-3, dqs_env(-1e-3), -1e-6),
                           (0.0, -1e-4, -1e-5), (2e-3, -2e-4, -2e-5))   # the last two: cloud top
        push!(pc_abs, abs(sbf(θs, ps, qs, qcs, dθ, dqv, dqv + dqc) - parcel_n2(dθ, dqv, dqc)))
    end
    @test all(<(2e-6), pc_abs)
    # cloud top, same θ above, 0.1 g/kg/m drier: the lifted parcel is buoyant
    # (vapour) despite its water (loading). Durran & Klemp's form, which
    # assumes a saturated environment, calls it stable.
    @test sbf(θs, ps, qs, qcs, 0.0, -1e-4, -1.1e-4) < 0
    @test (c.grav / T) * (-Γd + Γm) * (1 + Lv * rs / (Rd * T)) + c.grav * 1.1e-4 / (1 + qs + qcs) > 0

    # (e) against the fog column's parcel N² (RadiationFog.moist_stability:
    #     enthalpy-conserving displacement with the contract's saturation
    #     adjustment, θ_ρ): saturated layers of 8, 5 and 3 K/km
    rels = Float64[]
    for lapse in (8e-3, 5e-3, 3e-3)
        zf = collect(0.0:20.0:1000.0); nz = length(zf) - 1
        zc = (zf[1:end-1] .+ zf[2:end]) ./ 2
        T = 276.0 .- lapse .* zc
        p = 1.0e5 .* (T ./ 276.0) .^ (c.grav / (MT.R_d(c) * lapse))     # hydrostatic
        wet = Aeolus.RadiationFog.FogColumn(zf, T, 1.3 .* MT.saturation_specific_humidity.(T, p))
        @test all(wet.ql .> 1e-4)                 # both displaced parcels stay saturated
        k = 10
        N_parcel = 0.5 * (Aeolus.RadiationFog.moist_stability(wet, k - 1) +
                          Aeolus.RadiationFog.moist_stability(wet, k))
        mf = RegularGrid(; size = (1, 1, nz), extent = (20.0, 20.0, 1000.0), halo = (1, 1, 1))
        θf = Field(CellCenter(), mf); qvf = Field(CellCenter(), mf); qlf = Field(CellCenter(), mf)
        pv = zeros(nz + 2)
        for kk in 1:nz
            θf.data[:, :, kk+1] .= wet.T[kk] / MT.exner_function(wet.p[kk])
            qvf.data[:, :, kk+1] .= wet.qv[kk]; qlf.data[:, :, kk+1] .= wet.ql[kk]; pv[kk+1] = wet.p[kk]
        end
        Nf = Field(CellCenter(), mf)
        buoyancy_frequency_squared!(Nf, θf, qvf, qlf; θ₀ = θf.data[2, 2, k+1], g = c.grav, p = pv)
        N_dk = interior_view(Nf)[1, 1, k]
        buoyancy_frequency_squared!(Nf, θf, qvf; θ₀ = θf.data[2, 2, k+1], g = c.grav)
        @test interior_view(Nf)[1, 1, k] > N_dk + 0.9e-4     # the θ_v form overstates it
        push!(rels, abs(N_dk - N_parcel) / abs(N_parcel))
    end
    @test all(<(0.04), rels)
    @info "#239 cloudy-air N²" moist_adiabat_maxabs = Tuple(errs) θv_form_on_it = N2_qv fogcolumn_rel_diff = Tuple(rels) dk82_rel = maximum(dk_rel) lifted_parcel_absdiff = maximum(pc_abs)
end

end # testset
