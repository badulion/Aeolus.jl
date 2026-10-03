# Era VI, P56-a (#177): SGS diffusion of qv and of every transported scalar
# with κ_t.
#
#   1. qv now diffuses: step! (and step_rk3!) apply the closure's scalar flux
#      to qv; before #177 qv was advected but never diffused (a code bug).
#   2. Conservation of Σ qv dV and monotone decay of Σ qv² with a variable κt.
#   3. The flux-divergence kernel against the analytic steady state of
#      d/dz(κ dϕ/dz) = 0 with κ = κ₀(1 + z/H), fixed flux below, fixed value
#      above; 2nd-order convergence.
#   5. θ and momentum are unchanged by the qv diffusion.

using Test
using Aeolus
using Aeolus.Fields: interior_view, Field
using Aeolus.Grids: interior
using Random: Xoshiro, rand, randn
using Printf

# κ = κ₀ (or a prescribed interior field), ν = 0. (Own name: the closure
# test file defines a similar PrescribedKappa in the same namespace.)
struct ScalarKappa{K} <: AbstractSGSClosure
    κ::K
end
function Aeolus.Operators.sgs_coefficients!(f::SGSFields, c::ScalarKappa, sim)
    fill!(f.νh.data, 0)
    c.κ isa Real ? fill!(f.κh.data, c.κ) : (interior_view(f.κh) .= c.κ)
    return f
end

# Fill every storage cell (i, j) of a 1 × 1 × Nz column field from `col`
# (length Nz + 2Hz, storage order).
function _set_column!(f::Field, col)
    for k in axes(f.data, 3), j in axes(f.data, 2), i in axes(f.data, 1)
        f.data[i, j, k] = col[k]
    end
    return f
end

@testset "P56-a: SGS diffusion of qv and tracers with κ_t" begin

    @testset "1. qv diffuses at the discrete eigenvalue" begin
        # qv = sin(2πz/Lz), u = 0, κ ≡ κ₀: an exact eigenvector of the discrete
        # diffusion, eigenvalue λ = −κ₀ (4/dz²) sin²(π dz/Lz). Before #177
        # step! left qv untouched (amplitude 1, relative error ≈ 0.4 here).
        g = RegularGrid(; size = (4, 4, 32), extent = (1.0, 1.0, 2.0), halo = (2, 2, 2))
        κ0 = 6e-2
        λ = -κ0 * (4 / g.dz^2) * sin(π * g.dz / g.Lz)^2
        zc = [(k - 0.5) * g.dz for k in 1:g.Nz]
        prof = sin.(2π .* zc ./ g.Lz)
        amp(f) = sum(interior_view(f)[2, 3, :] .* prof) / sum(abs2, prof)
        γ = ARK2().γ
        R_ark2(z) = (1 + (1 - γ) * z / (1 - γ * z)) / (1 - γ * z)
        cases = (("step!", 0.02, 100, z -> 1 + z,
                  (s, h) -> step!(s, h; sgs = ScalarKappa(κ0))),
                 ("step_rk3!", 0.02, 100, z -> 1 + z + z^2 / 2 + z^3 / 6,
                  (s, h) -> step_rk3!(s, h; sgs = ScalarKappa(κ0))),
                 ("step_imex!", 0.5, 20, R_ark2,
                  (s, h) -> step_imex!(s, h, ARK2(); sgs = ScalarKappa(κ0), tol = 1e-14, maxit = 500)))
        for (name, h, nst, R, adv!) in cases
            sim = BoussinesqSimulation(g; moisture = true)     # ν = 0, θ₀ = 0, C_s = 0
            for k in 1:g.Nz
                interior_view(sim.qv)[:, :, k] .= prof[k]
                interior_view(sim.θ)[:, :, k] .= prof[k]
            end
            for _ in 1:nst
                adv!(sim, h)
            end
            rel = abs(amp(sim.qv) / R(h * λ)^nst - 1)
            @test rel <= 1e-10
            @test amp(sim.qv) < 0.75                       # it decayed
            # qv and θ obey the same equation here: identical decay
            @test interior_view(sim.qv) == interior_view(sim.θ)
            @printf("  [qv] %-10s %3d steps of %.2f s, hλ = %.4f: amplitude %.4f, rel. error vs R(hλ)^n %.1e\n",
                    name, nst, h, h * λ, amp(sim.qv), rel)
        end

        # molecular: ν∇²qv, as for θ (and as step_imex! always did)
        sim = BoussinesqSimulation(g; moisture = true, ν = κ0)
        for k in 1:g.Nz
            interior_view(sim.qv)[:, :, k] .= prof[k]
        end
        for _ in 1:100
            step!(sim, 0.02)
        end
        relν = abs(amp(sim.qv) / (1 + 0.02λ)^100 - 1)
        @test relν <= 1e-10
        @printf("  [qv] molecular ν∇²qv, 100 steps: rel. error %.1e\n", relν)
    end

    @testset "2. conservation: Σqv dV with a variable κt, Σqv² decays" begin
        g = RegularGrid(; size = (12, 10, 8), extent = (1.2, 1.0, 0.8), halo = (2, 2, 2))
        rng = Xoshiro(177)
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        θb = FieldBoundaryConditions(:periodic, :periodic, :noflux)
        κ = 1e-3 .* (0.2 .+ rand(rng, 12, 10, 8))
        sim = BoussinesqSimulation(g; moisture = true, bcs = bcs, θ_bcs = θb)
        interior_view(sim.qv) .= 1e-2 .* (1 .+ randn(rng, 12, 10, 8))
        S0 = sum(interior_view(sim.qv))
        V0 = sum(abs2, interior_view(sim.qv))
        prev = V0
        maxrel = 0.0
        monotone = true
        for _ in 1:100
            step!(sim, 0.5; sgs = ScalarKappa(κ))
            maxrel = max(maxrel, abs(sum(interior_view(sim.qv)) - S0) / abs(S0))
            V = sum(abs2, interior_view(sim.qv))
            monotone &= V <= prev
            prev = V
        end
        @test maxrel <= 1e-12
        @test monotone
        # the fluctuation part Σ(qv − q̄)² (Σqv² also holds the conserved N q̄²)
        N = length(interior_view(sim.qv))
        @test prev - S0^2 / N < 0.2 * (V0 - S0^2 / N)
        @printf("  [qv] 100 steps, variable κt: max|ΔΣqv|/Σqv = %.1e, Σqv² %.3e → %.3e\n",
                maxrel, V0, prev)
    end

    @testset "3. analytic steady state, κ = κ₀(1 + z/H)" begin
        # d/dz(κ dϕ/dz) = 0 on 0 < z < L, upward flux −κ ϕ' = F at z = 0, ϕ(L) = ϕ_top:
        #   ϕ(z) = ϕ_top + (F H/κ₀) ln((1 + L/H)/(1 + z/H)).
        # The discrete steady state of add_eddy_diffusion_scalar! (κ face
        # averages, ghost ϕ = 2ϕ_top − ϕ_N above, prescribed flux F/dz into the
        # bottom cell with a zero-gradient ghost) is solved directly.
        κ0, H, L, F, ϕtop = 2.0, 0.5, 1.0, 0.3, 1.0
        exact(z) = ϕtop + (F * H / κ0) * log((1 + L / H) / (1 + z / H))
        function steady(Nz)
            g = RegularGrid(; size = (1, 1, Nz), extent = (1.0, 1.0, L), halo = (1, 1, 1))
            Hz = g.Hz
            zs = [(k - Hz - 0.5) * g.dz for k in 1:Nz+2Hz]          # storage-level centres
            κ = Field(CellCenter(), g); _set_column!(κ, κ0 .* (1 .+ zs ./ H))
            ϕ = Field(CellCenter(), g); t = Field(CellCenter(), g)
            function T(x, top, flux)                                 # tendency of the column x
                col = zeros(Nz + 2Hz)
                col[Hz+1:Hz+Nz] .= x
                col[Hz] = x[1]                                       # zero gradient below
                col[Hz+Nz+1] = 2top - x[end]                         # ϕ(L) = top
                _set_column!(ϕ, col); fill!(t.data, 0)
                add_eddy_diffusion_scalar!(t, ϕ, κ)
                return interior_view(t)[1, 1, :] .+ (flux / g.dz) .* ((1:Nz) .== 1)
            end
            b = T(zeros(Nz), ϕtop, F)
            A = reduce(hcat, (T(Float64.((1:Nz) .== j), 0.0, 0.0) for j in 1:Nz))
            x = -(A \ b)
            zc = [(k - 0.5) * g.dz for k in 1:Nz]
            return maximum(abs.(x .- exact.(zc)))
        end
        e32, e64 = steady(32), steady(64)
        @test e64 <= 1e-3
        @test 3.5 <= e32 / e64 <= 4.5
        @printf("  [κ(z)] steady state: max error %.2e (Nz=32), %.2e (Nz=64), ratio %.2f\n",
                e32, e64, e32 / e64)
    end

    @testset "5. θ and momentum unchanged" begin
        # Decaying turbulence with the built-in Smagorinsky; θ₀ = 0, so qv has
        # no route back into the flow: u, v, w, θ are bit for bit the dry run.
        g = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        rng = Xoshiro(5)
        U, V, W, Θ = (randn(rng, 16, 16, 16) for _ in 1:4)
        Q = 1e-2 .* (1 .+ 0.1 .* randn(rng, 16, 16, 16))
        mk(moist) = (s = BoussinesqSimulation(g; C_s = 0.17, ν = 1e-4, moisture = moist);
                     interior_view(s.u) .= 0.1 .* U; interior_view(s.v) .= 0.1 .* V;
                     interior_view(s.w) .= 0.1 .* W; interior_view(s.θ) .= Θ;
                     moist && (interior_view(s.qv) .= Q); s)
        for (name, adv!) in (("step!", (s, h) -> step!(s, h)),
                             ("step_rk3!", (s, h) -> step_rk3!(s, h)))
            dry, wet = mk(false), mk(true)
            for _ in 1:20
                adv!(dry, 0.01); adv!(wet, 0.01)
            end
            @test dry.u.data == wet.u.data && dry.v.data == wet.v.data &&
                  dry.w.data == wet.w.data && dry.θ.data == wet.θ.data
            @test dry.νt.data == wet.νt.data && dry.κt.data == wet.κt.data
            @test sum(interior_view(wet.qv)) ≈ sum(Q) rtol = 1e-12
        end
    end
end
