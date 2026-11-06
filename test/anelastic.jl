using Test
using Random
using Aeolus
using Aeolus.Anelastic
using Aeolus.Grids: CellCenter, FaceZ, znodes, interior
using Aeolus.Fields: set!, interior_view

# =========================================================================
# Era II / Phase 27 — anelastic and pseudo-incompressible dynamics.
# (The warm-bubble benchmark of all three cores is validation/anelastic.jl.)
# =========================================================================

# _l2, _reldiff: shared with the warm-bubble validation (validation/anelastic.jl)
isdefined(@__MODULE__, :_reldiff) || include(joinpath(@__DIR__, "support", "anelastic_norms.jl"))

"Drop the mean of an array (the null space of every singular Poisson operator)."
_demean(a) = a .- sum(a) / length(a)

"Continuous hydrostatic p(z) for dp/dz = −g p/(R θ (p/p₀)^κ), RK4 with `nsub` sub-steps."
function _continuous_hydrostatic(θ, z, p_surf; R, cp, p₀, g, nsub = 1000)
    κ = R / cp
    f(zz, pp) = -g * pp / (R * θ(zz) * (pp / p₀)^κ)
    out = similar(z, Float64)
    p, zz = Float64(p_surf), 0.0
    for k in eachindex(z)
        h = (z[k] - zz) / nsub
        for _ in 1:nsub
            k1 = f(zz, p); k2 = f(zz + h / 2, p + h / 2 * k1)
            k3 = f(zz + h / 2, p + h / 2 * k2); k4 = f(zz + h, p + h * k3)
            p += h / 6 * (k1 + 2k2 + 2k3 + k4); zz += h
        end
        out[k] = p
    end
    return out
end

@testset "Era II — Anelastic & pseudo-incompressible" begin

@testset "Reference states: discrete hydrostatic balance, analytic profiles" begin
    g = RegularGrid(; size=(4,4,24), extent=(2000.0,2000.0,6000.0), halo=(2,2,2))
    refs = (isothermal_reference(g; T_surf=288.0, p_surf=1e5),
            isentropic_reference(g; θ_const=300.0, p_surf=1e5),
            constant_stability_reference(g; N=0.01, θ_surf=300.0, p_surf=1e5),
            reference_from_profile(g, z -> 300.0 + 0.004z, 1e5))
    # (p̄[k] − p̄[k−1])/Δz = −g ρ̄_f[k] on the model's own grid, to round-off
    @test maximum(r -> max_hydrostatic_residual(r) / (r.g * maximum(r.ρ̄)), refs) < 1e-13
    @test maximum(r -> maximum(abs, r.p̄ .- r.R_d .* r.ρ̄ .* r.T̄) / maximum(r.p̄), refs) < 1e-14

    r = refs[1]                              # isothermal: exactly ρ₀ exp(−z/H)
    H = r.R_d * 288.0 / r.g
    ζ = collect(znodes(g, CellCenter())) .- g.z0
    @test maximum(abs, r.p̄ .- 1e5 .* exp.(-ζ ./ H)) / 1e5 < 1e-14
    @test reference_scale_height(r) ≈ H rtol=1e-12
    @test maximum(abs, refs[3].N² .- 0.01^2) / 0.01^2 < 1e-12   # constant N² reproduced
    @test maximum(abs, refs[2].N²) == 0.0                      # neutral: identically zero

    # against the continuous atmosphere, with the MoistThermodynamics
    # constants (one source of truth for Exner and the ideal gas)
    MT = Aeolus.MoistThermodynamics
    c = MT.DEFAULT_CONSTANTS
    R, cp, p₀, gg = c.gas_constant / c.molmass_dry, c.cp_dry, c.p_ref, 9.81
    m = RegularGrid(; size = (4, 4, 16), extent = (250.0, 250.0, 100.0), halo = (2, 2, 2))
    θf = z -> 294.0 + max(0, z - 60.0) * 3e-3          # neutral layer under an inversion
    rp = reference_from_profile(m, θf, 101325.0; R_d = R, cp = cp, p₀ = p₀, g_accel = gg)
    zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    p_cont = _continuous_hydrostatic(θf, zc, 101325.0; R = R, cp = cp, p₀ = p₀, g = gg)
    @test maximum(abs.(rp.p̄ .- p_cont) ./ p_cont) < 1e-6     # measured ≲ 6e-8
    @test maximum(abs.(rp.π̄ .- MT.exner_function.(rp.p̄)) ./ rp.π̄) < 1e-12
    @test maximum(abs.(rp.ρ̄ .- MT.air_density.(rp.T̄, rp.p̄)) ./ rp.ρ̄) < 1e-12
end

@testset "Density-weighted Poisson: manufactured solution, symmetry, direct = CG (#309)" begin
    L, Lz = 1000.0, 1000.0
    g  = RegularGrid(; size=(8,8,8), extent=(L,L,Lz), halo=(2,2,2))
    # ρ̄ = exp(4ζ/Lz): > 3 between the extreme cell centres
    ρ̄c = [exp(4 * (z - g.z0) / Lz) for z in znodes(g, CellCenter())]
    ρ̄f = [exp(4 * (k - 1) * g.dz / Lz) for k in 1:g.Nz+1]
    @test maximum(ρ̄c) / minimum(ρ̄c) > 3
    s = DensityWeightedPoissonSolver(g, ρ̄c, ρ̄f; tol=1e-13, maxit=3000)
    φ = Field(CellCenter(), g); f = Field(CellCenter(), g); out = Field(CellCenter(), g)
    set!(φ, (x,y,z) -> cos(2π*x/L) * cos(2π*y/L) * cos(π*(z-g.z0)/Lz) +
                       0.3 * sin(4π*x/L) * cos(2π*(z-g.z0)/Lz))
    apply_weighted_laplacian!(f, φ, s)
    solve!(out, f, s)
    @test maximum(abs, _demean(collect(interior_view(out))) .- _demean(collect(interior_view(φ)))) < 1e-8

    # self-adjoint (face-averaged ρ̄): ⟨Lφ, ψ⟩ = ⟨φ, Lψ⟩
    rng = MersenneTwister(2027)
    ψ = Field(CellCenter(), g); Lφ = Field(CellCenter(), g); Lψ = Field(CellCenter(), g)
    interior_view(φ) .= randn(rng, size(interior_view(φ)))
    interior_view(ψ) .= randn(rng, size(interior_view(ψ)))
    apply_weighted_laplacian!(Lφ, φ, s); apply_weighted_laplacian!(Lψ, ψ, s)
    a1 = sum(interior_view(Lφ) .* interior_view(ψ)); a2 = sum(interior_view(φ) .* interior_view(Lψ))
    @test abs(a1 - a2) / (_l2(interior_view(Lφ)) * _l2(interior_view(ψ))) < 1e-13

    # the direct solver inverts the same operator exactly (neumann sides, ρθ weighting)
    gd = RegularGrid(; size=(8,6,10), extent=(2000.0,2600.0,9000.0), halo=(2,2,2))
    ref = isothermal_reference(gd; T_surf=300.0)
    d = DirectWeightedPoissonSolver(gd, ref; weighting=:ρθ, bc_x=:neumann, bc_y=:periodic)
    c = DensityWeightedPoissonSolver(gd, ref; weighting=:ρθ, bc_x=:neumann, bc_y=:periodic,
                                     tol=1e-14, maxit=5000)
    φd = Field(CellCenter(), gd); fd = Field(CellCenter(), gd)
    interior_view(φd) .= _demean(randn(rng, size(interior_view(φd))))
    apply_weighted_laplacian!(fd, φd, c)
    pd = Field(CellCenter(), gd); pc = Field(CellCenter(), gd)
    solve!(pd, fd, d); solve!(pc, fd, c)
    scale = maximum(abs, interior_view(φd))
    @test maximum(abs, interior_view(pd) .- interior_view(φd)) / scale < 1e-11
    @test maximum(abs, interior_view(pd) .- interior_view(pc)) / scale < 1e-10
end

@testset "Anelastic constraint after a step; ∇·u = −w ∂lnρ̄/∂z is not zero" begin
    L, Lz = 3000.0, 3000.0
    g   = RegularGrid(; size=(12,12,12), extent=(L,L,Lz), halo=(2,2,2))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ref = isentropic_reference(g; θ_const=300.0)
    sim = AnelasticSimulation(g; ref=ref, bcs=bcs)
    @test sim.solver isa DirectWeightedPoissonSolver
    set!(sim.θ, (x,y,z) -> 300.0 +
        2.0 * exp(-(((x-L/2)/500)^2 + ((y-L/2)/500)^2 + ((z-1000.0)/500)^2)))
    for n in 1:3
        step_anelastic!(sim, 2.0)
    end
    @test max_w(sim) > 1e-3                         # the bubble rises
    @test max_anelastic_divergence(sim) < 1e-10
    @test max_mass_flux_divergence(sim) < 1e-10
    predicted = maximum(abs, interior_view(sim.w)) / reference_scale_height(ref)
    @test 0.2 < max_divergence(sim) / predicted < 5.0
end

@testset "Boussinesq limit: discrepancy linear in Δρ/ρ̄" begin
    function limit_case(Lz)
        g   = RegularGrid(; size=(8,8,8), extent=(Lz,Lz,Lz), halo=(2,2,2))
        bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
        ref = isentropic_reference(g; θ_const=300.0)
        pert(x,y,z) = 2.0 * exp(-(((x-Lz/2)/(0.2Lz))^2 + ((y-Lz/2)/(0.2Lz))^2 +
                                  ((z-0.4Lz)/(0.2Lz))^2))
        b = BoussinesqSimulation(g; θ₀=300.0, bcs=bcs)
        set!(b.θ, (x,y,z) -> 300.0 + pert(x,y,z))
        a = AnelasticSimulation(g; ref=ref, bcs=bcs)
        set!(a.θ, (x,y,z) -> 300.0 + pert(x,y,z))
        for _ in 1:3
            step!(b, 1.0); step_anelastic!(a, 1.0)
        end
        return _reldiff(a.w, b.w)
    end
    shallow, medium = limit_case(20.0), limit_case(200.0)
    @test 0 < shallow < 1e-4
    @test 9.0 < medium / shallow < 11.0            # ×10 depth ⇒ ×10 discrepancy
end

@testset "Pseudo-incompressible constraint; the two systems differ when stratified" begin
    L, Lz = 4000.0, 8000.0
    g   = RegularGrid(; size=(12,12,12), extent=(L,L,Lz), halo=(2,2,2))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    ref = constant_stability_reference(g; N=0.012, θ_surf=300.0)
    zc  = collect(znodes(g, CellCenter()))
    function build(S)
        s = S(g; ref=ref, bcs=bcs)
        ii, jj, kk = interior(g)
        for (kl, k) in enumerate(kk), (jl, j) in enumerate(jj), (il, i) in enumerate(ii)
            x = g.x0 + (il - 0.5) * g.dx; y = g.y0 + (jl - 0.5) * g.dy
            s.θ.data[i,j,k] = ref.θ̄[kl] +
                3.0 * exp(-(((x-L/2)/700)^2 + ((y-L/2)/700)^2 + ((zc[kl]-2000.0)/700)^2))
        end
        for _ in 1:3
            step!(s, 2.0)
        end
        return s
    end
    pi_ = build(PseudoIncompressibleSimulation)
    an  = build(AnelasticSimulation)
    @test max_pseudo_incompressible_divergence(pi_) < 1e-10
    @test constraint_weights(pi_) == (ref.Pc_pad, ref.Pf_pad)
    # each violates the other's constraint once θ̄ varies with height
    @test max_anelastic_divergence(pi_) > 1e-8
    @test max_pseudo_incompressible_divergence(an) > 1e-8
end

end # @testset "Era II — Anelastic & pseudo-incompressible"
