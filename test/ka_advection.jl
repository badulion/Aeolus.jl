# KA-e (#302): the scalar-transport hot path and the momentum advection of
# AdvectionSchemes on launch!.
#
#   1. bit for bit against the pre-port serial loops (a transcript:
#      test/support/legacy_advection.jl) for the flux-form scalar sweep of
#      every reconstruction (WENO-Z5/7, TENO5/8, WENO-JS5, Centered2,
#      upwind), the reconstructed and upwind face fluxes, the positivity
#      limiter (with and without low-order fluxes, periodic wrap and BC
#      halos, positivity active), advect_positive! (RegularGrid and
#      StretchedGrid), the WENO-Z and skew-symmetric momentum advection, the
#      variance-conserving scalar advection and boundary_scalar_flux (every
#      flux-form scheme, levels, mass-flux correction, diffusion, weights,
#      face_fluxes); serially, threaded with every launch on the KA path
#      (`threshold = 0`) and on the default backend, on random fields;
#   2. advect_positive! and boundary_scalar_flux allocate nothing with a
#      workspace on a serial backend;
#   3. the positive-definite path refuses to allocate its scratch per call
#      when the backend runs threaded.

using Test
using Random: Xoshiro, randn, rand
using Aeolus
using Aeolus.Grids: FaceX, FaceY, FaceZ, CellCenter, interior
using Aeolus.Fields: Field, interior_view, set!
using Aeolus.Simulations: BoussinesqSimulation
using Aeolus.StretchedGrids: StretchedGrid
const ASK = Aeolus.AdvectionSchemes

isdefined(@__MODULE__, :LegacyAdvection) ||
    include(joinpath(@__DIR__, "support", "legacy_advection.jl"))
const LA = LegacyAdvection

const ADV_BACKENDS = ("serial" => CPUBackend(serial = true),
                      "threaded, threshold 0" => CPUBackend(threshold = 0),
                      "default" => CPUBackend())
adv_bits(A) = reinterpret(UInt64, vec(collect(Float64, A)))

function adv_field(rng, loc, g; scale = 1.0, offset = 0.0)
    f = Field(loc, g)
    f.data .= offset .+ scale .* randn(rng, size(f.data))
    return f
end
adv_copy(f::Field) = (h = Field(Aeolus.Fields.location(f), f.mesh); h.data .= f.data; h)
adv_copy(x::AbstractArray) = copy(x)
adv_data(x::Field) = x.data
adv_data(x::AbstractArray) = x

# Run `new!(outs...)` on every backend and `old!(outs...)` once, each on its
# own copy of `outs`; every output must agree bit for bit (halos included).
# A non-`nothing` return value (boundary_scalar_flux's totals) must agree too.
function adv_check(name, new!, old!, outs...)
    ref = map(adv_copy, outs)
    rref = old!(ref...)
    for (bname, b) in ADV_BACKENDS
        got = map(adv_copy, outs)
        rgot = with_backend(b) do
            new!(got...)
        end
        ok = all(adv_bits(adv_data(a)) == adv_bits(adv_data(r)) for (a, r) in zip(got, ref))
        if rref isa NamedTuple
            ok &= all(reinterpret(UInt64, Float64(getfield(rgot, s))) ==
                      reinterpret(UInt64, Float64(getfield(rref, s))) for s in keys(rref))
        end
        ok || @info "KA-e bit mismatch" name bname
        @test ok
    end
end

const ADV_RECONS = (("WENO-Z5", ASK.WENOZ(5, 2)), ("WENO-Z5 p=1", ASK.WENOZ(5, 1)),
                    ("WENO-Z7", ASK.WENOZ(7, 2)), ("TENO5", ASK.TENO(5)),
                    ("TENO8", ASK.TENO(8)), ("WENO-JS5", ASK.WENOJS(5)),
                    ("Centered2", ASK.Centered2()), ("upwind", ASK._UpwindFirst()))

adv_grid() = RegularGrid(Float64; size = (13, 11, 9), extent = (1.3, 1.7, 0.8), halo = (5, 5, 5))

# velocities of both signs; a positive scalar with zeros and steep fronts,
# so that the limiter is active
function adv_state(rng, g)
    u = adv_field(rng, FaceX(), g); v = adv_field(rng, FaceY(), g)
    w = adv_field(rng, FaceZ(), g; scale = 0.5)
    ϕ = Field(CellCenter(), g)
    ϕ.data .= max.(randn(rng, size(ϕ.data)), 0.0) .^ 3       # ~half the cells zero
    return u, v, w, ϕ
end

@testset "KA-e: scalar and momentum advection on launch! (bit for bit against the old loops)" begin
    rng = Xoshiro(302)
    g = adv_grid()
    u, v, w, ϕ = adv_state(rng, g)
    θ = adv_field(rng, CellCenter(), g; scale = 2.0, offset = 300.0)
    bcs_w = FieldBoundaryConditions(:periodic, :periodic, :wall)
    nlimited = 0

    for (rname, r) in ADV_RECONS
        @testset "$rname" begin
            tend = adv_field(rng, CellCenter(), g)
            adv_check("$rname sweep",
                      t -> ASK._advect_reconstructed!(t, u, v, w, θ, r, g),
                      t -> LA._advect_reconstructed!(t, u, v, w, θ, r, g), tend)
            Fx = Field(FaceX(), g); Fy = Field(FaceY(), g); Fz = Field(FaceZ(), g)
            adv_check("$rname face fluxes",
                      (a, b, c) -> ASK._reconstructed_fluxes!(a, b, c, u, v, w, θ, r, g),
                      (a, b, c) -> LA._reconstructed_fluxes!(a, b, c, u, v, w, θ, r, g), Fx, Fy, Fz)
            # advect_positive!: limiter active (large dt), periodic wrap and walls
            for (bn, bcs) in (("wrap", nothing), ("walls", bcs_w)), dt in (0.05, 0.4)
                ws = FluxLimiterWorkspace(g)
                adv_check("$rname advect_positive! $bn dt=$dt",
                          t -> advect_positive!(t, u, v, w, ϕ, dt; scheme = r, workspace = ws, bcs = bcs),
                          t -> LA.advect_positive!(t, u, v, w, ϕ, dt; scheme = r,
                                                   workspace = FluxLimiterWorkspace(g), bcs = bcs),
                          adv_field(rng, CellCenter(), g))
                LA.advect_positive!(Field(CellCenter(), g), u, v, w, ϕ, dt; scheme = r,
                                    workspace = ws, bcs = bcs)
                nlimited += count(<(1), interior_view(ws.ratio))
            end
        end
    end
    @info "KA-e: limited cells (ratio < 1) over the advect_positive! checks: $nlimited"
    @test nlimited > 0

    @testset "upwind fluxes, flux divergence, limiter phases" begin
        Lx = adv_field(rng, FaceX(), g); Ly = adv_field(rng, FaceY(), g); Lz = adv_field(rng, FaceZ(), g)
        adv_check("upwind fluxes",
                  (a, b, c) -> ASK._upwind_fluxes!(a, b, c, u, v, w, ϕ, g),
                  (a, b, c) -> LA._upwind_fluxes!(a, b, c, u, v, w, ϕ, g), Lx, Ly, Lz)
        Fx = adv_field(rng, FaceX(), g); Fy = adv_field(rng, FaceY(), g); Fz = adv_field(rng, FaceZ(), g)
        adv_check("flux divergence",
                  t -> ASK._add_flux_divergence!(t, Fx, Fy, Fz, g),
                  t -> LA._add_flux_divergence!(t, Fx, Fy, Fz, g), adv_field(rng, CellCenter(), g))
        adv_check("one-layer wrap", f -> ASK._wrap_one_layer!(f, g),
                  f -> LA._wrap_one_layer!(f, g), adv_field(rng, CellCenter(), g))
        # the bare limiter: no low-order fluxes, absurd fluxes (CFL ≫ 1), with
        # and without a BC halo for the ratio; and with upwind fluxes
        for lo in (nothing, (Lx, Ly, Lz)), bcs in (nothing, bcs_w)
            adv_check("limiter lo=$(lo !== nothing) bcs=$(bcs !== nothing)",
                      (a, b, c, R) -> apply_positivity_limiter!(a, b, c, ϕ, 10.0, g;
                                                                flux_lo = lo, ratio = R, bcs = bcs),
                      (a, b, c, R) -> LA.apply_positivity_limiter!(a, b, c, ϕ, 10.0, g;
                                                                   flux_lo = lo, ratio = R, bcs = bcs),
                      Fx, Fy, Fz, adv_field(rng, CellCenter(), g))
        end
    end

    @testset "StretchedGrid advect_positive! (centred)" begin
        sg = StretchedGrid(Float64; size = (13, 11, 9), extent = (1.3, 1.7, 0.8), halo = (3, 3, 3),
                           stretching = (:geometric, 1.08))
        su, sv, sw, sϕ = adv_state(rng, sg)
        sb = FieldBoundaryConditions(:periodic, :periodic, :wall)
        for dt in (0.05, 0.4)
            ws = FluxLimiterWorkspace(sg)
            adv_check("stretched advect_positive! dt=$dt",
                      t -> advect_positive!(t, su, sv, sw, sϕ, dt; scheme = :centered,
                                            workspace = ws, bcs = sb),
                      t -> LA.advect_positive!(t, su, sv, sw, sϕ, dt; scheme = :centered,
                                               workspace = FluxLimiterWorkspace(sg), bcs = sb),
                      adv_field(rng, CellCenter(), sg))
        end
    end

    @testset "momentum and variance-conserving advection" begin
        for (rn, r) in (("WENO-Z5", ASK.WENOZ(5, 2)), ("WENO-Z7", ASK.WENOZ(7, 2)), ("TENO5", ASK.TENO(5)))
            adv_check("momentum $rn",
                      (a, b, c) -> ASK._momentum_advect_recon!(a, b, c, u, v, w, r, g),
                      (a, b, c) -> LA._momentum_advect_recon!(a, b, c, u, v, w, r, g),
                      adv_field(rng, FaceX(), g), adv_field(rng, FaceY(), g), adv_field(rng, FaceZ(), g))
        end
        adv_check("momentum skew",
                  (a, b, c) -> ASK._momentum_advect_skew!(a, b, c, u, v, w, g),
                  (a, b, c) -> LA._momentum_advect_skew!(a, b, c, u, v, w, g),
                  adv_field(rng, FaceX(), g), adv_field(rng, FaceY(), g), adv_field(rng, FaceZ(), g))
        adv_check("variance conserving",
                  t -> ASK._advect_variance_conserving!(t, u, v, w, θ, g),
                  t -> LA._advect_variance_conserving!(t, u, v, w, θ, g), adv_field(rng, CellCenter(), g))
    end
end

# an open box after one step: a non-uniform stream through every open face,
# a non-zero mass-flux correction δ, Smagorinsky κt
function adv_open_sim()
    Nx, Ny, Nz = 12, 10, 8
    m = RegularGrid(size = (Nx, Ny, Nz), extent = (240.0, 200.0, 160.0), halo = (5, 5, 5))
    data = FunctionBoundaryData(u = (x, y, z, t) -> 2.0 + 0.002y + 0.001z,
                                v = (x, y, z, t) -> -0.5 + 0.001x,
                                w = (x, y, z, t) -> 0.05,
                                θ = (x, y, z, t) -> 300.0 + 0.004x - 0.003y + 0.01z)
    s = BoussinesqSimulation(m; bcs = open_boundary_conditions(m, data), θ₀ = 300.0,
                             C_s = 0.18, ν = 0.5)
    set!(s.u, (x, y, z) -> 2.0 + 0.3 * sin(2π * x / 240) * cos(2π * z / 160))
    set!(s.v, (x, y, z) -> -0.5 + 0.2 * cos(2π * y / 200))
    set!(s.θ, (x, y, z) -> 300.0 + 0.004x - 0.003y + 0.01z + 0.2 * sin(2π * x / 120) * sin(2π * y / 100))
    step!(s, 1.0)
    return s
end

@testset "KA-e: boundary_scalar_flux on launch! (bit for bit against the old loops)" begin
    s = adv_open_sim()
    m = s.u.mesh
    Nz = m.Nz
    @test only(open_boundary_states(s)).δ != 0
    rng = Xoshiro(3021)
    Wc = 1.0 .+ 0.1 .* rand(rng, size(s.θ.data, 3)); Wf = 1.0 .+ 0.1 .* rand(rng, size(s.θ.data, 3))
    d = (ν = s.ν, κh = s.κt, κv = s.κt)
    ws = FluxLimiterWorkspace(m)
    advect_positive!(Field(CellCenter(), m), s.u, s.v, s.w, s.θ, 0.5; scheme = :weno_z5, workspace = ws)
    cases = Any[(sch, "levels+correction+diffusion", (diffusivity = d,))
             for sch in (:centered, :upwind, :weno_js5, :weno_z5, :weno_z7, :teno5, :teno8)]
    for sch in (:centered, :weno_z5)
        push!(cases, (sch, "plain", (;)), (sch, "weights", (weights = (Wc, Wf), diffusivity = d)),
              (sch, "face_fluxes", (face_fluxes = (ws.Fx, ws.Fy, ws.Fz),)))
    end
    for (sch, kn, kw) in cases
        adv_check("bsf $sch $kn",
                  (l, c) -> boundary_scalar_flux(s, s.θ; scheme = sch, fill = false, levels = l,
                                                 correction_levels = c, kw...),
                  (l, c) -> LA.boundary_scalar_flux(s, s.θ; scheme = sch, fill = false, levels = l,
                                                    correction_levels = c, kw...),
                  zeros(Nz), zeros(Nz))
    end
    # the correction split is booked (not all zero) in the case checked
    c = zeros(Nz)
    boundary_scalar_flux(s, s.θ; fill = false, correction_levels = c)
    @test any(!iszero, c)
end

function _adv_pos_bytes(t, u, v, w, ϕ, dt, r, ws, bcs)
    advect_positive!(t, u, v, w, ϕ, dt; scheme = r, workspace = ws, bcs = bcs)
    return @allocated advect_positive!(t, u, v, w, ϕ, dt; scheme = r, workspace = ws, bcs = bcs)
end
function _adv_bsf_bytes(s, sch, l)
    boundary_scalar_flux(s, s.θ; scheme = sch, fill = false, levels = l)
    return @allocated boundary_scalar_flux(s, s.θ; scheme = sch, fill = false, levels = l)
end

@testset "KA-e: no allocation on the serial path" begin
    rng = Xoshiro(3022)
    g = adv_grid()
    u, v, w, ϕ = adv_state(rng, g)
    t = Field(CellCenter(), g)
    ws = FluxLimiterWorkspace(g)
    with_backend(CPUBackend(serial = true)) do
        for r in (ASK.WENOZ(5, 2), ASK.TENO(5), ASK.Centered2()),
            bcs in (nothing, FieldBoundaryConditions(:periodic, :periodic, :wall))
            @test _adv_pos_bytes(t, u, v, w, ϕ, 0.1, r, ws, bcs) == 0
        end
        s = adv_open_sim()
        for sch in (:centered, :upwind)
            @test _adv_bsf_bytes(s, sch, zeros(s.u.mesh.Nz)) == 0
        end
    end
end

@testset "KA-e: the positive-definite path needs its scratch when threaded" begin
    rng = Xoshiro(3023)
    g = adv_grid()
    u, v, w, ϕ = adv_state(rng, g)
    t = Field(CellCenter(), g)
    Fx = Field(FaceX(), g); Fy = Field(FaceY(), g); Fz = Field(FaceZ(), g)
    with_backend(CPUBackend(threshold = 0)) do
        if Threads.nthreads() > 1
            @test_throws ArgumentError advect_positive!(t, u, v, w, ϕ, 0.1)
            @test_throws ArgumentError apply_positivity_limiter!(Fx, Fy, Fz, ϕ, 0.1, g)
        else
            # one thread: every launch is serial, the scratch is allocated
            @test advect_positive!(t, u, v, w, ϕ, 0.1) === t
            @test apply_positivity_limiter!(Fx, Fy, Fz, ϕ, 0.1, g) isa Tuple
        end
        ws = FluxLimiterWorkspace(g)
        @test advect_positive!(t, u, v, w, ϕ, 0.1; workspace = ws) === t
        @test apply_positivity_limiter!(Fx, Fy, Fz, ϕ, 0.1, g; ratio = ws.ratio) isa Tuple
    end
    with_backend(CPUBackend(serial = true)) do
        @test advect_positive!(t, u, v, w, ϕ, 0.1) === t       # serial: allocates, allowed
    end
end
