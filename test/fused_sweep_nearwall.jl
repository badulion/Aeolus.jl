# The fused sweep near walls, at open faces and in the urban site (#533
# plan step 5): the terms that add to the tendencies next to a boundary
# run inside the cache-blocked pass and agree with their whole-field
# references (their `run_stage!` / operator, which stays).
#
#   1. open lateral faces: the boundary update (the data, the radiating
#      outflow from the stage-start layers, the mass correction) runs after
#      the sweep and before the projection, as after the separate predictor
#      passes, so an open box takes the fused step; step! and step_rk3!
#      agree with one loop per term, Dirichlet, Orlanski and convective
#      outflow, regular and stretched grids, dry and moist, serial and
#      threaded.
#
# Tolerances: every term evaluates the floating-point expressions of its
# whole-field reference, in the same order, so the agreement is round-off
# (`agree`, ROUNDOFF_RTOL = 1e-10 of the field's scale).

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Grids: interior
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Random: Xoshiro, randn

const NWs = Aeolus.Simulations

const _NW_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)))

_nw_state(s) = (s.u, s.v, s.w, s.p, s.θ, s.qv)
_nw_agree(a, b) = all(i -> agree(interior_view(_nw_state(a)[i]), interior_view(_nw_state(b)[i])),
                      eachindex(_nw_state(a))) && a.t ≈ b.t

@testset "fused sweep near walls and open faces (#533 step 5)" begin

@testset "open lateral faces: the fused step = one loop per term" begin
    Lx, Ly, Lz = 800.0, 600.0, 400.0
    # time-dependent data with a net imbalance (the mass correction acts)
    # and a θ, qv that differ from the interior (the radiation acts)
    data(moist) = FunctionBoundaryData(; u = (x, y, z, t) -> 2.0 + 0.3 * x / Lx + 0.5 * sin(2π * t / 60),
                                       v = (x, y, z, t) -> 0.5 + 0.2 * cos(2π * t / 45), w = 0.0,
                                       θ = (x, y, z, t) -> 300.0 + 0.01 * z + 0.2 * sin(2π * t / 30),
                                       (moist ? (qv = (x, y, z, t) -> 0.008 + 1e-3 * sin(2π * t / 40),) : ())...)
    reg = RegularGrid(size = (16, 12, 8), extent = (Lx, Ly, Lz))
    str = StretchedGrid(; size = (16, 12, 10), extent = (Lx, Ly, Lz), stretching = (:tanh, 1.5))
    cases = (("regular, Dirichlet", reg, :dirichlet, :open, false),
             ("regular, Orlanski, moist", reg, :orlanski, :open, true),
             ("regular, convective", reg, :convective, :open, false),
             # (a radiating top is refused on a z-stretched mesh)
             ("stretched, Orlanski sides, wall top", str, :orlanski, :wall, true),
             ("stretched, Dirichlet", str, :dirichlet, :open, false))
    for (name, m, outflow, top, moist) in cases, (bn, b) in _NW_BACKENDS, drv in (step!, step_rk3!)
        with_backend(b) do
            fields = moist ? (:u, :v, :w, :θ, :qv) : (:u, :v, :w, :θ)
            mk() = (s = BoussinesqSimulation(m; ν = 1.0, moisture = moist,
                                             bcs = open_boundary_conditions(m, data(moist); outflow = outflow,
                                                                            top = top, fields = fields));
                    set!(s.u, 2.0); set!(s.v, 0.5); set!(s.θ, 300.0);
                    rng = Xoshiro(533);
                    interior_view(s.θ) .+= 0.1 .* randn(rng, size(interior_view(s.θ)));
                    interior_view(s.u) .+= 0.1 .* randn(rng, size(interior_view(s.u)));
                    moist && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))));
                    s)
            a = NWs.set_loop_fusion!(mk(), :always)     # errors if any term has no fused form
            r = NWs.set_loop_fusion!(mk(), :never)
            fill!(a.tu.data, NaN)
            for _ in 1:4
                drv(a, 0.5); drv(r, 0.5)
            end
            ok = _nw_agree(a, r)
            ok || println("  [#533 s5] open faces mismatch: $name $bn $drv")
            @test ok
            @test all(isnan, a.tu.data)                 # the tendencies stayed in the sweep
            @test NWs.sweep_report(a)["substages_fused"] == "all"
        end
    end
end

end
