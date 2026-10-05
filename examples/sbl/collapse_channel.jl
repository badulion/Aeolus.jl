# =========================================================================
# Turbulence collapse in the flux-cooled half channel (P58-e, #193)
#
# The neutral half channel of P54-d (`half_channel_case`, 2πH × 2πH × H,
# -∂p/∂x = u*²/H, z0/H = 1e-4) with a prescribed surface cooling
# `surface_heat_flux = -Q`, the flux-prescribed MOST wall law
# (`MOSTFluxWallLaw`, P57-a/P58-e) and the stability-corrected Smagorinsky
# closure (`StabilityCorrected`, P55-e). A neutral run is spun up for
# 10 H/u*; from its state the cooling is switched on and each run is
# continued for 20 H/u*.
#
# The sweep is in units of the maximum sustainable heat flux (Van de
# Wiel et al. 2012, `max_sustainable_heat_flux`) of the neutral state at
# two heights:
#   Q_wall — at the wall model's height Δz/2 = H/64 (beyond it the wall
#            layer cannot carry the flux and the wall stress vanishes);
#   Q_bulk — at the channel height H (the channel-scale criterion of
#            Donda et al. 2015).
# For each run it prints the resolved TKE over 15-20 H/u* relative to the
# neutral run (whole domain and lowest third), the number of lowest-third
# levels with turbulent fraction γ < 0.2 (Transition, P59-c) at 20 H/u*,
# the first-level wind relative to the neutral U₀ and the wall stress.
#
#   julia --project=. examples/sbl/collapse_channel.jl        # ~4 min
#   FRACS=0,0.5,2 julia --project=. examples/sbl/collapse_channel.jl
#
# The theory page (docs/src/theory/turbulence_stability.md, "Maximum
# sustainable heat flux") quotes the table this prints.
# =========================================================================

using Aeolus
using Aeolus.SurfaceExchange: von_karman
using Printf

const N = (16, 16, 32)
const Z0 = 1e-4
const Θ = 300.0
const SF = BusingerDyerBeljaarsHoltslag()
# H/Q_wall of each run (0 = neutral)
const FRACS = parse.(Float64, split(get(ENV, "FRACS", "0,0.0159,0.0636,0.25,0.5,1,2"), ","))

closure(mesh) = StabilityCorrected(StaticSmagorinsky(0.1), mesh; length_scales = :isotropic)
mesh() = RegularGrid(; size = N, extent = (2π, 2π, 1.0))
plane_u1(s) = (g = s.mesh; sum(view(s.u.data, g.Hx+1:g.Hx+g.Nx, g.Hy+1:g.Hy+g.Ny, g.Hz+1)) / (g.Nx * g.Ny))

function main()
    κ = von_karman(); Nz = N[3]; h = 1 / (2Nz)
    U0 = log(h / Z0) / κ
    Qwall = max_sustainable_heat_flux(SF, U0, h, Z0, Z0, Θ)
    Qbulk = max_sustainable_heat_flux(SF, log(1 / Z0) / κ, 1.0, Z0, Z0, Θ)
    @printf("MSHF of the neutral state: Q_wall = %.1f K m/s (U₀ = %.2f u* at H/64), Q_bulk = %.2f K m/s\n",
            Qwall, U0, Qbulk)
    t0 = time()
    spin = half_channel_case(; N = N, closure = closure(mesh()))
    run_channel!(spin, 10.0)
    @printf("neutral spin-up to 10 H/u*: %.0f s\n\n", time() - t0)
    lowest = 1:cld(Nz, 3)
    rows = []
    for f in FRACS
        hc = f == 0 ? half_channel_case(; N = N, closure = closure(mesh())) :
             half_channel_case(; N = N, closure = closure(mesh()), surface_heat_flux = -f * Qwall,
                               θ_ref = Θ)
        hc.sim.u.data .= spin.sim.u.data; hc.sim.v.data .= spin.sim.v.data
        hc.sim.w.data .= spin.sim.w.data
        e = Float64[]; el = Float64[]; u1 = Float64[]; τ = Float64[]; nxt = Ref(1.0)
        t1 = time()
        run_channel!(hc, 20.0; callback = c -> begin
            if c.sim.t >= nxt[] - 1e-9
                p = resolved_tke_profile(c.sim.u, c.sim.v, c.sim.w)
                if c.sim.t >= 15 - 1e-9
                    push!(e, sum(p) / Nz); push!(el, sum(p[lowest]) / length(lowest))
                    push!(u1, plane_u1(c.sim)); push!(τ, c.τ_wall_bottom)
                end
                nxt[] += 1
            end
        end)
        γ = turbulent_fraction(hc.sim)
        push!(rows, (f = f, e = sum(e) / length(e), el = sum(el) / length(el),
                     nlam = count(<(0.2), γ[lowest]), u1 = sum(u1) / length(u1) / U0,
                     τ = sum(τ) / length(τ), wall = time() - t1))
    end
    neu = rows[findfirst(r -> r.f == 0, rows)]
    println("| H/H_max(wall) | H/H_max(bulk) | TKE/TKE₀ | TKE/TKE₀ lowest third | levels γ<0.2 (of $(length(lowest))) | U₁/U₀ | τ_w/u*² | wall time |")
    println("| --: | --: | --: | --: | --: | --: | --: | --: |")
    for r in rows
        @printf("| %.4g | %.3g | %.3f | %.3f | %d | %.3f | %.2f | %.0f s |\n", r.f, r.f * Qwall / Qbulk,
                r.e / neu.e, r.el / neu.el, r.nlam, r.u1, r.τ, r.wall)
    end
    @printf("\n(2 H_max(wall) predicts U₁/U₀ = 2^(1/3) = %.4f; total %.0f s)\n", 2^(1 / 3), time() - t0)
end

main()
