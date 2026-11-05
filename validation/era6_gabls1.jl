# P55-f (#191), ONE-g (#456): GABLS1 (Beare et al. 2006, BLM 118, 247) on
# the one model — the configuration examples/sbl/gabls1.toml with
# its components: the geostrophic forcing (`LargeScale`), the
# Monin-Obukhov surface layer (`SurfaceLayer`), the retargeted sponge and
# Smagorinsky with the Brown et al. (1994) stability correction in the
# stress form. Reduced to 32³ at 12.5 m and the first 1.25 h (the full
# 9 h, 64³ run is the config itself, its reference checks against the
# intercomparison ranges in its results.toml).
#
# The configuration (the protocol built from components, the retargeted
# sponge) stays tested in test/era6_gabls1.jl.

using Test
using Aeolus
using Aeolus.Components: Components, LargeScale, SurfaceLayer, Sponge, SGS, step_adaptive!, sgs_closure
using Aeolus.Fields: interior_view
using Statistics: mean

# gabls1_domain, _comp: shared with test/era6_gabls1.jl
isdefined(@__MODULE__, :gabls1_domain) || include(joinpath(@__DIR__, "..", "test", "support", "sbl_domains.jl"))

# Reduced CI run: 32³ (Δ = 12.5 m), 0-1.25 h, statistics over 1.0-1.25 h
# (after the spin-up of the initial neutral layer: u* peaks at 0.43 m/s
# near 15 min), with and without the stability correction.
const T_CI = 4500.0
const T_AVG_CI = 3600.0

# the plane-mean column integral ∫⟨θ⟩dz (K m)
column_heat(d) = (m = d.state.mesh; sum(interior_view(d.state.θ)) * m.dz / (m.Nx * m.Ny))

# SGS share of the vertical θ flux on the faces of the lower half of the BL,
# and the mean κv/κh of the cells below h (the closure's last coefficients)
function gabls1_sgs_share(d, h)
    s = d.state; m = s.mesh; clo = sgs_closure(d)
    κv = clo isa StabilityCorrected ? clo.κv : s.κt
    κh = s.κt
    sg = 0.0; tot = 0.0; r = 0.0; n = 0
    for k in 2:max(2, round(Int, h / 2 / m.dz)), j in 1:m.Ny, i in 1:m.Nx
        I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
        κf = 0.5 * (κv.data[I, J, K] + κv.data[I, J, K-1])
        g = (s.θ.data[I, J, K] - s.θ.data[I, J, K-1]) / m.dz
        sg += -κf * g
        tot += s.w.data[I, J, K] * 0.5 * (s.θ.data[I, J, K] + s.θ.data[I, J, K-1]) - κf * g
    end
    for k in 1:max(1, round(Int, h / m.dz)), j in 1:m.Ny, i in 1:m.Nx
        I, J, K = m.Hx + i, m.Hy + j, m.Hz + k
        κh.data[I, J, K] > 0 && (r += κv.data[I, J, K] / κh.data[I, J, K]; n += 1)
    end
    return (share = sg / tot, κ_ratio = r / max(n, 1))
end

function gabls1_ci(stab)
    d = gabls1_domain(32, stab)
    sl = _comp(d, SurfaceLayer)
    H0 = column_heat(d)
    heat = 0.0; maxres = 0.0
    hs = Float64[]; us = Float64[]; ws = Float64[]; sgs = Float64[]; rκ = Float64[]
    wmax = -Inf; n = 0; dt_prev = nothing
    tw = time()
    while d.state.t < T_CI - 1e-9
        _, dt, _ = step_adaptive!(d; dt_prev = dt_prev, t_end = T_CI)
        dt_prev = dt; n += 1
        # the θ budget of every step: d/dt ∫⟨θ⟩dz = ⟨w'θ'⟩_0 (no flux at the top)
        heat += mean(sl.wθ) * dt
        maxres = max(maxres, abs(column_heat(d) - H0 - heat))
        d.state.t >= T_AVG_CI || continue
        wmax = max(wmax, mean(sl.wθ))
        push!(us, Components.sl_ustar(sl)); push!(ws, mean(sl.wθ))
        n % 50 == 0 || continue
        p = Components.sl_profiles(sl, d)
        h = stress_bl_height(p.zf, p.τ)
        push!(hs, h)
        q = gabls1_sgs_share(d, h)
        push!(sgs, q.share); push!(rκ, q.κ_ratio)
    end
    return (h = mean(hs), u_star = mean(us), u_min = minimum(us), u_max = maximum(us),
            wθ = mean(ws), wθ_max = wmax, heat = heat, maxres = maxres,
            sgs_share = mean(sgs), κ_ratio = mean(rκ), nsteps = n, wall = time() - tw)
end

@testset "GABLS1 on the one model (P55-f, ONE-g)" begin

@testset "Reduced GABLS1: θ budget, u*, heat flux, the correction in the SBL" begin
    b = gabls1_ci("brown_et_al_1994")
    n = gabls1_ci("none")
    for (name, r) in (("brown94", b), ("none", n))
        @info "GABLS1 32³, 1.0-1.25 h, $name" r.h r.u_star r.u_min r.u_max r.wθ r.wθ_max r.maxres / abs(r.heat) r.sgs_share r.κ_ratio r.nsteps r.wall
        # θ budget: Δ∫⟨θ⟩dz = ∫⟨w'θ'⟩_0 dt to round-off (measured 1.7e-9 relative
        # on the former sbl run type)
        @test r.maxres <= 1e-6 * abs(r.heat)
        # after the spin-up u* is in [0.2, 0.4] m/s at every step (measured 0.26-0.29)
        @test 0.2 <= r.u_min && r.u_max <= 0.4
        # the cooled surface takes heat from the air at every step
        @test r.wθ_max < 0
    end
    # The correction acts in the SBL: κv/κh < 1 below h (measured 0.63; 1 by
    # construction without it) and the SGS share of the vertical heat flux
    # in the lower half of the BL is reduced (measured 0.108 against 0.153 on
    # the former sbl run type).
    @test b.κ_ratio < 0.8 && n.κ_ratio ≈ 1.0
    @test b.sgs_share < 0.9 * n.sgs_share
    # No assertion on the ordering of h(:brown94) and h(:none): over 1.0-1.25 h
    # they agree within their scatter at this resolution (P55-f: 178.7 ± 3.4 m
    # and 178.2 ± 3.9 m), the numbers are reported above.
end

end # testset
