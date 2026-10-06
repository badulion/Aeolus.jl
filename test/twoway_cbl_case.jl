# The two-way convective-boundary-layer case of N-nest-e (#263), shared by
# the suite (test/era2_nest_twoway_longrun.jl: 100 m parent, 2 h) and the
# benchmark (benchmarks/bench_twoway_cbl.jl: 50 m parent, 4 h).
#
# A scaled version of the two-way CBL of Hellsten et al. (2021), GMD 14,
# 3185, §4.1: a 3.2 × 3.2 × 1.6 km periodic parent with a surface heat flux
# of 0.1 K m/s, a 600 m mixed layer under a 5 K/km inversion, Smagorinsky
# SGS; the child covers 1.2 × 1.2 km × 800 m on the ground at r = 3, its
# sides and top nest faces. No mean wind.

using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view

const _TWL_θS = 300.0

"""
    _twl_cbl(feedback; Δ = 100.0, T = 7200.0, seed = 1, r = 3, every = 1)

Run the CBL case with `feedback` (`:one_way`, `:two_way` or a
`TwoWayFeedback`) for `T` seconds at parent spacing `Δ`. Returns the time
series (every `every` parent steps) of the total kinetic energy, the
child-mean `w` at 400 m, the 2Δ amplitude of θ along the parent rows
crossing the nest at a quarter of the child depth and the parent's θ over
the covered columns minus the rest in the child's lower half; the largest child
`|∇·u|` of every step and the largest relative error of the composite θ
budget after 30 min.
"""
function _twl_cbl(feedback; Δ = 100.0, T = 7200.0, seed = 1, r = 3, every = 1)
    N = round(Int, 3200 / Δ); Nz = round(Int, 1600 / Δ)
    pm = RegularGrid(Float64; size = (N, N, Nz), extent = (3200.0, 3200.0, 1600.0),
                     halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; θ₀ = _TWL_θS, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    for k in axes(p.θ.data, 3)
        z = (k - pm.Hz - 0.5) * pm.dz
        p.θ.data[:, :, k] .= _TWL_θS + (z > 600 ? 0.005 * (z - 600) : 0.0)
    end
    θi = interior_view(p.θ); nl = max(1, round(Int, 150 / Δ))
    θi[:, :, 1:nl] .+= 0.2 .* (rand(rng, size(θi, 1), size(θi, 2), nl) .- 0.5)
    nest = nested_grid(pm, ((1000.0, 2200.0), (1000.0, 2200.0), (0.0, 800.0)), r)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; θ₀ = _TWL_θS, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, p, nest; order = 3, region = :all)
    for (cf, sf) in ((c.u, st.u), (c.v, st.v), (c.w, st.w), (c.θ, st.θ))
        cf.data .= sf.data
    end
    run = NestedRun(p, c, nest; feedback, cfl_parent = CFLConstraints(C_advective = 0.5))
    cx, cy, cz = nest_covered_range(nest)
    # the composite θ − θ_s content (the subtraction keeps the round-off of
    # the 300 K background out of the budget)
    function heat()
        h = 0.0
        @inbounds for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
            (i in cx && j in cy && k in cz) && continue
            h += (p.θ.data[pm.Hx+i, pm.Hy+j, pm.Hz+k] - _TWL_θS) * pm.dx * pm.dy * pm.dz
        end
        return h + sum(interior_view(c.θ) .- _TWL_θS) * cm.dx * cm.dy * cm.dz
    end
    H0 = heat()
    ii, jj, kk = interior(cm)
    kc = cm.Hz + cm.Nz ÷ 2                            # child mid-height w faces (400 m)
    jrows = pm.Hy .+ cy                               # parent rows crossing the nest
    krow = pm.Hz + cz[1] + length(cz) ÷ 4             # at a quarter of the child depth
    out = (t = Float64[], ke = Float64[], wc = Float64[], noise = Float64[], dθ = Float64[],
           divmax = Ref(0.0), budget = Ref(0.0), finite = Ref(true))
    klow = pm.Hz .+ (1:length(cz) ÷ 2)                 # parent levels of the child's lower half
    t0 = time()
    while p.t < T - 1e-9
        step_nested!(run)
        out.divmax[] = max(out.divmax[], max_divergence(c))
        if p.t > 1800                                 # budget against ≥ 30 min of heating
            inj = 0.1 * 3200.0^2 * p.t
            out.budget[] = max(out.budget[], abs(heat() - H0 - inj) / inj)
        end
        run.stats.parent_steps % every == 0 || continue
        push!(out.t, p.t); push!(out.ke, kinetic_energy(p) + kinetic_energy(c))
        push!(out.wc, sum(view(c.w.data, ii, jj, kc)) / (length(ii) * length(jj)))
        # 2Δ mode of θ along the parent rows that cross the nest (x direction)
        a = 0.0
        for j in jrows
            s = 0.0
            for i in 1:pm.Nx
                s += (iseven(i) ? 1 : -1) * p.θ.data[pm.Hx+i, j, krow]
            end
            a += (s / pm.Nx)^2
        end
        push!(out.noise, sqrt(a / length(jrows)))
        # the parent's θ over the covered columns minus over the rest, in the
        # child's lower half (the buoyancy behind a secondary circulation)
        sc = 0.0; so = 0.0; nc = 0; no = 0
        @inbounds for k in klow, j in 1:pm.Ny, i in 1:pm.Nx
            v = p.θ.data[pm.Hx+i, pm.Hy+j, k]
            if (i in cx) && (j in cy)
                sc += v; nc += 1
            else
                so += v; no += 1
            end
        end
        push!(out.dθ, sc / nc - so / no)
    end
    out.finite[] = all(isfinite, p.u.data) && all(isfinite, c.u.data) &&
                   all(isfinite, p.θ.data) && all(isfinite, c.θ.data)
    return out, run, time() - t0
end

_twl_window(t, x, a, b) = (m = (t .>= a) .& (t .< b); sum(x[m]) / count(m))

# least-squares slope of y(t) and its standard error
function _twl_fit(t, y)
    n = length(t); tm = sum(t) / n; ym = sum(y) / n
    Stt = sum(abs2, t .- tm)
    s = sum((t .- tm) .* (y .- ym)) / Stt
    res = y .- (ym .+ s .* (t .- tm))
    se = sqrt(sum(abs2, res) / (n - 2) / Stt)
    return s, se
end

