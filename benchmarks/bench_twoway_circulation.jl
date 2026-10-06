# The #327 (M5) investigation of the two-way secondary circulation over a
# convective child: paired one-way / two-way runs of reduced CBL nests
# (parent at 100 m, 2 h, the 2nd hour averaged) and standalone periodic
# CBLs at the parent's and the child's spacing. The ensemble table on
# docs/src/theory/nesting_twoway.md was produced with this script (4–6 seeds
# per configuration, ≈ 70 min of single-core runs in total).
#
# Run one member:
#   julia --project=. benchmarks/bench_twoway_circulation.jl <case> <one_way|two_way> <seed> <out.txt>
# cases: orig (the suite's case), big (2.4 km child), hellsten (2.4 km × 500 m
# child, r = 2, lapse-rate start), otop / htop (child tops above the CBL),
# orig_r1 / hellsten_r1 / otop_r1 (r = 1), orig_b0 / orig_b4 (buffer 0 / 4),
# orig_noreflux, orig_nofc; standalone: sa_<ml600|lapse>_<Δ>, e.g.
# sa_ml600_33.3333333333 (the <fb> argument is ignored).
# Output: time-mean (2nd hour) profiles of the child-mean w and θ, the parent
# w and θ over the covered columns and over the rest.

using Printf, Random, Aeolus, Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view

const θS = 300.0
function setup(case)
    if case == "orig_r1"
        return (Lp = 3200.0, Hp = 1600.0, Lc = 1200.0, Hc = 800.0, r = 1, prof = :ml600)
    elseif startswith(case, "orig")
        return (Lp = 3200.0, Hp = 1600.0, Lc = 1200.0, Hc = 800.0, r = 3, prof = :ml600)
    elseif case == "big"
        return (Lp = 6400.0, Hp = 1600.0, Lc = 2400.0, Hc = 800.0, r = 3, prof = :ml600)
    elseif case == "hellsten"
        return (Lp = 6400.0, Hp = 1600.0, Lc = 2400.0, Hc = 500.0, r = 2, prof = :lapse)
    elseif case == "otop"
        return (Lp = 3200.0, Hp = 1600.0, Lc = 1200.0, Hc = 1200.0, r = 3, prof = :ml600)
    elseif case == "otop_r1"
        return (Lp = 3200.0, Hp = 1600.0, Lc = 1200.0, Hc = 1200.0, r = 1, prof = :ml600)
    elseif case == "htop"
        return (Lp = 6400.0, Hp = 1600.0, Lc = 2400.0, Hc = 1200.0, r = 2, prof = :lapse)
    elseif case == "orig_r1"
        return (Lp = 3200.0, Hp = 1600.0, Lc = 1200.0, Hc = 800.0, r = 1, prof = :ml600)
    elseif case == "hellsten_r1"
        return (Lp = 6400.0, Hp = 1600.0, Lc = 2400.0, Hc = 500.0, r = 1, prof = :lapse)
    elseif case == "hellsten_r3"
        return (Lp = 6400.0, Hp = 1600.0, Lc = 2400.0, Hc = 500.0, r = 3, prof = :lapse)
    end
    error("case $case")
end

function feedback_of(case, fb)
    fb == "one_way" && return :one_way
    case == "orig_b0" && return TwoWayFeedback(buffer = 0)
    case == "orig_b4" && return TwoWayFeedback(buffer = 4)
    case == "orig_noreflux" && return TwoWayFeedback(reflux = false)
    return :two_way
end

function run_case(case, fbs, seed, outfile; Δ = 100.0, T = 7200.0, tavg = 3600.0)
    S = setup(case)
    N = round(Int, S.Lp / Δ); Nz = round(Int, S.Hp / Δ)
    pm = RegularGrid(Float64; size = (N, N, Nz), extent = (S.Lp, S.Lp, S.Hp), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; θ₀ = θS, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    for k in axes(p.θ.data, 3)
        z = (k - pm.Hz - 0.5) * pm.dz
        p.θ.data[:, :, k] .= S.prof === :ml600 ? θS + (z > 600 ? 0.005 * (z - 600) : 0.0) :
                                                θS + 0.003 * z
    end
    θi = interior_view(p.θ); nl = max(1, round(Int, 150 / Δ))
    θi[:, :, 1:nl] .+= 0.2 .* (rand(rng, size(θi, 1), size(θi, 2), nl) .- 0.5)
    x0 = (S.Lp - S.Lc) / 2
    nest = nested_grid(pm, ((x0, x0 + S.Lc), (x0, x0 + S.Lc), (0.0, S.Hc)), S.r)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; θ₀ = θS, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, p, nest; order = 3, region = :all)
    for (cf, sf) in ((c.u, st.u), (c.v, st.v), (c.w, st.w), (c.θ, st.θ))
        cf.data .= sf.data
    end
    fb = feedback_of(case, fbs)
    run = NestedRun(p, c, nest; feedback = fb, flux_correction = case != "orig_nofc",
                    cfl_parent = CFLConstraints(C_advective = 0.5))
    cx, cy, cz = nest_covered_range(nest)
    ii, jj, kk = interior(cm)
    # accumulators (dt-weighted over t > tavg)
    wc = zeros(cm.Nz + 1); θc = zeros(cm.Nz)           # child-mean w (faces), θ
    wpc = zeros(pm.Nz + 1); wpo = zeros(pm.Nz + 1)      # parent mean w covered / outside cols
    θpc = zeros(pm.Nz); θpo = zeros(pm.Nz)               # parent mean θ covered / outside cols
    wcser = Float64[]; tser = Float64[]
    acc = 0.0
    incov(i, j) = (i in cx) && (j in cy)
    ncov = length(cx) * length(cy); nout = N * N - ncov
    t0 = time()
    while p.t < T - 1e-9
        tp = p.t
        step_nested!(run)
        dt = p.t - tp
        p.t <= tavg && continue
        acc += dt
        for k in 1:cm.Nz+1
            wc[k] += dt * sum(view(c.w.data, ii, jj, cm.Hz + k)) / (cm.Nx * cm.Ny)
        end
        for k in 1:cm.Nz
            θc[k] += dt * (sum(view(c.θ.data, ii, jj, cm.Hz + k)) / (cm.Nx * cm.Ny) - θS)
        end
        for k in 1:pm.Nz+1
            sc = 0.0; so = 0.0
            @inbounds for j in 1:N, i in 1:N
                v = p.w.data[pm.Hx+i, pm.Hy+j, pm.Hz+k]
                incov(i, j) ? (sc += v) : (so += v)
            end
            wpc[k] += dt * sc / ncov; wpo[k] += dt * so / nout
        end
        for k in 1:pm.Nz
            sc = 0.0; so = 0.0
            @inbounds for j in 1:N, i in 1:N
                v = p.θ.data[pm.Hx+i, pm.Hy+j, pm.Hz+k] - θS
                incov(i, j) ? (sc += v) : (so += v)
            end
            θpc[k] += dt * sc / ncov; θpo[k] += dt * so / nout
        end
        kmid = cm.Hz + (cm.Nz + 1) ÷ 2
        push!(tser, p.t); push!(wcser, sum(view(c.w.data, ii, jj, kmid)) / (cm.Nx * cm.Ny))
    end
    wall = time() - t0
    open(outfile, "w") do io
        println(io, "case=$case fb=$fbs seed=$seed wall=$(round(wall, digits=1)) steps=$(run.stats.parent_steps) substeps=$(run.stats.child_substeps) retries=$(run.stats.parent_retries)/$(run.stats.child_retries) finite=$(all(isfinite, p.u.data) && all(isfinite, c.u.data))")
        println(io, "dzc=$(cm.dz) dzp=$(pm.dz) Hc=$(S.Hc)")
        println(io, "wc ", join(round.(wc ./ acc, digits = 5), " "))
        println(io, "thc ", join(round.(θc ./ acc, digits = 5), " "))
        println(io, "wpc ", join(round.(wpc ./ acc, digits = 5), " "))
        println(io, "wpo ", join(round.(wpo ./ acc, digits = 5), " "))
        println(io, "thpc ", join(round.(θpc ./ acc, digits = 5), " "))
        println(io, "thpo ", join(round.(θpo ./ acc, digits = 5), " "))
        println(io, "tser ", join(round.(tser, digits = 1), " "))
        println(io, "wcser ", join(round.(wcser, digits = 4), " "))
    end
    println("done $case $fbs $seed wall $wall")
end

# standalone periodic CBLs: sa_<prof>_<Δ>, e.g. sa_ml600_33.333, sa_lapse_50
function run_standalone(case, seed, outfile; L = 2400.0, Hp = 1600.0, T = 7200.0, tavg = 3600.0)
    _, prof, ds = split(case, "_")
    Δ = parse(Float64, ds)
    N = round(Int, L / Δ); Nz = round(Int, Hp / Δ)
    m = RegularGrid(Float64; size = (N, N, Nz), extent = (L, L, Hp), halo = (3, 3, 3))
    s = BoussinesqSimulation(m; θ₀ = θS, C_s = 0.17, θ_flux_bottom = 0.1,
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    for k in axes(s.θ.data, 3)
        z = (k - m.Hz - 0.5) * m.dz
        s.θ.data[:, :, k] .= prof == "ml600" ? θS + (z > 600 ? 0.005 * (z - 600) : 0.0) : θS + 0.003 * z
    end
    θi = interior_view(s.θ); nl = max(1, round(Int, 150 / Δ))
    θi[:, :, 1:nl] .+= 0.2 .* (rand(rng, size(θi, 1), size(θi, 2), nl) .- 0.5)
    c = CFLConstraints(C_advective = 0.5)
    θc = zeros(Nz); acc = 0.0; dtp = nothing; t0 = time()
    while s.t < T - 1e-9
        tp = s.t
        _, dtp, _ = step_adaptive!(s, c; dt_prev = dtp, divergence_tol = Inf)
        dt = s.t - tp
        s.t <= tavg && continue
        acc += dt
        for k in 1:Nz
            θc[k] += dt * (sum(view(s.θ.data, m.Hx+1:m.Hx+N, m.Hy+1:m.Hy+N, m.Hz+k)) / N^2 - θS)
        end
    end
    open(outfile, "w") do io
        println(io, "case=$case seed=$seed wall=$(round(time() - t0, digits=1))")
        println(io, "dz=$(m.dz)")
        println(io, "th ", join(round.(θc ./ acc, digits = 5), " "))
    end
end

if startswith(ARGS[1], "sa_")
    run_standalone(ARGS[1], parse(Int, ARGS[3]), ARGS[4])
else
    run_case(ARGS[1], ARGS[2], parse(Int, ARGS[3]), ARGS[4])
end
