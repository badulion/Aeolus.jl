# Transcripts of the radiosity loops before KA-n (#305), from master at
# c617bc1, kept as the references of test/ka_radiation.jl:
#
#   * `legacy_aggregate_to_grid`: the facet → column scatter (with the
#     per-column dominant class from a Dict), replaced by the column gather;
#   * `legacy_view_factors`: the serial upper-triangle pair loop and the
#     serial row sums of the sky-view factor, replaced by the launch.
#
# Only the function names changed.

using Aeolus.Radiosity: RadiosityScene, ViewFactors, ShortwaveSolution, LongwaveSolution,
                        UrbanRadiationForcing, URBAN_STREET, nfacets, facet_areas,
                        exchange_area
using Aeolus.RadiationCore: STEFAN_BOLTZMANN

function legacy_aggregate_to_grid(s::RadiosityScene{T}, vf::ViewFactors{T};
                           sw::Union{Nothing, ShortwaveSolution{T}} = nothing,
                           lw::Union{Nothing, LongwaveSolution{T}} = nothing,
                           T_surface::Union{Nothing, AbstractMatrix} = nothing) where {T}
    nx, ny = size(s.heightmap)
    cell = s.dx * s.dy
    swa = zeros(T, nx, ny)
    lwn = zeros(T, nx, ny)
    aden = zeros(T, nx, ny)
    svf = zeros(T, nx, ny)
    Tw = zeros(T, nx, ny)
    cls = fill(URBAN_STREET, nx, ny)
    # Per column, the dominant class is the one holding the most facet area —
    # for a built column that is the roof, for a canyon column the ground or
    # the wall, whichever carries more surface.
    areaper = Dict{Tuple{Int, Int, Int}, T}()

    @inbounds for i in 1:nfacets(s)
        f = s.facets[i]
        ci, cj = f.col
        (1 <= ci <= nx && 1 <= cj <= ny) || continue
        a = f.area
        sw === nothing || (swa[ci, cj] += a * sw.absorbed[i])
        lw === nothing || (lwn[ci, cj] += a * lw.net_loss[i])
        aden[ci, cj] += a
        svf[ci, cj] += a * vf.F_sky[i]
        Tw[ci, cj] += a * f.temperature
        k = (ci, cj, f.material)
        areaper[k] = get(areaper, k, zero(T)) + a
    end

    @inbounds for j in 1:ny, i in 1:nx
        if aden[i, j] > 0
            svf[i, j] /= aden[i, j]
            Tw[i, j] /= aden[i, j]
        else
            svf[i, j] = one(T)
            Tw[i, j] = T(288.15)
        end
        swa[i, j] /= cell
        lwn[i, j] /= cell
        aden[i, j] /= cell
    end
    # Dominant class per column.
    best = Dict{Tuple{Int, Int}, T}()
    for ((i, j, m), a) in areaper
        if a > get(best, (i, j), T(-Inf))
            best[(i, j)] = a
            cls[i, j] = m
        end
    end

    swe = zeros(T, nx, ny)
    lwe = zeros(T, nx, ny)
    σ = T(STEFAN_BOLTZMANN)
    @inbounds for j in 1:ny, i in 1:nx
        m = cls[i, j]
        α = s.materials.albedo[m]
        ε = s.materials.emissivity[m]
        swe[i, j] = swa[i, j] / max(one(T) - α, eps(T))
        Tsfc = T_surface === nothing ? Tw[i, j] : T(T_surface[i, j])
        # the surface balance forms ε·lw_in - ε σ T⁴; we want that to equal -lw_net.
        lwe[i, j] = σ * Tsfc^4 - lwn[i, j] / max(ε, eps(T))
    end

    return UrbanRadiationForcing{T}(swa, lwn, aden, svf, swe, lwe, cls)
end

function legacy_view_factors(s::RadiosityScene{T}) where {T}
    n = nfacets(s)
    F = zeros(T, n, n)
    A = facet_areas(s)
    @inbounds for i in 1:n-1
        for j in i+1:n
            X = exchange_area(s, i, j)
            X == 0 && continue
            F[i, j] = X / A[i]
            F[j, i] = X / A[j]
        end
    end
    F_sky = Vector{T}(undef, n)
    @inbounds for i in 1:n
        acc = zero(T)
        for j in 1:n
            acc += F[i, j]
        end
        F_sky[i] = clamp(one(T) - acc, zero(T), one(T))
    end
    return F, F_sky
end
