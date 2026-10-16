# Transcripts of the column-physics loops before KA-m (#301), kept as the
# reference of test/ka_columns.jl. Each is the pre-port loop
# body as it was on the base of the port (KA-b #367 merged with master at
# c617bc1), with its serial loop order and its running accumulators.

module LegacyColumns

using Aeolus
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, PhasePartition, exner_function
using Aeolus.RadiationFog: fog_condensation
using Aeolus.P3Microphysics: p3_sediment!, p3_rain_sedimentation_velocities

"RadiationFog.fog_adjust! before KA-m: one sweep, one running total."
function fog_adjust!(θ, qv, ql, p, dt; τ = 0.0, levels = axes(θ, 3), c = DEFAULT_CONSTANTS)
    Nx, Ny, _ = size(θ)
    total = 0.0
    @inbounds for k in levels
        pk = p isa AbstractVector ? float(p[k]) : float(p)
        Π  = exner_function(pk, PhasePartition(0.0), c)
        for j in 1:Ny, i in 1:Nx
            T = θ[i, j, k] * Π
            r = fog_condensation(T, pk, qv[i, j, k], ql[i, j, k], dt; τ = τ, c = c)
            r.Δq == 0 && continue
            θ[i, j, k]  = r.T / Π
            qv[i, j, k] = r.qv
            ql[i, j, k] = r.ql
            total += r.Δq
        end
    end
    return total
end

"Microphysics.apply_kessler_step! before KA-m (serial loops, one shared column scratch)."
function apply_kessler_step!(qv, ql, qr, dt; k_auto = 1e-3, ql_c = 1e-3, k_acc = 2.2,
                             Vt = :spectrum, ρ_air = Aeolus.Superdroplet.DEFAULT_MICRO.ρ_air,
                             N0 = Aeolus.Microphysics.KESSLER_N0, mesh = qv.mesh)
    dz = mesh.dz; Δt = dt
    ii, jj, kk = Aeolus.Grids.interior(mesh)
    qld = ql.data; qrd = qr.data
    @inbounds for k in kk, j in jj, i in ii
        qle = qld[i,j,k]
        qre = qrd[i,j,k]
        auto = k_auto * max(qle - ql_c, 0.0)
        accr = k_acc * qle * (max(qre, 0.0) ^ (7/8))
        dqr = (auto + accr) * Δt
        dqr = min(dqr, max(qle, 0.0))
        qld[i,j,k] -= dqr
        qrd[i,j,k] += dqr
    end
    if Vt === :spectrum
        nz = length(kk)
        ρ = ρ_air isa AbstractVector ? collect(float.(ρ_air)) : fill(float(ρ_air), nz)
        col = zeros(eltype(qrd), nz)
        Nr = similar(col)
        function vel(f)
            Nr .= Aeolus.Microphysics.kessler_rain_number.(f[1], ρ; N0 = N0)
            return (p3_rain_sedimentation_velocities(f[1], Nr, ρ)[2],)
        end
        @inbounds for j in jj, i in ii
            for (n, k) in enumerate(kk); col[n] = qrd[i, j, k]; end
            any(>(0), col) || continue
            p3_sediment!((col,), vel, ρ, dz, Δt)
            for (n, k) in enumerate(kk); qrd[i, j, k] = col[n]; end
        end
        return (qv, ql, qr)
    end
    if Vt > 0
        α = min(Vt * Δt / dz, 0.5)
        @inbounds for k in kk, j in jj, i in ii
            in_from_above = k + 1 <= last(kk) ? qrd[i,j,k+1] : 0.0
            qrd[i,j,k] += α * (in_from_above - qrd[i,j,k])
        end
    end
    return (qv, ql, qr)
end

"Chemistry.apply_dry_deposition! before KA-m (one running total over the columns)."
function apply_dry_deposition!(c, v_d, dt; dz, k = 1)
    Nx, Ny, Nz = size(c)
    removed = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        kk = k isa Integer ? k : k[i, j]
        1 <= kk <= Nz || continue
        vd = v_d isa Real ? v_d : v_d[i, j]
        c0 = c[i, j, kk]
        c1 = c0 * exp(-vd * dt / dz)
        c[i, j, kk] = c1
        removed += (c0 - c1) * dz
    end
    return removed
end

end # module
