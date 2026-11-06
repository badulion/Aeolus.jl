# N-parent-b (#254): validation of the free-atmosphere top of a deep domain
# — gravity-wave absorption (Klemp & Lilly 1978, JAS 35, 78; Durran 2010
# §8.4): the reflected energy fraction of a simulated 2-D packet
# (N = 0.01 s⁻¹, λ_z = 1 km) from damping layers 1 and 2 λ_z deep,
# velocity-only and with θ damping (`ReferenceDamping`) added, against a
# rigid lid. The top stays tested in test/deep_top.jl (relaxation,
# locality and exactness of the targets, the fixed-gradient top, the
# budget).

using Test, Printf, Random
using FFTW: fft, fftfreq
using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Grids: znodes, xnodes
using Aeolus.Operators: sponge_profile

# The 2-D gravity-wave packet of test 3. Returns the reflected energy
# fraction: the time-integrated energy of the downward-propagating part
# (m > 0 for k > 0, ω > 0) over the time-integrated energy of the upward part
# (m < 0), both from the vertical wavenumber spectrum of the complex x-mode
# amplitude of w in a window below the damping layer. Both packets cross the
# window at the same |c_gz| (a reflection keeps k, ω and |m|), so the ratio
# of the time integrals is the energy ratio.
function _gw_reflection(; depth, θdamp, α, dz = 50.0, dt = 20.0, t_end = 22_000.0,
                        z_s = 6000.0, z_packet = 2000.0, σ = 600.0, window = (3000.0, 5500.0))
    N = 0.01; λ = 1000.0; Nx = 16
    k = 2π / λ; m = -2π / λ                          # m < 0: upward group velocity
    ω = N * k / sqrt(k^2 + m^2)
    H = z_s + depth
    Nz = round(Int, H / dz)
    g = RegularGrid(; size = (Nx, 2, Nz), extent = (λ, 2λ / Nx, H), halo = (2, 2, 2))
    θ0 = 300.0; ga = 9.81
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(g; ν = 0.0, θ₀ = θ0, g_accel = ga, bcs = bcs, θ_bcs = bcs,
                               sponge_α_max = α, sponge_z_start = z_s)
    θbar(z) = θ0 * (1 + N^2 * z / ga)
    A = 1e-3
    env(z) = exp(-(z - z_packet)^2 / (2σ^2))
    # velocity from a stream function on the C-grid: discretely divergence-free
    ψ(x, z) = (A / k) * env(z) * sin(k * x + m * z)
    set!(sim.u, (x, y, z) -> -(ψ(x, z + dz / 2) - ψ(x, z - dz / 2)) / dz)
    set!(sim.w, (x, y, z) -> (ψ(x + g.dx / 2, z) - ψ(x - g.dx / 2, z)) / g.dx)
    # b = g θ'/θ0 = (N²/ω) A sin φ (linear polarization of the upward wave)
    set!(sim.θ, (x, y, z) -> θbar(z) + θ0 / ga * (N^2 / ω) * A * env(z) * sin(k * x + m * z))
    zc = collect(znodes(g, CellCenter()))
    F = ReferenceDamping(:θ, sponge_profile(α, z_s, H), θbar.(zc))
    bf! = θdamp ? ((s, Δt) -> apply_forcing!(s, F, s.t, Δt)) : nothing
    kr = findall(z -> window[1] <= z <= window[2], zc)
    nk = length(kr)
    hann = [0.5 - 0.5cos(2π * (n - 0.5) / nk) for n in 1:nk]
    ms = 2π .* fftfreq(nk, 1 / dz)
    xs = collect(xnodes(g, CellCenter()))
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    Eup = 0.0; Edn = 0.0
    W = zeros(ComplexF64, nk)
    nsteps = round(Int, t_end / dt)
    for n in 0:nsteps
        if n % 5 == 0
            for (q, kk) in enumerate(kr)
                s = 0.0im
                for i in 1:Nx
                    wc = 0.5 * (sim.w.data[Hx+i, Hy+1, Hz+kk] + sim.w.data[Hx+i, Hy+1, Hz+kk+1])
                    s += wc * cis(-k * xs[i])
                end
                W[q] = 2s / Nx * hann[q]
            end
            Wm = fft(W)
            for i in 1:nk
                ms[i] < 0 ? (Eup += abs2(Wm[i])) : ms[i] > 0 ? (Edn += abs2(Wm[i])) : nothing
            end
        end
        n == nsteps && break
        step_rk3!(sim, dt; body_force! = bf!)
    end
    return Edn / Eup
end

@testset "N-parent-b: free-atmosphere top, gravity-wave absorption (#254)" begin

    # ---------------------------------------------------------------------
    @testset "3. gravity-wave absorption (Klemp & Lilly 1978)" begin
        # rigid lid, no damping: the measurement sees the full reflection
        R_lid = _gw_reflection(; depth = 1000.0, θdamp = false, α = 0.0)
        α = 1e-3                                   # λ_max = 0.1 N ≈ ω/7
        R = Dict{Tuple{Float64,Bool},Float64}()
        for depth in (1000.0, 2000.0), θd in (false, true)
            R[(depth, θd)] = _gw_reflection(; depth = depth, θdamp = θd, α = α)
        end
        @info @sprintf("gravity-wave reflection, α_max = %.0e s⁻¹: lid %.3f; depth 1 λz: uvw %.4f, +θ %.4f; depth 2 λz: uvw %.4f, +θ %.4f",
                       α, R_lid, R[(1000.0, false)], R[(1000.0, true)], R[(2000.0, false)], R[(2000.0, true)])
        @test R_lid > 0.9
        for depth in (1000.0, 2000.0)
            @test R[(depth, true)] < 0.05                     # the issue's target
            @test R[(depth, true)] < R[(depth, false)] / 4    # θ damping absorbs what velocity damping lets through
        end
        @test R[(2000.0, true)] < R[(1000.0, true)]
    end

end
