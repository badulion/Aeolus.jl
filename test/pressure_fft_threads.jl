# KA-h (#298): the spectral pressure solvers on threads.
#
#  * The kernels (copy in / out, the eigenvalue multiply, the DCT and
#    transform normalisation, the Thomas solve over wavenumbers) give the
#    bits of the broadcasts and loops they replaced, on every backend path.
#  * Threaded FFTW plans give the same bits at every FFTW thread count under
#    ESTIMATE (the suite's planner), for every solver and axis-BC
#    combination, so `init_threads!` may run FFTW on Julia's threads.
#
# The cross-process check at julia -t 1/4/24 is the "fft" section of the
# thread-invariance harness (test/support/thread_invariance_worker.jl).

using Test
using LinearAlgebra: mul!
using Random: Xoshiro, randn
using Aeolus
using Aeolus.Fields: Field, interior_view
using Aeolus.Grids: interior
using Aeolus.PressureSolvers: FFTPoissonSolver, CosineFFTPoissonSolver, solve!
using Aeolus.StretchedGrids: StretchedPoissonSolver, _mean_mode!

pft_bits(A) = reinterpret(UInt64, vec(copy(A)))

# ---- the solves as master wrote them before KA-h (broadcasts and loops) ----

function pft_old_solve!(p, source, s::FFTPoissonSolver)
    ii, jj, kk = interior(source.mesh)
    @views s.src_buffer .= source.data[ii, jj, kk]
    mul!(s.hat_buffer, s.forward_plan, s.src_buffer)
    s.hat_buffer .*= s.inv_eigenvalues
    mul!(s.src_buffer, s.inverse_plan, s.hat_buffer)
    @views p.data[ii, jj, kk] .= s.src_buffer
    return p
end

function pft_old_solve!(p, source, s::CosineFFTPoissonSolver)
    ii, jj, kk = interior(source.mesh)
    @views s.real_buf .= source.data[ii, jj, kk]
    s.plan_dct_fwd === nothing ? (s.real_buf2 .= s.real_buf) :
                                 mul!(s.real_buf2, s.plan_dct_fwd, s.real_buf)
    if s.plan_fft_fwd === nothing
        s.real_buf2 .*= s.inv_eigenvalues
    else
        mul!(s.hat_buf, s.plan_fft_fwd, s.real_buf2)
        s.hat_buf .*= s.inv_eigenvalues
        mul!(s.real_buf2, s.plan_fft_inv, s.hat_buf)
    end
    s.plan_dct_inv === nothing ? (s.real_buf .= s.real_buf2) :
                                 mul!(s.real_buf, s.plan_dct_inv, s.real_buf2)
    s.real_buf .*= s.dct_scale
    @views p.data[ii, jj, kk] .= s.real_buf
    return p
end

function pft_old_solve!(p, source, s::StretchedPoissonSolver)
    ii, jj, kk = interior(source.mesh)
    Nx, Ny, Nz = size(s.buf)
    @views s.buf .= source.data[ii, jj, kk]
    mul!(s.buf2, s.plan_fwd, s.buf)
    b = s.buf2
    @views s.col .= b[1, 1, :]
    for k in 1:Nz, q in 1:Ny, pp in 1:Nx            # level by level
        b[pp, q, k] = k == 1 ? b[pp, q, 1] * s.inv_den[pp, q, 1] :
                      (b[pp, q, k] - s.lower[k] * b[pp, q, k-1]) * s.inv_den[pp, q, k]
    end
    for k in Nz-1:-1:1, q in 1:Ny, pp in 1:Nx
        b[pp, q, k] -= s.cp[pp, q, k] * b[pp, q, k+1]
    end
    _mean_mode!(s)
    @views b[1, 1, :] .= s.col
    mul!(s.buf, s.plan_inv, s.buf2)
    @views p.data[ii, jj, kk] .= s.buf .* s.norm
    return p
end

# ---- the solver matrix ------------------------------------------------------

function pft_cases(Nx, Ny, Nz)
    mesh = RegularGrid(; size = (Nx, Ny, Nz), extent = (1.0, 1.3, 0.7), halo = (3, 3, 3))
    smesh = StretchedGrid(; size = (Nx, Ny, Nz), extent = (800.0, 800.0, 3000.0), halo = (3, 3, 3),
                            stretching = (:geometric, 1.06))
    cases = Pair{String,Any}["fft" => (mesh, () -> FFTPoissonSolver(mesh))]
    for bx in (:periodic, :neumann), by in (:periodic, :neumann), bz in (:periodic, :neumann)
        push!(cases, "cosine($bx, $by, $bz)" =>
              (mesh, () -> CosineFFTPoissonSolver(mesh; bc_x = bx, bc_y = by, bc_z = bz)))
    end
    for bx in (:periodic, :neumann), by in (:periodic, :neumann)
        push!(cases, "stretched($bx, $by)" =>
              (smesh, () -> StretchedPoissonSolver(smesh; bc_x = bx, bc_y = by)))
    end
    return cases
end

function pft_source(mesh)
    s = Field(mesh)
    s.data .= randn(Xoshiro(298), size(s.data))     # halos too: they must not be read
    return s
end

# Solve with a fresh solver (its plans made at the current FFTW thread count)
# and return the bits of the whole array, halos included (they are untouched).
function pft_solve(mesh, mk; old = false)
    src = pft_source(mesh)
    p = Field(mesh); fill!(p.data, 7.0)
    s = mk()
    old ? pft_old_solve!(p, src, s) : solve!(p, src, s)
    return p
end

# Run `f` with the FFTW policy at `n` threads, restoring the policy after.
function with_fftw_threads(f, n)
    st = Aeolus.Operators._FFTW_POLICY
    n0, ex0 = st.threads, st.threads_explicit
    Aeolus.set_fftw_policy!(threads = n)
    try
        return f()
    finally
        Aeolus.set_fftw_policy!(threads = n0)
        st.threads_explicit = ex0
    end
end

@testset "Spectral pressure solvers on threads (KA-h, #298)" begin
    @test Aeolus.fftw_policy().planner === :estimate      # set by runtests.jl

    @testset "kernels are bit-identical to the old broadcasts ($name)" for (name, b) in
            ("serial" => CPUBackend(serial = true), "threaded, threshold 0" => CPUBackend(threshold = 0),
             "default" => CPUBackend())
        for sz in ((37, 29, 23), (32, 24, 16))
            for (what, (mesh, mk)) in pft_cases(sz...)
                ref = pft_solve(mesh, mk; old = true)
                new = with_backend(() -> pft_solve(mesh, mk), b)
                @test pft_bits(new.data) == pft_bits(ref.data)
                @test new.data[1, 1, 1] == 7.0                # halos untouched
            end
        end
    end

    @testset "FFTW threads 1/2/4/12/24 give identical solves" begin
        for sz in ((64, 64, 32), (37, 29, 23))
            cases = pft_cases(sz...)
            ref = with_fftw_threads(1) do
                [pft_bits(pft_solve(mesh, mk).data) for (_, (mesh, mk)) in cases]
            end
            for n in (2, 4, 12, 24)
                got = with_fftw_threads(n) do
                    [pft_bits(pft_solve(mesh, mk).data) for (_, (mesh, mk)) in cases]
                end
                for (c, (what, _)) in enumerate(cases)
                    got[c] == ref[c] || @error "FFTW threads $n differ from 1 thread: $what at $sz"
                    @test got[c] == ref[c]
                end
            end
        end
    end

    @testset "the threaded solve still solves the discrete Poisson problem" begin
        # stretched, Neumann sides, FFTW on 4 threads, threaded kernels:
        # the residual of the grid's own Laplacian is round-off
        g = StretchedGrid(; size = (32, 24, 20), extent = (800.0, 600.0, 3000.0), halo = (3, 3, 3),
                            stretching = (:geometric, 1.08))
        src = pft_source(g)
        interior_view(src) .-= sum(interior_view(src) .* reshape(Aeolus.StretchedGrids.cell_thickness(g)[g.Hz+1:g.Hz+g.Nz], 1, 1, :)) /
                               (g.Nx * g.Ny * g.Lz)       # volume-weighted mean removed
        p = with_fftw_threads(4) do
            with_backend(CPUBackend(threshold = 0)) do
                solve!(Field(g), src, StretchedPoissonSolver(g; bc_x = :neumann, bc_y = :neumann))
            end
        end
        # mirror ghosts (Neumann on every face), then the conservative 7-point operator
        φ = copy(p.data); Hx, Hy, Hz = g.Hx, g.Hy, g.Hz; Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
        φ[Hx, :, :] .= φ[Hx+1, :, :];  φ[Hx+Nx+1, :, :] .= φ[Hx+Nx, :, :]
        φ[:, Hy, :] .= φ[:, Hy+1, :];  φ[:, Hy+Ny+1, :] .= φ[:, Hy+Ny, :]
        φ[:, :, Hz] .= φ[:, :, Hz+1];  φ[:, :, Hz+Nz+1] .= φ[:, :, Hz+Nz]
        dzc = Aeolus.StretchedGrids.cell_thickness(g); dzf = Aeolus.StretchedGrids.center_spacing(g)
        res = 0.0; scale = maximum(abs, interior_view(src))
        for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            L = (φ[i+1, j, k] - 2φ[i, j, k] + φ[i-1, j, k]) / g.dx^2 +
                (φ[i, j+1, k] - 2φ[i, j, k] + φ[i, j-1, k]) / g.dy^2
            fluxt = k == Hz + Nz ? 0.0 : (φ[i, j, k+1] - φ[i, j, k]) / dzf[k+1]
            fluxb = k == Hz + 1 ? 0.0 : (φ[i, j, k] - φ[i, j, k-1]) / dzf[k]
            L += (fluxt - fluxb) / dzc[k]
            res = max(res, abs(L - src.data[i, j, k]))
        end
        @test res < 1e-10 * scale
    end
end
