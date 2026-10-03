# Era VI, Phase 54 (P54-c, #168): the LES analysis toolbox.
#
#   * explicit filters (box, extracted from the dynamic Smagorinsky model;
#     Gaussian and sharp spectral, periodic, FFTW), with a solid mask;
#   * the shell-summed isotropic energy spectrum;
#   * the solenoidal random-phase initialiser that replaced the test-local
#     `_kolmogorov_field!` of era2_sgs_numerics.jl.
#
# What is asserted: exact identities (Parseval, a single Fourier mode, the
# analytic transfer functions of the filters, the discrete divergence of
# the initial field) and the bit-for-bit refactor of the dynamic model.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Random: Xoshiro, randn
using Statistics: mean
using Printf

# Frozen copy of `_dynamic_smagorinsky!` as it was on master (cc4a150),
# before its test filter became `box_filter!`'s stencil sum. The refactor
# must reproduce it bit for bit.
function _legacy_dynamic_smagorinsky!(nd, ud, vd, wd, ws::Aeolus.Operators.DynamicSmagorinskyWorkspace{T}, g,
                               plane::Bool, α2, cs2max) where {T}
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    Uc, Vc, Wc, magS = ws.Uc, ws.Vc, ws.Wc, ws.magS
    S11, S22, S33, S12, S13, S23 = ws.S11, ws.S22, ws.S33, ws.S12, ws.S13, ws.S23

    # Pass 1: cell-centred velocity and strain on the interior + one shell.
    # Local index l ∈ 1:N+2 is data index H - 1 + l.
    @inbounds for lk in 1:Nz+2, lj in 1:Ny+2, li in 1:Nx+2
        i = Hx - 1 + li; j = Hy - 1 + lj; k = Hz - 1 + lk
        Uc[li,lj,lk] = 0.5 * (ud[i,j,k] + ud[i+1,j,k])
        Vc[li,lj,lk] = 0.5 * (vd[i,j,k] + vd[i,j+1,k])
        Wc[li,lj,lk] = 0.5 * (wd[i,j,k] + wd[i,j,k+1])
        s11, s22, s33, s12, s13, s23, ms = Aeolus.Operators._strain_at(ud, vd, wd, i, j, k, idx, idy, idz)
        S11[li,lj,lk] = s11; S22[li,lj,lk] = s22; S33[li,lj,lk] = s33
        S12[li,lj,lk] = s12; S13[li,lj,lk] = s13; S23[li,lj,lk] = s23
        magS[li,lj,lk] = ms
    end

    # Pass 2: L_ij M_ij and M_ij M_ij at every interior cell.
    inv27 = one(T) / 27
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        li = i + 1; lj = j + 1; lk = k + 1
        Ut = zero(T); Vt = zero(T); Wt = zero(T)
        UU = zero(T); VV = zero(T); WW = zero(T); UV = zero(T); UW = zero(T); VW = zero(T)
        A11 = zero(T); A22 = zero(T); A33 = zero(T); A12 = zero(T); A13 = zero(T); A23 = zero(T)
        B11 = zero(T); B22 = zero(T); B33 = zero(T); B12 = zero(T); B13 = zero(T); B23 = zero(T)
        for dk in -1:1, dj in -1:1, di in -1:1
            a = li + di; b = lj + dj; c = lk + dk
            uu = Uc[a,b,c]; vv = Vc[a,b,c]; ww = Wc[a,b,c]
            Ut += uu; Vt += vv; Wt += ww
            UU += uu*uu; VV += vv*vv; WW += ww*ww
            UV += uu*vv; UW += uu*ww; VW += vv*ww
            s11 = S11[a,b,c]; s22 = S22[a,b,c]; s33 = S33[a,b,c]
            s12 = S12[a,b,c]; s13 = S13[a,b,c]; s23 = S23[a,b,c]
            ms = magS[a,b,c]
            A11 += s11; A22 += s22; A33 += s33; A12 += s12; A13 += s13; A23 += s23
            B11 += ms*s11; B22 += ms*s22; B33 += ms*s33
            B12 += ms*s12; B13 += ms*s13; B23 += ms*s23
        end
        Ut *= inv27; Vt *= inv27; Wt *= inv27
        L11 = UU*inv27 - Ut*Ut; L22 = VV*inv27 - Vt*Vt; L33 = WW*inv27 - Wt*Wt
        L12 = UV*inv27 - Ut*Vt; L13 = UW*inv27 - Ut*Wt; L23 = VW*inv27 - Vt*Wt
        tr = (L11 + L22 + L33) / 3
        L11 -= tr; L22 -= tr; L33 -= tr
        A11 *= inv27; A22 *= inv27; A33 *= inv27; A12 *= inv27; A13 *= inv27; A23 *= inv27
        magSt = sqrt(2 * (A11^2 + A22^2 + A33^2) + 4 * (A12^2 + A13^2 + A23^2))
        # L = T − τ̃ with τ = −2C_s²Δ²|S|S (deviatoric) gives L = C_s² M with
        # M = 2Δ²(~(|S|S) − α²|S̃|S̃). (The example's copy had the opposite
        # sign, so C_s² came out negative in forward scatter and was clipped.)
        c1 = 2 * Δ2 * α2 * magSt; c2 = 2 * Δ2 * inv27
        M11 = c2*B11 - c1*A11; M22 = c2*B22 - c1*A22; M33 = c2*B33 - c1*A33
        M12 = c2*B12 - c1*A12; M13 = c2*B13 - c1*A13; M23 = c2*B23 - c1*A23
        ws.LM[i,j,k] = L11*M11 + L22*M22 + L33*M33 + 2 * (L12*M12 + L13*M13 + L23*M23)
        ws.MM[i,j,k] = M11*M11 + M22*M22 + M33*M33 + 2 * (M12*M12 + M13*M13 + M23*M23)
    end

    # Averaging of the identity and the clip to [0, cs_max²].
    cs2 = ws.cs2
    if plane
        @inbounds for k in 1:Nz
            lm = zero(T); mm = zero(T)
            for j in 1:Ny, i in 1:Nx
                lm += ws.LM[i,j,k]; mm += ws.MM[i,j,k]
            end
            c = mm > 0 ? clamp(lm / mm, zero(T), T(cs2max)) : zero(T)
            for j in 1:Ny, i in 1:Nx
                cs2[i,j,k] = c
            end
        end
    else
        @inbounds for I in eachindex(cs2)
            mm = ws.MM[I]
            cs2[I] = mm > 0 ? clamp(ws.LM[I] / mm, zero(T), T(cs2max)) : zero(T)
        end
    end

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        nd[Hx+i, Hy+j, Hz+k] = cs2[i,j,k] * Δ2 * magS[i+1, j+1, k+1]
    end
    return nothing
end

_periodic() = FieldBoundaryConditions(:periodic)

# Random interior values at every location, halos filled periodically.
function _random_fields(mesh, rng)
    fs = (Field(CellCenter(), mesh), Field(FaceX(), mesh), Field(FaceY(), mesh), Field(FaceZ(), mesh))
    for f in fs
        interior_view(f) .= randn(rng, size(interior_view(f)))
        fill_halos!(f, _periodic())
    end
    return fs
end

@testset "Era VI — LES analysis toolbox (P54-c, #168)" begin

    @testset "dynamic Smagorinsky: bit for bit after extracting the box filter" begin
        # The test filter of the dynamic model is now `box_filter!`'s
        # stencil sum (one source of truth). Same field, same workspace
        # layout: L_ij M_ij, M_ij M_ij, C_s² and ν_t must agree with `==`.
        rng = Xoshiro(11)
        for (sz, ext, avg, α) in (((16, 12, 10), (1.0, 0.75, 0.5), :plane, 3),
                                  ((16, 12, 10), (1.0, 0.75, 0.5), :local, 3),
                                  ((12, 12, 12), (2π, 2π, 2π), :plane, 2))
            mesh = RegularGrid(; size = sz, extent = ext, halo = (3, 3, 3))
            _, u, v, w = _random_fields(mesh, rng)
            new = DynamicSmagorinskyWorkspace(mesh); old = DynamicSmagorinskyWorkspace(mesh)
            νn = Field(CellCenter(), mesh); νo = Field(CellCenter(), mesh)
            compute_dynamic_smagorinsky_νt!(νn, u, v, w; workspace = new, averaging = avg, filter_ratio = α)
            _legacy_dynamic_smagorinsky!(νo.data, u.data, v.data, w.data, old, mesh,
                                         avg === :plane, float(α)^2, 0.25)
            @test new.LM == old.LM && new.MM == old.MM
            @test new.cs2 == old.cs2
            @test νn.data == νo.data
        end
    end

    @testset "box filter: transfer function, mean, staggered locations" begin
        # A discrete 3-point top hat multiplies the mode e^{ikx} by
        # (1 + 2cos kΔ)/3 per direction; n = 5 by (1 + 2cos kΔ + 2cos 2kΔ)/5.
        mesh = RegularGrid(; size = (16, 16, 16), extent = (2π, 2π, 2π), halo = (2, 2, 2))
        Δ = mesh.dx
        for (loc, n) in ((CellCenter(), 3), (FaceX(), 3), (FaceZ(), 3), (FaceY(), (3, 5, 1)))
            f = Field(loc, mesh); out = Field(loc, mesh)
            set!(f, (x, y, z) -> cos(2x + 3y - z))
            fill_halos!(f, _periodic())
            box_filter!(out, f, n)
            nn = n isa Integer ? (n, n, n) : n
            G(k, m) = m == 1 ? 1.0 : m == 3 ? (1 + 2cos(k * Δ)) / 3 : (1 + 2cos(k * Δ) + 2cos(2k * Δ)) / 5
            ratio = G(2, nn[1]) * G(3, nn[2]) * G(1, nn[3])
            @test maximum(abs, interior_view(out) .- ratio .* interior_view(f)) < 1e-14
        end
        # the domain mean survives filtering (periodic halos)
        for f in _random_fields(mesh, Xoshiro(2))
            out = Field(Aeolus.Fields.location(f), mesh)
            box_filter!(out, f, 3)
            @test abs(mean(interior_view(out)) - mean(interior_view(f))) < 1e-14
        end
        f, = _random_fields(mesh, Xoshiro(3))
        @test_throws ArgumentError box_filter!(f, f)
        @test_throws ArgumentError box_filter!(Field(CellCenter(), mesh), f, 4)
        @test_throws ArgumentError box_filter!(Field(CellCenter(), mesh), f, 7)   # halo 2 < 3
    end

    @testset "box filter with a solid mask: renormalised over fluid neighbours" begin
        mesh = RegularGrid(; size = (12, 12, 12), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        cell = falses(size(Field(CellCenter(), mesh).data))
        ii, jj, kk = Aeolus.Grids.interior(mesh)
        for k in kk[1:4], j in jj[3:6], i in ii[5:8]          # a block "building"
            cell[i, j, k] = true
        end
        for loc in (CellCenter(), FaceX(), FaceY(), FaceZ())
            m = staggered_solid_mask(cell, loc)
            f = Field(loc, mesh); set!(f, (x, y, z) -> sin(2π * x) * cos(2π * y) + z)
            fill_halos!(f, _periodic())
            a = Field(loc, mesh); b = Field(loc, mesh)
            box_filter!(a, f, 3; mask = m)
            # garbage (and NaN) inside the solid must not reach the result
            g = Field(loc, mesh); g.data .= f.data
            g.data[m] .= ifelse.(isodd.(1:count(m)), 1e30, NaN)
            box_filter!(b, g, 3; mask = m)
            @test a.data == b.data
            # a field that is constant over the fluid is reproduced exactly
            c = Field(loc, mesh); c.data .= ifelse.(m, -7e20, 2.5)
            box_filter!(b, c, 3; mask = m)
            ai = interior_view(b); mi = view(m, ii, jj, kk)
            @test all(==(2.5), ai[.!mi]) && all(==(0), ai[mi])
        end
        # a face is solid when either neighbouring cell is
        mx = staggered_solid_mask(cell, FaceX())
        @test mx[ii[5], jj[3], kk[1]] && mx[ii[9], jj[3], kk[1]] && !mx[ii[10], jj[3], kk[1]]
        @test !mx[ii[4], jj[3], kk[1]]
        @test count(staggered_solid_mask(cell, FaceZ())) == count(cell) + 16
        @test_throws DimensionMismatch box_filter!(Field(CellCenter(), mesh), Field(CellCenter(), mesh);
                                                   mask = falses(3, 3, 3))
    end

    @testset "Gaussian and spectral cut-off filters" begin
        mesh = RegularGrid(; size = (32, 24, 16), extent = (2.0, 1.5, 1.0), halo = (2, 2, 2))
        # Gaussian: a single Fourier mode is multiplied by exp(−k²Δ²/24)
        for (loc, Δ) in ((CellCenter(), 0.2), (FaceY(), 0.13), (FaceZ(), (0.1, 0.2, 0.3)))
            kv = (2π * 3 / 2.0, 2π * 2 / 1.5, 2π * 4 / 1.0)
            f = Field(loc, mesh); out = Field(loc, mesh)
            set!(f, (x, y, z) -> 0.7 + cos(kv[1] * x + kv[2] * y + kv[3] * z + 0.3))
            gaussian_filter!(out, f, Δ)
            Δv = Δ isa Real ? (Δ, Δ, Δ) : Δ
            G = exp(-sum(kv .^ 2 .* Δv .^ 2) / 24)
            @test maximum(abs, interior_view(out) .- (0.7 .+ G .* (interior_view(f) .- 0.7))) < 1e-10
        end
        # sharp cut-off: modes inside k_c kept, outside removed; idempotent
        f = Field(CellCenter(), mesh); out = Field(CellCenter(), mesh); out2 = Field(CellCenter(), mesh)
        kin = 2π * 2 / 2.0; kout = 2π * 9 / 1.0
        set!(f, (x, y, z) -> sin(kin * x) + 0.5cos(kout * z))
        spectral_cutoff_filter!(out, f, 20.0)
        @test maximum(abs, interior_view(out) .- [sin(kin * x) for x in Aeolus.Grids.xnodes(mesh, CellCenter()),
                                                     _ in 1:24, _ in 1:16]) < 1e-12
        for ff in _random_fields(mesh, Xoshiro(4))
            loc = Aeolus.Fields.location(ff)
            o1 = Field(loc, mesh); o2 = Field(loc, mesh); o3 = Field(loc, mesh)
            spectral_cutoff_filter!(o1, ff, 25.0)
            spectral_cutoff_filter!(o2, o1, 25.0)
            @test maximum(abs, interior_view(o2) .- interior_view(o1)) < 1e-13 * maximum(abs, interior_view(o1))
            @test abs(mean(interior_view(o1)) - mean(interior_view(ff))) < 1e-14
            gaussian_filter!(o3, ff, 0.15)
            @test abs(mean(interior_view(o3)) - mean(interior_view(ff))) < 1e-14
            # the filters remove energy, never add it
            @test sum(abs2, interior_view(o1)) < sum(abs2, interior_view(ff))
            @test sum(abs2, interior_view(o3)) < sum(abs2, interior_view(ff))
        end
    end

    @testset "energy spectrum: Parseval and a single Fourier mode" begin
        for (sz, ext) in (((32, 32, 32), (2π, 2π, 2π)), ((24, 16, 20), (3.0, 2.0, 2.5)))
            mesh = RegularGrid(; size = sz, extent = ext, halo = (2, 2, 2))
            _, u, v, w = _random_fields(mesh, Xoshiro(5))
            interior_view(u) .+= 0.3                         # a mean flow lands in shell 0
            k, E = isotropic_energy_spectrum(u, v, w)
            Δk = k[2] - k[1]
            ctr(f, s) = 0.5 .* (interior_view(f) .+ circshift(interior_view(f), s))
            KE = 0.5 * mean(ctr(u, (-1, 0, 0)) .^ 2 .+ ctr(v, (0, -1, 0)) .^ 2 .+ ctr(w, (0, 0, -1)) .^ 2)
            @test abs(sum(E) * Δk - KE) < 1e-12 * KE
            @test Δk ≈ 2π / maximum(ext)
            @test E[1] * Δk ≈ 0.5 * sum(f -> mean(interior_view(f))^2, (u, v, w)) rtol = 1e-12   # shell 0 = the mean flow
        end
        # u = sin(k0 y): all of ½⟨u²⟩ = 1/4 in shell k0
        mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (2, 2, 2))
        u = Field(FaceX(), mesh); v = Field(FaceY(), mesh); w = Field(FaceZ(), mesh)
        for k0 in (1, 5, 12)
            set!(u, (x, y, z) -> sin(k0 * y))
            k, E = isotropic_energy_spectrum(u, v, w)
            @test E[k0 + 1] ≈ 0.25 rtol = 1e-12
            @test maximum(abs, E[[1:k0; k0+2:end]]) < 1e-12
        end
        # a w = cos(k0 x) on z-faces: the centre interpolation in z is exact
        set!(u, 0.0); set!(w, (x, y, z) -> cos(3x))
        k, E = isotropic_energy_spectrum(u, v, w)
        @test E[4] ≈ 0.25 rtol = 1e-12
    end

    @testset "isotropic random field: solenoidal, spectrum as requested" begin
        mesh = RegularGrid(; size = (32, 32, 32), extent = (2π, 2π, 2π), halo = (3, 3, 3))
        sim = BoussinesqSimulation(mesh)
        # von Kármán–Pao-like target with a peak at k = 3
        Etarget(k) = k^4 / (1 + (k / 3)^2)^(17 / 6)
        isotropic_random_field!(sim, Etarget; rng = Xoshiro(9))
        # discretely divergence-free under the C-grid divergence of the
        # pressure projection, relative to the velocity-gradient scale
        gscale = maximum(abs, diff(interior_view(sim.u); dims = 1)) / mesh.dx
        @test max_divergence(sim) < 1e-12 * gscale
        @test max_divergence(sim) < 1e-12
        # every shell holds exactly E(k) Δk (construction), at all k up to π/Δ
        k, E = isotropic_energy_spectrum(sim.u, sim.v, sim.w)
        rel = abs.(E[2:17] ./ Etarget.(k[2:17]) .- 1)
        @info @sprintf("  random field 32³: max shell error %.1e (k = 1–16), max|∇·u|/|∂u/∂x| %.1e",
                       maximum(rel), max_divergence(sim) / gscale)
        @test maximum(rel[4:end]) < 0.05                    # the issue's criterion, k ≥ 4
        @test maximum(rel) < 1e-10                          # it is exact by construction
        @test maximum(E[18:end]) < 1e-20                    # spherical truncation at π/Δ
        @test abs(E[1]) < 1e-25                             # no mean flow
        # isotropy: the components share the energy (random, ~3 % at 32³)
        ke = [mean(abs2, interior_view(f)) for f in (sim.u, sim.v, sim.w)]
        @test maximum(ke) / minimum(ke) < 1.15
        # halos filled; another seed gives another field with the same spectrum
        @test sim.u.data[1:3, 4:35, 4:35] == sim.u.data[33:35, 4:35, 4:35]
        u1 = copy(sim.u.data)
        isotropic_random_field!(sim, Etarget; rng = Xoshiro(10))
        @test sim.u.data != u1
        @test isotropic_energy_spectrum(sim.u, sim.v, sim.w)[2] ≈ E rtol = 1e-10
        # a non-cubic box: still solenoidal; a kmax below π/Δ is honoured
        m2 = RegularGrid(; size = (32, 16, 24), extent = (2.0, 1.0, 1.5), halo = (2, 2, 2))
        s2 = BoussinesqSimulation(m2)
        isotropic_random_field!(s2, k -> exp(-k / 20); rng = Xoshiro(1), kmax = 30.0)
        k2, E2 = isotropic_energy_spectrum(s2.u, s2.v, s2.w)
        @test max_divergence(s2) < 1e-12
        n30 = floor(Int, 30.0 / (k2[2] - k2[1]))
        @test all(abs.(E2[2:n30+1] ./ exp.(-k2[2:n30+1] ./ 20) .- 1) .< 1e-10)
        @test maximum(E2[n30+2:end]) < 1e-25
    end

    @testset "rescale_to_spectrum!: shells set, phases and solenoidality kept" begin
        mesh = RegularGrid(; size = (24, 24, 24), extent = (1.0, 1.0, 1.0), halo = (2, 2, 2))
        sim = BoussinesqSimulation(mesh; ν = 1e-3)
        isotropic_random_field!(sim, k -> 1e-4 * k^2 * exp(-k / 20); rng = Xoshiro(3))
        for _ in 1:10
            step!(sim, 2e-3)                    # build some phase structure
        end
        # phases kept: asking for 4 × the current spectrum doubles the field
        u0 = copy(interior_view(sim.u))
        kk, E0 = isotropic_energy_spectrum(sim.u, sim.v, sim.w)
        Δk = kk[2] - kk[1]
        rescale_to_spectrum!(sim.u, sim.v, sim.w, k -> 4 * E0[round(Int, k / Δk) + 1]; kmax = 1e3)
        @test maximum(abs, interior_view(sim.u) .- 2 .* u0) < 1e-12 * maximum(abs, u0)
        Enew(k) = 0.01 * k^(-5/3)
        rescale_to_spectrum!(sim.u, sim.v, sim.w, Enew)
        fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
        k, E = isotropic_energy_spectrum(sim.u, sim.v, sim.w)
        @test E[2:13] ≈ Enew.(k[2:13]) rtol = 1e-10
        @test max_divergence(sim) < 1e-12 * maximum(abs, interior_view(sim.u)) / mesh.dx
        a = copy(interior_view(sim.u))
        rescale_to_spectrum!(sim.u, sim.v, sim.w, Enew)       # idempotent
        @test interior_view(sim.u) ≈ a rtol = 1e-12
    end
end
