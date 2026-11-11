# Copies of the kernels as they were at f18feb68 (the baseline of #554, before
# E1 16ad00c9 and E2 cf704ba3), evaluated into Aeolus.PrognosticSGS under new
# names, for interleaved timing (tke_update_strain_bench.jl). Checked against
# bench/all (#566): the helpers they call (_dual_raw, _fold_src, _images,
# _DualBox, _AllIn, _DualUniform, plane_map!, _surface!, ...) keep their
# signatures; the separate-loop kernels _update! and _dual_strain! still
# exist next to the TKE pass of #561.
@eval Aeolus.PrognosticSGS begin
struct _DualLevelsOld{T, V}
    idx2::T; idy2::T
    dzc::V; dzf::V
end
@inline _hc(m::_DualLevelsOld, x, k) = @inbounds x * m.dzc[k]
@inline _hf(m::_DualLevelsOld, x, k) = @inbounds x * m.dzf[k]
@inline _izt(m::_DualLevelsOld, k) = @inbounds 1 / m.dzf[k+1]
@inline _izb(m::_DualLevelsOld, k) = @inbounds 1 / m.dzf[k]
@inline _iwt(m::_DualLevelsOld, k) = @inbounds 1 / m.dzc[k]
@inline _iwb(m::_DualLevelsOld, k) = @inbounds 1 / m.dzc[k-1]
@inline _per_volume(m::_DualLevelsOld, x, k) = @inbounds x * (1 / m.dzc[k])
_dual_metrics_old(g::RegularGrid) = _DualUniform(1 / g.dx^2, 1 / g.dy^2, 1 / g.dz^2)
_dual_metrics_old(g::StretchedGrid) = _DualLevelsOld(1 / g.dx^2, 1 / g.dy^2, g.dzc, g.dzf)

@inline function _dual_strain_body_old!(I, J, K, d, u, v, w, m, bx, Hx, Nx, lox, hix, Hy, Ny, loy, hiy,
                                    Hz, Nz, loz, hiz)
    T = eltype(d)
    acc = zero(T)
    if (Hx + 1 < I < Hx + Nx) & (Hy + 1 < J < Hy + Ny) & (Hz + 1 < K < Hz + Nz)
        acc = _dual_raw(u, v, w, m, _AllIn(), I, J, K)
    else
        for nz in 1:3
            sz = _fold_src(K, Hz, Nz, loz, hiz, nz)
            sz == 0 && continue
            for ny in 1:3
                sy = _fold_src(J, Hy, Ny, loy, hiy, ny)
                sy == 0 && continue
                for nx in 1:3
                    sx = _fold_src(I, Hx, Nx, lox, hix, nx)
                    sx == 0 && continue
                    acc += _dual_raw(u, v, w, m, bx, sx, sy, sz)
                end
            end
        end
    end
    @inbounds d[I, J, K] = _per_volume(m, acc, K)
    return nothing
end

function _dual_strain_old!(D, u, v, w, bcs, g)
    d = D.data
    fill!(d, zero(eltype(d)))
    ii, jj, kk = interior(g)
    bx = _DualBox(first(ii), last(ii), first(jj), last(jj), first(kk), last(kk))
    lox, hix = _images(bcs.west, g.Hx, g.Nx)
    loy, hiy = _images(bcs.south, g.Hy, g.Ny)
    loz, hiz = _images(bcs.bottom, g.Hz, g.Nz)
    launch!(Backends.backend(d), _dual_strain_body_old!, (ii, jj, kk), d, u.data, v.data, w.data,
            _dual_metrics_old(g), bx, g.Hx, g.Nx, lox, hix, g.Hy, g.Ny, loy, hiy, g.Hz, g.Nz, loz, hiz)
    return D
end

function _update_old!(c::TKEClosure{T}, Δt::T, pbs) where {T}
    g = mesh(c.e)
    V0 = T(_tke_volume0(g))
    par = (Hx = g.Hx, Hy = g.Hy, Hz = g.Hz, Nz = g.Nz, V0 = V0, Δt = Δt, pb1 = T(pbs[1]),
           pb2 = T(pbs[2]), ck = c.c_k, cl = c.c_l, cε0 = c.c_ε0, cε1 = c.c_ε1,
           sfac = c.surface_factor, exact = c.dissipation === :exact)
    parts = Vector{NTuple{8,T}}(undef, g.Nz)
    Backends.plane_map!(parts, Backends.backend(c.e.data), _tke_update_plane_old, interior(g),
               c.e.data, c.ν.data, c.N2.data, c.D.data, c.tend.data, c.Pw.data, c.Δk, c.vrel,
               c.l_wall, c.first_level, c.mask, par)
    sPs, sPb, sT, sPw, sε, sB, nclip, eclip = ntuple(n -> ordered_sum(map(p -> p[n], parts)), Val(8))
    st = c.stats
    st[1] += nclip; st[2] += eclip
    st[4] = sPs * V0; st[5] = sPb * V0; st[6] = sε * V0; st[7] = sB * V0; st[8] = sT * V0
    st[9] += 1
    st[10] = sPw * V0
    c.surface === nothing || _surface!(c, Δt)
    return c
end
function _tke_update_plane_old(K, ir, jr, ed, nd, n2, dd, td, pw, Δk, vrel, l_wall, first_level, mask, p)
    T = eltype(ed)
    acc = ntuple(_ -> zero(T), Val(8))
    for J in jr, I in ir
        acc = map(+, acc, _tke_update_body_old(I, J, K, ed, nd, n2, dd, td, pw, Δk, vrel, l_wall,
                                           first_level, mask, p))
    end
    return acc
end
@inline function _tls_old(e, N2, Δ; l_wall = oftype(float(Δ), Inf), c_l = 0.76)
    sq = sqrt(max(e, zero(e)))
    lmax = min(Δ, l_wall)
    if N2 > 0
        N = sqrt(N2)
        l = min(lmax, c_l * sq / N)
        s = max(sq / lmax, N / c_l)
    else
        l = lmax
        s = sq / lmax
    end
    return l, s
end


@inline function _tke_update_body_old(I, J, K, ed, nd, n2, dd, td, pw, Δk, vrel, l_wall, first_level,
                                  mask, p)
    T = eltype(ed)
    z = zero(T)
    i, j, k = I - p.Hx, J - p.Hy, K - p.Hz
    @inbounds begin
        if _solid(mask, I, J, K)
            ed[I, J, K] = z
            return (z, z, z, z, z, z, z, z)
        end
        Δt = p.Δt
        Δ = Δk[k]; vr = vrel[k]; V = p.V0 * vr
        e = ed[I, J, K]
        N2 = n2[I, J, K]
        l, s = _tls_old(e, N2, Δ; l_wall = _lwall(l_wall, i, j, k, T), c_l = p.cl)
        fac = _first(first_level, i, j, k) ? p.sfac : one(T)
        cε = tke_dissipation_coefficient(l, Δ; c_ε0 = p.cε0, c_ε1 = p.cε1, factor = fac)
        Ps = nd[I, J, K] * dd[I, J, K] + pw[I, J, K]
        κv = (1 + 2l / Δ) * nd[I, J, K]
        Psf = (k == 1 ? p.pb1 : z) + (k == p.Nz ? p.pb2 : z)
        Pb = -κv * N2 + Psf
        # sources: P_s and the positive buoyancy production; sinks, linear in
        # e: dissipation c_ε s e, the buoyant destruction κ_t N² = r_N e
        # (r_N = (1+2l/Δ) c_k N²/s, finite as e → 0) and a stable surface flux
        src = Ps + (N2 < 0 ? -κv * N2 : z) + max(Psf, z)
        rb = (N2 > 0 && s > 0 ? (1 + 2l / Δ) * p.ck * N2 / s : z) +
             (Psf < 0 ? -Psf / max(e, floatmin(T)) : z)
        tr = td[I, J, K]
        estar = e + Δt * (tr + src)
        nclip = z; eclip = z
        if estar < 0
            nclip = one(T); eclip = -(estar * V)
            estar = z
        end
        rε = cε * s
        if p.exact && l > 0 && estar > 0
            e1 = inv(inv(sqrt(estar)) + cε * Δt / (2l))^2
            enew = e1 / (1 + Δt * rb)
            dε = (estar - e1) / Δt
            dB = (e1 - enew) / Δt
        else
            R = rε + rb
            enew = estar / (1 + Δt * R)
            rem = (estar - enew) / Δt
            dε = R > 0 ? rem * (rε / R) : z
            dB = R > 0 ? rem * (rb / R) : z
        end
        ed[I, J, K] = enew
        return (Ps * vr, Pb * vr, tr * vr, pw[I, J, K] * vr, dε * vr, dB * vr, nclip, eclip)
    end
end

end
