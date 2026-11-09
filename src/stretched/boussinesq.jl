# =========================================================================
# #151 (N-grid) — the Boussinesq core on a StretchedGrid.
# =========================================================================
#
# Included into `module StretchedGrids`. `BoussinesqSimulation(g)` runs on a
# vertically stretched mesh with the one set of finite-volume kernels of
# `Operators.FiniteVolume` (momentum and scalar advection, the Laplacian of
# a z-face field, dry and moist buoyancy, Smagorinsky, eddy viscosity and
# eddy diffusion): they read the grid's per-level vertical metrics
# (`grid_metrics(g)`, `FiniteVolume.LevelZ`) and evaluate exactly the
# stretched stencils this file used to carry as copies of the RegularGrid
# kernels (folded in KA-d, #299; the control volumes, the volume-flux
# transport and the energy argument are documented at the kernels).
#
# What stays here is what has no RegularGrid twin in FiniteVolume:
#
#   * the vertical metrics of the core      `Simulations._dzc/_dzf/_dz_min`
#   * the Δz of N² (the TKE closure)        `Operators._n2_dz`
#
# Control volumes of the staggered grid in z (storage index k):
#
#   u, v, θ, p  at zc[k], control volume [zf[k],  zf[k+1]], height dzc[k]
#   w           at zf[k], control volume [zc[k-1], zc[k]],  height dzf[k]

import ..Simulations: _dzc, _dzf, _dz_min, _check_boussinesq_mesh

# --- the vertical metrics of the core -------------------------------------

_dzc(g::StretchedGrid, k) = @inbounds g.dzc[k]
_dzf(g::StretchedGrid, k) = @inbounds g.dzf[k]
_dz_min(g::StretchedGrid) = minimum(@view g.dzc[g.Hz+1:g.Hz+g.Nz])
_check_boussinesq_mesh(::StretchedGrid) = nothing

# --- N² of the stratification (the TKE closure's buoyancy term, #259) ------

import ..Operators: _n2_dz

"""
    _n2_dz(m::StretchedGrid)

The Δz of ``N^2`` on a StretchedGrid: the cell thickness `dzc` of every
storage level (`Operators._n2_body!` reads it per level).

``N^2_k = (g/θ_0)(θ_{v,k+1/2} - θ_{v,k-1/2})/Δz_c[k]``: the face values of
the uniform form (the mean of the two adjacent centres) differenced over
the cell thickness. Then ``N^2_k = (g/θ_0)(θ_{k+1} - θ_{k-1})/(2Δz_c[k])``
is exactly the per-cell share of the stretched κ_t operator's
potential-energy change (`FiniteVolume._add_eddy_diffusion_scalar!` with the
stretched metrics: face
gradients over `dzf`, divergence over `dzc`), so ``Σ(−κ_t N^2)V`` is the
buoyancy flux of the SGS heat flux, as on the uniform grid. For a linear
θ it is ``Γ(Δz_c[k-1]/2 + Δz_c[k] + Δz_c[k+1]/2)/(2Δz_c[k])``: Γ on the
uniform levels, Γ(1 + 1.5e-3) on a geometric part of ratio 1.08, and
first order where the ratio jumps (+1.7 % in the last 25 m cell under the
first stretched one of the nesting parent, #259). With condensate
(`Operators._Cloudy`, #239) the per-cell stencil is the uniform one's
(condensate loading, the saturated N² in cells with ``q_c > 0``), over
the same ``Δz_c[k]``.
"""
_n2_dz(m::StretchedGrid) = m.dzc
