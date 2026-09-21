# Multiple dispatch on `Location`

Julia's multiple dispatch is used aggressively in Aeolus to select the
correct discrete stencil for a given operator. This is the single most
important design choice in the codebase — it is what lets the same
function name (`∂x`, `divergence`, `advect_centered!`) mean the right
thing for both the collocated and staggered layouts, and what will let
a future spectral-element backend slot in without touching call sites.

## The pattern

Every `Field` carries its staggered [`Location`](@ref
Aeolus.Grids.Location) as a type parameter:

```julia
Field{L, T, A, M}    # L <: Location, T <: AbstractFloat, A <: Array, M <: AbstractMesh
```

So `Field{FaceX}` is a different concrete type from `Field{CellCenter}`
— even though both wrap a 3D `Array{Float64,3}`. When we call

```julia
∂x(dpdx, p)
```

Julia looks at the concrete types of `dpdx` and `p` and picks *the*
method whose signature matches most specifically.

## Example: `∂x`

The public `∂x` in `Operators.jl` is a one-liner:

```julia
∂x(out::Field, f::Field) = _∂x!(out, f, mesh(f))
```

The heavy lifting is done by `_∂x!` in `FiniteVolume.jl`, where we have
*three* methods:

```julia
# 1. Cell-center → cell-center (2nd-order centered)
function _∂x!(out::Field{CellCenter}, f::Field{CellCenter}, g::RegularGrid)
    ...
end

# 2. Cell-center → x-face (2-point gradient — pressure gradient)
function _∂x!(out::Field{FaceX}, f::Field{CellCenter}, g::RegularGrid)
    ...
end

# 3. X-face → cell-center (2-point difference — divergence contribution)
function _∂x!(out::Field{CellCenter}, f::Field{FaceX}, g::RegularGrid)
    ...
end
```

The compiler resolves which method to call at *compile time* based on
`typeof(out)` and `typeof(f)`. The three stencils are physically
different — different neighbours, different index shifts — but at the
call site we write just `∂x(out, f)` for all three.

## Why not one polymorphic function with branches?

We could write

```julia
function ∂x!(out, f)
    lout = location(out); lin = location(f)
    if lin isa CellCenter && lout isa CellCenter
        # ... centered stencil ...
    elseif lin isa CellCenter && lout isa FaceX
        # ... gradient stencil ...
    ...
```

This is worse for three reasons:

  * **Type instability.** The compiler can't know which branch runs
    until runtime; the outer function returns `Union{...}` — slower
    code, heap allocations.
  * **Extensibility fails.** Adding a new stencil (e.g. for a spectral
    element) requires editing the polymorphic function — not just
    adding a new method somewhere.
  * **The stencils live together with the wrong stencils.** A bug in
    one branch endangers all of them.

Multiple dispatch is *precisely* the right tool for this — a family of
routines that share a name but have unrelated implementations selected
by the types of their arguments.

## The `Operators` façade + `FiniteVolume` kernels split

The public API in `Operators.jl` looks like:

```julia
∂x(out, f) = _∂x!(out, f, mesh(f))
```

so it takes 2 arguments; the `FiniteVolume` kernel takes 3 arguments —
the extra one is the mesh. This is a *second axis* of dispatch: the
mesh type. When we add a spectral-element backend the new methods will
be

```julia
_∂x!(out::Field{...}, f::Field{...}, g::SpectralElementMesh) = ...
```

with the same `Location` combinations but a different mesh type. The
compiler picks the right implementation from *the combination of all
type parameters* — mesh, input location, output location.

Concretely, adding a spectral-element backend (Phase 5) means:

1. Define `SpectralElementMesh <: AbstractMesh`.
2. Implement `totalsize`, `interior`, `xnodes`, `ynodes`, `znodes` for
   it.
3. Add kernel methods `_∂x!(out, f, mesh::SpectralElementMesh)` etc.
4. Add an SEM-native `AbstractPressureSolver` implementation.

*None* of `Operators.jl`, `Simulations.jl`, or user code changes. This is
the property of the design that we most want to preserve as the code
grows.

## Dispatch table for the C-grid FV backend

The complete set of first-derivative stencils on `RegularGrid`:

| Method                                                  | Stencil                                                        | Physical role         |
| :------------------------------------------------------ | :------------------------------------------------------------- | :-------------------- |
| `_∂x!(CellCenter, CellCenter, RegularGrid)`             | `(fᵢ₊₁ - fᵢ₋₁) / (2Δx)`                                       | diagnostic            |
| `_∂x!(FaceX, CellCenter, RegularGrid)`                  | `(fᵢ - fᵢ₋₁) / Δx`                                             | pressure gradient     |
| `_∂x!(CellCenter, FaceX, RegularGrid)`                  | `(fᵢ₊₁ - fᵢ) / Δx`                                             | divergence            |
| `_∂y!(...)`, `_∂z!(...)`                                | identical up to permutation                                    |                       |
| `_laplacian!(L, L, RegularGrid) where {L}`              | 7-point centered                                               | diffusion / Poisson   |
| `_divergence!(CellCenter, FaceX, FaceY, FaceZ, ...)`    | staggered, `Δu/Δx + Δv/Δy + Δw/Δz`                            | pressure-projection RHS |
| `_divergence!(CellCenter, CellCenter³, ...)`            | 2nd-order centered                                             | diagnostic            |
| `_advect_centered!(CellCenter, FaceX, FaceY, FaceZ, CellCenter, ...)` | flux-form staggered                              | θ advection           |
| `_advect_centered!(CellCenter⁵, ...)`                   | advection-form                                                 | passive tracer        |
| `_momentum_advect!(FaceX, FaceY, FaceZ, FaceX, FaceY, FaceZ, ...)` | flux-form staggered momentum                        | Boussinesq momentum   |

## Take-away

Every place you see `_∂x!(out, f, mesh)` in Aeolus is a promise:
whichever concrete stencil belongs to that combination of locations
and mesh type, Julia will pick it, and the compiler will inline it as
if we had written a specialized function by hand. This is why the code
reads at the physical-equation level (`add_buoyancy!`, `divergence`)
while executing at the tight-inner-loop level.
