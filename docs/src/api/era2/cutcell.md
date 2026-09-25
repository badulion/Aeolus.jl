# Cut-cell IBM and ghost-cell walls (Phases 21–22)

Buildings intersect grid cells at arbitrary angles. A cut-cell
treatment carries the *fluid* volume and wetted area of each partial
cell, so the discrete divergence remains conservative on a body-fitted
boundary; ghost-cell reconstruction then imposes the wall condition to
second order rather than by first-order masking.

```@autodocs
Modules = [Aeolus.CutCell]
```
