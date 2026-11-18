# Rendering the notebooks

`render.jl` exports the notebooks to static HTML in `assets/`. Rerun it after you change a notebook, then commit the updated HTML with the notebook.

From the repository root:

```
julia --project -e 'using Pkg; Pkg.instantiate()'    # first time only
julia --project scripts/render.jl                    # all notebooks
julia --project scripts/render.jl radiative_balance.jl numerical_fluid_dynamics.jl
```

Each notebook runs from top to bottom with its controls at their defaults. If a cell errors, the script stops and leaves the existing HTML unchanged.

For `oceananigans_near_global_demo.jl` the script also turns on "Run the simulation", so the page includes the results. This runs the full 120-day simulation: a few minutes on a GPU, longer on a CPU. The page reports the hardware it ran on.
