# ComputationalThinkingforClimate.jl
MIT 6.C25[J] Real World Computation with Julia: Climate Science, Fall 2026

## Notebooks

Work through the notebooks in this order. Each one has a static HTML render that you can read in the browser without installing anything.

1. [`radiative_balance.jl`](radiative_balance.jl) ([HTML](https://numericalearth.github.io/ComputationalThinkingforClimate.jl/assets/radiative_balance.html)): build an energy balance model of Earth's surface temperature, one process at a time (solar heating, reflection, radiative cooling, an atmosphere, CO₂ and water vapor feedback), then extend it to temperature as a function of latitude.
2. [`numerical_fluid_dynamics.jl`](numerical_fluid_dynamics.jl) ([HTML](https://numericalearth.github.io/ComputationalThinkingforClimate.jl/assets/numerical_fluid_dynamics.html)): from the Navier–Stokes equations to a finite-volume solver for linear advection and the Burgers equation, with centered and upwind fluxes, time steppers and the CFL condition.
3. [`oceananigans_near_global_demo.jl`](oceananigans_near_global_demo.jl) ([HTML](https://numericalearth.github.io/ComputationalThinkingforClimate.jl/assets/oceananigans_near_global_demo.html)): use [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) to simulate an idealized global ocean on a CPU or GPU, and watch baroclinic instability form jets and eddies that carry heat toward the poles.

To update the HTML after you change a notebook, run `julia --project scripts/render.jl [notebook.jl ...]`.

## Opening a notebook

Install [Julia](https://julialang.org/downloads/), then clone this repository and start Julia in it:

```
git clone https://github.com/xkykai/ComputationalThinkingforClimate.jl
cd ComputationalThinkingforClimate.jl
julia --project
```

In Julia, run:

```julia
using Pkg; Pkg.instantiate()  # first time only
using Pluto; Pluto.run()
```

Pluto opens in your browser. Enter the path of a notebook (for example `radiative_balance.jl`) and click **Open**. Each notebook installs its own packages the first time it runs, which can take several minutes.
