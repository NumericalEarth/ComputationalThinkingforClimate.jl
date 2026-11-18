# ComputationalThinkingforClimate.jl
MIT 6.C25[J] Real World Computation with Julia: Climate Science, Fall 2026

## Notebooks

- [`radiative_balance.jl`](radiative_balance.jl) ([HTML](https://raw.githack.com/NumericalEarth/ComputationalThinkingforClimate.jl/main/assets/radiative_balance.html)): An introduction to Earth's radiative balance model
- [`numerical_fluid_dynamics.jl`](numerical_fluid_dynamics.jl) ([HTML](https://raw.githack.com/NumericalEarth/ComputationalThinkingforClimate.jl/main/assets/numerical_fluid_dynamics.html)): An introduction to basics of finite-volume numerical fluid dynamics, where we build numerical solvers for the linear advection and Burgers equation from scratch
- [`oceananigans_near_global_demo.jl`](oceananigans_near_global_demo.jl) ([HTML](https://raw.githack.com/NumericalEarth/ComputationalThinkingforClimate.jl/main/assets/oceananigans_near_global_demo.html)): A showcase of [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) by solving an idealized baroclinic instability problem on a sphere with different grids (lat-lon, rotated lat-lon, tripolar)

To update the HTML after you change a notebook, run `julia --project scripts/render.jl [notebook.jl ...]`.

## Opening a notebook

Install [Julia](https://julialang.org/downloads/), then clone this repository and start Julia in it:

```
git clone https://github.com/NumericalEarth/ComputationalThinkingforClimate.jl
cd ComputationalThinkingforClimate.jl
julia --project
```

In Julia, run:

```julia
using Pkg; Pkg.instantiate()  # first time only
using Pluto; Pluto.run()
```

Pluto opens in your browser. Enter the path of a notebook (for example `radiative_balance.jl`) and click **Open**.
