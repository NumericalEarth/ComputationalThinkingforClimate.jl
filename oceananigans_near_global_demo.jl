### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 27e48d3d-859a-48bc-8324-00e73496a7ff
begin
    using Oceananigans
    using Oceananigans.Units
    using Oceananigans.Architectures: synchronize
    using Oceananigans.Operators: ζ₃ᶠᶠᶜ
    using Oceananigans.ImmersedBoundaries: mask_immersed_field!
    using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
    using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
    import ConservativeRegridding
    using CairoMakie
    using PlutoUI
    using Printf
    using Base64
end

# ╔═╡ 4621f04d-a10d-4a6e-9428-789a580cae3d
begin
    arch = if Sys.isapple() && Sys.ARCH === :aarch64
        using Metal
        Metal.functional() ? GPU(Metal.MetalBackend()) : CPU()
    elseif !isnothing(Sys.which("nvidia-smi"))
        using CUDA
        CUDA.functional() ? GPU() : CPU()
    elseif !isnothing(Sys.which("rocminfo"))
        using AMDGPU
        AMDGPU.functional() ? GPU(AMDGPU.ROCBackend()) : CPU()
    else
        CPU()
    end

    Oceananigans.defaults.FloatType = Float32
    arch
end

# ╔═╡ ae9a3c8c-e3d7-4f6a-af26-91efa98b9db6
md"""
# Oceananigans.jl: fast and friendly ocean-flavored fluid dynamics

Previously, we built a solver for one equation in one dimension. Now, let's try using [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) to run a model of the global ocean in 3D and on a rotating sphere! Oceananigans.jl is a finite-volume fluid dynamics solver, written entirely in Julia, that runs on both CPUs and GPUs. It is developed as part of the [Climate Modeling Alliance (CliMA)](https://clima.caltech.edu/), a multi-institution effort to build a next-generation climate model in Julia. Oceananigans.jl has been designed from the ground up for GPUs, making it currently the [fastest ocean model on Earth](https://clima.caltech.edu/2024/04/15/the-worlds-fastest-ocean-model/) ([Silvestri et al., 2023](https://arxiv.org/abs/2309.06662)).

Oceananigans.jl is also the foundation of [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl), a high-resolution atmosphere model for simulating clouds and storms. Through [NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl), Oceananigans.jl can be coupled to atmosphere, sea-ice and land models such as Breeze.jl and [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl), and through [ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl) to [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl), CliMA's global atmosphere model. Many more packages build on or interact with Oceananigans.jl. Some examples include [OceanBioME.jl](https://github.com/OceanBioME/OceanBioME.jl) for ocean biogeochemistry, [ClimaSeaIce.jl](https://github.com/CliMA/ClimaSeaIce.jl) for sea ice, and [Terrarium.jl](https://github.com/NumericalEarth/Terrarium.jl) for land.

In this notebook, we will set up an idealized simulation that aims to illustrate one of the most important fluid dynamical processes in the ocean, **baroclinic instability**, which is responsible for transporting heat from the equator to higher latitudes ([Marshall & Plumb, 2008](https://www.sciencedirect.com/book/9780125586917/atmosphere-ocean-and-climate-dynamics); [Vallis, 2017](https://doi.org/10.1017/9781107588431)). In this simulation, the planet has no land and has a 3-km-deep ocean that is warm near the equator and cold near the poles, with a sharp change of temperature, a **front**, at 45° latitude in each hemisphere. Within weeks, the fronts break into waves that roll up into eddies. A fluid on a rotating planet removes a temperature contrast in this way, which is the mechanism for the formation of storms in the atmosphere and the ocean, as in the [video](https://youtu.be/Ei8x3y-Ddq0) in `numerical_fluid_dynamics.jl`.
"""

# ╔═╡ 3a691689-54cc-4bd9-a4e0-2159e8a2dd53
TableOfContents()

# ╔═╡ 2ce0dfa9-f800-4eb0-b809-49826e7eef76
md"""
## Setting up the software

First, we load the required packages:
"""

# ╔═╡ 53ee3e80-44ce-46d8-b680-f07e98b42055
md"""
If you have an Apple Silicon Mac, an NVIDIA GPU or an AMD GPU with working drivers, the notebook will use it. Otherwise it runs on the CPU. In this notebook, we will do the calculations in single precision (`Float32`) to support all hardware, since Apple Metal GPUs don't support double precision natively yet. For publication-worthy runs, using double precision is highly recommended.

Owing to the design of Oceananigans.jl and Julia, apart from `arch`, nothing else in the notebook changes between the CPU and the GPU. Internally, Oceananigans.jl uses [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), which allows developers to "write once, run everywhere".

Given the machine this notebook is running on, we will use **$(arch isa GPU ? Dict(:MetalBackend => "Apple GPU", :CUDABackend => "NVIDIA GPU", :ROCBackend => "AMD GPU")[nameof(typeof(arch.device))] : "CPU")** for the subsequent simulation.
"""

# ╔═╡ 7dcb171d-c8ef-4f57-aa51-51994eab1868
md"""
## Choosing a grid

There are many types of grids you can choose from in Oceananigans.jl, such as `RectilinearGrid` for a box or a channel, `LatitudeLongitudeGrid`, `RotatedLatitudeLongitudeGrid`, `TripolarGrid`, `ConformalCubedSphereGrid` and `YinYangGrid` for a sphere, and `ImmersedBoundaryGrid`, which adds a sea floor, coastlines or islands to any of these. Since we are simulating fluid dynamics on a sphere, in this notebook we showcase three options appropriate for the needs of this simulation.

1. A **latitude-longitude grid** uses lines of constant latitude and longitude, like a map. Its boxes get narrower toward the poles, and at a pole their width is zero, so the equations break down there. Our grid stops at 85°N and 85°S and has walls there.
2. A **rotated latitude-longitude grid** is the same grid with its poles moved, here to 70°E, 55°N and the opposite point in the Southern Hemisphere.
3. A **tripolar grid** keeps a latitude-longitude grid in the south, with a wall near 80°S, and puts two poles in the north, at 70°E and 250°E. This grid is well suited to simulations of Earth because its northern poles can be "hidden" on land (in Siberia and Canada). Our idealized planet has no land, so we add two small islands to cover them. Many ocean models used for climate projections run on a tripolar grid.
"""

# ╔═╡ 8f261ecd-4258-4224-9157-a8b965cccdfe
function island_bottom(λ, φ)
    Δλ = mod(λ - 70 + 90, 180) - 90   # degrees of longitude to the nearer northern pole of the tripolar grid
    on_island = (Δλ / 8)^2 + ((φ - 55) / 4)^2 < 2
    return on_island ? 100 : -3000
end

# ╔═╡ d759e126-1023-4519-8e91-62caf37c608f
function make_grid(arch, grid_type; resolution, FT = Oceananigans.defaults.FloatType)
    size = (360 ÷ resolution, 170 ÷ resolution, 10)
    halo = (7, 7, 7)
    z = (-3000, 0)

    grid = if grid_type == :latlon
        LatitudeLongitudeGrid(arch, FT; size, halo, z, longitude = (0, 360), latitude = (-85, 85))
    elseif grid_type == :rotated
        RotatedLatitudeLongitudeGrid(arch, FT; size, halo, z, longitude = (0, 360), latitude = (-85, 85), north_pole = (70, 55))
    else
        ImmersedBoundaryGrid(TripolarGrid(arch, FT; size, halo, z), GridFittedBottom(island_bottom))
    end

    synchronize(arch)
    return grid
end

# ╔═╡ 69ecde40-72d9-4d04-b70d-91c3efb14ce7
md"""
`make_grid` builds each grid with boxes `resolution` degrees wide and 10 levels over a depth of 3 km. The `halo` is a layer of extra boxes around the edge of the grid, so that the boxes at the edge have neighbors. `island_bottom` gives the height of the sea floor: +100 m, dry land, near the two northern poles of the tripolar grid, and −3000 m everywhere else.

Below we show a schematic of the three different grids at 5°, seen from above 60°N. Look for the hole at the pole of the latitude-longitude grid, the shifted hole of the rotated grid, and the two northern poles of the tripolar grid. Between those two poles the tripolar grid folds over the North Pole: each grid line on one side of the fold continues into its mirror image on the other side. The red lines show where they join.
"""

# ╔═╡ bba74ad1-42f3-4f30-aa0a-36448f169dc0
cpu_grids = Dict(t => make_grid(CPU(), t; resolution = 5) for t in (:latlon, :rotated, :tripolar));

# ╔═╡ 0add26c5-b5ba-4282-93fc-da7d61de70d0
md"""
Pick one grid of your choice! We will run the simulation with this grid.

**Grid:** $(@bind grid_type Select([:latlon => "Latitude-longitude", :rotated => "Rotated latitude-longitude", :tripolar => "Tripolar"]))
"""

# ╔═╡ a80dc238-4349-40e8-a4a0-9a5e76aaf61a
md"""
## Setting up the simulation

We can set up the complete `model` and `simulation` in a single cell:
"""

# ╔═╡ 80e5c2e7-6db2-4d80-bef1-c7632da09ed9
begin
    grid = make_grid(arch, grid_type; resolution = 2)

    model = HydrostaticFreeSurfaceModel(grid;
                                        coriolis = HydrostaticSphericalCoriolis(),
                                        buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState()),
                                        tracers = (:T, :S),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps = 60),
                                        momentum_advection = WENOVectorInvariant(order = 9),
                                        tracer_advection = WENO(order = 7))

    T₀(φ) = 15 * (1 - tanh((abs(φ) - 45) / 8))
    Tᵢ(λ, φ, z) = T₀(φ) + rand()
    Sᵢ(λ, φ, z) = 28 - 5e-3 * z + rand()
    set!(model, T = Tᵢ, S = Sᵢ)

    simulation = Simulation(model; Δt = 10minutes, stop_time = 120days)
    synchronize(arch)   # wait for the GPU to finish
end

# ╔═╡ 5c15ae8f-bcfb-4f5d-bf28-b376328b7f6e
md"""
To run the simulation, we just call
```julia
run!(simulation)
```

Before that, let's look at some of the arguments in the code snippet:

### Grid

With `resolution = 2`, the grid has 180 boxes in longitude, 85 in latitude and 10 in depth. Each box is about 220 km wide at the equator and 300 m thick.

### Fluid equations

`HydrostaticFreeSurfaceModel` solves a variant of the Navier–Stokes equations with approximations that suit a model whose grid boxes are kilometers wide in the parameter regime of Earth's ocean. Since Earth's ocean is much wider than it is deep, at the scales of our grid resolution vertical accelerations are small and the vertical part of ``F = ma`` reduces to a balance where the pressure at any depth equals the weight per unit area of the water above it. This is called **hydrostatic balance**.

Vertical accelerations matter only for motions that are about as tall as they are wide, such as small-scale turbulence. To resolve these motions for Earth's ocean, a model must solve the full rotating Boussinesq equations (`NonhydrostaticModel` in Oceananigans.jl) on a grid with boxes about 1 m wide. The ocean holds about 1.3 billion km³ of water, so a global model at 1 m resolution needs about 10¹⁸ boxes, far more than any computer can store today! Global models therefore use the hydrostatic equations and represent the smaller motions with turbulence models.

| argument | choice | explanation |
|:---------|:-------|:--------|
| `coriolis` | `HydrostaticSphericalCoriolis()` | the Coriolis force on a rotating sphere |
| `buoyancy` | `SeawaterBuoyancy` with TEOS-10 | the realistic density of Earth's seawater computed from its temperature, salinity and depth |
| `tracers` | `(:T, :S)` | tracers carried by the flow, here temperature and salinity (more tracers can be added) |
| `free_surface` | `SplitExplicitFreeSurface` | a sea surface that moves up and down |
| `momentum_advection`, `tracer_advection` | `WENO` (weighted essentially non-oscillatory) schemes | how to compute the advection terms |

### Initial condition

To approximate the state of Earth's ocean in an idealized form, we initialize the temperature to be 30 °C in the tropics and 0 °C near the poles, the same at every depth. Salinity increases with depth, so the water at the bottom is denser than the water at the top. To kickstart the turbulence, we add random noise of up to 1 °C and 1 g/kg. The model starts at rest.

### Simulation time

The time step is 10 minutes and the run lasts 120 days.
"""

# ╔═╡ b0e06ea4-3962-44c3-b250-1067d559454b
let
    φ = -85:0.5:85
    fig = Figure(size = (700, 300))
    ax = Axis(fig[1, 1], xlabel = "latitude (°)", ylabel = "temperature (°C)", xticks = -90:30:90, titlesize = 16, title = "Initial temperature (without the noise)")
    lines!(ax, φ, T₀.(φ), color = :gray50, linewidth = 2)
    fig
end

# ╔═╡ 05401c11-43ad-4c09-8a1e-7ea95d128c09
md"""
## Saving simulation outputs

A model's variables come in two kinds. **Prognostic** variables, such as the velocities, the temperature ``T`` and the salinity ``S``, are the ones the model steps forward in time. **Diagnostic** variables, such as the kinetic energy or the vorticity, are computed from the prognostic ones when we need them.

Here we save the (prognostic) temperature and two (diagnostic) quantities. The **speed** ``\sqrt{u^2 + v^2}`` shows the currents. The **vorticity** ``\zeta = \partial v / \partial x - \partial u / \partial y`` is the vertical component of the curl of the velocity. It measures how fast the water spins, so it shows eddies in the ocean and storms in the atmosphere clearly.
"""

# ╔═╡ 84c481fd-6ad9-417b-877a-befc212cd4ae
begin
    u, v, w = model.velocities
    T = model.tracers.T
    speed = @at (Center, Center, Center) sqrt(u^2 + v^2)
    ζ = @at (Center, Center, Center) KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)
    synchronize(arch)
end;

# ╔═╡ ab84cfbc-3718-4847-b220-c8869caaa98d
md"""
### Lazy operations

In Oceananigans.jl, most diagnostics are **lazy**, meaning that you write a recipe for the set of operations acting on the prognostic variables, and the model runs it only when a result is needed. The speed we defined above is an example:
```julia
speed = @at (Center, Center, Center) sqrt(u^2 + v^2)
```
This line doesn't compute anything. It builds an object that remembers the steps: interpolate `u` and `v` to the box centers, square them, add them and take the square root. The same holds for derivatives such as `∂x(u)` as well as averages and integrals such as `Average(T, dims = 1)`. For the vorticity, we write
```julia
ζ = @at (Center, Center, Center) KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)
```
which is also a lazy operation. It only defines a recipe that computes the vorticity at the box corners from `u` and `v`, then interpolates it to the box centers. To get numbers out of a recipe, we need to wrap it in a `Field` and call `compute!`, which the output writer does automatically.

Lazy operations are useful because:

- They save memory. We never store the speed or the vorticity for the entire ocean, since we only want to save the surface layer. A GPU has much less memory than a computer's main memory, so memory efficiency matters even more for GPU code.
- They are fast. The speed is computed in one loop over the grid, on the CPU or the GPU, with no temporary arrays for `u^2`, `v^2` or their sum.
- They are always up to date. A recipe reads `u` and `v` at the moment it runs, so you define it once before the run and every output uses the current velocities.
"""

# ╔═╡ 81097ca3-5023-4387-8a24-3af2c7c643fc
md"""
We can also add a callback that prints the progress of the simulation at an interval of our choice, set with `TimeInterval` or `IterationInterval`.
"""

# ╔═╡ c4e14bdc-d063-4150-9a62-e48ecfb6787b
function progress(sim)
    u = sim.model.velocities.u
    @info @sprintf("%s: max|u| = %.2f m/s, wall time %s", prettytime(sim), maximum(abs, u), prettytime(sim.run_wall_time))
    return nothing
end

# ╔═╡ 43f051bb-1dbe-4f3d-b11f-94cde9e01815
output_dir = mkpath(joinpath(@__DIR__, "demo_output"))

# ╔═╡ 40f45db1-996f-4eb0-ab50-db6fa1c90fd0
run_name = "$(grid_type)_$(360 ÷ size(grid, 1))deg"

# ╔═╡ 7611f1fd-8680-423c-b52a-001b24d0601c
output_file = let
    add_callback!(simulation, progress, TimeInterval(10days))
    simulation.output_writers[:surface] = JLD2Writer(model, (; T, speed, ζ); dir = output_dir, filename = "surface_$run_name",
                                                     indices = (:, :, size(grid, 3)), schedule = TimeInterval(1day),
                                                     overwrite_files = true)
    simulation.output_writers[:surface].filepath
end

# ╔═╡ cbbb6280-4384-46d9-8e1f-1d7d3e5a26d1
md"""
**Run the simulation:** $(@bind run_simulation CheckBox(default = false))

With a GPU the run takes 1 to 3 minutes, and on a CPU several minutes more. Progress messages will appear under the cell as the simulation is running. While the box is checked, a change to the model starts a new run. For rapid iteration, uncheck the box before you make any edits. Each run saves its output under `run_name`, which records the grid and the resolution. If an earlier run with the same `run_name` saved its output, the plots in [Results](#results) show it without a new run.
"""

# ╔═╡ 076f663d-281c-4c97-9f1d-f63ce011337f
simulation_done = if run_simulation
    run!(simulation)
    true
else
    isfile(output_file)
end

# ╔═╡ 75ce0ac0-6a5c-4e8d-a1ea-84ef5598a2ef
md"""
## Results $(html"<a id='results'></a>")
"""

# ╔═╡ b499be47-5368-4c3b-9e19-3c9a8d786ac0
results = if simulation_done
    (T = FieldTimeSeries(output_file, "T"),
     speed = FieldTimeSeries(output_file, "speed"),
     ζ = FieldTimeSeries(output_file, "ζ"))
end;

# ╔═╡ f34fe12c-88e9-416a-9b7f-89c27f705696
md"""
### The planet

The temperature (left), the speed (middle) and the vorticity (right) at the sea surface. Red (positive) vorticity means the water spins counterclockwise, and blue (negative) means clockwise. On the tripolar grid, the white patches are the islands.

**Day:** $(@bind planet_day PlutoUI.Slider(0:(isnothing(results) ? 0 : length(results.T.times) - 1), default = 0, show_value = true))

**View from longitude:** $(@bind view_longitude PlutoUI.Slider(-180:10:180, default = -30, show_value = true))

**View from latitude:** $(@bind view_latitude PlutoUI.Slider(-90:5:90, default = 35, show_value = true))
"""

# ╔═╡ 566e0aca-d2b0-4071-aaa1-061cd858c3bc
md"""
Initially, nothing much appears to be happening in the temperature. Within the first few days, a zonal (east-west) current known as a **jet** forms along each front, where the temperature gradient is largest. The jet flows east near the surface and develops because the planet rotates.

After one to two weeks the fronts start to meander. The meanders then grow, break and roll up into eddies, with tongues of warm water that reach toward the poles and tongues of cold water that reach toward the equator.

We can also make a movie of the simulation.

**Make the movie:** $(@bind make_movie CheckBox(default = false))
"""

# ╔═╡ 3494a6b6-d761-4b1b-a398-891bbec22c5a
movie_file = joinpath(output_dir, "ocean_planet_$run_name.mp4");

# ╔═╡ 1ae2ed62-a8e9-4ae6-a5b6-dcf9f537e273
md"""
### Zonal means

The simulation above shows baroclinic instability, a process found throughout planetary fluid dynamics. It is the main source of the eddies in the ocean and the storms in the midlatitude atmosphere.

To see its effects more clearly, we average each surface field around each latitude circle, the **zonal mean**, and plot the result as a function of time. On the latitude-longitude grid each row of boxes is a latitude circle. On the other two grids the rows don't follow latitude circles, so `zonal_mean` first regrids the field onto a latitude-longitude grid with 2° boxes. The regridding is **conservative**: each new box gets the area-weighted average of the old boxes that overlap it, so the regridded field holds the same amount of heat.
"""

# ╔═╡ 5c675a7b-7396-4b32-8296-3db55ba1dc75
begin
    zonal_grid = LatitudeLongitudeGrid(Float64; size = (180, 90, 1), longitude = (0, 360), latitude = (-90, 90), z = (-1, 0))

    zonal_regridder = if grid_type != :latlon
        source_grid = make_grid(CPU(), grid_type; resolution = 360 ÷ size(grid, 1), FT = Float64)   # the regridder needs Float64 coordinates
        ConservativeRegridding.Regridder(zonal_grid, source_grid)
    end
end

# ╔═╡ b30b3f25-170f-42fa-9803-3f4b4d4ba965
function zonal_mean(fts)
    ocean = similar(fts[1])
    fill!(ocean, 1)
    mask_immersed_field!(ocean)

    if isnothing(zonal_regridder)
        φ = φnodes(fts.grid, Center())
        latitude_rows = field -> interior(field, :, :, 1)
    else
        φ = φnodes(zonal_grid, Center())
        regridded = CenterField(zonal_grid)
        latitude_rows = field -> interior(ConservativeRegridding.regrid!(regridded, zonal_regridder, field), :, :, 1)
    end

    ocean_per_row = sum(latitude_rows(ocean), dims = 1)
    means = [sum(latitude_rows(fts[n]), dims = 1) ./ ocean_per_row for n in eachindex(fts.times)]
    return φ, permutedims(reduce(vcat, means))
end

# ╔═╡ 51d43b83-7abf-48c1-9f2f-029134078c8f
if isnothing(results)
    md"Run the simulation to see this plot."
else
    let
        days = results.speed.times ./ day
        φ, zonal_speed = zonal_mean(results.speed)

        fig = Figure(size = (700, 420))
        ax = Axis(fig[1, 1], xlabel = "time (days)", ylabel = "latitude (°)", yticks = -90:30:90, titlesize = 16, title = "Zonal-mean surface speed")
        hm = heatmap!(ax, days, φ, zonal_speed', colormap = :speed, colorrange = (0, 2))
        Colorbar(fig[1, 2], hm, label = "speed (m/s)")
        fig
    end
end

# ╔═╡ 1bf163ef-ab3b-450f-9902-80c9ca82945e
md"""
The jets appear at ±45° in the zonal-mean speed. In the first two weeks, a weaker streak moves from each jet toward the pole: a wave that forms as the water adjusts from rest. After about three weeks the bands get stronger and wider, as the eddies add their own zonal currents.
"""

# ╔═╡ 3d02997c-1fb7-4b7f-ba10-396ff7f83bc3
if isnothing(results)
    md"Run the simulation to see this plot."
else
    let
        days = results.T.times ./ day
        φ, zonal_T = zonal_mean(results.T)

        fig = Figure(size = (1100, 420))
        ax1 = Axis(fig[1, 1], xlabel = "latitude (°)", ylabel = "temperature (°C)", xticks = -90:30:90, titlesize = 22, title = "Zonal-mean surface temperature")
        lines!(ax1, φ, zonal_T[:, 1], color = :gray50, linewidth = 2, label = "day 0")
        lines!(ax1, φ, zonal_T[:, end], color = :firebrick, linewidth = 2, label = "day $(round(Int, days[end]))")
        axislegend(ax1, position = :cb)

        ax2 = Axis(fig[1, 2], xlabel = "time (days)", ylabel = "latitude (°)", yticks = -90:30:90, titlesize = 22, title = "Change since day 0")
        hm = heatmap!(ax2, days, φ, (zonal_T .- zonal_T[:, 1])', colormap = :balance, colorrange = (-15, 15))
        Colorbar(fig[1, 3], hm, label = "temperature change (°C)")
        fig
    end
end

# ╔═╡ 511f6251-8445-4484-9d89-e94d102c8940
md"""
From the figures above, we see that the ocean eddies spread the surface temperature out compared to its initial condition. However, the heat spreads poleward over a narrower band of latitude than it does in the atmosphere. This is because ocean eddies are typically much smaller than the storms in the atmosphere, so they mix heat over a shorter distance.

Overall, Earth's climate is an intricate balance between the dynamics and the thermodynamics of the atmosphere and the ocean. To understand the climate system and make predictions about the future, climate scientists need to develop atmosphere and ocean models and run them for centuries to millennia. Many open questions remain in both the dynamics and the thermodynamics of the climate system, and we hope this series of classes has given you a flavor of what that work looks like!
"""

# ╔═╡ 27a20176-a359-4a41-8385-5a098885489a
md"""
## Exercises

### Compare the grids

Run all three grids. Compare the maps, the zonal-mean temperature and the wall time in the progress messages. Do the three grids give the same physics? Which one costs the most?
"""

# ╔═╡ bb926df3-cfcd-4a75-a603-42628aa7cece
details("Hint", md"""
On an Apple M-series GPU, not counting compilation, 120 days took about 1 minute on the latitude-longitude and rotated grids and 1.4 minutes on the tripolar grid. The results are close. The grid is a numerical choice, and a good model gives the same physics on any reasonable grid. The tripolar grid does extra work at the fold, where the two halves of its northern part meet.
""")

# ╔═╡ 43c1aadb-311a-47f3-b4b8-bb4d7fc77c4f
md"""
### Break the CFL condition

The **CFL (Courant–Friedrichs–Lewy) condition** says that in one time step, nothing in the model (a current or a wave) may travel farther than about one grid box. Set `Δt = 30minutes` in the model cell and run it. What happens, and how soon? Then find the longest time step that works.
"""

# ╔═╡ f6d14578-452b-4f8f-add9-1f89ed8b22e9
details("Hint", md"""
In our tests on the latitude-longitude grid, a step of 20 minutes ran, and a step of 30 minutes stopped with an error within the first day of model time. On a GPU the error is a `DomainError`: some number in the model has become nonsense, and a square root or a power fails.
""")

# ╔═╡ 59869cca-4917-4757-aa13-e62ff9330efd
md"""
### Halve the grid spacing

Set `resolution = 1` in the model cell. Predict how much longer the run takes: there are 4 times as many boxes, and the smallest box is half as wide. Then run it, and compare the eddies and the zonal-mean temperature with the 2° run.
"""

# ╔═╡ 61c05b91-c56d-4373-927e-4da6881ca2be
details("Hint", md"""
With a 10-minute step the 1° model blows up, because the smallest boxes are half as wide. Set `Δt = 5minutes`. The run then has 4 times the boxes and twice the steps: 8 times the work. If we also doubled the number of vertical levels, it would be 16 times.
""")

# ╔═╡ 7b1a6aed-13de-4739-ae0c-522ac9e0d3fe
md"""
### Remove the noise

On the latitude-longitude grid, delete `+ rand()` from both initial conditions and change `run_name` (for example, add `"_no_noise"`) so the earlier output is kept. What do you predict? Run it and look at the maps.
"""

# ╔═╡ 6b549874-6cc2-4078-ad99-83af21f3df0d
details("Hint", md"""
The front stays the same at every longitude and no eddies form. The equations don't prefer any longitude, so a state that is the same at every longitude stays that way. The front is unstable, but an instability needs a seed to grow from, and a real ocean or atmosphere always has one. The rotated and tripolar grids are not the same at every longitude, so their numerical errors can seed eddies even without the noise.
""")

# ╔═╡ 652c59a9-0fb1-4d74-bf75-cf6688ee45fe
md"""
## Further reading

- G. L. Wagner et al. (2025). [High-level, high-resolution ocean modeling at all scales with Oceananigans](https://arxiv.org/abs/2502.14148). The source of this model setup.
- A. Ramadhan et al. (2020). [Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs](https://doi.org/10.21105/joss.02018). *Journal of Open Source Software* 5(53), 2018.
- S. Silvestri et al. (2023). [Oceananigans.jl: A Julia library that achieves breakthrough resolution, memory and energy efficiency in global ocean simulations](https://arxiv.org/abs/2309.06662). Global ocean simulations on GPUs at resolutions down to 488 m.
- S. Silvestri et al. (2024). [A new WENO-based momentum advection scheme for simulations of ocean mesoscale turbulence](https://doi.org/10.1029/2023MS004130). *Journal of Advances in Modeling Earth Systems* 16(7), e2023MS004130. The `WENOVectorInvariant` scheme used in this notebook.
- S. Silvestri et al. (2025). [A GPU-based ocean dynamical core for routine mesoscale-resolving climate simulations](https://doi.org/10.1029/2024MS004465). *Journal of Advances in Modeling Earth Systems* 17(4), e2024MS004465. The `HydrostaticFreeSurfaceModel` used in this notebook, simulating a decade of near-global ocean per wall-clock day at 8 km resolution on 64 GPUs.
- [Oceananigans.jl documentation](https://clima.github.io/OceananigansDocumentation/stable/), with more examples to run.
- J. Marshall and R. A. Plumb (2008). [*Atmosphere, Ocean and Climate Dynamics: An Introductory Text*](https://www.sciencedirect.com/book/9780125586917/atmosphere-ocean-and-climate-dynamics), Academic Press. Includes rotating-tank experiments of baroclinic instability.
- G. K. Vallis (2017). [*Atmospheric and Oceanic Fluid Dynamics*](https://doi.org/10.1017/9781107588431), 2nd edition, Cambridge University Press. See the chapters on baroclinic instability.
"""

# ╔═╡ 3ae7b80b-b2ac-481b-9d3d-aeda8f6cc9e6
md"""
## Appendix: plotting helpers

These functions draw a field on a globe. `orthographic` is the projection that a camera far from the planet would see. `lon_lat` gives the longitude and latitude of every box, on any of the three grids.
"""

# ╔═╡ cd2ec92c-6e1f-462a-b585-2598b3c42155
function orthographic(λ, φ; λ₀ = -30, φ₀ = 35)
    visible = @. sind(φ₀) * sind(φ) + cosd(φ₀) * cosd(φ) * cosd(λ - λ₀) > 0
    x = @. cosd(φ) * sind(λ - λ₀)
    y = @. cosd(φ₀) * sind(φ) - sind(φ₀) * cosd(φ) * cosd(λ - λ₀)
    return x, y, visible
end

# ╔═╡ 880d45fc-e31a-4367-8779-f612ce736fe8
function lon_lat(grid, ℓx, ℓy)
    λ = Array(λnodes(grid, ℓx, ℓy, Center()))
    φ = Array(φnodes(grid, ℓx, ℓy, Center()))
    λ isa AbstractMatrix && return λ, φ
    return λ .+ 0 .* φ', 0 .* λ .+ φ'
end

# ╔═╡ c05b017e-4779-489a-962d-91926577e07f
close_seam(a) = vcat(a, a[1:1, :])   # repeat the first meridian, because the grid wraps around in longitude

# ╔═╡ 497984f7-5009-477f-adbf-cf80e6ab72fd
let
    fig = Figure(size = (1200, 430))
    titles = Dict(:latlon => "Latitude-longitude", :rotated => "Rotated latitude-longitude", :tripolar => "Tripolar")
    for (i, t) in enumerate((:latlon, :rotated, :tripolar))
        λ, φ = close_seam.(lon_lat(cpu_grids[t], Face(), Face()))
        x, y, visible = orthographic(λ, φ; λ₀ = -20, φ₀ = 60)
        x[.!visible] .= NaN
        ax = Axis(fig[1, i], aspect = DataAspect(), titlesize = 22, title = titles[t])
        hidedecorations!(ax)
        hidespines!(ax)
        for j in axes(x, 2)
            lines!(ax, x[:, j], y[:, j], color = :black, linewidth = 0.5)
        end
        for k in axes(x, 1)
            lines!(ax, x[k, :], y[k, :], color = :black, linewidth = 0.5)
        end
        if t == :tripolar
            Nx, Ny = size(x, 1) - 1, size(x, 2)
            for k in 1:Nx÷2
                mirror = [k, Nx + 2 - k]
                lines!(ax, x[mirror, Ny], y[mirror, Ny], color = :red, linewidth = 1)
            end
        end
        lines!(ax, Circle(Point2f(0, 0), 1), color = :black, linewidth = 1)
    end
    fig
end

# ╔═╡ da1dfe12-48ef-405e-9000-b661203a2a05
land_to_nan(field, mask = field) = ifelse.(isnan.(mask) .| (mask .== 0), NaN32, field)

# ╔═╡ 83b1054a-d95a-4bca-a219-c1eaed0ae83a
function globe!(ax, λ, φ, data; λ₀ = -30, φ₀ = 35, kw...)
    λ, φ = close_seam(λ), close_seam(φ)
    xyv = lift(λ₀ -> orthographic(λ, φ; λ₀, φ₀), convert(Observable, λ₀))
    x = @lift $xyv[1]
    y = @lift $xyv[2]
    color = lift((d, (_, _, visible)) -> ifelse.(visible, close_seam(d), NaN32), convert(Observable, data), xyv)
    plot = surface!(ax, x, y, @lift(zero($x)); color, shading = NoShading, nan_color = :transparent, kw...)
    lines!(ax, Circle(Point2f(0, 0), 1), color = :black, linewidth = 1)
    return plot
end

# ╔═╡ 9d223bad-ca93-40b1-b021-d83064222dbe
function surface_figure(results, n; λ₀, φ₀ = 35)
    λ, φ = lon_lat(results.T.grid, Center(), Center())
    Tₙ = @lift land_to_nan(interior(results.T[$n], :, :, 1))
    speedₙ = @lift land_to_nan(interior(results.speed[$n], :, :, 1), $Tₙ)
    ζₙ = @lift land_to_nan(interior(results.ζ[$n], :, :, 1), $Tₙ) ./ 1e-5

    fig = Figure(size = (1500, 520))
    ax1 = Axis(fig[1, 1], aspect = DataAspect(), titlesize = 22, title = "Surface temperature")
    ax2 = Axis(fig[1, 3], aspect = DataAspect(), titlesize = 22, title = "Surface speed")
    ax3 = Axis(fig[1, 5], aspect = DataAspect(), titlesize = 22, title = "Surface vorticity")
    hidedecorations!.((ax1, ax2, ax3))
    hidespines!.((ax1, ax2, ax3))
    hm1 = globe!(ax1, λ, φ, Tₙ; λ₀, φ₀, colormap = :thermal, colorrange = (0, 30))
    hm2 = globe!(ax2, λ, φ, speedₙ; λ₀, φ₀, colormap = :speed, colorrange = (0, 2))
    hm3 = globe!(ax3, λ, φ, ζₙ; λ₀, φ₀, colormap = :balance, colorrange = (-2, 2))
    Colorbar(fig[1, 2], hm1, label = "temperature (°C)")
    Colorbar(fig[1, 4], hm2, label = "speed (m/s)")
    Colorbar(fig[1, 6], hm3, label = "vorticity (10⁻⁵ s⁻¹)")
    Label(fig[0, :], @lift("Day $($n - 1)"), fontsize = 24, font = :bold)
    return fig
end

# ╔═╡ 523a3e0a-df64-4772-ac00-b8cd94321626
if isnothing(results)
    md"Run the simulation to see this plot."
else
    surface_figure(results, Observable(planet_day + 1); λ₀ = view_longitude, φ₀ = view_latitude)
end

# ╔═╡ 16adfbd5-1653-496d-b6fc-ca10d8417b48
function embed_mp4(path; width = 900)
    data = base64encode(read(path))
    HTML("""<video controls autoplay loop width="$width"><source src="data:video/mp4;base64,$data" type="video/mp4"></video>""")
end

# ╔═╡ 00226ea4-663e-4034-878d-2401028768dc
if isnothing(results)
    md"Run the simulation to see the movie."
elseif make_movie
    let
        n = Observable(1)
        fig = surface_figure(results, n; λ₀ = @lift(-30 + 2 * ($n - 1)))
        CairoMakie.record(fig, movie_file, 1:length(results.T.times); framerate = 15) do i
            n[] = i
        end
        embed_mp4(movie_file)
    end
elseif isfile(movie_file)
    embed_mp4(movie_file)
else
    md"Check the box to make the movie."
end

# ╔═╡ 91623e61-db88-4a1e-b7e6-25c23b3e1260
let
	changes = md"""
	- converted to a Pluto notebook with explanations and exercises
	- grid chosen from a menu
	- tripolar grid islands wrap in longitude, so both northern poles are covered
	- runs on Metal, CUDA, AMDGPU or CPU, in `Float32`
	- runs for 120 days
	- saves surface temperature, speed and vorticity
	- adds plots, a movie and the zonal-mean temperature
	"""
	md"""
	!!! info "License and attribution"
	    The model setup is adapted from [`listings/baroclinic_wave.jl`](https://github.com/glwagner/OceananigansPaper/blob/70ae7ea425e0a9315f00d8d2b420318e676ed543/listings/baroclinic_wave.jl) in [OceananigansPaper](https://github.com/glwagner/OceananigansPaper), under the Apache License, Version 2.0 (full text below).

	    $(details("Changes from the original", changes))
	"""
end

# ╔═╡ 28b9a1be-96ca-40f7-8bc2-4e9cf3994fe6
details("Apache License, Version 2.0 (full text)", Base.Text(raw"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
"""))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
Base64 = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ConservativeRegridding = "8e50ac2c-eb48-49bc-a402-07c87b949343"
Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
Oceananigans = "9e8cae18-63c1-5223-a75c-80ca9d6e9a09"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
SeawaterPolynomials = "d496a93d-167e-4197-9f49-d3af4ff8fe40"

[compat]
AMDGPU = "~2.7.0"
CUDA = "~6.2.2"
CairoMakie = "~0.15.17"
ConservativeRegridding = "~0.2.21"
Metal = "~1.10.0"
Oceananigans = "~0.117.0"
PlutoUI = "~0.7.84"
SeawaterPolynomials = "~0.3.11"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.4"
manifest_format = "2.1"
project_hash = "d5dc27af3a659f19b645f881503a757749836c22"

[[deps.AMDGPU]]
deps = ["AbstractFFTs", "AcceleratedKernels", "Adapt", "Atomix", "BFloat16s", "CEnum", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLD_jll", "LLVM", "LLVM_jll", "Libdl", "LinearAlgebra", "Pkg", "PrecompileTools", "Preferences", "PrettyTables", "Printf", "ROCmDeviceLibs_jll", "Random", "Random123", "RandomNumbers", "SparseArrays", "StaticArraysCore", "Statistics", "UnsafeAtomics"]
git-tree-sha1 = "c451e0febf762a9019d8a9747a31e8cf3f655fd7"
registries = "General"
uuid = "21141c5a-9bdb-4563-92ae-f87d6854732e"
version = "2.7.0"

    [deps.AMDGPU.extensions]
    AMDGPUChainRulesCoreExt = "ChainRulesCore"
    AMDGPUEnzymeCoreExt = "EnzymeCore"
    AMDGPUSparseMatricesCSRExt = "SparseMatricesCSR"
    AMDGPUSpecialFunctionsExt = "SpecialFunctions"

    [deps.AMDGPU.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
registries = "General"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
registries = "General"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.AcceleratedKernels]]
deps = ["ArgCheck", "GPUArraysCore", "KernelAbstractions", "Markdown", "UnsafeAtomics"]
git-tree-sha1 = "f75592c5dc92f7f75e544b156eba7165d4418193"
registries = "General"
uuid = "6a4ca0a5-0e36-4168-a932-d9be78d558f1"
version = "0.4.4"

    [deps.AcceleratedKernels.extensions]
    AcceleratedKernelsoneAPIExt = "oneAPI"

    [deps.AcceleratedKernels.weakdeps]
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "8c0bc12397dabf5d2dfc4b5763ec463854cc44d3"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.46"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "dee8388a9bec89c8713ad176f96960ae2f98598a"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.4"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptiveArrayPools]]
deps = ["Preferences", "Printf", "Random"]
git-tree-sha1 = "86d0ae34bc46373170064ef33c12a303994ab736"
registries = "General"
uuid = "4f381ef7-9af0-4cbe-99d4-cf36d7b0f233"
version = "0.3.12"
weakdeps = ["CUDA", "Metal"]

    [deps.AdaptiveArrayPools.extensions]
    AdaptiveArrayPoolsCUDAExt = "CUDA"
    AdaptiveArrayPoolsMetalExt = "Metal"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
registries = "General"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
registries = "General"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
registries = "General"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "1bbcf97714051f302b3beafab65bcdeb6044c4df"
registries = "General"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.2.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
registries = "General"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
registries = "General"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
registries = "General"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "5f866fe1b4890785975a933a452946507660a9d6"
registries = "General"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.4"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
registries = "General"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

[[deps.BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "091d591a060e43df1dd35faab3ca284925c48e46"
registries = "General"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.3.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
registries = "General"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "0db159493e83ecb2d122f9c40138d89b70d43963"
registries = "General"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.1.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
registries = "General"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CUDA]]
deps = ["CUDACore", "CUDATools", "Reexport", "cuBLAS", "cuFFT", "cuRAND", "cuSOLVER", "cuSPARSE"]
git-tree-sha1 = "6b7605ef4d1bfc851f6743c278eb01842b3ebb99"
registries = "General"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "6.2.2"

[[deps.CUDACore]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVPTX_LLVM_Backend_jll", "PrecompileTools", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "StaticArrays"]
git-tree-sha1 = "32a29bdca1244fc7fa3ae88c04305905b8ae89c5"
registries = "General"
uuid = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
version = "6.2.2"

    [deps.CUDACore.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDACore.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDATools]]
deps = ["CUDACore", "CUDA_Compiler_jll", "CUPTI", "Crayons", "GPUCompiler", "LLVM", "NVML", "NVTX", "PrecompileTools", "Preferences", "PrettyTables", "Printf", "Statistics", "demumble_jll"]
git-tree-sha1 = "4c97d66f3507212fef6b624ef90d4290048f5e23"
registries = "General"
uuid = "9ec180c6-1c07-47c7-9e6e-ebefa4d1f6d0"
version = "6.2.2"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c32d22f2f563ce192c88a44b09c2b569f1e7a980"
registries = "General"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.4.4+1"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "901cc2d895842b2ddef81715cb70fffc3b5028ae"
registries = "General"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.3.1+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "159b1c1f03e6355bb9c6d8954e5f51019788dd26"
registries = "General"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "2.1.1"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2e0352eb2a8321e46e1de54059bed9be8fd9391c"
registries = "General"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.23.0+1"

[[deps.CUPTI]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox"]
git-tree-sha1 = "dde1ce6735c79e113d46b62abce2159a5c02bd37"
registries = "General"
uuid = "9e67e8f6-ba02-4b6c-a7db-3b11ae1e7ab7"
version = "6.2.2"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
registries = "General"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "FreeType2_jll", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "2d9e0c818e6e5af309192a5b3638c8848c7119b5"
registries = "General"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.17"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0cfb779e24fdced77a05df1dbe99200821a9f563"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+1"

[[deps.ChainRulesCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "295fa9b898db109bc56df62c0fc20bd105cec00e"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.8"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "3496e2b359c27793d12bec54ec94be7983a6ceb2"
registries = "General"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.2"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "d4101e848e8d3f585d61d244c2fe0c80a70e6b3b"
registries = "General"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.1.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
registries = "General"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.ChunkSplitters]]
git-tree-sha1 = "1c52c8e2673edc030191177ff1aee42d25149acb"
registries = "General"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.2.0"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
registries = "General"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
registries = "General"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
registries = "General"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "060adaf7a5c2e14bca025edf0eea0206ffdebd87"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.32.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "af523e59de94de05b9b367a2ca4d63233aacd222"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.5"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "96fce5c80e0834e96ab6d24fb384fd945c9cb2d4"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.2"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "0737a3e59f9d94957f5b177fa2ca40901636b85b"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.4"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "b56df0c58a9d37b4a926f84f3f4f283a5987680e"
registries = "General"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.9"

[[deps.ConservativeRegridding]]
deps = ["ChunkSplitters", "ConstructionBase", "DocStringExtensions", "Extents", "GeoInterface", "GeometryOps", "GeometryOpsCore", "LinearAlgebra", "ProgressMeter", "SciMLPublic", "SmallCollections", "SortTileRecursiveTree", "SparseArrays", "StableTasks", "StaticArrays"]
git-tree-sha1 = "b02c0e025f75e769c31d0c9e43a3887089716707"
registries = "General"
uuid = "8e50ac2c-eb48-49bc-a402-07c87b949343"
version = "0.2.21"

    [deps.ConservativeRegridding.extensions]
    ConservativeRegriddingClimaCoreExt = "ClimaCore"
    ConservativeRegriddingHealpixExt = "Healpix"
    ConservativeRegriddingInterfacesExt = "Interfaces"
    ConservativeRegriddingNCDatasetsExt = "NCDatasets"
    ConservativeRegriddingOceananigansExt = ["KernelAbstractions", "Oceananigans"]
    ConservativeRegriddingRingGridsExt = "RingGrids"

    [deps.ConservativeRegridding.weakdeps]
    ClimaCore = "d414da3d-4745-48bb-8d80-42e94e092884"
    Healpix = "9f4e344d-96bc-545a-84a3-ae6b9e1b672b"
    Interfaces = "85a1e053-f937-4924-92a5-1367d23b7b87"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    NCDatasets = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
    Oceananigans = "9e8cae18-63c1-5223-a75c-80ca9d6e9a09"
    RingGrids = "d1845624-ad4f-453b-8ff4-a8db365bf3a7"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "a692f5e257d332de1e554e4566a4e5a8a72de2b2"
registries = "General"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.4"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
registries = "General"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
registries = "General"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.CpuId]]
deps = ["Markdown", "PrecompileTools"]
git-tree-sha1 = "5dbd5150b96f29553fc3b553c6bfb0eadf7ac204"
registries = "General"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.3"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.CubedSphere]]
deps = ["Distances", "LinearAlgebra", "Printf", "Random", "StaticArrays", "Statistics", "TaylorSeries"]
git-tree-sha1 = "641b7bfdb81feda05316db4e8d3d57358aead4cd"
registries = "General"
uuid = "7445602f-e544-4518-8976-18f8e8ae6cdb"
version = "0.3.6"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "e1e28f37cbdca4b594ad86ecfe2ee8f25796ebff"
registries = "General"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.7.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
registries = "General"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "478d3450b1cc56521d1d99ad923acf43f8813c6d"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.134"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
registries = "General"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
registries = "General"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
registries = "General"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f2659a735f9360fedfa29925812380f322679e93"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.10.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
registries = "General"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "fdcc9d4a253e0041d37c95487f9ae027c3ec002d"
registries = "General"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.6"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
registries = "General"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "16aa62b6b2c5c65a642155bd1bce90633b8d3130"
registries = "General"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
registries = "General"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
registries = "General"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FastInterpolations]]
deps = ["AdaptiveArrayPools", "HelpPlots", "LinearAlgebra", "Preferences", "Printf"]
git-tree-sha1 = "f7bcd496f0a2e4e1174e7e1f667f27fb671fd7df"
registries = "General"
uuid = "9ea80cae-fc13-4c00-8066-6eaedb12f34b"
version = "0.4.20"

    [deps.FastInterpolations.extensions]
    FastInterpolationsChainRulesCoreExt = "ChainRulesCore"
    FastInterpolationsColorVectorSpaceExt = ["ColorTypes", "ColorVectorSpace"]
    FastInterpolationsEnzymeExt = "Enzyme"
    FastInterpolationsForwardDiffExt = "ForwardDiff"
    FastInterpolationsRecipesBaseExt = "RecipesBase"
    FastInterpolationsSymbolicsExt = ["Symbolics", "SymbolicUtils"]

    [deps.FastInterpolations.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
    ColorVectorSpace = "c3611d14-8923-5661-9e6a-0046d554d3a4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
registries = "General"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
registries = "General"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
registries = "General"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "831f86fde78ab7f6470d6ba4d1161a11fb6b4ec0"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.2"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "e1e9ba3a8f5d71e6ae698f138a1362aeabb42e9f"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.18.3+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
registries = "General"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
registries = "General"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e894be022a2eba58463939f1bf1f7f7006f27f52"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.19+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "574e443e2545f8622b716676c9358ca8a7ea57b4"
registries = "General"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.5.16"
weakdeps = ["JLD2"]

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "0b2b3e8b8fb7c1f5ddbf498be2eb2b3b37c9c0ed"
registries = "General"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.1"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "REPL", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "5e54ec63c34bcc878558b173c411b8efe6b08344"
registries = "General"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.23.0"

    [deps.GPUCompiler.weakdeps]
    AMDGPU_LLVM_Backend_jll = "cc5c0156-bd05-5a77-8a68-bb0aafb29019"
    LLVMDowngrader_jll = "f52de702-fb25-5922-94ba-81dd59b07444"
    NVPTX_LLVM_Backend_jll = "ef6e0fe3-e6ef-59c0-bde6-4989574699e0"

[[deps.GPUToolbox]]
git-tree-sha1 = "799f1f2dd18bf93968a2637227ba4d09115bd5fa"
registries = "General"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "3.5.0"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "7528a7956248c723d01a0a9b0447bf254bf4da52"
registries = "General"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.5"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "20f6b13b28c6304104968374f38c24ef71a8ea16"
registries = "General"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.6.2"

    [deps.GeoInterface.extensions]
    GeoInterfaceMakieExt = ["Makie", "GeometryBasics"]
    GeoInterfaceRecipesBaseExt = "RecipesBase"

    [deps.GeoInterface.weakdeps]
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "ec46c5825710fa1a15d468acb2d93cc939a7a5fe"
registries = "General"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.13"
weakdeps = ["Extents", "GeoInterface", "IntervalSets"]

    [deps.GeometryBasics.extensions]
    ExtentsExt = "Extents"
    GeometryBasicsGeoInterfaceExt = "GeoInterface"
    IntervalSetsExt = "IntervalSets"

[[deps.GeometryOps]]
deps = ["AbstractTrees", "AdaptivePredicates", "CoordinateTransformations", "DataAPI", "DelaunayTriangulation", "ExactPredicates", "Extents", "GeoFormatTypes", "GeoInterface", "GeometryOpsCore", "LinearAlgebra", "PrecompileTools", "Random", "SortTileRecursiveTree", "StaticArrays", "Statistics", "Tables"]
git-tree-sha1 = "b0d208d68332d90d632d9bb4303e0b55ca8fb13c"
registries = "General"
uuid = "3251bfac-6a57-4b6d-aa61-ac1fef2975ab"
version = "0.1.47"

    [deps.GeometryOps.extensions]
    GeometryOpsDataFramesExt = "DataFrames"
    GeometryOpsDimensionalDataExt = "DimensionalData"
    GeometryOpsFlexiJoinsExt = "FlexiJoins"
    GeometryOpsLibGEOSExt = "LibGEOS"
    GeometryOpsMakieExt = "Makie"
    GeometryOpsProjExt = "Proj"
    GeometryOpsTGGeometryExt = "TGGeometry"

    [deps.GeometryOps.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"
    FlexiJoins = "e37f2e79-19fa-4eb7-8510-b63b51fe0a37"
    LibGEOS = "a90b1aa1-3769-5649-ba7e-abc5a9d163eb"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Proj = "c94c279d-25a6-4763-9509-64d165bea63e"
    TGGeometry = "d7e755d2-3c95-4bcf-9b3c-79ab1a78647b"

[[deps.GeometryOpsCore]]
deps = ["DataAPI", "GeoInterface", "StableTasks", "Tables"]
git-tree-sha1 = "8f32a94cf3d80a716cae6025af6cbc4e68421da5"
registries = "General"
uuid = "05efe853-fabf-41c8-927e-7063c8b9f013"
version = "0.1.12"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3efbbc027441271444dcd0c0a46f2d119dc4329"
registries = "General"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "6.1.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "ea9f1fb37ff10597353b56f3ba0ebd525acef6e0"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.90.1+0"

[[deps.Glob]]
git-tree-sha1 = "246c628cec062230b7d183aab88841fa94fcabe9"
registries = "General"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.5.0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
registries = "General"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "ef70da5e123a06a29e2d6ddff0f09985bc226491"
registries = "General"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.3"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f700c32c1e9e11b576592043f270e6b33868a56f"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.15000.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
registries = "General"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HelpPlots]]
git-tree-sha1 = "07d389d81ca31e42d68bd88cfa1e900a63b7d378"
registries = "General"
uuid = "5880b2e8-ad0a-48c9-bacd-30dc9c1357e0"
version = "1.1.0"

    [deps.HelpPlots.extensions]
    HelpPlotsExt = "Plots"

    [deps.HelpPlots.weakdeps]
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "cbbd99d516990d05c02aa132aa077b2bf50ae323"
registries = "General"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.15.0+0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "c28dceeadc1c9c699656a3ac9d63f79cfaa17472"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.31"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
registries = "General"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
registries = "General"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
registries = "General"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "f0f005f997dfb8c5fe23920d99458a9619873893"
registries = "General"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.10"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
registries = "General"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
registries = "General"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
registries = "General"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
registries = "General"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
registries = "General"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
registries = "General"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
registries = "General"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "1c531bf0f8a5c60a340926e058fd3f209b5eef5d"
registries = "General"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.12"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticMakieExt = "Makie"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "0db6aea5b64caa1c47e9fdd27394f0296f05e8bf"
registries = "General"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.15"

    [deps.IntervalSets.extensions]
    IntervalSetsMakieExt = "Makie"
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "d22e8bdb3f03c300a103ff23c0dd5a654c6701de"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.7"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
registries = "General"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "123311cf9834c6ee6a8347bba9af68f255615391"
registries = "General"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.11.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecCore", "ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "9ce2e7c49ae4a7035b7d60db3a553b1f6c16875d"
registries = "General"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.7"

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

    [deps.JLD2.weakdeps]
    UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "06d90f699285d055e34f0b014636a75a17ff2252"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.13.4"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]
    JSONSerializationExt = ["Serialization"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
registries = "General"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e06b451acfb1c6fa5ea753b5042207226271f455"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.1+0"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
registries = "General"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "e51d16f9dcff134c4ecf79e7cfa764c59d2d648e"
registries = "General"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.45"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "19fa81989298fc2fb103fee29b03c5284f17e8a5"
registries = "General"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.13"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "b13076e4fff6ea0d0a5ddd9b939a11ca2027755a"
registries = "General"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.10.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d7c65411dda4bfa0beb7490476122b4f805cf31"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.3.0+0"

[[deps.LLD_jll]]
deps = ["Artifacts", "Libdl", "Zlib_jll", "libLLVM_jll"]
uuid = "d55e3150-da41-5e91-b323-ecfd1eec6109"
version = "20.1.8+2"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "3bcfa5cd59aecfc0680d8b5b8ea38237b009f112"
registries = "General"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.13.2"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMDowngrader_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "e8b622923cbf4a80cdd0d22d8ad4b20d112797df"
registries = "General"
uuid = "f52de702-fb25-5922-94ba-81dd59b07444"
version = "0.8.3+0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "d77aea19c9a71059a021acd99b0a4343e9661d94"
registries = "General"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.47+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
registries = "General"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LLVM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML", "Zlib_jll", "Zstd_jll", "libLLVM_jll"]
git-tree-sha1 = "18aec222aa5f605cce5012fdc3e04768060037a4"
registries = "General"
uuid = "86de99a1-58d6-5da7-8064-bd56ce2e322c"
version = "21.1.2+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
registries = "General"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
registries = "General"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
registries = "General"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MPI]]
deps = ["Distributed", "DocStringExtensions", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "PkgVersion", "PrecompileTools", "Serialization", "Sockets"]
git-tree-sha1 = "7b7af40eaafefa3b85178e4028b5a094880e5abd"
registries = "General"
uuid = "da04e1cc-30fd-572f-bb4f-1f8673147195"
version = "0.20.29"

    [deps.MPI.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    OneAPIExt = "oneAPI"

    [deps.MPI.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MPIABI_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "42e2cada9f35500b4d227106ce2b60e85d9f15bc"
registries = "General"
uuid = "b5ada748-db0f-5fc0-8972-9331c762740c"
version = "1.0.1+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "08cb47f95a2f1b795e43a9e4f9783e68ecdc1ca6"
registries = "General"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "5.0.3+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8e98d5d80b87403c311fd51e8455d4546ba7a5f8"
registries = "General"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.12"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "09a99227205419c00e734f413ee98c18bc1b02e4"
registries = "General"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.7+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FastInterpolations", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "8ff1cd11162828e9f517e32d83b49d2af63391c6"
registries = "General"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.17"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
registries = "General"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
registries = "General"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.Metal]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CodecBzip2", "Crayons", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMDowngrader_jll", "LinearAlgebra", "ObjectiveC", "PrecompileTools", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "SHA", "ScopedValues", "StaticArrays", "Statistics", "UUIDs"]
git-tree-sha1 = "e5f7aa85211cc925b67d6704d07aab1b99b8ce7a"
registries = "General"
uuid = "dde4c033-4e86-420c-a63e-0dd931031962"
version = "1.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.Metal.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "3350d35d0ebcf7ccee0c66290b8e95d82419ef6f"
registries = "General"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.5+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "d9d20d6eb2a3290ba5c70a1c013b45305cc39c9e"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
registries = "General"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.11.5"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
registries = "General"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.NVML]]
deps = ["CEnum", "CUDACore", "GPUToolbox", "Libdl"]
git-tree-sha1 = "f59c423c6f40a753f69b3107c7944d06784933e2"
registries = "General"
uuid = "611af6d1-644e-4c5d-bd58-854d7d1254b9"
version = "6.2.2"

[[deps.NVPTX_LLVM_Backend_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "6a1ec02404e66d69b1b6bc884299a4c80f932c6d"
registries = "General"
uuid = "ef6e0fe3-e6ef-59c0-bde6-4989574699e0"
version = "22.1.7+1"

[[deps.NVTX]]
deps = ["JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "a9083c3e469e63cca454d1fc3b19472d9d92c14a"
registries = "General"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.3"
weakdeps = ["Colors"]

    [deps.NVTX.extensions]
    NVTXColorsExt = "Colors"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
registries = "General"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "676c0f1c1c34224d6869675047a54b42fdc7b8c5"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.5"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
registries = "General"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.ObjectiveC]]
deps = ["CEnum", "Libdl", "Preferences"]
git-tree-sha1 = "4abd3c3e24a1631d1e512673cda4835a612f88ab"
registries = "General"
uuid = "e86c9b32-1129-44ac-8ea0-90d5bb39ded9"
version = "6.0.1"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
registries = "General"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.Oceananigans]]
deps = ["Adapt", "BFloat16s", "Crayons", "CubedSphere", "Dates", "Distances", "DocStringExtensions", "FFTW", "GPUArraysCore", "Glob", "InteractiveUtils", "JLD2", "KernelAbstractions", "Krylov", "LinearAlgebra", "Logging", "MPI", "MuladdMacro", "OffsetArrays", "OrderedCollections", "Printf", "ReactantCore", "Rotations", "SeawaterPolynomials", "SparseArrays", "StaticArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "d2dbf4b74c3c95d01dd622101135723cd8f70d8c"
registries = "General"
uuid = "9e8cae18-63c1-5223-a75c-80ca9d6e9a09"
version = "0.117.0"

    [deps.Oceananigans.extensions]
    OceananigansAMDGPUExt = ["AMDGPU", "AbstractFFTs"]
    OceananigansCUDAExt = ["CUDA", "GPUArraysCore"]
    OceananigansConservativeRegriddingExt = "ConservativeRegridding"
    OceananigansEnzymeExt = "Enzyme"
    OceananigansMakieExt = "Makie"
    OceananigansMetalExt = ["Metal", "AbstractFFTs"]
    OceananigansNCCLExt = ["NCCL", "CUDA"]
    OceananigansNCDatasetsExt = "NCDatasets"
    OceananigansOneAPIExt = "oneAPI"
    OceananigansReactantExt = ["Reactant", "KernelAbstractions", "ConstructionBase", "AbstractFFTs"]
    OceananigansZarrExt = "Zarr"

    [deps.Oceananigans.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AbstractFFTs = "621f4979-c628-5d54-868e-fcf4e3e8185c"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ConservativeRegridding = "8e50ac2c-eb48-49bc-a402-07c87b949343"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    NCDatasets = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    Zarr = "0a941bbe-ad1d-11e8-39d9-ab76183a1d99"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.OffsetArrays]]
git-tree-sha1 = "750d0f2b5aac238d50dfee5f7a44fae3175a3305"
registries = "General"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.18.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "38a93f17e431141c6470bb67a88952a7c4f0e928"
registries = "General"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.34+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
registries = "General"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "OpenJPH_jll", "Zlib_jll", "Zstd_jll", "libdeflate_jll"]
git-tree-sha1 = "afa66dc1246b28b68600d053715e00a379268980"
registries = "General"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.5.3+0"

[[deps.OpenJPH_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "1b1f3cabb1130670812b848f4623d69d578ea805"
registries = "General"
uuid = "95196869-6d61-5868-8eee-c38c33804e0c"
version = "0.32.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "0dba69ef957d790d41484ca7c20212c5ac806738"
registries = "General"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.14+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.47.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "32b657a0d57c310a1a172bfc8c8cf68c5e674323"
registries = "General"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.5"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
registries = "General"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
registries = "General"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "e5ae52d1f4f19a9591d615cb0f1ef8d415b40933"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
registries = "General"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "9869376035f47756b8b8e711fd0d06a06d51da6f"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.84"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
registries = "General"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "eeac4c9337f876c3c32b8da732141953567f57c9"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.1"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"
    PrettyTablesWriteDocxExt = "WriteDocx"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    WriteDocx = "d049ceea-54ee-41d7-a26f-ba29db3b6599"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
registries = "General"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Printf"]
git-tree-sha1 = "ce5ba3edfdeba107a721748d53b5f5887f415853"
registries = "General"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.13.0"

    [deps.ProgressMeter.extensions]
    ProgressMeterDistributedExt = "Distributed"
    ProgressMeterIJuliaExt = "IJulia"

    [deps.ProgressMeter.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
registries = "General"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "4d8c1b7c3329c1885b857abb50d08fa3f4d9e3c8"
registries = "General"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.7"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.ROCmDeviceLibs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "26ae7b27acf9bb6357dca3d90bc12ef5ba07f023"
registries = "General"
uuid = "873c0968-716b-5aa7-bb8d-d1e2e2aeff2d"
version = "7.0.2+0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
registries = "General"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
registries = "General"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
registries = "General"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
registries = "General"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.ReactantCore]]
deps = ["ExpressionExplorer", "MacroTools"]
git-tree-sha1 = "1fe4e616752a0c4818ae88a504fc59b0fb016659"
registries = "General"
uuid = "a3311ec8-5e00-46d5-b541-4f83e724a433"
version = "0.1.26"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
registries = "General"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "c5e8a0a52c0d5d043214de04f11f4d9c2c6ccdb8"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.13"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "5680a9276685d392c87407df00d57c9924d9f11e"
registries = "General"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.1"

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

    [deps.Rotations.weakdeps]
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
registries = "General"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0651e0ac84b2027c5f227c058d37f81b5be373bd"
registries = "General"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.8.2"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
registries = "General"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
registries = "General"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SeawaterPolynomials]]
git-tree-sha1 = "b700ec85301a323666d1c99b2ecf4670dd8b213e"
registries = "General"
uuid = "d496a93d-167e-4197-9f49-d3af4ff8fe40"
version = "0.3.11"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "57aa595158717ef165e6f5ab639fe2e3178c0a2b"
registries = "General"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
registries = "General"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
registries = "General"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2c79185e5261b159474903598c571ddd9f12ff7b"
registries = "General"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.6"

[[deps.SmallCollections]]
deps = ["BitIntegers", "CpuId", "LinearAlgebra", "Random"]
git-tree-sha1 = "7b5124b89a89d7c0030e4fe28256a1df680538be"
registries = "General"
uuid = "2b935e18-93d6-40ac-ae34-fe54c09f00e1"
version = "0.6.3"

    [deps.SmallCollections.extensions]
    BangBangExt = "BangBang"
    StaticArraysExt = "StaticArrays"

    [deps.SmallCollections.weakdeps]
    BangBang = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortTileRecursiveTree]]
deps = ["AbstractTrees", "Extents", "GeoInterface"]
git-tree-sha1 = "f9aa6616a9b3bd01f93f27c010f1d25fc5a094a9"
registries = "General"
uuid = "746ee33f-1797-42c2-866d-db2fce69d14d"
version = "0.1.4"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "88ef0d98204515bc53dd4aaea5f7788882d1b66c"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
registries = "General"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
registries = "General"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "8a8a52f887f38d44014e24ee46b8a7311c625f6d"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.24"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools", "StyledStrings"]
git-tree-sha1 = "9e3a02d73a8f0a9be04e677dbef758aab40f8bea"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.6.1"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "4daaddbe37aa80b4551bf7c56adf53e7cfcdf35a"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.4"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    AdaptExt = "Adapt"
    GPUArraysCoreExt = "GPUArraysCore"
    KernelAbstractionsExt = "KernelAbstractions"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"
    StaticArraysExt = "StaticArrays"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "b04bfab37484defe300b2f9ed64469abab2431ff"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.12.0"

    [deps.StructUtils.extensions]
    StructUtilsDBInterfaceExt = ["DBInterface", "Tables"]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DBInterface = "a10d1c49-ce27-4219-8d33-6db1a4562965"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaylorSeries]]
deps = ["LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4a5ddc4036946d3a7900b5776b0872e36d848a29"
registries = "General"
uuid = "6aa5eb33-94cf-58f4-a9d0-e4b2c4fc25ea"
version = "0.20.10"

    [deps.TaylorSeries.extensions]
    TaylorSeriesIAExt = "IntervalArithmetic"
    TaylorSeriesJLD2Ext = "JLD2"
    TaylorSeriesRATExt = "RecursiveArrayTools"
    TaylorSeriesSAExt = "StaticArrays"

    [deps.TaylorSeries.weakdeps]
    IntervalArithmetic = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
registries = "General"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
registries = "General"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
registries = "General"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
registries = "General"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "eb3b85ca904a1d1647b15acc091b113b8874b648"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "1f0f9f401753701a7e4113b5056ca38d33875b55"
registries = "General"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.29.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "21b39bfb1fab6156b61fbcba4c86c57b6216d2c3"
registries = "General"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.2"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
registries = "General"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
registries = "General"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b826aedca5fce2e3c91904e338c279e5c1a2da74"
registries = "General"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.15.4+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "ce91b54a5f90c4f27df6950706234b34e49cf79e"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.14+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a54cf54677c9f6eac0d97e9d757c233b56bc21b9"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.13+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.2+0"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.cuBLAS]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "LLVM", "LinearAlgebra"]
git-tree-sha1 = "2ba506e5126e4b1326b6d0d75540efe1dac55cd7"
registries = "General"
uuid = "182d3088-87b7-4494-8cad-fc6afaa545bc"
version = "6.2.2"

    [deps.cuBLAS.extensions]
    EnzymeCoreExt = "EnzymeCore"

    [deps.cuBLAS.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.cuFFT]]
deps = ["AbstractFFTs", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "cc12ff818c71e2c3edd20b74daae897c02907703"
registries = "General"
uuid = "533571aa-0936-420e-b4be-9c66f5f626ca"
version = "6.2.2"

[[deps.cuRAND]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "Random", "Random123", "RandomNumbers"]
git-tree-sha1 = "958b553148093fa78fde00bfb95e26824847db42"
registries = "General"
uuid = "20fd9a0b-12d5-4c2f-a8af-7c34e9e60431"
version = "6.2.2"

[[deps.cuSOLVER]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "SparseArrays", "cuBLAS", "cuSPARSE"]
git-tree-sha1 = "1fedf6078bf590157cc03af6e11265e27204c595"
registries = "General"
uuid = "887afef0-6a32-4de5-add4-7827692ba8fc"
version = "6.2.2"

[[deps.cuSPARSE]]
deps = ["Adapt", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "KernelAbstractions", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "b971d958ec72453cce96b58ae38c91575805015a"
registries = "General"
uuid = "b26da814-b3bc-49ef-b0ee-c816305aa060"
version = "6.2.2"

    [deps.cuSPARSE.extensions]
    SparseMatricesCSRExt = "SparseMatricesCSR"

    [deps.cuSPARSE.weakdeps]
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
registries = "General"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
registries = "General"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libLLVM_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "Zlib_jll", "Zstd_jll"]
uuid = "8f36deef-c2a5-5394-99ed-8e07531fb29a"
version = "20.1.8+2"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "21e05cef1411bfe6b22a1ca8cbdc64a297d4b447"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdeflate_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "11c07205bc3b3b7833b8fb36beb4e4764d0126a7"
registries = "General"
uuid = "46979653-d7f6-5232-b59e-dd310c4598de"
version = "1.26.0+1"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "e067c8bae65bb40866552296a2d5b4dc65b8928f"
registries = "General"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.80.702+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "21cbc8fc2ac0bb182ed461e556d9c062ae6a3164"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.24.1+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "52d3b9475133c3bc8c0a7f90f18bfc8cdb5a443a"
registries = "General"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.70.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ed2d8d4ad96c03d3abe3382cdbbdea9489b63636"
registries = "General"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2023.1.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.5+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─ae9a3c8c-e3d7-4f6a-af26-91efa98b9db6
# ╟─3a691689-54cc-4bd9-a4e0-2159e8a2dd53
# ╟─2ce0dfa9-f800-4eb0-b809-49826e7eef76
# ╠═27e48d3d-859a-48bc-8324-00e73496a7ff
# ╟─53ee3e80-44ce-46d8-b680-f07e98b42055
# ╠═4621f04d-a10d-4a6e-9428-789a580cae3d
# ╟─7dcb171d-c8ef-4f57-aa51-51994eab1868
# ╠═8f261ecd-4258-4224-9157-a8b965cccdfe
# ╠═d759e126-1023-4519-8e91-62caf37c608f
# ╟─69ecde40-72d9-4d04-b70d-91c3efb14ce7
# ╟─bba74ad1-42f3-4f30-aa0a-36448f169dc0
# ╟─497984f7-5009-477f-adbf-cf80e6ab72fd
# ╟─0add26c5-b5ba-4282-93fc-da7d61de70d0
# ╟─a80dc238-4349-40e8-a4a0-9a5e76aaf61a
# ╠═80e5c2e7-6db2-4d80-bef1-c7632da09ed9
# ╟─5c15ae8f-bcfb-4f5d-bf28-b376328b7f6e
# ╟─b0e06ea4-3962-44c3-b250-1067d559454b
# ╟─05401c11-43ad-4c09-8a1e-7ea95d128c09
# ╠═84c481fd-6ad9-417b-877a-befc212cd4ae
# ╟─ab84cfbc-3718-4847-b220-c8869caaa98d
# ╟─81097ca3-5023-4387-8a24-3af2c7c643fc
# ╠═c4e14bdc-d063-4150-9a62-e48ecfb6787b
# ╠═43f051bb-1dbe-4f3d-b11f-94cde9e01815
# ╠═40f45db1-996f-4eb0-ab50-db6fa1c90fd0
# ╠═7611f1fd-8680-423c-b52a-001b24d0601c
# ╟─cbbb6280-4384-46d9-8e1f-1d7d3e5a26d1
# ╠═076f663d-281c-4c97-9f1d-f63ce011337f
# ╟─75ce0ac0-6a5c-4e8d-a1ea-84ef5598a2ef
# ╠═b499be47-5368-4c3b-9e19-3c9a8d786ac0
# ╟─f34fe12c-88e9-416a-9b7f-89c27f705696
# ╟─523a3e0a-df64-4772-ac00-b8cd94321626
# ╟─566e0aca-d2b0-4071-aaa1-061cd858c3bc
# ╟─3494a6b6-d761-4b1b-a398-891bbec22c5a
# ╟─00226ea4-663e-4034-878d-2401028768dc
# ╟─1ae2ed62-a8e9-4ae6-a5b6-dcf9f537e273
# ╠═5c675a7b-7396-4b32-8296-3db55ba1dc75
# ╠═b30b3f25-170f-42fa-9803-3f4b4d4ba965
# ╟─51d43b83-7abf-48c1-9f2f-029134078c8f
# ╟─1bf163ef-ab3b-450f-9902-80c9ca82945e
# ╟─3d02997c-1fb7-4b7f-ba10-396ff7f83bc3
# ╟─511f6251-8445-4484-9d89-e94d102c8940
# ╟─27a20176-a359-4a41-8385-5a098885489a
# ╟─bb926df3-cfcd-4a75-a603-42628aa7cece
# ╟─43c1aadb-311a-47f3-b4b8-bb4d7fc77c4f
# ╟─f6d14578-452b-4f8f-add9-1f89ed8b22e9
# ╟─59869cca-4917-4757-aa13-e62ff9330efd
# ╟─61c05b91-c56d-4373-927e-4da6881ca2be
# ╟─7b1a6aed-13de-4739-ae0c-522ac9e0d3fe
# ╟─6b549874-6cc2-4078-ad99-83af21f3df0d
# ╟─652c59a9-0fb1-4d74-bf75-cf6688ee45fe
# ╟─3ae7b80b-b2ac-481b-9d3d-aeda8f6cc9e6
# ╠═cd2ec92c-6e1f-462a-b585-2598b3c42155
# ╠═880d45fc-e31a-4367-8779-f612ce736fe8
# ╠═c05b017e-4779-489a-962d-91926577e07f
# ╠═da1dfe12-48ef-405e-9000-b661203a2a05
# ╠═83b1054a-d95a-4bca-a219-c1eaed0ae83a
# ╠═9d223bad-ca93-40b1-b021-d83064222dbe
# ╠═16adfbd5-1653-496d-b6fc-ca10d8417b48
# ╟─91623e61-db88-4a1e-b7e6-25c23b3e1260
# ╟─28b9a1be-96ca-40f7-8bc2-4e9cf3994fe6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
