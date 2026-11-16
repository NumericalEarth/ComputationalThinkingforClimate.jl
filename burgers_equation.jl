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

# ╔═╡ ae9a3c8c-e3d7-4f6a-af26-91efa98b9db6
begin
    using CairoMakie, PlutoUI, LinearAlgebra, NCDatasets
    using PlutoUI: Slider
end

# ╔═╡ 91623e61-db88-4a1e-b7e6-25c23b3e1260
html"""
<div style="
position: absolute;
width: calc(100% - 30px);
border: 50vw solid #282936;
border-top: 200px solid #282936;
border-bottom: none;
box-sizing: content-box;
left: calc(-50vw + 15px);
top: -200px;
height: 200px;
pointer-events: none;
"></div>

<div style="
height: 200px;
width: 100%;
background: #282936;
color: #fff;
padding-top: 68px;
">
<span style="
font-family: Vollkorn, serif;
font-weight: 700;
font-feature-settings: 'lnum', 'pnum';
">
<p style="text-align: center; font-size: 2rem;">
<em> Simulating the fluid dynamics of the Earth </em>
</p>

<style>
body {
overflow-x: hidden;
}
</style>"""

# ╔═╡ 3a691689-54cc-4bd9-a4e0-2159e8a2dd53
TableOfContents(title = "📚 Table of Contents", indent = true, depth = 3, aside = true)

# ╔═╡ cbbb6280-4384-46d9-8e1f-1d7d3e5a26d1
md"""
In the previous lesson, we discussed aspects of how the radiative processes that drive Earth's weather and climate could be modeled, and how they affect Earth's temperature. However, the thermodynamics are only one half of the full picture. As the atmophere and ocean are fluids, they flow around the globe as well, distributing various tracers across the globe with their motions.

In this lesson, we will explore the fluid dynamical aspects of the Earth, and illustrate some of the considerations a computational fluid dynamicist thinks about when trying to model such processes. 
"""

# ╔═╡ 2ce0dfa9-f800-4eb0-b809-49826e7eef76
md"""
## A planet in motion
"""

# ╔═╡ 27e48d3d-859a-48bc-8324-00e73496a7ff
html"""
<iframe style="width: 100%; aspect-ratio: 16 / 9;" src="https://www.youtube-nocookie.com/embed/Ei8x3y-Ddq0?vq=hd2160" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
"""

# ╔═╡ 4621f04d-a10d-4a6e-9428-789a580cae3d
md"""
The video above shows a high-resolution simulation of Earth's atmosphere, ocean, and clouds, using 100% Julia! Below are some of the packages that are used in building and running the models:

- Atmosphere: [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
- Ocean: [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)
- Atmosphere--Ocean coupling: [NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl)
- Cloud Microphysics: [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl)
- Radiative Transfer: [NumericalRadiation.jl](https://github.com/NumericalEarth/NumericalRadiation.jl)

What's behind a simulation like this? Below we will explore some of the ingredients which are required to be able to achieve this.
"""

# ╔═╡ a80dc238-4349-40e8-a4a0-9a5e76aaf61a
md"""
## Energy balance of the Earth

In the previous lesson, we approximated the energy equilibrium of the Earth due to solar forcing with a _globally averaged_ temperature. Averaged over a year, however, the tropics absorb more sunlight than the extratropics, primarily because the Sun's rays hit the surface at a steeper angle in the tropics.
"""

# ╔═╡ 80e5c2e7-6db2-4d80-bef1-c7632da09ed9
sunlight = let
    S₀ = 1366 / 4          # average sunlight, from Monday [W m⁻²]
    α = 0.3                # albedo
    σ = 5.670374419e-8     # Stefan-Boltzmann constant [W m⁻² K⁻⁴]
    ε = 0.745              # emissivity of the atmosphere
    φ = -90:90
    P₂(s) = (3s^2 - 1) / 2
    absorbed = @. (1 - α) * S₀ * (1 - 0.477 * P₂(sind(φ)))
    local_balance(S) = (S / ((1 - ε / 2) * σ))^(1 / 4) - 273.15
    (; φ, absorbed, mean_absorbed = (1 - α) * S₀, T = local_balance.(absorbed))
end;

# ╔═╡ 5c15ae8f-bcfb-4f5d-bf28-b376328b7f6e
let (; φ, absorbed, mean_absorbed, T) = sunlight
    # observed surface temperature, 1961-1990 (Jones et al., 1999)
    observed = NCDataset(download("https://crudata.uea.ac.uk/cru/data/temperature/absolute.nc")) do ds
        monthly = ds["tem"][:, :, :]   # longitude × latitude × month [°C]
        (; φ = ds["lat"][:], T = vec(sum(monthly, dims = (1, 3))) / (size(monthly, 1) * size(monthly, 3)))
    end

    fig = Figure(size = (900, 360))
    ax1 = Axis(fig[1, 1], xlabel = "latitude (°)", ylabel = "W m⁻²", xticks = -90:30:90,
               title = "Sunlight absorbed, annual mean")
    band!(ax1, φ, fill(mean_absorbed, length(φ)), max.(absorbed, mean_absorbed), color = (:firebrick, 0.3))
    band!(ax1, φ, min.(absorbed, mean_absorbed), fill(mean_absorbed, length(φ)), color = (:royalblue, 0.3))
    lines!(ax1, φ, absorbed, color = :black, linewidth = 2)
    hlines!(ax1, mean_absorbed, color = :gray40, linestyle = :dash)

    ax2 = Axis(fig[1, 2], xlabel = "latitude (°)", ylabel = "°C", xticks = -90:30:90,
               title = "Surface temperature, annual mean")
    lines!(ax2, φ, T, color = :black, linewidth = 2, label = "local radiative equilibrium")
    lines!(ax2, observed.φ, observed.T, color = :firebrick, linewidth = 2, label = "observed, 1961–1990")
    axislegend(ax2, position = :cb)
    fig
end

# ╔═╡ b0e06ea4-3962-44c3-b250-1067d559454b
md"""
The figures above shows the zonally-averaged absorbed sunlight (left) and the annual mean surface temperature respectively. They are obtained by doing a similar radiative balance calculation where absorbed sunlight equals to outgoing radiation as the previous lesson to obtain the equilibrium.

In the left panel, the black line shows the sunlight absorbed at different latitudes computed with pure thermodynamic considerations. The dotted line shows the globally averaged energy balance due to solar radiation. The red regions in the tropics indicate regions where there is energy surplus with respect to the global average, while the blue regions show the energy deficit in higher latitudes.

The black line in the right panel shows the _local radiative equilibrium_, which is the equilibrium temperature of the Earth if Earth's energy balance across latitudes is only driven by radiative balance. In this calculation we have assumed that albedo and atmospheric composition is the same everywhere on Earth. The results indicate that the Equator would sit near $(round(Int, maximum(sunlight.T))) °C and the poles near $(round(Int, minimum(sunlight.T))) °C.

However, we know that to not be the case! The observed contrast is shown in the red line ([Jones et al., 1999](https://doi.org/10.1029/1999RG900002)). From this, we see that the observations suggest a smaller temperature contrast across latitudes everywhere except over regions of Antarctica which is fully covered by thick ice sheets.

How do we explain such a contrast? The other ingredient that determines the energy balance of the Earth system is the fluid dynamics of the atmosphere and ocean, which can be seen in the opening video. Atmospheric and oceanic fluid dynamics reduce the temperature contrast by carrying heat from the tropics toward the poles. In the mid-latitudes they carry more than 5 PW (1 PW = 10¹⁵ W), most of it in the atmosphere ([Trenberth & Caron, 2001](https://doi.org/10.1175/1520-0442(2001)014<3433:EOMAAO>2.0.CO;2)). All the power plants in the world produced about 32,000 TWh of electricity in 2025 ([Ember, 2026](https://ember-energy.org/latest-insights/global-electricity-review-2026/electricity-demand-and-supply-trends/)), an average of 3.6 TW, which is about 1400 times less.

In many ways, the Earth system is analogous to a heat engine ([Lorenz, 1955](https://doi.org/10.3402/tellusa.v7i2.8796); [Barry et al., 2002](https://doi.org/10.1038/415774a)). Sunlight heats the tropics more than the poles, and the atmosphere converts part of this heat into the kinetic energy of the winds, creating the weather that we are familiar with. A hurricane is a heat engine of its own: it takes heat from the warm ocean surface and gives off heat at the cold top of the storm ([Emanuel, 1986](https://doi.org/10.1175/1520-0469(1986)043<0585:AASITF>2.0.CO;2); [Emanuel, 2006](https://doi.org/10.1063/1.2349743)). Similarly, the ocean is also a heat engine, albeit a much poorer one, because it is heated and cooled at the same surface level. In contrast, most of the energy that drives its currents comes from the winds and the tides ([Wunsch & Ferrari, 2004](https://doi.org/10.1146/annurev.fluid.36.050802.122121)).

Now the question is: How does a fluid behave? What governs the behavior of all fluids, including the atmosphere and the ocean? Below we will begin to explore the rich world of fluid mechanics and how to simulate them!
"""

# ╔═╡ cb9899d1-0837-42b3-88fd-6090fb831e6f
md"""
## Newton's laws for fluids: Navier--Stokes equations

Newton's laws apply themselves to all classical and nonrelativistic matter, including fluids.
The now infamous (Euler--)Navier--Stokes(--Poisson) equations are no more than Newton's 2nd Law applied on fluids.

Take a small parcel of fluid such as air or water. It obeys ``F = ma`` like any other object. Dividing both sides of the equation by its mass (or density) we obtain an expression for its acceleration. The acceleration of a parcel is caused by various forces that act on it. For a fluid parcel in a rotating frame such as ours, some of the forces it experiences are:

$\begin{align}
\text{\color{brown}{acceleration of a parcel}} = & + \text{\color{blue}{pressure force}} \newline
& + \text{\color{red}{buoyancy force}} \newline
& + \text{\color{purple}{Coriolis force}} \newline
& + \text{\color{gray}{viscosity (internal friction)}}
\end{align}$

Newton's second law in this form holds only in an inertial frame, one that does not accelerate or rotate. We measure winds and currents on the rotating Earth, which is a non-inertial frame, so the equation needs additional *fictitious* forces such as the Coriolis and centrifugal forces. The centrifugal force only changes the effective gravity slightly, so for Earth-like planets it is generally negligible.

Mathematically, a variant of the Navier--Stokes equations appropriate for describing incompressible geophysical fluids (such as the ocean) is known as the _rotating Boussinesq equations_. In a coordinate system that rotates with the Earth, at each point in space and time, such a fluid parcel can be described by its velocity vector field ``\boldsymbol{u} = (u, v, w)`` together with its buoyancy ``b``,
```math
b = -g\,\frac{\rho - \rho_0}{\rho_0},
```
which measures how much lighter the fluid is than a reference density ``\rho_0``. Fluid with ``b > 0`` is lighter than the reference and rises.

```math
\begin{align}
\cssId{eq-momentum}{\underbrace{\frac{\partial \boldsymbol{u}}{\partial t} + (\boldsymbol{u}\cdot\nabla)\boldsymbol{u}}_{\text{acceleration}}}
&= \underbrace{-\nabla p}_{\text{pressure}}
+ \underbrace{b\,\hat{\boldsymbol{z}}}_{\text{buoyancy}}
\underbrace{-\,f\,\hat{\boldsymbol{z}}\times\boldsymbol{u}}_{\text{Coriolis}}
+ \underbrace{\nu\nabla^2\boldsymbol{u}}_{\text{friction}}, \tag{1} \newline
\cssId{eq-buoyancy}{\frac{\partial b}{\partial t} + (\boldsymbol{u}\cdot\nabla) b} &= \kappa\nabla^2 b, \tag{2} \newline
\cssId{eq-incompressibility}{\underbrace{\nabla\cdot\boldsymbol{u}}_{\text{incompressibility}}} &= 0. \tag{3}
\end{align}
```

Here ``p`` is the pressure divided by ``\rho_0``, ``f`` is the Coriolis parameter, ``\nu`` is the viscosity and ``\kappa`` is the diffusivity of buoyancy.


| Term | Name | What it does |
|:-----|:-----|:-------------|
| ``\partial \boldsymbol{u}/\partial t + (\boldsymbol{u}\cdot\nabla)\boldsymbol{u}`` | acceleration | the "``a``" of ``F = ma`` |
| ``-\nabla p`` | pressure gradient | pushes fluid from high pressure to low pressure, which drives the winds |
| ``b\,\hat{\boldsymbol{z}}`` | buoyancy | light (warm) fluid rises and heavy (cold) fluid sinks, as in the rising air under small clouds |
| ``-f\,\hat{\boldsymbol{z}}\times\boldsymbol{u}`` | Coriolis | turns moving fluid to the right of its direction of motion in the Northern Hemisphere and to the left in the Southern Hemisphere, and makes storms spin |
| ``\nu\nabla^2\boldsymbol{u}`` | friction | smooths out sharp differences in velocity, at scales of about a millimeter in air |
| ``\kappa\nabla^2 b`` | diffusion | smooths out sharp differences in buoyancy |


You might be wondering: why does acceleration look like ``\partial \boldsymbol{u}/\partial t + (\boldsymbol{u}\cdot\nabla)\boldsymbol{u}``? Why do we need ``(\boldsymbol{u}\cdot\nabla)\boldsymbol{u}``?
"""

# ╔═╡ 4f5cf0d4-2ea8-4e62-841d-1b09111769be
md"""
### Acceleration: the material derivative of a moving parcel

The _Eulerian_ velocity field ``\boldsymbol{u}(\boldsymbol{x}, t)`` described above gives the velocity at a fixed location ``\boldsymbol{x}``. 

To see where the two terms of the acceleration come from, take one dimension and follow a parcel. Its velocity is the field at its own position, and its acceleration is the rate at which that velocity changes. In a short time ``\Delta t``, a parcel at ``x`` with velocity ``u`` travels a distance ``\Delta x = u\,\Delta t``, and its velocity changes by

```math
\Delta u = u(x + \Delta x,\ t + \Delta t) - u(x, t) \approx \frac{\partial u}{\partial t}\,\Delta t + \frac{\partial u}{\partial x}\,\Delta x .
```

Divide by ``\Delta t``, use ``\Delta x = u\,\Delta t``, and let ``\Delta t \to 0``:

```math
\frac{Du}{Dt} \equiv \lim_{\Delta t \to 0} \frac{\Delta u}{\Delta t} = \frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}.
```

``D/Dt`` is called the **material derivative**, which describes the rate of change following the material as it travels. Below we illustrate why this is with a schematic.
"""

# ╔═╡ 829221ca-5325-4e0b-9d65-bce670004840
let
    blue, orange = Makie.wong_colors()[1:2]
    halfwidth(x) = 1 - 0.55 * (1 + tanh(1.2 * (x - 2))) / 2
    speed(x) = 0.5 / halfwidth(x)
    xs = range(0, 4, length = 200)
    x₁, x₂ = 1.0, 2.3

    fig = Figure(size = (1000, 400))
    ax1 = Axis(fig[1, 1], aspect = DataAspect(), title = "Steady flow through a narrowing channel",
               limits = (0, 4, -1.2, 1.2))
    hidedecorations!(ax1)
    hidespines!(ax1)
    band!(ax1, xs, halfwidth.(xs), fill(1.2, 200), color = :gray85)
    band!(ax1, xs, fill(-1.2, 200), -halfwidth.(xs), color = :gray85)
    lines!(ax1, xs, halfwidth.(xs), color = :black, linewidth = 2)
    lines!(ax1, xs, -halfwidth.(xs), color = :black, linewidth = 2)
    for xa in (0.1, 3.45), s in (-0.55, 0, 0.55)
        arrows2d!(ax1, [xa], [s * halfwidth(xa)], [0.42 * speed(xa)], [0], color = :gray55,
                  shaftwidth = 2.5, tipwidth = 9, tiplength = 9)
    end
    lines!(ax1, [x₁, x₂], [0, 0], color = :gray40, linestyle = :dash)
    arrows2d!(ax1, [x₁, x₂], [0, 0], 1.1 .* speed.([x₁, x₂]), [0, 0], color = orange,
              shaftwidth = 4, tipwidth = 13, tiplength = 12)
    scatter!(ax1, [x₁, x₂], [0, 0], markersize = 26, color = [:lightskyblue, blue], strokecolor = :black, strokewidth = 1.5)
    text!(ax1, x₁ + 0.1, 0.09, text = L"u(x,\, t)", align = (:left, :bottom), color = orange, fontsize = 17)
    text!(ax1, x₂ + 0.1, 0.09, text = L"u(x + \Delta x,\ t + \Delta t)", align = (:left, :bottom), color = orange, fontsize = 17)
    text!(ax1, x₁, -0.13, text = L"\text{time } t", align = (:center, :top), fontsize = 16)
    text!(ax1, x₂, -0.13, text = L"\text{time } t + \Delta t", align = (:center, :top), fontsize = 16)
    bracket!(ax1, x₁, -0.36, x₂, -0.36, text = L"\Delta x = u\,\Delta t", orientation = :down, fontsize = 17)

    ax2 = Axis(fig[1, 2], limits = (0, 4, 0.4, 3.2),
               xticks = ([1.6, 3.4], [L"x", L"x + \Delta x"]), yticks = ([1, 2.4], [L"t", L"t + \Delta t"]),
               xticklabelsize = 18, yticklabelsize = 18, xgridvisible = false, ygridvisible = false)
    lines!(ax2, [1.6, 1.6, 3.4], [1, 2.4, 2.4], color = :gray40, linestyle = :dash)
    arrows2d!(ax2, [1.6], [1], [1.8], [1.4], color = blue, shaftwidth = 3, tipwidth = 12, tiplength = 12)
    scatter!(ax2, [1.6, 3.4], [1, 2.4], markersize = 20, color = [:lightskyblue, blue], strokecolor = :black, strokewidth = 1.5)
    text!(ax2, 1.5, 1.7, text = L"\frac{\partial u}{\partial t}\,\Delta t", align = (:right, :center), fontsize = 18)
    text!(ax2, 2.5, 2.5, text = L"\frac{\partial u}{\partial x}\,\Delta x", align = (:center, :bottom), fontsize = 18)
    text!(ax2, 2.65, 1.55, text = "path of the parcel", align = (:left, :top), color = blue, rotation = 0.68)
    colsize!(fig.layout, 1, Relative(0.62))
    fig
end

# ╔═╡ 0d0ce3b4-b845-4f7f-bbb4-9a8ea9f296ed
md"""

In the material derivative, the first term ``\partial u/\partial t`` is the rate at which the velocity changes at a fixed location. The second term ``u\,\partial u/\partial x`` is the rate at which the parcel's velocity changes as it moves to a place where the velocity is different. It is called the _advection_ term, and in three dimensions it becomes ``(\boldsymbol{u}\cdot\nabla)\boldsymbol{u}``.

In the channel example above, the flow is steady, i.e., at any location ``x`` the velocity never changes, so ``\partial u/\partial t = 0`` everywhere. The parcel still speeds up as it moves into the narrow part of the channel, where the flow is faster. All of its acceleration comes from the advection term.

The advection term is the product of ``u`` with the gradient of ``u``, which makes the equations nonlinear. Much of the difficulty of proving, understanding, and simulating the properties of the Navier--Stokes equations lie can be attributed to this innocent-looking term. It is also responsible for moving energy across a wide range of length scales, which is the source of fluid turbulence.

Trying to capture the turbulent effects created by the advection term in various scenarios e.g. over an airfoil, around a building, in the ocean without explicitly resolving them when computing the Navier--Stokes equations is known as turbulence modeling. Turbulence modeling is a rich and ongoing topic of modern research, with broad applications in various fields such as aerospace engineering, earth system modeling, and astrophysics. Independent of whether the Navier--Stokes Millennium problem is solved, this _turbulence closure problem_ remains one of the most challenging frontiers in all of physics and engineering!
"""

# ╔═╡ 05401c11-43ad-4c09-8a1e-7ea95d128c09
md"""
### Shock formation

Let's now look at a scenario where there are no (external or internal) forces in the system. Therefore, we only have the material derivative

```math
\cssId{eq-material}{\frac{Du}{Dt} = 0 .} \tag{4}
```

Each parcel keeps the velocity it starts with. Write the initial velocity as ``u(x, 0) = u_0(x)``. The parcel that starts at ``x_0`` with velocity ``u_0(x_0)`` is later at

```math
x(t) = x_0 + u_0(x_0)\,t .
```

Let's initialize the flow with an initial velocity ``u_0(x)`` of

```math
u_0(x) = 1 + \tfrac{1}{2}\sin x, \qquad 0 \le x < 2\pi .
```
Under this initialization, there are some locations where the flow is locally converging/diverging due to the velocity differences.

We use a periodic domain where a parcel that leaves at ``x = 2\pi`` comes back in at ``x = 0``.
"""

# ╔═╡ 84c481fd-6ad9-417b-877a-befc212cd4ae
u₀(x) = 1 + sin(x) / 2

# ╔═╡ c4e14bdc-d063-4150-9a62-e48ecfb6787b
md"""
**Time:** $(@bind time_parcels Slider(0:0.1:3, default = 0, show_value = true))
"""

# ╔═╡ 43f051bb-1dbe-4f3d-b11f-94cde9e01815
let t = time_parcels
    fig = Figure(size = (800, 620))
    ax1 = Axis(fig[1, 1], ylabel = "time", title = "Paths of the parcels (color: velocity)", limits = (0, 2π, 0, 3),
               xticks = 0:π/2:2π)
    for x₀ in range(0, 2π, length = 49)[1:end-1], shift in (0, -2π)
        lines!(ax1, [x₀, x₀ + 3u₀(x₀)] .+ shift, [0, 3], color = fill(u₀(x₀), 2), colormap = :viridis,
               colorrange = (0.5, 1.5), linewidth = 1)
    end
    hlines!(ax1, t, color = :black, linewidth = 2)
    hlines!(ax1, 2, color = :gray40, linestyle = :dot)
    text!(ax1, 0.1, 2.03, text = "paths first cross at t = 2", color = :gray30, fontsize = 12)
    hidexdecorations!(ax1, grid = false)

    ax2 = Axis(fig[2, 1], xlabel = "x", ylabel = "u", title = "Velocity of the parcels at t = $t", limits = (0, 2π, 0.3, 1.7),
               xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]))
    if t > 2
        fold = acos(2 / t)
        edges = sort([x₀ + u₀(x₀) * t for x₀ in (π - fold, π + fold)])
        for shift in (0, -2π)
            vspan!(ax2, edges[1] + shift, edges[2] + shift, color = (:firebrick, 0.2))
        end
        text!(ax2, mod(edges[1], 2π) - 0.15, 0.9, text = "three velocities\nat one place", align = (:right, :center), color = :firebrick)
    end
    x₀ = range(0, 2π, length = 400)
    parcels = range(0, 2π, length = 49)[1:end-1]
    for shift in (0, -2π)
        lines!(ax2, x₀ .+ u₀.(x₀) .* t .+ shift, u₀.(x₀), color = :black, linewidth = 2)
        scatter!(ax2, parcels .+ u₀.(parcels) .* t .+ shift, u₀.(parcels), color = u₀.(parcels), colormap = :viridis,
                 colorrange = (0.5, 1.5), markersize = 9, strokecolor = :black, strokewidth = 0.5)
    end
    linkxaxes!(ax1, ax2)
    rowsize!(fig.layout, 1, Relative(0.45))
    fig
end

# ╔═╡ 7611f1fd-8680-423c-b52a-001b24d0601c
md"""
As you move the slider, the fast parcels (yellow) catch up with the slow parcels (purple) ahead of them, and the front of the wave gets steeper. At ``t = 2`` the front is vertical. For ``t > 2``, the solution ``u`` is not unique for all locations, giving multiple solutions at one place.

This is unphysical as a real fluid can only have one local velocity per location. Realistically, the fast fluid piles into the slow fluid instead, creating a sharp jump, called a **shock**. In compressible fluids, shocks (such as a sonic boom) can form and propagate. Many phenomena such as breaking waves and traffic jams can also be explained and described in the same way.
"""

# ╔═╡ 076f663d-281c-4c97-9f1d-f63ce011337f
md"""
## Numerical methods for fluid dynamics

Let's now begin to see how to build certain components of a fluid dynamical solver from scratch. There are 5 equations in [(1)](#eq-momentum), [(2)](#eq-buoyancy), and [(3)](#eq-incompressibility) which contain 5 variables ``(u, v, w, p, b)``, thus we are able to solve the system! However, the incompressibility equation [(3)](#eq-incompressibility) requires a separate treatment from Equations [(1)](#eq-momentum) and [(2)](#eq-buoyancy) since it does not explicitly depend on time. In many popular methods, this is treated by using a fractional step method ([Chorin, 1968](https://doi.org/10.1090/S0025-5718-1968-0242392-2); [Kim & Moin, 1985](https://doi.org/10.1016/0021-9991(85)90148-2)) where Equations [(1)](#eq-momentum) and [(2)](#eq-buoyancy) are time-stepped first before solving for the diagnostic constraint in [(3)](#eq-incompressibility). 

For simplicity, we will not be looking at the treatment of Equation [(3)](#eq-incompressibility) today. We will also further simplify the Boussinesq equations by omitting the buoyancy equation, solving for a system with a homogeneous "fluid". We will also remove effects of viscosity so that the system is inviscid. We also reduce the system down to one dimension and consider an irrotational system with no Coriolis term. With these simplifications, the Navier--Stokes equations reduce to the **Burgers equation** which are a simplified form of the Navier--Stokes equations without the pressure term and the incompressibility criteria. Nonetheless, many important considerations for constructing a fluid solver can be demonstrated with the Burgers equation as well.
"""

# ╔═╡ 96cef172-fb67-4e7b-a35a-8c3cf693b099
md"""
### Inviscid Burgers equation

You have already seen the Burgers equation [(4)](#eq-material) in the shock formation example. Expanding it in full, the inviscid Burgers' equation is 
```math
\cssId{eq-burgers}{\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} = 0 .} \tag{5}
```

Now we will try to solve this equation using the _finite volume method_, which approximates the continuous equation with a discrete grid where the domain is divided into small cells and each cell stores the average of ``u`` over that cell, rather than its value at a single point. 

In general, finite volume works with equations in the _flux form_, where the cell averages change only through the fluxes across cell faces, so whatever leaves one cell enters its neighbor.

In the flux form, the inviscid Burgers equation can be written as 
```math
\cssId{eq-burgers-flux}{\frac{\partial u}{\partial t} + \frac{\partial F}{\partial x} = 0, \qquad F = \frac{u^2}{2},} \tag{6}
```
where we have used the chain rule ``u\,\partial u/\partial x = \partial(u^2/2)/\partial x`` to write the equation in terms of the flux ``F``.

``F`` is rate at which ``u`` is carried past a point. If we integrate the equation from ``x = A`` to ``x = B``, we see that
```math
\frac{d}{dt}\int_A^B u\,dx = F(A) - F(B),
```
where the amount of ``u`` between ``A`` and ``B`` changes only by what flows in at one end and out at the other.
"""

# ╔═╡ 75ce0ac0-6a5c-4e8d-a1ea-84ef5598a2ef
md"""
### Linear advection

Let's start with an even simpler problem first and come back to Burgers later. Suppose the wind is blowing at constant speed ``a`` and it carries a passive tracer along, such as smoke with concentration ``c(x, t)``:

```math
\cssId{eq-linear-advection}{\frac{\partial c}{\partial t} + \frac{\partial F}{\partial x} = 0, \qquad F = a\,c .} \tag{7}
```

This is the _linear advection equation_. This is a useful system to study because its exact solution is known: the initial shape moves at speed ``a`` without change, ``c(x, t) = c_0(x - a t)`` (you can check this by substituting it into Equation [(7)](#eq-linear-advection)). We can use this analytical solution to compare with our numerical results.
"""

# ╔═╡ b499be47-5368-4c3b-9e19-3c9a8d786ac0
md"""
### Spatial discretization on a grid

To solve continuous equations numerically, we have to discretize the system. Various discretization strategies form the basis of various methods, such as finite difference, finite element, finite volume, spectral element, boundary element.
In finite volume, we cut the domain into ``N`` boxes, called **cells**, of width ``\Delta x``, and store one number for each cell: ``c_j``, the average of ``c`` in cell ``j``.

**Number of cells:** $(@bind cell_count Slider(5:5:100, default = 10, show_value = true))
"""

# ╔═╡ f34fe12c-88e9-416a-9b7f-89c27f705696
let N = cell_count
    blue, orange = Makie.wong_colors()[1:2]
    smooth(x) = 0.55 + 0.3 * sin(2π * x) + 0.12 * sin(6π * x + 1)
    Δx = 1 / N
    centers = Δx/2:Δx:1
    averages = [sum(smooth, range(xc - Δx / 2, xc + Δx / 2, length = 50)) / 50 for xc in centers]

    fig = Figure(size = (800, 410))
    ax = Axis(fig[1, 1], xlabel = "x", limits = (0, 1, 0, 1.15), title = "A function and its cell averages")
    barplot!(ax, centers, averages, gap = 0, color = (blue, 0.25), strokecolor = (blue, 0.8), strokewidth = 1)
    lines!(ax, 0:0.002:1, smooth, color = :gray30, linewidth = 2)
    if N ≤ 25
        j = N ÷ 2
        scatter!(ax, centers[j-1:j+1], averages[j-1:j+1], color = blue, markersize = 10)
        for (k, label) in zip(j-1:j+1, (L"c_{j-1}", L"c_j", L"c_{j+1}"))
            text!(ax, centers[k], averages[k] + 0.03, text = label, align = (:center, :bottom), fontsize = 18)
        end
        faces = [centers[j] - Δx / 2, centers[j] + Δx / 2]
        vlines!(ax, faces, color = :black, linewidth = 2)
        arrows2d!(ax, faces .- 0.3Δx, [0.3, 0.3], [0.6Δx, 0.6Δx], [0, 0], color = :red,
                  shaftwidth = 4, tipwidth = 13, tiplength = 12)
        text!(ax, faces[1] - 0.35Δx, 0.3, text = L"F_{j-1/2}", align = (:right, :center), color = :red, fontsize = 18)
        text!(ax, faces[2] + 0.35Δx, 0.3, text = L"F_{j+1/2}", align = (:left, :center), color = :red, fontsize = 18)
        bracket!(ax, faces[1], 0.18, faces[2], 0.18, text = L"\Delta x", orientation = :down, fontsize = 16)
    end
    fig
end

# ╔═╡ 523a3e0a-df64-4772-ac00-b8cd94321626
md"""
To find how ``c_j`` changes in time, apply the flux form to cell ``j``. The amount in the cell is ``c_j\,\Delta x``, and it changes only by what flows through the two **faces** of the cell:

```math
\Delta x\,\frac{d c_j}{dt} = F_{j-1/2} - F_{j+1/2} .
```

``F_{j-1/2}`` is the flux through the left face and ``F_{j+1/2}`` the flux through the right face.

No approximation has been made so far, and the equation has a useful property. The flux that leaves cell ``j`` through its right face enters cell ``j+1`` through its left face, so when you add up all the cells every flux cancels and the total amount ``\sum_j c_j\,\Delta x`` is conserved exactly. This is why the finite volume method is able to conserve ``c`` by construction. This is a useful property for many fluid applications where we do not want tracers in the fluid to leak out of the system, especially when ocean models are usually run for thousands of simulation years!

The mathematical approximation we are making here enters via the fluxes. We store averages in cells, but the flux needs the value of ``c`` at a face. How do we estimate that?
"""

# ╔═╡ 566e0aca-d2b0-4071-aaa1-061cd858c3bc
md"""
### Centered flux

An obvious estimate for the value at a face is the average of the two cells next to it:

```math
c_{j+1/2} = \frac{c_j + c_{j+1}}{2}, \qquad F_{j+1/2} = a\,\frac{c_j + c_{j+1}}{2} .
```
"""

# ╔═╡ 3494a6b6-d761-4b1b-a398-891bbec22c5a
begin
    struct LinearAdvection{FT}
        a :: FT   # wind speed
    end

    flux(eq::LinearAdvection, c) = eq.a * c
end

# ╔═╡ 00226ea4-663e-4034-878d-2401028768dc
md"""
The domain is periodic, so cell ``N + 1`` is cell 1 and cell 0 is cell ``N``.
"""

# ╔═╡ d4db5136-89ef-4b01-8f65-2d92ce74adeb
periodic(c, j) = c[mod1(j, length(c))]

# ╔═╡ 1ae2ed62-a8e9-4ae6-a5b6-dcf9f537e273
md"""
A way to compute the flux at a face is also a type. `face_flux(eq, c, j, scheme)` returns ``F_{j+1/2}``, the flux through the right face of cell ``j``.
"""

# ╔═╡ 5430b003-fe8c-4e43-bce8-a9423705ef70
md"""
The **tendency** is the rate of change of every cell, which is just the right hand side of the equation. `tendency!` writes it into an array `dc` that we allocate once and reuse at every step. By Julia convention, a `!` at the end of a function name means that the function modifies one of its arguments.
"""

# ╔═╡ 511f6251-8445-4484-9d89-e94d102c8940
md"""
Dividing the flux balance for cell ``j`` by ``\Delta x`` gives

```math
\frac{dc_j}{dt} = -\,\frac{F_{j+1/2} - F_{j-1/2}}{\Delta x} .
```

With the centered flux ``F_{j+1/2} = a\,(c_j + c_{j+1})/2``, we get
```math
\frac{dc_j}{dt} = -\,a\,\frac{c_{j+1} - c_{j-1}}{2\,\Delta x}, \qquad j = 1, \dots, N,
```
with ``c_0 = c_N`` and ``c_{N+1} = c_1`` since the domain is periodic. We note from this system of equations that the only derivative left is in time. The right-hand side is just arithmetic on the numbers ``c_1, \dots, c_N``. Therefore, this is a set of ``N`` coupled ordinary differential equations, where each ``c_j`` equation is coupled to the ``c_{j+1}`` and ``c_{j-1}`` equations.

Using the finite volume method, we have now turned a partial differential equation into a system of ordinary differential equations. This is usually helpful because we can now use an ODE solver to discretize the time dimension and complete the procedure.
"""

# ╔═╡ 86bd1203-b030-4f70-b9f4-601af8e96e42
md"""
### Solving ODEs

For notation simplicity in ODE analysis, let's ignore the spatial indices for a brief moment, and write the ODE(s) as
```math
\frac{dc}{dt} = G(c),
```
where ``c`` is the list of cell averages and ``G`` is the tendency.

The simplest way to solve an ODE is **forward Euler**. We use the superscript ``n`` to index time, so ``c^n`` is the list of cell averages at time ``t^n = n\,\Delta t``. Then we approximate the time derivative by the change over one step:

```math
\frac{dc}{dt} \approx \frac{c^{n+1} - c^{n}}{\Delta t} .
```

Setting this equal to the tendency evaluated at the current time, ``G(c^n)``, and solving for ``c^{n+1}`` gives

```math
c^{n+1} = c^{n} + \Delta t\,G(c^{n}) .
```
Using this equation, for a given ``c^n``, we would be able to obtain ``c^{n+1}`` by substituting the right hand side. Doing this iteratively allows us to step the equations forward in time given initial condition ``c^0``. Thus, we can now solve the equation numerically!
"""

# ╔═╡ 1889409d-d5d5-45cb-8386-1a525acd3e90
begin
    struct ForwardEuler end

    allocate_workspace(::ForwardEuler, c) = (; dc = similar(c))

    function advance!(c, Δt, G!, ::ForwardEuler, workspace)
        (; dc) = workspace
        G!(dc, c)
        @. c += Δt * dc
        return c
    end
end

# ╔═╡ 425e9917-fa9e-45df-a046-69dc8d64bb05
md"""
`simulate` allocates the storage for every state once, before the loop, then takes time steps until `stop_time` and writes each new state into place.
"""

# ╔═╡ 0c9e15a0-cdc8-43ef-a2d5-10f720707ce4
md"""
### The first run

The domain has length 1 and 100 cells. The wind blows to the right with ``a = 1``, so the exact solution comes back to where it started at ``t = 1``. The initial condition has a smooth bump and a top hat with sharp edges.
"""

# ╔═╡ ddfa0a8c-34ca-4817-8986-3c4a698f8c4e
begin
    N = 100
    Δx = 1 / N
    x = Δx/2:Δx:1   # cell centers
    a = 1.0
end;

# ╔═╡ 27a20176-a359-4a41-8385-5a098885489a
initial_tracer(x) = exp(-((x - 0.25) / 0.07)^2) + (0.55 < x < 0.8)

# ╔═╡ f6d14578-452b-4f8f-add9-1f89ed8b22e9
c₀ = initial_tracer.(x);

# ╔═╡ 59869cca-4917-4757-aa13-e62ff9330efd
md"""
The time step is set through the **Courant number**

```math
C = \frac{a\,\Delta t}{\Delta x}
```

the fraction of a cell that the wind crosses in one time step. We choose ``C`` and get ``\Delta t`` from it.

**Courant number:** $(@bind courant_centered Slider([0.01, 0.1, 0.25, 0.5, 1.0], default = 0.5, show_value = true))
"""

# ╔═╡ 7b1a6aed-13de-4739-ae0c-522ac9e0d3fe
md"""
**Time:** $(@bind time_centered Slider(0:0.02:1, default = 0.1, show_value = true))
"""

# ╔═╡ 6b549874-6cc2-4078-ad99-83af21f3df0d
begin
    scheme_colors = NamedTuple{(:upwind, :centered, :third_order, :weno)}(Tuple(Makie.wong_colors()[1:4]))

    state_at(run, t) = run.states[argmin(abs.(run.times .- t))]

    tidy(x) = 10 ≤ abs(x) < 1e6 ? round(Int, x) : round(x, sigdigits = 2)

    exact_tracer(t) = (xs = 0:0.001:1; (xs, initial_tracer.(mod.(xs .- a * t, 1))))

    function draw_profiles!(ax, x, curves; exact = nothing)
        isnothing(exact) || lines!(ax, exact..., color = :black, linewidth = 1, label = "exact")
        for curve in curves
            lines!(ax, x, curve.values; color = curve.color, label = curve.label,
                   linestyle = get(curve, :linestyle, :solid), linewidth = 1.5)
        end
        return ax
    end

    legend_below!(position, ax) = Legend(position, ax, orientation = :horizontal, framevisible = false, tellwidth = false)
end;

# ╔═╡ 3ae7b80b-b2ac-481b-9d3d-aeda8f6cc9e6
md"""
The run fails. Wiggles a few cells wide appear near the sharp edges of the top hat and grow. On the logarithmic axis the growth is a straight line, so it is exponential, and at ``C = 0.5`` the largest value is above ``10^8`` after one trip around the domain.

A smaller time step slows the growth but leaves it in place. At ``C = 0.1`` the wiggles grow more slowly. At ``C = 0.01`` the solution survives one trip, full of wiggles that are still growing.

All the while, the total amount of ``c`` is conserved to the last digit:
"""

# ╔═╡ 880d45fc-e31a-4367-8779-f612ce736fe8
md"""
A method can conserve the total exactly and still be unstable.
"""

# ╔═╡ c05b017e-4779-489a-962d-91926577e07f
md"""
### Is the time stepper to blame?

Our method has two parts, the flux in space and the stepper in time, and either could be at fault. To find out, change the time stepper and nothing else. A more accurate one is the third-order **Runge–Kutta** method, RK3, which takes three Euler steps and blends them.
"""

# ╔═╡ 83b1054a-d95a-4bca-a219-c1eaed0ae83a
begin
    struct RK3 end

    allocate_workspace(::RK3, c) = (; dc = similar(c), c¹ = similar(c), c² = similar(c))

    function advance!(c, Δt, G!, ::RK3, workspace)
        (; dc, c¹, c²) = workspace
        G!(dc, c)
        @. c¹ = c + Δt * dc
        G!(dc, c¹)
        @. c² = 3//4 * c + 1//4 * (c¹ + Δt * dc)
        G!(dc, c²)
        @. c = 1//3 * c + 2//3 * (c² + Δt * dc)
        return c
    end
end

# ╔═╡ 149f84fd-935e-4089-9c49-0a663368d586
function simulate(c₀, G!, stepper; Δt, stop_time)
    nsteps = round(Int, stop_time / Δt)
    Δt = convert(eltype(c₀), Δt)
    times  = [n * Δt for n in 0:nsteps]
    states = [similar(c₀) for n in 0:nsteps]
    c = copy(c₀)
    workspace = allocate_workspace(stepper, c)
    states[1] .= c
    for n in 1:nsteps
        advance!(c, Δt, G!, stepper, workspace)
        states[n + 1] .= c
    end
    return (; times, states)
end

# ╔═╡ 16adfbd5-1653-496d-b6fc-ca10d8417b48
md"""
`simulate`, `tendency!` and `face_flux` are unchanged. Julia picks the right `advance!` from the type of the `stepper` argument, which is multiple dispatch at work.
"""

# ╔═╡ 0d11739c-14a6-4435-980e-22f93a422833
md"""
**Time:** $(@bind time_rk3 Slider(0:0.05:3, default = 0.5, show_value = true))
"""

# ╔═╡ d4349e75-f356-4696-87c4-67465e6eb687
md"""
The blow-up is gone, and the run stays stable for as long as you like. The solution is still wrong. It overshoots above 1 and undershoots below 0, and ripples that start at the sharp edges of the top hat spread over the whole domain, the smooth bump included. If ``c`` is the concentration of smoke or water vapor, a negative value has no meaning, and the part of a model that makes clouds or rain from it will fail.

The time stepper cured the instability. The wiggles come from the centered flux.
"""

# ╔═╡ db567831-19b6-4ca7-8abf-05e81e3d00cd
details("Why does the centered flux make ripples?", let
    blue, orange = Makie.wong_colors()[1:2]
    fig = Figure(size = (700, 260))
    ax = Axis(fig[1, 1], xlabel = "cell", ylabel = "c", limits = (0.5, 8.5, -1.6, 1.6), xticks = 1:8,
              title = "The shortest wave on the grid: +1, −1, +1, −1, ...")
    scatterlines!(ax, 1:8, [(-1)^j for j in 1:8], color = blue, markersize = 12)
    lines!(ax, [3, 5], [-1, -1], color = orange, linewidth = 3)
    text!(ax, 4, -1.1, text = L"c_{j+1} - c_{j-1} = 0", align = (:center, :top), color = orange, fontsize = 18)
    text!(ax, 4, 1.1, text = L"c_j", align = (:center, :bottom), fontsize = 18)
    [md"""
    With the centered flux, the tendency of cell ``j`` is ``-a\,(c_{j+1} - c_{j-1}) / (2\Delta x)``. It uses the two neighbors of the cell and not the cell itself.

    In the shortest wave that the grid can hold, the two neighbors of every cell are equal, so the tendency is zero everywhere: the scheme can't see this wave, and the wave doesn't move. Waves a little longer than this move, but too slowly.

    A sharp edge is made of waves of all lengths. The long ones travel at the right speed and the short ones are left behind. They are the ripples.
    """, fig]
end)

# ╔═╡ 48bee702-7b53-4558-ba2b-461d13c4d75c
md"""
### Second try: the upwind flux

Advection carries information with the flow. If the wind blows from left to right, the fluid that crosses the face ``j + 1/2`` in the next instant is the fluid that is in cell ``j`` now. The fluid in cell ``j + 1`` is moving away from the face and has no influence on what crosses it, yet the centered flux gives it half the weight.

So take the value at the face from the side that the wind comes from, the **upwind** side:

```math
F_{j+1/2} = \begin{cases} a\,c_j & \text{if } a > 0 \\ a\,c_{j+1} & \text{if } a < 0 . \end{cases}
```

**The wind blows** $(@bind wind_sign Select([1 => "from left to right (a > 0)", -1 => "from right to left (a < 0)"]))
"""

# ╔═╡ 8f2520be-30e5-442b-8c4c-5c04df6589da
let
    blue, orange = Makie.wong_colors()[1:2]
    donor = wind_sign > 0 ? 2 : 3
    labels = (L"c_{j-1}", L"c_j", L"c_{j+1}", L"c_{j+2}")
    fig = Figure(size = (700, 250))
    ax = Axis(fig[1, 1], aspect = DataAspect(), limits = (-0.2, 4.2, -0.75, 1.75))
    hidedecorations!(ax)
    hidespines!(ax)
    for k in 1:4
        poly!(ax, Rect(k - 1, 0, 1, 1), color = k == donor ? (blue, 0.45) : :gray95, strokecolor = :gray40, strokewidth = 1)
        text!(ax, k - 0.5, 0.5, text = labels[k], align = (:center, :center), fontsize = 22)
    end
    lines!(ax, [2, 2], [-0.15, 1.15], color = :black, linewidth = 4)
    text!(ax, 2, -0.2, text = L"\text{face } j + 1/2", align = (:center, :top), fontsize = 18)
    arrows2d!(ax, [2 - 0.9wind_sign], [1.45], [1.8wind_sign], [0], color = :gray40, shaftwidth = 4, tipwidth = 14, tiplength = 14)
    text!(ax, 2 - 1.05wind_sign, 1.45, text = "wind", align = (wind_sign > 0 ? :right : :left, :center), color = :gray40, fontsize = 16)
    text!(ax, 4.1, -0.45, text = wind_sign > 0 ? L"F_{j+1/2} = a\,c_j" : L"F_{j+1/2} = a\,c_{j+1}", align = (:right, :center),
          color = blue, fontsize = 22)
    fig
end

# ╔═╡ 519f12cd-3b0e-4c21-8cce-9b6c1bac10e2
md"""
In code, the scheme asks the equation for the `wave_speed` at the face, and takes the flux from the left cell or the right cell. For linear advection the speed is ``a``.
"""

# ╔═╡ 4e670940-e9c0-4926-84b7-e0e41a00e92b
md"""
Run it with plain forward Euler, the stepper that failed before.

**Courant number:** $(@bind courant_upwind Slider(0.1:0.1:1.2, default = 0.5, show_value = true))
"""

# ╔═╡ 2d1daf3c-c577-41d2-a988-774999f12fdc
md"""
**Time:** $(@bind time_upwind Slider(0:0.05:3, default = 1, show_value = true))
"""

# ╔═╡ 64819290-2e59-434b-86fa-279532560ffd
md"""
At ``C = 0.5`` the run is stable with the simplest time stepper, and ``c`` stays between 0 and 1 forever, without wiggles or negative values.

Its fault is a different one: the shapes are smeared. The edges of the top hat spread out, and the bump gets lower and wider with every trip. After one trip around the domain the two schemes look like this:
"""

# ╔═╡ 406fbeee-1583-4c7e-a318-5009d186c1d4
md"""
Wiggles and smearing are the two typical faults of an advection scheme, and much of the effort that goes into the numerics of a climate model is spent on schemes that have little of both.

To see where the smearing comes from, write the upwind flux (for ``a > 0``) in a different way:

```math
a\,c_j = \underbrace{a\,\frac{c_j + c_{j+1}}{2}}_{\text{centered flux}} - \underbrace{\frac{a}{2}\,(c_{j+1} - c_j)}_{\text{correction}} .
```

The upwind flux is the centered flux plus a correction. The correction moves ``c`` from the cell that has more to the cell that has less, so it smooths, like friction, and that is what removes the wiggles. Its size is proportional to ``c_{j+1} - c_j``, which gets smaller as the grid gets finer.

The equation we set out to solve has no friction in it. This friction comes from the method, and the ocean model in the next notebook relies on it.
"""

# ╔═╡ e8d57055-b03a-4a43-9c27-bb51dfa174f6
md"""
### Third try: a sharper upwind flux

The upwind scheme above is called **first-order**, because its error is proportional to ``\Delta x``. After one trip the bump has lost more than 40% of its height, which is far too much for a model that has to carry heat around a planet for centuries. There are two ways to reduce the error: make ``\Delta x`` smaller, or make a better estimate of the value at the face.

#### A finer grid

`advect` carries the same initial condition around the domain on a grid of `N` cells, at ``C = 0.5``.
"""

# ╔═╡ d874cea2-7214-4fe3-8906-32e2c769c747
md"""
**Number of cells:** $(@bind fine_cell_count Slider([100, 200, 500, 1000, 2000], default = 100, show_value = true))
"""

# ╔═╡ 3ae070e5-6dbb-47ba-89f2-312dc7f9268f
md"""
The smearing shrinks slowly: with 1000 cells the bump is still 9% too low. The cost rises much faster. The Courant number is fixed, so ten times the cells also means ten times the time steps, and the run takes 100 times the work. In three dimensions the same refinement takes 10,000 times the work.

#### A better value at the face

The first-order scheme uses ``c_j`` as the value at the face, as if ``c`` were flat inside each cell. A parabola describes a smooth function much better. Take the two cells on the upwind side of the face and one cell on the downwind side, and find the parabola that has the right average in each of the three. For ``a > 0``, its value at the face is

```math
c_{j+1/2} = \frac{-c_{j-1} + 5c_j + 2c_{j+1}}{6} .
```
"""

# ╔═╡ 1cbcb64e-f233-421b-bdf4-ef6be8a696e6
let
    blue, _, green = Makie.wong_colors()[1:3]
    f(x) = 0.15 + 0.9 * exp(-((x - 3.6) / 1.7)^2)
    average(k) = sum(f, range(k - 1, k, length = 200)) / 200
    c₁, c₂, c₃, c₄ = average.(1:4)
    parabola(x) = c₂ + (c₃ - c₁) / 2 * (x - 1.5) + (c₁ - 2c₂ + c₃) / 2 * ((x - 1.5)^2 - 1 / 12)

    fig = Figure(size = (800, 400))
    ax = Axis(fig[1, 1], limits = (-0.1, 4.1, 0, 1.25), xticks = (0.5:3.5, [L"c_{j-1}", L"c_j", L"c_{j+1}", L"c_{j+2}"]),
              xticklabelsize = 20, ylabel = "c", xgridvisible = false, title = "Two estimates of the value at the face j + 1/2 (wind from the left)")
    barplot!(ax, 0.5:3.5, [c₁, c₂, c₃, c₄], gap = 0, color = (:gray60, 0.2), strokecolor = :gray50, strokewidth = 1)
    lines!(ax, 0:0.01:4, f, color = :black, linewidth = 1, label = "the function")
    lines!(ax, [1, 2], [c₂, c₂], color = blue, linewidth = 3, label = "first order: flat in cell j")
    lines!(ax, 0:0.01:3, parabola, color = green, linewidth = 3, label = "third order: parabola through three cells")
    vlines!(ax, 2, color = :black, linewidth = 2)
    scatter!(ax, [2, 2], [c₂, parabola(2)], color = [blue, green], markersize = 16, strokecolor = :black, strokewidth = 1)
    arrows2d!(ax, [0.2], [1.15], [1], [0], color = :gray40, shaftwidth = 4, tipwidth = 14, tiplength = 14)
    text!(ax, 1.3, 1.15, text = "wind", align = (:left, :center), color = :gray40, fontsize = 16)
    legend_below!(fig[2, 1], ax)
    fig
end

# ╔═╡ 1290d041-a09b-445c-b5ff-e484be47f3ad
md"""
In the code, `c₁`, `c₂` and `c₃` are the three cells in the order in which the wind meets them.
"""

# ╔═╡ f732b713-9754-490d-8c6a-1693cdaaafe7
md"""
The error of this scheme is proportional to ``\Delta x^3``, so it is called **third-order**. It needs RK3: with forward Euler it blows up, as the centered flux did.
"""

# ╔═╡ 43688d97-1248-4756-98a5-152054c8a16c
md"""
With the same 100 cells the bump keeps 96% of its height. First-order upwind needs about 2000 cells for that, at 400 times the work.

The cost is a small wiggle at each sharp edge, where ``c`` now runs from about −0.06 to 1.06. Like the first-order flux, this one is the centered flux plus a correction:

```math
\frac{-c_{j-1} + 5c_j + 2c_{j+1}}{6} = \frac{c_j + c_{j+1}}{2} - \frac{c_{j+1} - 2c_j + c_{j-1}}{6} .
```

The correction is now a difference of two differences, which measures how sharply the data bends. It is very small where the data is smooth and large only for features a few cells wide, so the friction acts where the centered flux made its ripples and almost nowhere else.

#### A scheme that looks at the data

A cleverer fixed formula can't remove the last wiggles. Godunov proved in 1959 that a scheme which uses the same formula everywhere is either first-order or makes wiggles at sharp features. The **WENO** schemes (weighted essentially non-oscillatory) escape the theorem because their formula depends on the data. They compute several estimates of the face value from different groups of cells and give the most weight to the group in which the data is smoothest, so near a jump they read only the cells on its smooth side.

The folded cell below has a fifth-order WENO scheme. It is one more method for `face_flux`, and the rest of the solver is unchanged.
"""

# ╔═╡ dd9201fc-3a6f-485a-9403-81605597e83c
md"""
WENO keeps 99% of the bump and stays between 0 and 1. Schemes of this family carry heat, salt and momentum in the ocean model of the next notebook.
"""

# ╔═╡ 98e876ad-55f2-446b-b9fb-bb78c30e8a61
md"""
### The limit on the time step

Go back to the first-order upwind run and move its Courant number slider. At ``C = 1`` the answer is exact, because each time step moves the data by one cell. At ``C = 1.1`` the run blows up. The limit ``C = 1`` is the **CFL condition**, named after Courant, Friedrichs and Lewy, who found it in 1928:

```math
C = \frac{a\,\Delta t}{\Delta x} \le 1 \qquad\text{or}\qquad \Delta t \le \frac{\Delta x}{a} .
```

The fluid must not cross more than one cell in one time step. Three arguments lead to this limit.
"""

# ╔═╡ 9570641e-f3d1-4a2c-944a-8696313faf13
md"""
#### 1. The scheme must be able to see where the fluid comes from

**Courant number:** $(@bind courant_path Slider(0.1:0.1:1.5, default = 0.5, show_value = true))
"""

# ╔═╡ a07d7f34-e4d4-49a5-be93-56cd84b2ee34
let C = courant_path
    blue = Makie.wong_colors()[1]
    ok = C ≤ 1
    path_color = ok ? :seagreen : :firebrick
    fig = Figure(size = (800, 330))
    ax = Axis(fig[1, 1], limits = (-2.5, 1.5, -0.35, 1.4),
              xticks = (-2:1, [L"x_{j-2}", L"x_{j-1}", L"x_j", L"x_{j+1}"]), yticks = ([0, 1], [L"t^n", L"t^{n+1}"]),
              xticklabelsize = 18, yticklabelsize = 18,
              title = ok ? "C = $C: the parcel starts between the two cells that the scheme reads" :
                           "C = $C: the parcel starts outside the cells that the scheme reads")
    poly!(ax, Point2f[(0, 1), (-1, 0), (0, 0)], color = (blue, 0.25))
    scatter!(ax, repeat(-2:1, 2), repeat([0, 1], inner = 4), color = :gray40, markersize = 10)
    scatter!(ax, [-1, 0, 0], [0, 0, 1], color = blue, markersize = 16)
    lines!(ax, [-C, 0], [0, 1], color = path_color, linewidth = 3)
    scatter!(ax, [-C], [0], color = path_color, markersize = 16, marker = :diamond)
    text!(ax, -C, -0.08, text = "the parcel starts here", align = (:center, :top), color = path_color)
    text!(ax, 0.08, 1.05, text = L"c_j^{n+1}", align = (:left, :bottom), fontsize = 18)
    text!(ax, 0.08, 0.45, text = "the scheme reads\nthe two blue cells", align = (:left, :center), color = blue)
    fig
end

# ╔═╡ a4ca19fb-8262-4d56-a0dc-c42e029e2995
md"""
The new value in cell ``j`` is computed from the old values in cells ``j - 1`` and ``j`` (the blue triangle). The fluid that arrives at ``x_j`` after one step started at ``x_j - a\,\Delta t``. If ``C > 1``, it started further away than cell ``j - 1``. The scheme has no access to the information it needs, and no formula can repair that.
"""

# ╔═╡ b681d694-31fd-4d02-b794-e39fa928f792
md"""
#### 2. A weighted average can't overshoot

Write out one upwind step for ``a > 0``:

```math
c_j^{n+1} = c_j^n - C\,(c_j^n - c_{j-1}^n) = (1 - C)\,c_j^n + C\,c_{j-1}^n .
```

If ``0 \le C \le 1``, the new value is a weighted average of two old values, with weights between 0 and 1. An average is never larger than the largest of its inputs or smaller than the smallest. So no new maximum or minimum can ever appear. That is why the upwind solution stays between 0 and 1.

If ``C > 1``, one weight is negative. The scheme then extrapolates beyond the two values, and every step overshoots a little more than the last.
"""

# ╔═╡ a1be5695-b54c-48a2-b153-207b4f231e8e
md"""
#### 3. Eigenvalues

The tendency is linear in ``c``, so one time step is a matrix multiplication:

```math
c^{n+1} = A\,c^{n}, \qquad\text{and after } n \text{ steps}\qquad c^{n} = A^n\,c^{0} .
```

From 18.06: powers of a matrix grow if any eigenvalue has ``|\lambda| > 1``, and stay bounded if all of them are inside or on the unit circle.

We can build ``A`` without any algebra. Column ``j`` of the tendency matrix is the tendency of a vector that is 1 in cell ``j`` and 0 elsewhere.
"""

# ╔═╡ 4b79a694-99fa-459a-aa82-bcdaf1372a6a
md"""
**Method:** $(@bind time_step_method Select(["centered + forward Euler", "upwind + forward Euler", "centered + RK3", "third-order upwind + forward Euler", "third-order upwind + RK3", "centered + backward Euler"]))

**Courant number:** $(@bind courant_eigenvalues Slider(0.1:0.1:2, default = 0.5, show_value = true))
"""

# ╔═╡ 68c244a3-9089-4d61-988b-216967b43790
md"""
Each dot is one eigenvalue of ``A``. Green dots are inside or on the unit circle. Red dots are outside, and each red dot is a pattern that grows at every step.

| method | where the eigenvalues lie | stable for |
|:--|:--|:--|
| centered + forward Euler | on a vertical line through 1 | no ``C`` |
| upwind + forward Euler | on a circle of radius ``C`` centered at ``1 - C`` | ``C \le 1`` |
| centered + RK3 | just inside the circle | ``C \le 1.7`` |
| third-order upwind + forward Euler | on a loop that leaves the circle next to 1 | no ``C`` |
| third-order upwind + RK3 | inside the circle | ``C \le 1.6`` |

For the centered flux with forward Euler, each eigenvalue is ``1 + i\,\omega\,\Delta t``, the multiplier of the oscillation test, and that is why the first run spiraled out. For the upwind flux with forward Euler and ``C \le 1``, the matrix ``A`` has no negative entries and its columns add up to 1, which makes it a Markov matrix.
"""

# ╔═╡ 7649806a-f858-4357-933d-7724a098be72
details("Where do these shapes come from?", md"""
The eigenvectors of ``A`` are the waves ``c_j = e^{i k x_j}``, the Fourier modes from 18.03. Put one of them into the scheme and it comes out multiplied by a number, which is the eigenvalue.

For the centered flux with forward Euler:

```math
\lambda_k = 1 - \frac{C}{2}\left(e^{ik\Delta x} - e^{-ik\Delta x}\right) = 1 - i\,C\sin(k\Delta x), \qquad |\lambda_k|^2 = 1 + C^2\sin^2(k\Delta x) \ge 1 .
```

For the upwind flux with forward Euler:

```math
\lambda_k = 1 - C\left(1 - e^{-ik\Delta x}\right) = (1 - C) + C\,e^{-ik\Delta x} ,
```

a circle of radius ``C`` centered at ``1 - C``.

This calculation is called **von Neumann stability analysis**.
""")

# ╔═╡ 88758f9f-4599-4300-8ce1-c713f4d36955
md"""
#### What the limit costs

The time step must shrink with the grid spacing: ``\Delta t \le \Delta x / a``. Suppose you halve the grid spacing of a three-dimensional model to see finer detail. You get ``2 \times 2 \times 2 = 8`` times as many cells, and you need twice as many time steps. The cost goes up 16 times.

In a real model, ``a`` is the fastest signal of any kind: the fastest wind or current, or the fastest wave. This is the main reason that climate models, which must run for centuries, use cells that are tens of kilometers wide.
"""

# ╔═╡ 587f6b66-dd66-477d-afd4-228d5010f637
md"""
### Back to the Burgers equation

The solver is ready for the Burgers equation:

```math
\frac{\partial u}{\partial t} + \frac{\partial F}{\partial x} = 0, \qquad F = \frac{u^2}{2} .
```

Two things change. The flux is ``u^2/2``, and the speed that carries information is ``u`` itself, for which we use the average of the two cells next to the face. That takes three lines of code.
"""

# ╔═╡ a2fcf8e5-44ce-442e-916d-30df3ec10f53
begin
    struct Burgers end

    flux(::Burgers, u) = u^2 / 2
    wave_speed(::Burgers, uᴸ, uᴿ) = (uᴸ + uᴿ) / 2
end

# ╔═╡ 3020f6fd-4c81-4063-873e-1231c4f7eadb
begin
    struct CenteredFlux end

    face_flux(eq, c, j, ::CenteredFlux) = (flux(eq, periodic(c, j)) + flux(eq, periodic(c, j + 1))) / 2
end

# ╔═╡ ffddcac5-7d51-46dc-baff-acb815bb28b1
begin
    struct UpwindFlux end

    wave_speed(eq::LinearAdvection, cᴸ, cᴿ) = eq.a

    function face_flux(eq, c, j, ::UpwindFlux)
        cᴸ, cᴿ = periodic(c, j), periodic(c, j + 1)
        return wave_speed(eq, cᴸ, cᴿ) ≥ 0 ? flux(eq, cᴸ) : flux(eq, cᴿ)
    end
end

# ╔═╡ 637ff736-78f3-4cb0-85d6-f966d032027a
begin
    struct UpwindThirdOrder end

    function face_flux(eq, c, j, ::UpwindThirdOrder)
        from_left = wave_speed(eq, periodic(c, j), periodic(c, j + 1)) ≥ 0
        c₁, c₂, c₃ = from_left ? (periodic(c, j - 1), periodic(c, j), periodic(c, j + 1)) :
                                 (periodic(c, j + 2), periodic(c, j + 1), periodic(c, j))
        return flux(eq, (-c₁ + 5c₂ + 2c₃) / 6)
    end
end

# ╔═╡ 3472f397-335c-425b-b19d-09842fc91885
begin
    struct WENOFlux end

    function weno_face_value(c₁, c₂, c₃, c₄, c₅)
        q₁ = (2c₁ - 7c₂ + 11c₃) / 6
        q₂ = (-c₂ + 5c₃ + 2c₄) / 6
        q₃ = (2c₃ + 5c₄ - c₅) / 6
        β₁ = 13//12 * (c₁ - 2c₂ + c₃)^2 + 1//4 * (c₁ - 4c₂ + 3c₃)^2
        β₂ = 13//12 * (c₂ - 2c₃ + c₄)^2 + 1//4 * (c₂ - c₄)^2
        β₃ = 13//12 * (c₃ - 2c₄ + c₅)^2 + 1//4 * (3c₃ - 4c₄ + c₅)^2
        ε = oftype(c₃, 1e-6)
        w₁, w₂, w₃ = 1//10 / (ε + β₁)^2, 6//10 / (ε + β₂)^2, 3//10 / (ε + β₃)^2
        return (w₁ * q₁ + w₂ * q₂ + w₃ * q₃) / (w₁ + w₂ + w₃)
    end

    function face_flux(eq, c, j, ::WENOFlux)
        from_left = wave_speed(eq, periodic(c, j), periodic(c, j + 1)) ≥ 0
        stencil = from_left ? (periodic(c, j - 2), periodic(c, j - 1), periodic(c, j), periodic(c, j + 1), periodic(c, j + 2)) :
                              (periodic(c, j + 3), periodic(c, j + 2), periodic(c, j + 1), periodic(c, j), periodic(c, j - 1))
        return flux(eq, weno_face_value(stencil...))
    end
end

# ╔═╡ b861c7bb-b662-4ddd-9bce-3385bde2ca54
md"""
Run the flow from the start of the notebook, ``u_0(x) = 1 + \frac{1}{2}\sin x``, with three methods. The fastest parcel moves at 1.5, so the time step is ``\Delta t = C\,\Delta x / 1.5``.
"""

# ╔═╡ ac62481e-fde7-4345-9214-97046d49ec27
md"""
**Time:** $(@bind time_burgers Slider(0:0.1:6, default = 1.5, show_value = true))
"""

# ╔═╡ ff0c1dda-0fd1-4f16-af64-a832e6702948
function burgers_exact(t; n = 4000)
    x₀ = range(0, 2π, length = n)
    ξ = @. x₀ + t * sin(x₀) / 2                 # position in the frame that moves at speed 1
    keep = @. ifelse(x₀ < π, ξ < π, ξ > π)      # parcels that have not reached the shock at ξ = π
    position = mod.(ξ[keep] .+ t, 2π)
    order = sortperm(position)
    return position[order], u₀.(x₀[keep])[order]
end

# ╔═╡ e6c40bdb-afb8-414b-ad8f-65f988bd92d5
md"""
Before ``t = 2`` the solution is smooth and all three methods agree with the exact solution. Any reasonable scheme works while the solution is smooth. At ``t = 2`` the shock forms, and the methods part.

| method | after the shock forms |
|:--|:--|
| centered + forward Euler | blows up within a short time |
| centered + RK3 | survives, but overshoots to 2, when no parcel ever had a velocity above 1.5, and leaves noise at the scale of the grid behind the shock |
| upwind + forward Euler | gives a clean jump that stays within the range of the initial data, at the right place (the dotted line) and moving at the right speed |

First-order upwind does well here because a shock steepens itself, which works against the smearing. Of the sharper schemes, third-order upwind overshoots to 1.7 at the shock, and WENO gives a jump as clean as that of first-order upwind.

The trouble starts when the flow makes a feature smaller than a grid cell, and real flows always do. The smallest swirls in the atmosphere are about a millimeter across, while the cells of a climate model are tens of kilometers wide, a gap that no computer will close. A model of the atmosphere or the ocean therefore has to give a sensible answer when the flow has features that the grid can't hold, and the advection scheme decides whether it does.
"""

# ╔═╡ c9a1bd7e-69cb-4926-8043-3ddfd536568c
md"""
### Why the flux form matters

We wrote the Burgers equation in two ways:

```math
\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} = 0 \qquad\text{and}\qquad \frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\!\left(\frac{u^2}{2}\right) = 0 .
```

They are the same equation. We could apply the upwind idea to the first form directly, with the difference taken on the upwind side:
"""

# ╔═╡ 4474c72e-97e1-4ccd-88a8-04d987e59127
function advective_tendency!(du, u, Δx)
    for j in eachindex(u)
        du[j] = -u[j] * (u[j] - periodic(u, j - 1)) / Δx
    end
    return du
end

# ╔═╡ 03cbe4a0-0550-4536-add4-a59466629330
md"""
**Time:** $(@bind time_forms Slider(0:0.5:10, default = 6, show_value = true))
"""

# ╔═╡ e407ff9a-dd14-4e3b-b16c-84033c10ec80
md"""
## Exercises

### 1. A fifth-order upwind scheme

A curve through five cells gives a better value at the face than a parabola through three. For wind from the left,

```math
c_{j+1/2} = \frac{2c_{j-2} - 13c_{j-1} + 47c_j + 27c_{j+1} - 3c_{j+2}}{60} .
```

Write the `face_flux` method, for wind from the left only. The plot then shows your scheme after ten trips around the domain.
"""

# ╔═╡ 6bb4e870-0ea1-4888-aa12-0f655801d77e
begin
    struct UpwindFifthOrder end

    face_flux(eq, c, j, ::UpwindFifthOrder) = missing # YOUR CODE HERE
end

# ╔═╡ fc87feae-8962-40fd-873e-d5068e1b4287
function tendency!(dc, c, eq, scheme, Δx)
    F(j) = face_flux(eq, c, j, scheme)
    for j in eachindex(c)
        dc[j] = (F(j - 1) - F(j)) / Δx
    end
    return dc
end

# ╔═╡ 61c05b91-c56d-4373-927e-4da6881ca2be
centered_euler = let
    Δt = courant_centered * Δx / a
    G!(dc, c) = tendency!(dc, c, LinearAdvection(a), CenteredFlux(), Δx)
    simulate(c₀, G!, ForwardEuler(); Δt, stop_time = 1)
end;

# ╔═╡ 652c59a9-0fb1-4d74-bf75-cf6688ee45fe
let run = centered_euler
    c = state_at(run, time_centered)
    amplitude = maximum(abs, c)
    fig = Figure(size = (800, 640))
    ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "c", limits = (0, 1, -1, 2),
               title = "Centered flux + forward Euler, C = $courant_centered, t = $time_centered: largest |c| = $(tidy(amplitude))")
    draw_profiles!(ax1, x, [(label = "centered + forward Euler", values = c, color = scheme_colors.centered)];
                   exact = exact_tracer(time_centered))
    legend_below!(fig[2, 1], ax1)

    ax2 = Axis(fig[3, 1], xlabel = "time", ylabel = "largest |c|", yscale = log10, limits = (0, 1, 0.5, 1e9))
    lines!(ax2, run.times, maximum.(abs, run.states), color = scheme_colors.centered)
    vlines!(ax2, time_centered, color = :gray40, linestyle = :dash)
    rowsize!(fig.layout, 3, Relative(0.28))
    fig
end

# ╔═╡ cd2ec92c-6e1f-462a-b585-2598b3c42155
sum(centered_euler.states[end]) * Δx - sum(c₀) * Δx

# ╔═╡ 28b9a1be-96ca-40f7-8bc2-4e9cf3994fe6
centered_rk3 = let
    Δt = 0.5 * Δx / a
    G!(dc, c) = tendency!(dc, c, LinearAdvection(a), CenteredFlux(), Δx)
    simulate(c₀, G!, RK3(); Δt, stop_time = 3)
end;

# ╔═╡ c003091a-8a14-4a9d-898a-86cd27fa926d
let
    c = state_at(centered_rk3, time_rk3)
    fig = Figure(size = (800, 410))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "c", limits = (0, 1, -0.6, 1.6),
              title = "Centered flux + RK3, C = 0.5, t = $time_rk3: c between $(round(minimum(c), digits = 2)) and $(round(maximum(c), digits = 2))")
    draw_profiles!(ax, x, [(label = "centered + RK3", values = c, color = scheme_colors.centered, linestyle = :dash)];
                   exact = exact_tracer(time_rk3))
    legend_below!(fig[2, 1], ax)
    fig
end

# ╔═╡ b195ea93-3a9d-4607-b8e5-6d0c7ffdb8fe
upwind_euler = let
    Δt = courant_upwind * Δx / a
    G!(dc, c) = tendency!(dc, c, LinearAdvection(a), UpwindFlux(), Δx)
    simulate(c₀, G!, ForwardEuler(); Δt, stop_time = 3)
end;

# ╔═╡ 18a73da5-75cc-4d8d-848d-bcba85bc16cc
let
    c = state_at(upwind_euler, time_upwind)
    fig = Figure(size = (800, 410))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "c", limits = (0, 1, -0.6, 1.6),
              title = "Upwind flux + forward Euler, C = $courant_upwind, t = $time_upwind: c between $(tidy(minimum(c))) and $(tidy(maximum(c)))")
    draw_profiles!(ax, x, [(label = "upwind + forward Euler", values = c, color = scheme_colors.upwind)];
                   exact = exact_tracer(time_upwind))
    legend_below!(fig[2, 1], ax)
    fig
end

# ╔═╡ a192d708-a790-4902-86cb-7ebb9d110a48
let
    upwind = let
        G!(dc, c) = tendency!(dc, c, LinearAdvection(a), UpwindFlux(), Δx)
        simulate(c₀, G!, ForwardEuler(); Δt = 0.5 * Δx / a, stop_time = 1)
    end
    fig = Figure(size = (800, 410))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "c", limits = (0, 1, -0.6, 1.6), title = "After one trip around the domain (C = 0.5)")
    draw_profiles!(ax, x, [(label = "centered + RK3", values = state_at(centered_rk3, 1), color = scheme_colors.centered, linestyle = :dash),
                           (label = "upwind + forward Euler", values = upwind.states[end], color = scheme_colors.upwind)];
                   exact = exact_tracer(1))
    legend_below!(fig[2, 1], ax)
    fig
end

# ╔═╡ 63038188-e602-45f8-a355-3faeca092ad7
function advect(N, scheme, stepper; trips = 1)
    Δx = 1 / N
    x = Δx/2:Δx:1
    G!(dc, c) = tendency!(dc, c, LinearAdvection(a), scheme, Δx)
    run = simulate(initial_tracer.(x), G!, stepper; Δt = 0.5 * Δx / a, stop_time = trips)
    return (; x, c = run.states[end])
end

# ╔═╡ b11a6e32-8568-41b3-b071-082f36aa33c8
let N = fine_cell_count
    (; x, c) = advect(N, UpwindFlux(), ForwardEuler())
    kept = round(Int, 100 * maximum(c[x .< 0.5]))
    fig = Figure(size = (800, 410))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "c", limits = (0, 1, -0.6, 1.6),
              title = "First-order upwind, $N cells, one trip: the bump keeps $kept% of its height, for $((N ÷ 100)^2) × the work")
    draw_profiles!(ax, x, [(label = "upwind + forward Euler", values = c, color = scheme_colors.upwind)]; exact = exact_tracer(1))
    legend_below!(fig[2, 1], ax)
    fig
end

# ╔═╡ 57f29cd6-c8af-49cf-8d78-3882f0f353a3
let
    first_order = advect(N, UpwindFlux(), ForwardEuler())
    third_order = advect(N, UpwindThirdOrder(), RK3())
    fig = Figure(size = (800, 410))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "c", limits = (0, 1, -0.6, 1.6), title = "$N cells, after one trip around the domain (C = 0.5)")
    draw_profiles!(ax, x, [(label = "upwind + forward Euler", values = first_order.c, color = scheme_colors.upwind),
                           (label = "third-order upwind + RK3", values = third_order.c, color = scheme_colors.third_order, linestyle = :dash)];
                   exact = exact_tracer(1))
    legend_below!(fig[2, 1], ax)
    fig
end

# ╔═╡ 947d749e-1847-4d8b-92ca-9be2f25ec1cd
let
    third_order = advect(N, UpwindThirdOrder(), RK3())
    weno = advect(N, WENOFlux(), RK3())
    range_of(c) = "$(round(minimum(c), digits = 2) + 0.0) to $(round(maximum(c), digits = 2))"
    fig = Figure(size = (800, 410))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "c", limits = (0, 1, -0.6, 1.6),
              title = "After one trip: third-order upwind from $(range_of(third_order.c)), WENO from $(range_of(weno.c))")
    draw_profiles!(ax, x, [(label = "third-order upwind + RK3", values = third_order.c, color = scheme_colors.third_order),
                           (label = "WENO + RK3", values = weno.c, color = scheme_colors.weno)];
                   exact = exact_tracer(1))
    legend_below!(fig[2, 1], ax)
    fig
end

# ╔═╡ d0ca6d74-be30-4ec1-8aeb-88fe0fd2a444
function tendency_matrix(scheme, N)
    L = zeros(N, N)
    e = zeros(N)
    for j in 1:N               # column j is the tendency of a unit spike in cell j
        e .= 0
        e[j] = 1
        tendency!(view(L, :, j), e, LinearAdvection(1.0), scheme, 1 / N)
    end
    return L
end

# ╔═╡ 8c734e9a-af1f-4a95-8df6-07999529b0ea
function step_matrix(method, C; N = 40)
    scheme = startswith(method, "centered") ? CenteredFlux() :
             startswith(method, "upwind") ? UpwindFlux() : UpwindThirdOrder()
    M = (C / N) * tendency_matrix(scheme, N)   # Δt × tendency matrix, for Δx = 1/N and a = 1
    endswith(method, "forward Euler") && return I + M
    endswith(method, "RK3") && return I + M + M^2 / 2 + M^3 / 6
    return inv(I - M)                          # backward Euler
end

# ╔═╡ 22910abe-d941-48a0-b4c6-2a6be24b6f5d
let
    λ = eigvals(step_matrix(time_step_method, courant_eigenvalues))
    largest = maximum(abs, λ)
    unstable = abs.(λ) .> 1 + 1e-9
    fig = Figure(size = (620, 560))
    ax = Axis(fig[1, 1], xlabel = "Re λ", ylabel = "Im λ", aspect = DataAspect(), limits = (-2.2, 2.2, -2.2, 2.2),
              title = "$time_step_method, C = $courant_eigenvalues\nlargest |λ| = $(round(largest, digits = 3)), after 200 steps: × $(round(largest^200, sigdigits = 2))")
    lines!(ax, cos.(0:0.01:2π), sin.(0:0.01:2π), color = :black)
    scatter!(ax, real.(λ), imag.(λ), color = ifelse.(unstable, :firebrick, :seagreen), markersize = 9)
    fig
end

# ╔═╡ 9feb44a6-a77c-4c49-bf8d-c23190f01678
burgers = let
    N = 200
    Δx = 2π / N
    x = Δx/2:Δx:2π
    Δt = 0.5 * Δx / 1.5
    run(scheme, stepper) = simulate(u₀.(x), (du, u) -> tendency!(du, u, Burgers(), scheme, Δx), stepper; Δt, stop_time = 6)

    (; x, Δx,
       upwind_euler = run(UpwindFlux(), ForwardEuler()),
       centered_rk3 = run(CenteredFlux(), RK3()),
       centered_euler = run(CenteredFlux(), ForwardEuler()))
end;

# ╔═╡ d73085ca-0234-411c-8a01-b550084b7895
let t = time_burgers
    (; x, upwind_euler, centered_rk3, centered_euler) = burgers
    first_nonfinite = findfirst(u -> !all(isfinite, u), centered_euler.states)
    blown_up = !isnothing(first_nonfinite) && centered_euler.times[first_nonfinite] ≤ t
    curves = [(label = "centered + forward Euler", values = blown_up ? fill(NaN, length(x)) : state_at(centered_euler, t), color = :gray50),
              (label = "centered + RK3", values = state_at(centered_rk3, t), color = scheme_colors.centered, linestyle = :dash),
              (label = "upwind + forward Euler", values = state_at(upwind_euler, t), color = scheme_colors.upwind)]

    fig = Figure(size = (800, 660))
    title = "Burgers equation, t = $t" * (blown_up ? ". Centered + forward Euler blew up at t = $(round(centered_euler.times[first_nonfinite], digits = 2))" : "")
    ax1 = Axis(fig[1, 1]; xlabel = "x", ylabel = "u", limits = (0, 2π, 0.2, 2.3), title,
               xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]))
    t ≥ 2 && vlines!(ax1, mod(π + t, 2π), color = :gray40, linestyle = :dot)
    draw_profiles!(ax1, x, curves; exact = burgers_exact(t))
    legend_below!(fig[2, 1], ax1)

    ax2 = Axis(fig[3, 1], xlabel = "time", ylabel = "largest u", limits = (0, 6, 1.3, 2.5))
    lines!(ax2, centered_euler.times, maximum.(centered_euler.states), color = :gray50)
    lines!(ax2, centered_rk3.times, maximum.(centered_rk3.states), color = scheme_colors.centered, linestyle = :dash)
    lines!(ax2, upwind_euler.times, maximum.(upwind_euler.states), color = scheme_colors.upwind)
    vlines!(ax2, t, color = :gray40, linestyle = :dash)
    rowsize!(fig.layout, 3, Relative(0.25))
    fig
end

# ╔═╡ 3e601fe9-9eed-40fb-8e86-14b218286b44
two_forms = let
    (; x, Δx) = burgers
    Δt = 0.5 * Δx / 1.5
    (flux_form = simulate(u₀.(x), (du, u) -> tendency!(du, u, Burgers(), UpwindFlux(), Δx), ForwardEuler(); Δt, stop_time = 10),
     advective_form = simulate(u₀.(x), (du, u) -> advective_tendency!(du, u, Δx), ForwardEuler(); Δt, stop_time = 10))
end;

# ╔═╡ 7ce90062-0426-454f-b160-7e62c963b1e4
let t = time_forms
    (; x, Δx) = burgers
    (; flux_form, advective_form) = two_forms
    vermillion = Makie.wong_colors()[6]
    fig = Figure(size = (800, 660))
    ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "u", limits = (0, 2π, 0.4, 1.6), title = "Two upwind schemes for the Burgers equation, t = $t",
               xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]))
    t ≥ 2 && vlines!(ax1, mod(π + t, 2π), color = :gray40, linestyle = :dot)
    draw_profiles!(ax1, x, [(label = "flux form", values = state_at(flux_form, t), color = scheme_colors.upwind),
                            (label = "advective form", values = state_at(advective_form, t), color = vermillion)];
                   exact = burgers_exact(t))
    legend_below!(fig[2, 1], ax1)

    ax2 = Axis(fig[3, 1], xlabel = "time", ylabel = "total / initial total", limits = (0, 10, 0.9, 1.02))
    total(run) = [sum(u) * Δx for u in run.states] ./ (sum(run.states[1]) * Δx)
    lines!(ax2, flux_form.times, total(flux_form), color = scheme_colors.upwind)
    lines!(ax2, advective_form.times, total(advective_form), color = vermillion)
    vlines!(ax2, t, color = :gray40, linestyle = :dash)
    rowsize!(fig.layout, 3, Relative(0.25))
    fig
end

# ╔═╡ 6388e1dd-b7ad-46a8-9104-526a3c9c7c7c
md"""
Both solutions are smooth and stable, and without the exact solution you couldn't tell from the picture which one is wrong. In the advective form the shock moves too slowly and the total amount of ``u`` leaks away: $(round(100 * (1 - sum(two_forms.advective_form.states[end]) / sum(two_forms.advective_form.states[1])), digits = 1))% is gone at ``t = 10``. A finer grid leaves the error in the position of the shock unchanged.

In the flux form, what leaves one cell enters the next, so the total is exact, and an exact total forces the shock to move at the right speed. A climate model runs for centuries and must not lose or create heat, salt or water along the way, which is why ocean and atmosphere models are built on fluxes through the faces of cells.

The next notebook, `ocean_planet.jl`, runs a model of a global ocean with [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl). Its script makes the same choices as this solver: a grid of cells, a flux at the faces, a time stepper and a time step.
"""

# ╔═╡ 05f57e2a-21a3-40a6-96e7-4cb086df7ee7
details("Hint", md"""
The flux is the `flux` of the face value:
```julia
function face_flux(eq, c, j, ::UpwindFifthOrder)
    c₁, c₂, c₃, c₄, c₅ = (periodic(c, k) for k in j-2:j+2)
    return flux(eq, (2c₁ - 13c₂ + 47c₃ + 27c₄ - 3c₅) / 60)
end
```
""")

# ╔═╡ 0099469c-4c36-45a6-999e-e2f58368bd38
if ismissing(face_flux(LinearAdvection(a), c₀, 1, UpwindFifthOrder()))
    md"Write `face_flux` for `UpwindFifthOrder` to see the plot."
else
    let
        fig = Figure(size = (800, 410))
        ax = Axis(fig[1, 1], xlabel = "x", ylabel = "c", limits = (0, 1, -0.6, 1.6), title = "After ten trips around the domain (C = 0.5)")
        draw_profiles!(ax, x, [(label = "third-order upwind + RK3", values = advect(N, UpwindThirdOrder(), RK3(); trips = 10).c, color = scheme_colors.third_order),
                               (label = "fifth-order upwind + RK3", values = advect(N, UpwindFifthOrder(), RK3(); trips = 10).c, color = Makie.wong_colors()[6]),
                               (label = "WENO + RK3", values = advect(N, WENOFlux(), RK3(); trips = 10).c, color = scheme_colors.weno)];
                       exact = exact_tracer(10))
        legend_below!(fig[2, 1], ax)
        fig
    end
end

# ╔═╡ 791bec76-e199-4883-a0c1-f8c8fba38a22
md"""
How much of the bump does each scheme keep after ten trips? Does the higher order remove the wiggles at the edges of the top hat? WENO reads the same five cells as your scheme.

### 2. A wind that changes sign

Solve the Burgers equation with ``u_0(x) = \sin x``. The flow is to the right in one half of the domain and to the left in the other half. Before you run it: where does the shock form, and how fast does it move?
"""

# ╔═╡ ab84215c-7bd0-4844-992d-eb2cabf98089
sign_change = let
    N = 200
    Δx = 2π / N
    x = Δx/2:Δx:2π
    run = simulate(sin.(x), (du, u) -> tendency!(du, u, Burgers(), UpwindFlux(), Δx), ForwardEuler(); Δt = 0.5 * Δx, stop_time = 3)
    (; x, run)
end;

# ╔═╡ 0c825e83-6859-4dad-9763-f6c3dec95d9e
md"""
**Time:** $(@bind time_sign_change Slider(0:0.1:3, default = 0, show_value = true))
"""

# ╔═╡ 6e6d532c-d6ce-485c-97e9-ef75102d35c6
let
    fig = Figure(size = (800, 340))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u", limits = (0, 2π, -1.2, 1.2), title = "Burgers equation with u₀ = sin x, t = $time_sign_change",
              xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]))
    hlines!(ax, 0, color = :gray70)
    draw_profiles!(ax, sign_change.x, [(label = "upwind + forward Euler", values = state_at(sign_change.run, time_sign_change), color = scheme_colors.upwind)])
    fig
end

# ╔═╡ e4efa161-4d66-428b-86c1-e5f17491d3a9
details("Hint", md"""
Fluid arrives at ``x = \pi`` from both sides, so the shock forms there, at ``t = 1/\max(-u_0') = 1``. The values on its two sides are equal and opposite, so their mean is zero and the shock doesn't move. At ``x = 0`` the fluid moves apart and the profile flattens.

The upwind scheme needs nothing new for this problem. `wave_speed` picks the left cell where the flow is to the right and the right cell where it is to the left.
""")

# ╔═╡ 489ca359-87b8-430d-ac01-0dda30f0e95c
md"""
### 3. How fast does the error shrink?

Carry the smooth bump ``c_0(x) = e^{-((x - 0.5)/0.1)^2}`` once around the domain with `N` cells, and measure the mean error ``\frac{1}{N}\sum_j |c_j - c_0(x_j)|``. Fill in the function. The plot shows the error against ``N`` on logarithmic axes.
"""

# ╔═╡ 8792bee1-15f3-4488-8f58-a5f06c08e99e
function bump_error(N, scheme, stepper)
    Δx = 1 / N
    x = Δx/2:Δx:1
    bump = @. exp(-((x - 0.5) / 0.1)^2)
    return missing # YOUR CODE HERE
end

# ╔═╡ 0af11048-6e25-4b8a-821d-b54af3dd50b2
details("Hint", md"""
```julia
run = simulate(bump, (dc, c) -> tendency!(dc, c, LinearAdvection(1.0), scheme, Δx), stepper; Δt = 0.5 * Δx, stop_time = 1)
return sum(abs, run.states[end] - bump) / N
```
""")

# ╔═╡ 7183c932-16ca-4b72-9362-f3c83a377a4b
if ismissing(bump_error(20, UpwindFlux(), ForwardEuler()))
    md"Fill in `bump_error` to see the plot."
else
    let
        Ns = [25, 50, 100, 200, 400]
        fig = Figure(size = (700, 480))
        ax = Axis(fig[1, 1], xlabel = "number of cells N", ylabel = "mean error", xscale = log10, yscale = log10, xticks = Ns)
        scatterlines!(ax, Ns, [bump_error(N, UpwindFlux(), ForwardEuler()) for N in Ns], color = scheme_colors.upwind, label = "upwind + forward Euler")
        scatterlines!(ax, Ns, [bump_error(N, CenteredFlux(), RK3()) for N in Ns], color = scheme_colors.centered, label = "centered + RK3")
        scatterlines!(ax, Ns, [bump_error(N, UpwindThirdOrder(), RK3()) for N in Ns], color = scheme_colors.third_order, label = "third-order upwind + RK3")
        lines!(ax, Ns, 5 ./ Ns, color = :gray50, linestyle = :dot, label = "∝ 1/N")
        lines!(ax, Ns, 120 ./ Ns .^ 2, color = :gray50, linestyle = :dash, label = "∝ 1/N²")
        lines!(ax, Ns, 2500 ./ Ns .^ 3, color = :gray50, linestyle = :dashdot, label = "∝ 1/N³")
        axislegend(ax, position = :lb)
        fig
    end
end

# ╔═╡ c09bb892-2f43-4ec5-9256-af2543e2eaab
md"""
When you double the number of cells, by what factor does the error of each scheme fall? A first-order scheme approaches a factor of 2, a second-order scheme 4 and a third-order scheme 8, which is where the names come from.
"""

# ╔═╡ c5c0863f-a83f-44b4-9c42-0cb758b77d5e
md"""
## Optional: a time stepper with no limit

Forward Euler computes the tendency at the old time. **Backward Euler** computes it at the new time:

```math
c^{n+1} = c^{n} + \Delta t\,G(c^{n+1}) .
```

The unknown is on both sides. For linear advection the tendency is a matrix, ``G(c) = L\,c``, so each step solves a linear system, as in 18.06:

```math
(I - \Delta t\,L)\,c^{n+1} = c^{n} .
```

**Courant number:** $(@bind courant_backward Slider([0.5, 1, 2, 5, 10], default = 5, show_value = true))
"""

# ╔═╡ 8c12d86f-b478-484e-87e6-8d29e7dcdc3e
backward_euler = let
    Δt = courant_backward * Δx / a
    L = tendency_matrix(CenteredFlux(), N)
    A = lu(I - Δt * L)
    nsteps = round(Int, 1 / Δt)
    states = [similar(c₀) for n in 0:nsteps]
    states[1] .= c₀
    for n in 1:nsteps
        ldiv!(states[n + 1], A, states[n])
    end
    states
end;

# ╔═╡ 9a4c5a4c-c25d-4916-9fba-495002f66344
let
    fig = Figure(size = (800, 410))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "c", limits = (0, 1, -0.6, 1.6),
              title = "Centered flux + backward Euler, C = $courant_backward, after one trip ($(length(backward_euler) - 1) steps)")
    draw_profiles!(ax, x, [(label = "centered + backward Euler", values = backward_euler[end], color = scheme_colors.centered)];
                   exact = exact_tracer(1))
    legend_below!(fig[2, 1], ax)
    fig
end

# ╔═╡ 47a8bb91-8196-48eb-b6d4-63ad62117f0c
md"""
Backward Euler is stable for every time step, even with the centered flux that failed before. In the eigenvalue plot, the eigenvalues of "centered + backward Euler" are inside or on the unit circle for every ``C``. (The small sawtooth in the plot above is the shortest wave on the grid, which the centered flux can't see.)

It is still a poor advection solver, for two reasons. Each step costs the solution of a linear system, and for the Burgers equation the system is nonlinear and costs more. And a long time step smears the solution badly, so to follow a wave you still need ``C`` near 1 or below.

Models use methods like this one for processes that are fast and whose details don't matter. In an ocean model those are the vertical mixing and, in some models, the waves on the sea surface.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NCDatasets = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
CairoMakie = "~0.15.17"
NCDatasets = "~0.14.15"
PlutoUI = "~0.7.84"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.4"
manifest_format = "2.1"
project_hash = "d4748cb2b7ef344c02c56868c9dd53b35f9d8f38"

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
git-tree-sha1 = "eec8a0ad4e4b673afba213931504c360fffb8f97"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.8.0"
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

    [deps.AdaptiveArrayPools.extensions]
    AdaptiveArrayPoolsCUDAExt = "CUDA"
    AdaptiveArrayPoolsMetalExt = "Metal"

    [deps.AdaptiveArrayPools.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"

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

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
registries = "General"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "37b6026a10d828f720288059f3471e675a06bb13"
registries = "General"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.8+0"

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

[[deps.CFTime]]
deps = ["Dates", "Printf"]
git-tree-sha1 = "fad2f199d1f1ae0c8e820ab68f81f6dbf62e60b2"
registries = "General"
uuid = "179af706-886a-5703-950a-314cd64e0468"
version = "0.2.10"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "1472c64b9dc55740edd9c4a4fa466c40babccfba"
registries = "General"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.2.0"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "47c2237c07bf37534f620515e28cc6cf021e6913"
registries = "General"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.3+0"

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

[[deps.CommonDataModel]]
deps = ["CFTime", "DataStructures", "Dates", "DiskArrays", "Preferences", "Printf", "Statistics"]
git-tree-sha1 = "54d50f8a0a68ff4160564dfa09fc5de32465cfca"
registries = "General"
uuid = "1fbeeb36-5f17-413c-809b-666fb144f157"
version = "0.4.4"

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

[[deps.ConstructionBase]]
git-tree-sha1 = "b1bc10667320dd5367aff450a992c951ea38fbbc"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.1"
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

[[deps.DiskArrays]]
deps = ["ConstructionBase", "LRUCache", "Mmap", "OffsetArrays"]
git-tree-sha1 = "49bd7d5b2c773a078b076f0d491c9c30b4715586"
registries = "General"
uuid = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
version = "0.4.25"

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
git-tree-sha1 = "38ef52f18cbdfa783ec6c25acc23fb1e8525ec16"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.10.2"

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

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "f4d904e7ff5aadd8e7fe9ab8544a0a295dfa8904"
registries = "General"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.3"

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

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "462ab5e6578928370f632705e96af82f6c1c09a1"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "ec46c5825710fa1a15d468acb2d93cc939a7a5fe"
registries = "General"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.13"

    [deps.GeometryBasics.extensions]
    ExtentsExt = "Extents"
    GeometryBasicsGeoInterfaceExt = "GeoInterface"
    IntervalSetsExt = "IntervalSets"

    [deps.GeometryBasics.weakdeps]
    Extents = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"

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

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "aws_c_s3_jll", "dlfcn_win32_jll", "libaec_jll", "mpif_jll"]
git-tree-sha1 = "d791a1c4c3c8f383b248a799be31105d00310204"
registries = "General"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "2.2.4+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "59d266e426ab9695f625fe2b6a51c2d3a04f1ecd"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.15001.0+0"

[[deps.HelpPlots]]
git-tree-sha1 = "4b09f61ba5ab9242250855d44cf8747d0d43b981"
registries = "General"
uuid = "5880b2e8-ad0a-48c9-bacd-30dc9c1357e0"
version = "1.1.1"

    [deps.HelpPlots.extensions]
    HelpPlotsExt = "Plots"

    [deps.HelpPlots.weakdeps]
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "b60662344c62a67fa474dc50b7b06e38b23b39fa"
registries = "General"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.14.0+2"

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
git-tree-sha1 = "b731f8bd40983d4e2ae6d3e73b6e8fbab272949e"
registries = "General"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.14"

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

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "19fa81989298fc2fb103fee29b03c5284f17e8a5"
registries = "General"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.13"

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

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
registries = "General"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

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
git-tree-sha1 = "52702dc5aba6392de19d734750c37ffb08515dc0"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.5"

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

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7dc3b0621d7942324cc3ed3272b4a491fd5cffa7"
registries = "General"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.11.1+0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

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

[[deps.NCDatasets]]
deps = ["CFTime", "CommonDataModel", "DataStructures", "Dates", "DiskArrays", "NetCDF_jll", "NetworkOptions", "Printf"]
git-tree-sha1 = "5eb7747d10437f5acb2675c1f865ffa0353f3f9c"
registries = "General"
uuid = "85f8d34a-cbdd-5861-8df4-14fed0d494ab"
version = "0.14.15"

    [deps.NCDatasets.extensions]
    NCDatasetsMPIExt = "MPI"

    [deps.NCDatasets.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "676c0f1c1c34224d6869675047a54b42fdc7b8c5"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.5"

[[deps.NetCDF_jll]]
deps = ["Artifacts", "Blosc_jll", "Bzip2_jll", "HDF5_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML", "XML2_jll", "Zlib_jll", "Zstd_jll", "libaec_jll", "libzip_jll"]
git-tree-sha1 = "62410b2b2ce08416a435f550c404a42a2e7e2b4b"
registries = "General"
uuid = "7243133f-43d8-5620-bbf4-c2c921802cf3"
version = "401.1000.102+1"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
registries = "General"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
registries = "General"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

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
git-tree-sha1 = "1e697ad6f68af89cd3651bbe9b4a7b5d800adf5a"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.4.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

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
git-tree-sha1 = "dd58036d3b9b072a1ad7c911b5da397862f8aa63"
registries = "General"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.3"

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

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

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
git-tree-sha1 = "15f266d7b3039ca0608d2c53bb32225b96932310"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.14"

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

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

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

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

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
git-tree-sha1 = "d1a08b3cb5d1470a2b6dc98ba644f1436d5a740f"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.2"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

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
git-tree-sha1 = "bf7e2d0d9ed9ac2a652fd598c0f784910cc759f2"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.5"

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

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "4daaddbe37aa80b4551bf7c56adf53e7cfcdf35a"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.4"

    [deps.StructArrays.extensions]
    AdaptExt = "Adapt"
    GPUArraysCoreExt = "GPUArraysCore"
    KernelAbstractionsExt = "KernelAbstractions"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"
    StaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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
git-tree-sha1 = "caba7c595e8a8c4e9fb72c9b08c14f7a131b7a58"
registries = "General"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.10"

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
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
registries = "General"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

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

[[deps.aws_c_auth_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_http_jll", "aws_c_sdkutils_jll"]
git-tree-sha1 = "7945f94b714d20b55aee2510b511afee0ee17ad2"
registries = "General"
uuid = "2b3700d1-4306-52e2-a478-c162f0c514be"
version = "1.0.1+0"

[[deps.aws_c_cal_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "60a8d3fafed6f7ed8b3114793885a03330aa8e51"
registries = "General"
uuid = "70f11efc-bab2-57f1-b0f3-22aad4e67c4b"
version = "1.0.0+0"

[[deps.aws_c_common_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f5ce8191ad02a5a4127d37ab5f474aee5cf91ad4"
registries = "General"
uuid = "73048d1d-b8c4-5092-a58d-866c5e8d1e50"
version = "1.0.3+1"

[[deps.aws_c_compression_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "6efd591b9fae7919266d3b342a18a6e6280850f9"
registries = "General"
uuid = "73a04cd5-f3d7-5bac-9290-e8adb709f224"
version = "1.0.0+1"

[[deps.aws_c_http_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_compression_jll", "aws_c_io_jll"]
git-tree-sha1 = "e4108b4a2201cb9987ddd36714fdcfd572d6bb24"
registries = "General"
uuid = "3254fc65-9028-534d-aa9d-d76d128babc6"
version = "1.0.0+0"

[[deps.aws_c_io_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_common_jll", "s2n_tls_jll"]
git-tree-sha1 = "b816756403a8db80e32fc23a4a1aae240d2168ca"
registries = "General"
uuid = "13c41daa-f319-5298-b5eb-5754e0170d52"
version = "1.0.0+0"

[[deps.aws_c_s3_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_auth_jll", "aws_c_common_jll", "aws_c_http_jll", "aws_checksums_jll", "s2n_tls_jll"]
git-tree-sha1 = "1ebf5d4fa9c5248c3c1066124ef41d06de05a03a"
registries = "General"
uuid = "bd1f34fb-993f-5903-a121-aaf302eed6d4"
version = "1.3.1+0"

[[deps.aws_c_sdkutils_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "0162699e5aa99420f04a9606a9d7b38a872a81ee"
registries = "General"
uuid = "1282aa60-004d-510b-9f52-12498d409daa"
version = "1.0.0+0"

[[deps.aws_checksums_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "ee1f5427815c98355b0b0454b5fe79806ffdf935"
registries = "General"
uuid = "b2a88e68-78e7-5e94-8c20-c02986ec140e"
version = "1.0.1+1"

[[deps.dlfcn_win32_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e141d67ffe550eadfb5af1bdbdaf138031e4805f"
registries = "General"
uuid = "c4b69c83-5512-53e3-94e6-de98773c479f"
version = "1.4.2+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
registries = "General"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "31e4c16405815484827f75a38fb7eb17432e72e3"
registries = "General"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.8+0"

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

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c4b294e893e8ebbb8788b68b1d56737c0f01557b"
registries = "General"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.12.1+0"

[[deps.mpif_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML"]
git-tree-sha1 = "52e322f7a6d418576cdeb043dd2a7f7ad9a8abd0"
registries = "General"
uuid = "9aeb927a-4695-514f-a259-621a69f20ec0"
version = "1.0.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.70.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.5+0"

[[deps.s2n_tls_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f9611b1b99beb089154f19a4031579f3a818cbd"
registries = "General"
uuid = "cddc5d3d-934d-5d3a-9747-62fc12ea3f48"
version = "1.7.12+0"

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
# ╟─91623e61-db88-4a1e-b7e6-25c23b3e1260
# ╟─ae9a3c8c-e3d7-4f6a-af26-91efa98b9db6
# ╟─3a691689-54cc-4bd9-a4e0-2159e8a2dd53
# ╟─cbbb6280-4384-46d9-8e1f-1d7d3e5a26d1
# ╟─2ce0dfa9-f800-4eb0-b809-49826e7eef76
# ╟─27e48d3d-859a-48bc-8324-00e73496a7ff
# ╟─4621f04d-a10d-4a6e-9428-789a580cae3d
# ╟─a80dc238-4349-40e8-a4a0-9a5e76aaf61a
# ╠═80e5c2e7-6db2-4d80-bef1-c7632da09ed9
# ╟─5c15ae8f-bcfb-4f5d-bf28-b376328b7f6e
# ╟─b0e06ea4-3962-44c3-b250-1067d559454b
# ╟─cb9899d1-0837-42b3-88fd-6090fb831e6f
# ╟─4f5cf0d4-2ea8-4e62-841d-1b09111769be
# ╟─829221ca-5325-4e0b-9d65-bce670004840
# ╟─0d0ce3b4-b845-4f7f-bbb4-9a8ea9f296ed
# ╟─05401c11-43ad-4c09-8a1e-7ea95d128c09
# ╠═84c481fd-6ad9-417b-877a-befc212cd4ae
# ╟─c4e14bdc-d063-4150-9a62-e48ecfb6787b
# ╟─43f051bb-1dbe-4f3d-b11f-94cde9e01815
# ╟─7611f1fd-8680-423c-b52a-001b24d0601c
# ╟─076f663d-281c-4c97-9f1d-f63ce011337f
# ╟─96cef172-fb67-4e7b-a35a-8c3cf693b099
# ╟─75ce0ac0-6a5c-4e8d-a1ea-84ef5598a2ef
# ╟─b499be47-5368-4c3b-9e19-3c9a8d786ac0
# ╟─f34fe12c-88e9-416a-9b7f-89c27f705696
# ╟─523a3e0a-df64-4772-ac00-b8cd94321626
# ╟─566e0aca-d2b0-4071-aaa1-061cd858c3bc
# ╠═3494a6b6-d761-4b1b-a398-891bbec22c5a
# ╟─00226ea4-663e-4034-878d-2401028768dc
# ╠═d4db5136-89ef-4b01-8f65-2d92ce74adeb
# ╟─1ae2ed62-a8e9-4ae6-a5b6-dcf9f537e273
# ╠═3020f6fd-4c81-4063-873e-1231c4f7eadb
# ╟─5430b003-fe8c-4e43-bce8-a9423705ef70
# ╠═fc87feae-8962-40fd-873e-d5068e1b4287
# ╟─511f6251-8445-4484-9d89-e94d102c8940
# ╟─86bd1203-b030-4f70-b9f4-601af8e96e42
# ╠═1889409d-d5d5-45cb-8386-1a525acd3e90
# ╟─425e9917-fa9e-45df-a046-69dc8d64bb05
# ╠═149f84fd-935e-4089-9c49-0a663368d586
# ╟─0c9e15a0-cdc8-43ef-a2d5-10f720707ce4
# ╠═ddfa0a8c-34ca-4817-8986-3c4a698f8c4e
# ╠═27a20176-a359-4a41-8385-5a098885489a
# ╠═f6d14578-452b-4f8f-add9-1f89ed8b22e9
# ╟─59869cca-4917-4757-aa13-e62ff9330efd
# ╠═61c05b91-c56d-4373-927e-4da6881ca2be
# ╟─7b1a6aed-13de-4739-ae0c-522ac9e0d3fe
# ╟─6b549874-6cc2-4078-ad99-83af21f3df0d
# ╟─652c59a9-0fb1-4d74-bf75-cf6688ee45fe
# ╟─3ae7b80b-b2ac-481b-9d3d-aeda8f6cc9e6
# ╠═cd2ec92c-6e1f-462a-b585-2598b3c42155
# ╟─880d45fc-e31a-4367-8779-f612ce736fe8
# ╟─c05b017e-4779-489a-962d-91926577e07f
# ╠═83b1054a-d95a-4bca-a219-c1eaed0ae83a
# ╟─16adfbd5-1653-496d-b6fc-ca10d8417b48
# ╠═28b9a1be-96ca-40f7-8bc2-4e9cf3994fe6
# ╟─0d11739c-14a6-4435-980e-22f93a422833
# ╟─c003091a-8a14-4a9d-898a-86cd27fa926d
# ╟─d4349e75-f356-4696-87c4-67465e6eb687
# ╟─db567831-19b6-4ca7-8abf-05e81e3d00cd
# ╟─48bee702-7b53-4558-ba2b-461d13c4d75c
# ╟─8f2520be-30e5-442b-8c4c-5c04df6589da
# ╟─519f12cd-3b0e-4c21-8cce-9b6c1bac10e2
# ╠═ffddcac5-7d51-46dc-baff-acb815bb28b1
# ╟─4e670940-e9c0-4926-84b7-e0e41a00e92b
# ╠═b195ea93-3a9d-4607-b8e5-6d0c7ffdb8fe
# ╟─2d1daf3c-c577-41d2-a988-774999f12fdc
# ╟─18a73da5-75cc-4d8d-848d-bcba85bc16cc
# ╟─64819290-2e59-434b-86fa-279532560ffd
# ╟─a192d708-a790-4902-86cb-7ebb9d110a48
# ╟─406fbeee-1583-4c7e-a318-5009d186c1d4
# ╟─e8d57055-b03a-4a43-9c27-bb51dfa174f6
# ╠═63038188-e602-45f8-a355-3faeca092ad7
# ╟─d874cea2-7214-4fe3-8906-32e2c769c747
# ╟─b11a6e32-8568-41b3-b071-082f36aa33c8
# ╟─3ae070e5-6dbb-47ba-89f2-312dc7f9268f
# ╟─1cbcb64e-f233-421b-bdf4-ef6be8a696e6
# ╟─1290d041-a09b-445c-b5ff-e484be47f3ad
# ╠═637ff736-78f3-4cb0-85d6-f966d032027a
# ╟─f732b713-9754-490d-8c6a-1693cdaaafe7
# ╟─57f29cd6-c8af-49cf-8d78-3882f0f353a3
# ╟─43688d97-1248-4756-98a5-152054c8a16c
# ╟─3472f397-335c-425b-b19d-09842fc91885
# ╟─947d749e-1847-4d8b-92ca-9be2f25ec1cd
# ╟─dd9201fc-3a6f-485a-9403-81605597e83c
# ╟─98e876ad-55f2-446b-b9fb-bb78c30e8a61
# ╟─9570641e-f3d1-4a2c-944a-8696313faf13
# ╟─a07d7f34-e4d4-49a5-be93-56cd84b2ee34
# ╟─a4ca19fb-8262-4d56-a0dc-c42e029e2995
# ╟─b681d694-31fd-4d02-b794-e39fa928f792
# ╟─a1be5695-b54c-48a2-b153-207b4f231e8e
# ╠═d0ca6d74-be30-4ec1-8aeb-88fe0fd2a444
# ╠═8c734e9a-af1f-4a95-8df6-07999529b0ea
# ╟─4b79a694-99fa-459a-aa82-bcdaf1372a6a
# ╟─22910abe-d941-48a0-b4c6-2a6be24b6f5d
# ╟─68c244a3-9089-4d61-988b-216967b43790
# ╟─7649806a-f858-4357-933d-7724a098be72
# ╟─88758f9f-4599-4300-8ce1-c713f4d36955
# ╟─587f6b66-dd66-477d-afd4-228d5010f637
# ╠═a2fcf8e5-44ce-442e-916d-30df3ec10f53
# ╟─b861c7bb-b662-4ddd-9bce-3385bde2ca54
# ╠═9feb44a6-a77c-4c49-bf8d-c23190f01678
# ╟─ac62481e-fde7-4345-9214-97046d49ec27
# ╟─ff0c1dda-0fd1-4f16-af64-a832e6702948
# ╟─d73085ca-0234-411c-8a01-b550084b7895
# ╟─e6c40bdb-afb8-414b-ad8f-65f988bd92d5
# ╟─c9a1bd7e-69cb-4926-8043-3ddfd536568c
# ╠═4474c72e-97e1-4ccd-88a8-04d987e59127
# ╠═3e601fe9-9eed-40fb-8e86-14b218286b44
# ╟─03cbe4a0-0550-4536-add4-a59466629330
# ╟─7ce90062-0426-454f-b160-7e62c963b1e4
# ╟─6388e1dd-b7ad-46a8-9104-526a3c9c7c7c
# ╟─e407ff9a-dd14-4e3b-b16c-84033c10ec80
# ╠═6bb4e870-0ea1-4888-aa12-0f655801d77e
# ╟─05f57e2a-21a3-40a6-96e7-4cb086df7ee7
# ╟─0099469c-4c36-45a6-999e-e2f58368bd38
# ╟─791bec76-e199-4883-a0c1-f8c8fba38a22
# ╠═ab84215c-7bd0-4844-992d-eb2cabf98089
# ╟─0c825e83-6859-4dad-9763-f6c3dec95d9e
# ╟─6e6d532c-d6ce-485c-97e9-ef75102d35c6
# ╟─e4efa161-4d66-428b-86c1-e5f17491d3a9
# ╟─489ca359-87b8-430d-ac01-0dda30f0e95c
# ╠═8792bee1-15f3-4488-8f58-a5f06c08e99e
# ╟─0af11048-6e25-4b8a-821d-b54af3dd50b2
# ╟─7183c932-16ca-4b72-9362-f3c83a377a4b
# ╟─c09bb892-2f43-4ec5-9256-af2543e2eaab
# ╟─c5c0863f-a83f-44b4-9c42-0cb758b77d5e
# ╠═8c12d86f-b478-484e-87e6-8d29e7dcdc3e
# ╟─9a4c5a4c-c25d-4916-9fba-495002f66344
# ╟─47a8bb91-8196-48eb-b6d4-63ad62117f0c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
