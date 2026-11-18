# Render notebooks to static HTML in assets/. See README.md in this folder.

using Pluto

const ROOT = dirname(@__DIR__)
const NOTEBOOKS = ["radiative_balance.jl", "numerical_fluid_dynamics.jl", "oceananigans_near_global_demo.jl"]

# Bond values to set before export, so the HTML shows the simulation results.
const BONDS = Dict("oceananigans_near_global_demo.jl" => [:run_simulation => true])

function render(name)
    name = basename(name)
    path = cp(joinpath(ROOT, name), joinpath(mktempdir(), name))

    session = Pluto.ServerSession()
    notebook = Pluto.SessionActions.open(session, path; run_async = false)

    bonds = get(BONDS, name, [])
    for (sym, value) in bonds
        notebook.bonds[sym] = Pluto.BondValue(value)
    end
    Pluto.set_bond_values_reactive(; session, notebook, bound_sym_names = first.(bonds), run_async = false)

    any(c -> c.errored, notebook.cells) && error("$name has errored cells")

    html = joinpath(mkpath(joinpath(ROOT, "assets")), replace(name, r"\.jl$" => ".html"))
    write(html, Pluto.generate_html(notebook))
    Pluto.SessionActions.shutdown(session, notebook; async = false)
    @info "Wrote $html"
end

foreach(render, isempty(ARGS) ? NOTEBOOKS : ARGS)
