# Render notebooks to static HTML in assets/.
#
#     julia --project scripts/render.jl                  # all notebooks
#     julia --project scripts/render.jl radiative_balance.jl

using Pluto

const ROOT = dirname(@__DIR__)
const NOTEBOOKS = ["radiative_balance.jl", "numerical_fluid_dynamics.jl", "oceananigans_near_global_demo.jl"]

# Bond values to set before export, so the HTML shows the simulation results.
const BONDS = Dict("oceananigans_near_global_demo.jl" => Dict(:run_simulation => true))

function render(name)
    workdir = mktempdir()
    path = cp(joinpath(ROOT, name), joinpath(workdir, name))

    session = Pluto.ServerSession()
    notebook = Pluto.SessionActions.open(session, path; run_async = false)

    bonds = get(BONDS, name, Dict())
    if !isempty(bonds)
        for (sym, value) in bonds
            notebook.bonds[sym] = Pluto.BondValue(value)
        end
        Pluto.set_bond_values_reactive(; session, notebook, bound_sym_names = collect(keys(bonds)), run_async = false)
    end

    errored = [c for c in notebook.cells if c.errored]
    isempty(errored) || @warn "$name: $(length(errored)) cells errored" cells = [c.code for c in errored]

    html = joinpath(mkpath(joinpath(ROOT, "assets")), replace(name, r"\.jl$" => ".html"))
    write(html, Pluto.generate_html(notebook))
    Pluto.SessionActions.shutdown(session, notebook; async = false)
    @info "Wrote $html"
end

foreach(render, isempty(ARGS) ? NOTEBOOKS : ARGS)
