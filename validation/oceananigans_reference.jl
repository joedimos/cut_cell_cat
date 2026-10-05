# Optional integration harness. Needs Julia, network, and package installation.
# This script is not run by the Python-only CI job.
using Pkg
const PIN = "819a245b837041d01fcf273d74a8b45db81ea14a"
const output = length(ARGS) == 1 ? ARGS[1] : "oceananigans.csv"
Pkg.activate(mktempdir())
Pkg.add(PackageSpec(url="https://github.com/CliMA/Oceananigans.jl", rev=PIN))
using Oceananigans
using DelimitedFiles

N, κ, Δt, steps = 32, 0.1, 0.001, 20
grid = RectilinearGrid(size=N, z=(0, 1), topology=(Flat, Flat, Bounded))
model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(κ=κ), tracers=:c,
                           timestepper=:RungeKutta3)
# The function supplies cell-average values for this uniform cosine profile.
set!(model, c=z -> 0.5 + 0.2 * cos(π*z) * sinc(1/(2N)))
simulation = Simulation(model; Δt, stop_iteration=steps)
run!(simulation)
z = collect(znodes(model.tracers.c))
c = vec(Array(interior(model.tracers.c)))
open(output, "w") do io
    println(io, "# Oceananigans commit=" * PIN)
    println(io, "z,c,time")
    writedlm(io, hcat(z, c, fill(model.clock.time, N)), ',')
end
cp(joinpath(dirname(Base.active_project()), "Manifest.toml"), output * ".Manifest.toml"; force=true)
println("Wrote ", output)
