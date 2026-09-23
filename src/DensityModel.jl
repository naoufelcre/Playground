module DensityModel

using Gridap
using GridapEmbedded
using Gridap.TensorValues
using IterativeSolvers
using Preconditioners
using EvolvingDomains
using EvolvingDomains.Geometric
using EvolvingDomains.Kinematic
using Dates
using JSON
using LinearAlgebra
using Random
using SparseArrays

using StaticAssembler

include("Parameters.jl");include("Geom.jl")
include("StaticForms.jl");using .StaticForms;include("DensityVarForm.jl")
include("Helpers/Helpers.jl");using .Helpers
include("TensorHelpers.jl");using .TensorHelpers
include("blocks/InitBlock.jl")
include("State.jl");include("Backends.jl");include("Evolution.jl")
include("blocks/ExportBlock.jl")


_default_run_config(seed::AbstractString) = (
    seed=String(seed),
    horizon=1.0,
    save_stride=10,
    compact=true,
    export_images=false,
)

_run_id(model, seed) = "$(model)_$(replace(seed, r"[^A-Za-z0-9._-]" => "-"))_" *
    "$(Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))_" *
    randstring("abcdefghijklmnopqrstuvwxyz0123456789", 6)

function _validate_config_keys(values, allowed, section)
    values isa AbstractDict || throw(ArgumentError("$section must be a JSON object"))
    unknown = sort!([string(key) for key in keys(values) if string(key) ∉ allowed])
    isempty(unknown) || throw(ArgumentError(
        "Unknown $section key(s): $(join(unknown, ", "))"))
    return values
end

function load_run_config(selection::AbstractString)
    value = strip(selection)
    isempty(value) && return _default_run_config(
        randstring("abcdefghijklmnopqrstuvwxyz0123456789", 12))

    path = expanduser(value)
    if !isfile(path)
        endswith(lowercase(value), ".json") &&
            throw(ArgumentError("JSON specification not found: $value"))
        return _default_run_config(value)
    end

    spec = try
        JSON.parsefile(path)
    catch exception
        exception isa InterruptException && rethrow()
        throw(ArgumentError(
            "Could not read JSON specification '$value': $(sprint(showerror, exception))"))
    end
    _validate_config_keys(spec, ("seed", "horizon", "export"), "configuration")

    seed = get(spec, "seed", randstring("abcdefghijklmnopqrstuvwxyz0123456789", 12))
    seed isa String && !isempty(strip(seed)) ||
        throw(ArgumentError("seed must be a non-empty string"))

    horizon = get(spec, "horizon", 1.0)
    horizon isa Real && !(horizon isa Bool) && isfinite(horizon) && horizon > 0 ||
        throw(ArgumentError("horizon must be a finite number greater than zero"))

    export_config = get(spec, "export", Dict{String,Any}())
    _validate_config_keys(
        export_config, ("save_stride", "compact", "images"), "export")

    save_stride = get(export_config, "save_stride", 100)
    save_stride isa Integer && !(save_stride isa Bool) &&
        0 < save_stride <= typemax(Int) ||
        throw(ArgumentError("export.save_stride must be a positive integer"))

    compact = get(export_config, "compact", true)
    compact isa Bool || throw(ArgumentError("export.compact must be true or false"))
    export_images = get(export_config, "images", false)
    export_images isa Bool || throw(ArgumentError("export.images must be true or false"))

    return (
        seed=String(strip(seed)),
        horizon=Float64(horizon),
        save_stride=Int(save_stride),
        compact,
        export_images,
    )
end

function _write_run_config(output_dir, config)
    mkpath(output_dir)
    open(joinpath(output_dir, "config.json"), "w") do io
        JSON.print(io, Dict(
            "seed" => config.seed,
            "horizon" => config.horizon,
            "export" => Dict(
                "save_stride" => config.save_stride,
                "compact" => config.compact,
                "images" => config.export_images,
            ),
        ), 2)
        println(io)
    end
    return nothing
end


# One simulation step, in order:
#   discretize!       geometry + projected-operator cache
#   restrict_fields!  coefficient fields (ρ, ε) + master-DOF initial guess
#   assemble_system!  projected operator EᵀAE and right-hand side Eᵀb
#   solve_velocity!   CG with incomplete LDL; GMRES if the factor is indefinite
#   recover_velocity! extend the master-DOF solution to all grid nodes
#   evolve!           advance level set → strain → advect → swap buffers
#   maintain          island filtering + level-set reinitialization
#
function main(; seed::String="seeded-holes-1", horizon=1.0,
              nsteps::Union{Nothing,Int}=nothing,
              output_dir::Union{Nothing,String}=nothing,
              save_stride::Int=100, compact::Bool=true,
              export_images::Bool=false)
    horizon isa Real && !(horizon isa Bool) && isfinite(horizon) && horizon > 0 ||
        throw(ArgumentError("horizon must be a finite number greater than zero"))
    save_stride > 0 || throw(ArgumentError("save_stride must be positive"))
    state, backend, run = initialize(; seed, horizon, nsteps)
    tmap = nothing
    step = 0
    field_range = (0.0, 0.125)
    stress_field = similar(state.ρ.data)
    stress_time = Float64[]
    stress_hist = Float64[]
    vm_hist = Float64[]
    mass_hist = Float64[]
    meandens_hist = Float64[]
    rel(v) = (r = first(v); iszero(r) ? v : v ./ r)

    function export_state!(frame, stride=save_stride)
        isnothing(output_dir) && return nothing
        export_blocks!(frame, stride, output_dir, basename(normpath(output_dir)),
            state.info, compact, export_images, state.t, state.geom, state.ρ,
            state.ε, state.v_grid)
    end

    export_state!(0)

    while state.t < run.horizon && (isnothing(run.max_steps) || step < run.max_steps)
        step += 1
        iteration_start = time()
        #println("\nstep $step, t = $(state.t) / $(run.horizon), threads = $(Threads.nthreads())")
        discretize!(backend, state)
        coefficients = restrict_fields!(backend, state)
        operator, rhs = assemble_system!(backend, coefficients)
        velocity = solve_velocity!(backend, operator, rhs)
        recover_velocity!(backend, state, velocity)

        v_extended = extend(state.geom, state.v_grid)
        timestep = stable_timestep(
            state, v_extended, run.horizon - state.t)
        state.Δt = timestep.Δt

        tmap = evolve!(state, tmap, backend.plan.refresh, v_extended)
        state.t = min(run.horizon, state.t + state.Δt)
        maintain_geometry!(state, run.reinit_freq, run.min_island_nodes, step)
        iteration_time = time() - iteration_start
        # ponytail: node quadrature masked by level set, not cut-cell quadrature
        stress = stress_norm_squared(state)
        stress_field!(stress_field, state)
        push!(stress_time, state.t)
        push!(stress_hist, stress)
        push!(vm_hist, von_mises_stress(state))
        ρd = state.ρ.data
        lsv = state.geom.levelset
        n_in = 0
        m_in = 0.0
        @inbounds for i in eachindex(ρd, lsv)
            if lsv[i] < 0
                n_in += 1
                m_in += ρd[i]
            end
        end
        dA = state.info.spacing[1] * state.info.spacing[2]
        push!(mass_hist, m_in * dA)
        push!(meandens_hist, m_in / n_in)
        Y = hcat(rel(stress_hist), rel(vm_hist), rel(mass_hist), rel(meandens_hist))
        EvolvingDomains.plot(state.geom, stress_time, Y;
            field=stress_field, colorrange=field_range,
            labels=["∫|σ|²", "∫σ_vm", "mass", "mean ρ"],
            ylabel="value / value₀", curvetitle="relative traces",
            xrange=(0.0, run.horizon),
            label="stress |σ| - step $step, " *
                  "$(round(iteration_time; sigdigits=3)) s/iter, " *
                  "t = ~$(round(state.t;sigdigits=4)) / $(run.horizon), " *
                  "CFL = $(round(timestep.cfl; sigdigits=3)), " *
                  "∫|σ|² = $(round(stress; sigdigits=3)), " *
                  "threads = $(Threads.nthreads())")
        export_state!(step)
    end
    !isnothing(output_dir) && step % save_stride != 0 && export_state!(step, 1)
    if state.t < run.horizon
        @warn "Stopped at the accepted-step limit before reaching the horizon" step=step time=state.t horizon=run.horizon
    end

    return (
        density=copy(state.ρ.data),
        strain=ntuple(component -> copy(state.ε[component].data), 3),
        velocity=copy(state.v_grid.data),
    )
end


function cli_main(input::IO=stdin, output::IO=stdout, error_output::IO=stderr)::Cint
    try
        print(output, "Seed or JSON specification path (blank = generate seed):\n> ")
        flush(output)
        eof(input) && throw(ArgumentError("No seed or JSON specification provided"))
        config = load_run_config(readline(input))

        run_id = _run_id("density", config.seed)
        output_dir = joinpath(pwd(), "export", run_id)
        _write_run_config(output_dir, config)

        println(output, "\nSeed: $(config.seed)")
        println(output, "Export: $output_dir")
        println(output, "Starting simulation...")
        flush(output)

        main(; seed=config.seed, horizon=config.horizon, output_dir,
            save_stride=config.save_stride, compact=config.compact,
            export_images=config.export_images)

        println(output, "\nDensityModel completed.")
        println(output, "Results: $output_dir")
        return 0
    catch exception
        if exception isa InterruptException
            println(error_output, "\nDensityModel interrupted.")
            return 130
        end
        println(error_output, "Error: ", sprint(showerror, exception))
        return 1
    end
end

function julia_main()::Cint
    Base.invokelatest(
        Base.include, Main, joinpath(pkgdir(DensityModel), "src", "CellShapeModel.jl"))
    return Base.invokelatest(getfield(Main, :CellShapeModel).cli_main)
end


if abspath(PROGRAM_FILE) == @__FILE__
    exit(cli_main())
end

end
