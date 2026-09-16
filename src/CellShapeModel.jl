let project_root = normpath(joinpath(@__DIR__, ".."))
    (Base.active_project() == joinpath(project_root, "Project.toml") ||
     project_root in LOAD_PATH) ||
        pushfirst!(LOAD_PATH, project_root)
end

module CellShapeModel

using Dates
using EvolvingDomains
using JSON
using Random
using DensityModel

include("CellShapeVarForm.jl")
using .CellShapeVarForm

const DEFAULT_DIRECTIONAL_ACTIVITY = 1.0

initialize(; kwargs...) = DensityModel.initialize(; kwargs...)

function _parse_directional_activity(value)
    text = strip(value)
    isempty(text) && return DEFAULT_DIRECTIONAL_ACTIVITY
    β = tryparse(Float64, text)
    !isnothing(β) && isfinite(β) && β >= 0 ||
        throw(ArgumentError("β must be a finite nonnegative number"))
    return β
end

function load_cli_config(args=String[]; input::IO=stdin, output::IO=stdout)
    length(args) <= 2 || throw(ArgumentError(
        "usage: julia --project=. src/CellShapeModel.jl [seed-or-json] [β]"))

    selection = if isempty(args)
        print(output, "Seed or JSON specification path (blank = generate seed):\n> ")
        flush(output)
        eof(input) && throw(ArgumentError("No seed or JSON specification provided"))
        readline(input)
    else
        args[1]
    end

    β = if length(args) == 2
        _parse_directional_activity(args[2])
    elseif isempty(args)
        print(output, "Directional activity β [$DEFAULT_DIRECTIONAL_ACTIVITY]:\n> ")
        flush(output)
        eof(input) ? DEFAULT_DIRECTIONAL_ACTIVITY :
            _parse_directional_activity(readline(input))
    else
        DEFAULT_DIRECTIONAL_ACTIVITY
    end

    config = DensityModel.load_run_config(selection)
    return (; config..., β)
end

function _write_nematic_config(output_dir, β)
    open(joinpath(output_dir, "nematic.json"), "w") do io
        JSON.print(io, Dict(
            "model" => "active density-strain Kelvin-Voigt",
            "reference_density" => 1.0,
            "constitutive_density_floor" => CellShapeVarForm.MIN_CONSTITUTIVE_DENSITY,
            "directional_activity" => β,
            "area" => "1 / density",
            "shape" => "dev(strain)",
            "active_stress" =>
                "A * (zeta_m(density) * I + 2 * beta * density * dev(strain))",
            "relaxation" => "mechanical deformation only",
            "indicator" => "density * principal strain difference",
            "postprocessing" => Dict(
                "nematic_xx" => "0.5 * (strain_xx - strain_yy)",
                "nematic_xy" => "strain_xy",
                "density_order" =>
                    "density * hypot(strain_xx - strain_yy, 2 * strain_xy)",
            ),
        ), 2)
        println(io)
    end
    return nothing
end

function _observables!(state)
    return observables!(
        state.ε_target[1].data,
        state.ε_target[3].data,
        state.ε_target[2].data,
        state.ρ,
        state.ε,
    )
end

function _constitutive_coefficients(state, base_coefficients)
    density = EvolvingDomains.extend(state.geom, state.ρ)
    strain = ntuple(i -> EvolvingDomains.extend(state.geom, state.ε[i]), 3)
    return (
        density.data,
        strain[1].data,
        strain[2].data,
        strain[3].data,
        base_coefficients[5],
    )
end

"""
    main(; seed, β=1, ...)

Run the opt-in active density-strain Kelvin-Voigt model. Density represents
cell area through `1/ρ`; `dev(ε)` is the small shape strain.
"""
function main(; seed::String="seeded-holes-1",
              β=DEFAULT_DIRECTIONAL_ACTIVITY,
              horizon=1.0,
              nsteps::Union{Nothing,Int}=nothing,
              output_dir::Union{Nothing,String}=nothing,
              save_stride::Int=100,
              compact::Bool=true,
              export_images::Bool=false)
    horizon isa Real && !(horizon isa Bool) && isfinite(horizon) && horizon > 0 ||
        throw(ArgumentError("horizon must be a finite number greater than zero"))
    save_stride > 0 || throw(ArgumentError("save_stride must be positive"))

    state, _, run = initialize(; seed, horizon, nsteps)
    form = augment_form(DensityModel.kelvin_voigt_form(state.p), state.p, β)
    backend = DensityModel.ProjectedBackend(form=form)
    tmap = nothing
    step = 0
    sim_id = isnothing(output_dir) ? "nematic" : basename(normpath(output_dir))

    function export_state!(frame, stride=save_stride)
        isnothing(output_dir) && return nothing
        DensityModel.export_blocks!(
            frame, stride, output_dir, sim_id, state.info,
            compact, export_images, state.t, state.geom, state.ρ,
            state.ε, state.v_grid)
        return nothing
    end

    export_state!(0)

    while state.t < run.horizon &&
          (isnothing(run.max_steps) || step < run.max_steps)
        step += 1
        iteration_start = time()
        DensityModel.discretize!(backend, state)
        base_coefficients = DensityModel.restrict_fields!(backend, state)
        coefficients = _constitutive_coefficients(state, base_coefficients)
        operator, rhs = DensityModel.assemble_system!(backend, coefficients)
        velocity = DensityModel.solve_velocity!(backend, operator, rhs)
        DensityModel.recover_velocity!(backend, state, velocity)

        v_extended = EvolvingDomains.extend(state.geom, state.v_grid)
        timestep = DensityModel.stable_timestep(
            state, v_extended, run.horizon - state.t)
        state.Δt = timestep.Δt

        tmap = CellShapeVarForm.evolve!(
            state, tmap, backend.plan.refresh, v_extended)
        state.t = min(run.horizon, state.t + state.Δt)
        DensityModel.maintain_geometry!(
            state, run.reinit_freq, run.min_island_nodes, step)
        iteration_time = time() - iteration_start

        _, _, indicator = _observables!(state)
        # ponytail: node quadrature masked by level set, not cut-cell quadrature
        stress = CellShapeVarForm.stress_norm_squared(state, state.p, β)
        EvolvingDomains.plot(state.geom;
            field=indicator,
            label="density × principal strain difference - step $step, " *
                  "$(round(iteration_time; sigdigits=3)) s/iter, " *
                  "t = $(state.t) / $(run.horizon), " *
                  "max = $(round(maximum(indicator); sigdigits=3)), " *
                  "∫|σ|² = $(round(stress; sigdigits=3)), " *
                  "threads = $(Threads.nthreads())")
        export_state!(step)
    end

    !isnothing(output_dir) && step % save_stride != 0 && export_state!(step, 1)
    if state.t < run.horizon
        @warn "Stopped at the accepted-step limit before reaching the horizon" step=step time=state.t horizon=run.horizon
    end

    nematic_xx, nematic_xy, density_order = _observables!(state)
    return (
        density=copy(state.ρ.data),
        strain=ntuple(component -> copy(state.ε[component].data), 3),
        nematic=(xx=copy(nematic_xx), xy=copy(nematic_xy)),
        density_order=copy(density_order),
        velocity=copy(state.v_grid.data),
    )
end

function _usage(io)
    println(io, "Usage:")
    println(io, "  julia --project=. src/CellShapeModel.jl [seed-or-json] [β]")
    println(io, "  β defaults to $DEFAULT_DIRECTIONAL_ACTIVITY")
end

function cli_main(args=ARGS; input::IO=stdin, output::IO=stdout,
                  error_output::IO=stderr)::Cint
    if length(args) == 1 && args[1] in ("-h", "--help")
        _usage(output)
        return 0
    end
    try
        config = load_cli_config(args; input, output)
        run_id = DensityModel._run_id("cell-shape", config.seed)
        output_dir = joinpath(pwd(), "export", run_id)
        DensityModel._write_run_config(output_dir, config)
        _write_nematic_config(output_dir, config.β)

        println(output, "\nSeed: $(config.seed)")
        println(output, "Directional activity β: $(config.β)")
        println(output, "Export: $output_dir")
        println(output, "Starting active density-strain simulation...")
        flush(output)

        main(; seed=config.seed, β=config.β, horizon=config.horizon,
             output_dir, save_stride=config.save_stride,
             compact=config.compact, export_images=config.export_images)

        println(output, "\nActive density-strain simulation completed.")
        println(output, "Results: $output_dir")
        return 0
    catch exception
        if exception isa InterruptException
            println(error_output, "\nDensityModel interrupted.")
            return 130
        end
        println(error_output, "Error: ", sprint(showerror, exception))
        _usage(error_output)
        return 1
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(CellShapeModel.cli_main())
end
