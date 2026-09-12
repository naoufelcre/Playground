include("InitBlock/Patterns.jl")
using .Patterns

function init_simulation(; seed::String,
        horizon = 1.0, reinit_freq = 1, min_island_nodes = 5)

    p    = AdimensionalParameters(ξ=0.1, A=0.1, ν=0.49, m=2)
    geom = random_blob_geometry(; seed, nx=301, ny=301)
    info = grid_info(geom.grid)
    Δt   = MIN_TIMESTEP

    ε, ε_target = initialize_strain(info), initialize_strain(info)
    ρ        = initialize_density(info, SmoothPatchyTissuePattern(; seed), geom)
    @inbounds for i in eachindex(ρ.data)
        geom.levelset[i] < 0.0 || continue
        ε[1].data[i] = ε[2].data[i] = 0.5 * (inv(ρ.data[i]) - 1.0)
    end
    α        = friction_field(info, RadialFrictionPattern())
    ρ_target = CartesianMeshField(zeros(Float64, prod(info.dims)), info)
    v_grid   = CartesianMeshField(zeros(VectorValue{2,Float64}, prod(info.dims)), info)

    return (; p, geom, info, horizon, Δt, reinit_freq, min_island_nodes,
              ε, ε_target, ρ, ρ_target, α, v_grid)
end

"""
    initialize(; seed, horizon, nsteps) -> (state, backend, run)

Build the full simulation from a seed: physical state (random-blob geometry,
patchy density, radial substrate friction), projected backend (kernel + CG
buffers), and horizon-based run configuration. `nsteps` optionally caps the
accepted steps for tests and diagnostics.
"""
function initialize(; nsteps=nothing, seed::String, horizon=1.0)
    BLAS.set_num_threads(1)
    init = init_simulation(; seed, horizon)
    state = SimulationState(init.geom, init.info, init.p, init.ε, init.ε_target,
                            init.ρ, init.ρ_target, init.α, init.v_grid, 0.0, init.Δt)
    backend = ProjectedBackend(form=kelvin_voigt_form(init.p))
    run = (horizon=init.horizon,
           max_steps=nsteps,
           reinit_freq=init.reinit_freq,
           min_island_nodes=init.min_island_nodes)
    return state, backend, run
end
