# Simulation state. Everything here is once-per-step plumbing; the numerical
# hot loops live in Geom.jl, TensorHelpers.jl, and StaticAssembler.

# Wraps a CartesianMeshField interpolator as an EvolvingDomains VelocitySource.
struct FieldVelocity{I} <: AbstractVelocitySource
    itp::I
end

EvolvingDomains.Kinematic.get_velocity(v::FieldVelocity, x, t) = v.itp(x[1], x[2])
EvolvingDomains.Kinematic.is_time_dependent(::FieldVelocity) = true

"""
    SimulationState

The evolving physical fields and time. `ε`/`ε_target` and `ρ`/`ρ_target` are
double buffers swapped by `swap_buffers!` after each advection step.
"""
mutable struct SimulationState{G,I,P,E,R,V}
    geom::G
    info::I
    p::P
    ε::E
    ε_target::E
    ρ::R
    ρ_target::R
    α::R
    v_grid::V
    t::Float64
    Δt::Float64
end

function swap_buffers!(state)
    state.ε, state.ε_target = state.ε_target, state.ε
    state.ρ, state.ρ_target = state.ρ_target, state.ρ
    return state
end