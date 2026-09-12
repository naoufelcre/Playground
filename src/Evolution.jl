# Domain evolution and geometric health control. Operation order matters:
# extend → advance geometry → transport-map refresh → advection → strain update
# (uses the already-advanced level set) → buffer swap.

const CFL_TARGET = 0.5
const DEFORMATION_TARGET = 0.2
const MAX_TIMESTEP = 2e-3
const TIMESTEP_GROWTH = 1.25

function stable_timestep(state, v_extended, remaining)
    nx, ny = state.info.dims
    dx, dy = state.info.spacing
    nx > 1 && ny > 1 || throw(ArgumentError("adaptive stepping requires at least 2 nodes per dimension"))
    length(v_extended.data) == nx * ny || throw(DimensionMismatch("velocity and grid dimensions differ"))

    max_advection_rate = 0.0
    max_deformation_rate_squared = 0.0
    velocity = v_extended.data
    @inbounds for j in 1:ny, i in 1:nx
        idx = i + (j - 1) * nx
        v = velocity[idx]
        all(isfinite, v) || throw(DomainError(v, "adaptive stepping requires finite velocity"))
        max_advection_rate = max(max_advection_rate, abs(v[1]) / dx + abs(v[2]) / dy)

        im, ip = max(i - 1, 1), min(i + 1, nx)
        jm, jp = max(j - 1, 1), min(j + 1, ny)
        dvx_dx = (velocity[ip + (j - 1) * nx][1] - velocity[im + (j - 1) * nx][1]) / ((ip - im) * dx)
        dvy_dx = (velocity[ip + (j - 1) * nx][2] - velocity[im + (j - 1) * nx][2]) / ((ip - im) * dx)
        dvx_dy = (velocity[i + (jp - 1) * nx][1] - velocity[i + (jm - 1) * nx][1]) / ((jp - jm) * dy)
        dvy_dy = (velocity[i + (jp - 1) * nx][2] - velocity[i + (jm - 1) * nx][2]) / ((jp - jm) * dy)
        shear = 0.5 * (dvy_dx + dvx_dy)
        deformation_rate_squared = dvx_dx^2 + dvy_dy^2 + 2shear^2
        max_deformation_rate_squared = max(max_deformation_rate_squared, deformation_rate_squared)
    end
    max_deformation_rate = sqrt(max_deformation_rate_squared)

    candidates = (
        MAX_TIMESTEP,
        TIMESTEP_GROWTH * state.Δt,
        iszero(max_advection_rate) ? Inf : CFL_TARGET / max_advection_rate,
        iszero(max_deformation_rate) ? Inf : DEFORMATION_TARGET / max_deformation_rate,
    )
    names = (:maximum, :growth, :advection, :deformation)
    limiter_index = argmin(candidates)
    unconstrained = candidates[limiter_index]
    Δt = min(remaining, max(MIN_TIMESTEP, unconstrained))
    limiter = Δt == remaining ? :horizon :
              unconstrained < MIN_TIMESTEP ? :minimum : names[limiter_index]
    isfinite(Δt) && Δt > 0 || error("adaptive timestep is not positive and finite")
    state.t + Δt > state.t || error("adaptive timestep is too small to advance floating-point time")

    return (;
        Δt,
        limiter,
        cfl=Δt * max_advection_rate,
        deformation=Δt * max_deformation_rate,
    )
end

function evolve!(state, tmap, static_geometry, v_extended)
    geom = state.geom
    vel_source = FieldVelocity(get_interpolator(v_extended))

    advance!(geom, v_extended.data, state.Δt)

    tmap = if isnothing(tmap)
        FastTransportMap(geom, vel_source, state.Δt, static_geometry)
    else
        update_fast_transport_map!(tmap, geom, vel_source, state.Δt, static_geometry)
    end
    advect_fields!([state.ρ_target], [state.ρ], tmap)
    advect_intensive!(state.ε_target, state.ε, tmap)
    update_strain!(state.ε_target, v_extended, state.Δt, state.info, geom.levelset)
    swap_buffers!(state)
    return tmap
end

"""
    maintain_geometry!(state, reinit_freq, min_island_nodes, step)

Every `reinit_freq` steps: filter small negative-phase islands, invalidate the
geometry cache if nodes were flipped, and reinitialize the level set.
"""
function maintain_geometry!(state, reinit_freq, min_island_nodes, step)
    reinit_freq > 0 && step % reinit_freq == 0 || return nothing
    topo_stats = filter_small_phase_islands!(
        state.geom.levelset, state.info;
        phase=:negative,
        min_component_size=min_island_nodes,
        connectivity=:edge,
    )
    topo_stats.n_flipped_nodes > 0 && invalidate!(state.geom.cache)
    reinitialize!(state.geom)
    return topo_stats
end
