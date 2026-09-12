module TensorHelpers

using Gridap
using Gridap.TensorValues
using Gridap.Geometry: get_node_coordinates, num_nodes
import EvolvingDomains
using EvolvingDomains.Geometric
using Base.Threads
using StaticArrays: MVector, SVector

export initialize_strain, update_strain!,
       FastTransportMap, update_fast_transport_map!, advect_fields!,
       advect_intensive!

function initialize_strain(info)
    n_nodes = prod(info.dims)
    return [CartesianMeshField(zeros(Float64, n_nodes), info) for _ in 1:3]
end

function update_strain!(ε::Vector{<:CartesianMeshField}, v_extended, Δt, info, ls_values;
                        deviatoric=false)
    nx, ny = info.dims
    n_nodes = prod(info.dims)
    vx = v_extended.data
    dx, dy = info.spacing
    @threads for idx in 1:n_nodes
        # Only update if inside the domain (ls < 0 = tissue/bulk)
        if ls_values[idx] < 0
            i = (idx - 1) % nx + 1
            j = (idx - 1) ÷ nx + 1
            im = max(i - 1, 1)
            ip = min(i + 1, nx)
            jm = max(j - 1, 1)
            jp = min(j + 1, ny)
            dv_dx_1 = (vx[ip + (j - 1) * nx][1] - vx[im + (j - 1) * nx][1]) / ((ip - im) * dx)
            dv_dx_2 = (vx[ip + (j - 1) * nx][2] - vx[im + (j - 1) * nx][2]) / ((ip - im) * dx)
            dv_dy_1 = (vx[i + (jp - 1) * nx][1] - vx[i + (jm - 1) * nx][1]) / ((jp - jm) * dy)
            dv_dy_2 = (vx[i + (jp - 1) * nx][2] - vx[i + (jm - 1) * nx][2]) / ((jp - jm) * dy)

            deformation_xx = deviatoric ? 0.5 * (dv_dx_1 - dv_dy_2) : dv_dx_1
            deformation_yy = deviatoric ? -deformation_xx : dv_dy_2
            deformation_xy = 0.5 * (dv_dx_2 + dv_dy_1)
            spin = 0.5 * (dv_dy_1 - dv_dx_2)
            sine, cosine = sincos(2.0 * spin * Δt)
            mean = 0.5 * (ε[1].data[idx] + ε[2].data[idx])
            difference = 0.5 * (ε[1].data[idx] - ε[2].data[idx])
            shear = ε[3].data[idx]
            rotated_difference = cosine * difference + sine * shear
            rotated_shear = -sine * difference + cosine * shear

            ε[1].data[idx] = mean + rotated_difference + Δt * deformation_xx
            ε[2].data[idx] = mean - rotated_difference + Δt * deformation_yy
            ε[3].data[idx] = rotated_shear + Δt * deformation_xy
        else
            # Optionally zero out values in holes to avoid accumulation/bleeding
            ε[1].data[idx] = 0.0
            ε[2].data[idx] = 0.0
            ε[3].data[idx] = 0.0
        end
    end
end

function advect_intensive!(targets::Vector{<:CartesianMeshField},
                            sources::Vector{<:CartesianMeshField}, map)
    length(targets) == length(sources) ||
        throw(DimensionMismatch("intensive source and target counts differ"))
    for target in targets
        fill!(target.data, 0.0)
    end

    @threads for k in eachindex(map.active_indices)
        target_idx = map.active_indices[k]
        ray_base = 4 * (k - 1)
        for component in eachindex(targets)
            value = 0.0
            for ray in 1:4
                indices = map.backward_indices[ray_base + ray]
                weights = map.backward_weights[ray_base + ray]
                @inbounds for m in 1:16
                    weights[m] > 0.0 &&
                        (value += 0.25 * weights[m] * sources[component].data[indices[m]])
                end
            end
            targets[component].data[target_idx] = value
        end
    end
    return targets
end

mutable struct FastTransportMap
    active_indices::Vector{Int}
    source_mask::BitVector
    target_mask::BitVector
    backward_indices::Vector{NTuple{16,Int}}
    backward_weights::Vector{NTuple{16,Float64}}
    demand_map::Vector{Float64}
    leak_source_indices::Vector{Int}
    leakage_indices::Vector{NTuple{16,Int}}
    leakage_rays::Vector{Point{2,Float64}}
    leakage_weights::Vector{NTuple{16,Float64}}
    coordinates::Vector{Point{2,Float64}}
    grid_meta
end

function _mark_active_nodes!(mask, cell_states, cell_nodes)
    length(cell_states) == size(cell_nodes, 2) ||
        throw(DimensionMismatch("cell states and connectivity differ"))
    fill!(mask, false)
    @inbounds for cell in eachindex(cell_states)
        (cell_states[cell] == -1 || cell_states[cell] == 0) || continue
        for local_node in axes(cell_nodes, 1)
            mask[cell_nodes[local_node, cell]] = true
        end
    end
    return mask
end

function FastTransportMap(geom, velocity, dt::Real, static_geometry=nothing)
    map = FastTransportMap(Int[], BitVector(), BitVector(),
                           NTuple{16,Int}[], NTuple{16,Float64}[],
                           Float64[], Int[], NTuple{16,Int}[],
                           Point{2,Float64}[], NTuple{16,Float64}[],
                           Point{2,Float64}[], nothing)
    update_fast_transport_map!(map, geom, velocity, dt, static_geometry)
end

function _trace_leakage_rays!(map::FastTransportMap, coords, velocity, dt, meta,
                              indices_buffer, weights_buffer, n_nodes::Int)
    nleaks = 0
    for node_idx in 1:n_nodes
        map.source_mask[node_idx] && map.demand_map[node_idx] < 1.0 - 1e-12 &&
            (nleaks += 1)
    end
    resize!(map.leak_source_indices, nleaks)
    resize!(map.leakage_indices, nleaks)
    resize!(map.leakage_rays, nleaks)
    resize!(map.leakage_weights, nleaks)

    trace_ray = EvolvingDomains.Kinematic.SemiLagrangian.trace_ray
    weights! = EvolvingDomains.Kinematic.SemiLagrangian.compute_conservative_weights!
    leak_id = 0
    for node_idx in 1:n_nodes
        if map.source_mask[node_idx] && map.demand_map[node_idx] < 1.0 - 1e-12
            leak_id += 1
            ray = trace_ray(coords[node_idx], velocity, dt)
            weights!(indices_buffer, weights_buffer, ray, meta, map.target_mask)
            map.leak_source_indices[leak_id] = node_idx
            map.leakage_indices[leak_id] = Tuple(indices_buffer)
            map.leakage_rays[leak_id] = ray
            map.leakage_weights[leak_id] = Tuple(weights_buffer)
        end
    end
    return nothing
end

function update_fast_transport_map!(map::FastTransportMap, geom, velocity, dt::Real,
                                    static_geometry=nothing)
    grid = geom.grid
    meta = grid_info(grid)
    if isempty(map.coordinates)
        map.coordinates = vec(collect(get_node_coordinates(grid)))
    end
    coords = map.coordinates
    n_nodes = Int(num_nodes(grid))
    isnothing(geom.cache.prev_cut) && throw(ArgumentError(
        "FastTransportMap requires the previous geometry cut; call ensure_cut!(geom) before advance!"))
    active_previous = active_indices = current_cell_states = nothing
    if isnothing(static_geometry)
        active_previous = get_active_indices(geom, :prev)
        active_indices = get_active_indices(geom, :current)
    else
        cutgeo = ensure_cut!(geom)
        raw_states = cutgeo.ls_to_bgcell_to_inoutcut
        current_cell_states = eltype(raw_states) <: AbstractVector ? raw_states[1] : raw_states
    end
    dx, dy = meta.spacing
    offsets = SVector(
        VectorValue(-0.25 * dx, -0.25 * dy), VectorValue(0.25 * dx, -0.25 * dy),
        VectorValue(-0.25 * dx, 0.25 * dy), VectorValue(0.25 * dx, 0.25 * dy))

    trace_ray = EvolvingDomains.Kinematic.SemiLagrangian.trace_ray
    weights! = EvolvingDomains.Kinematic.SemiLagrangian.compute_conservative_weights!
    indices_buffer = MVector{16,Int}(undef)
    weights_buffer = MVector{16,Float64}(undef)
    resize!(map.source_mask, n_nodes)
    resize!(map.target_mask, n_nodes)
    if isnothing(static_geometry)
        resize!(map.active_indices, length(active_indices))
        copyto!(map.active_indices, active_indices)
        fill!(map.source_mask, false)
        fill!(map.target_mask, false)
        map.source_mask[active_previous] .= true
        map.target_mask[active_indices] .= true
    else
        _mark_active_nodes!(
            map.source_mask, static_geometry.cell_states, static_geometry.coefficient_nodes)
        _mark_active_nodes!(
            map.target_mask, current_cell_states, static_geometry.coefficient_nodes)
        empty!(map.active_indices)
        for node in eachindex(map.target_mask)
            map.target_mask[node] && push!(map.active_indices, node)
        end
    end
    nrays = length(map.active_indices) * 4
    resize!(map.backward_indices, nrays)
    resize!(map.backward_weights, nrays)
    resize!(map.demand_map, n_nodes)
    fill!(map.demand_map, 0.0)
    ray_id = 0
    for node_idx in map.active_indices
        for offset in offsets
            ray_id += 1
            x_dep = trace_ray(coords[node_idx] + offset, velocity, -dt)
            weights!(indices_buffer, weights_buffer, x_dep, meta, map.source_mask)
            map.backward_indices[ray_id] = Tuple(indices_buffer)
            map.backward_weights[ray_id] = Tuple(weights_buffer)
            for m in 1:16
                source_idx = indices_buffer[m]
                map.source_mask[source_idx] &&
                    (map.demand_map[source_idx] += 0.25 * weights_buffer[m])
            end
        end
    end

    _trace_leakage_rays!(
        map, coords, velocity, dt, meta, indices_buffer, weights_buffer, n_nodes)

    map.grid_meta = meta
    map
end

function advect_fields!(targets::Vector{<:CartesianMeshField},
                         sources::Vector{<:CartesianMeshField}, map::FastTransportMap)
    length(targets) == length(sources) ||
        throw(DimensionMismatch("conservative source and target counts differ"))
    for target in targets
        fill!(target.data, 0.0)
    end

    source_data = [field.data for field in sources]
    target_data = [field.data for field in targets]

    @threads for k in eachindex(map.active_indices)
        target_idx = map.active_indices[k]
        ray_base = (k - 1) * 4
        for component in eachindex(targets)
            value = 0.0
            for ray_offset in 1:4
                indices = map.backward_indices[ray_base + ray_offset]
                weights = map.backward_weights[ray_base + ray_offset]
                for m in 1:16
                    w = weights[m]
                    if w > 0
                        s_idx = indices[m]
                        req = map.demand_map[s_idx]
                        scale = req > 1.0 ? (1.0 / req) : 1.0
                        value += 0.25 * w * scale * source_data[component][s_idx]
                    end
                end
            end
            target_data[component][target_idx] = value
        end
    end

    # This pass is intentionally serial, matching EvolvingDomains' leakage pass.
    for (k, s_idx) in enumerate(map.leak_source_indices)
        if any(component -> abs(source_data[component][s_idx]) > 1e-15,
               eachindex(sources))
            req = map.demand_map[s_idx]
            indices = map.leakage_indices[k]
            weights = map.leakage_weights[k]
            for m in 1:16
                target_idx = indices[m]
                w = weights[m]
                map.target_mask[target_idx] || continue
                for component in eachindex(targets)
                    target_data[component][target_idx] +=
                        w * (1.0 - req) * source_data[component][s_idx]
                end
            end
        end
    end

    targets
end

end
