using Gridap
using EvolvingDomains
using Gridap.Geometry: get_node_coordinates
using Random

# Synthetic geometry: N random non-overlapping blobs on the [0,1]² box, ported
# from AMD_tools GeometryGenerator.random_blobs. The string seed drives a
# MersenneTwister, so the same seed reproduces the same level set across
# processes and sessions. Positive level set = lesion/void (matches the
# `sdf = where(mask, +, −)` convention of AMD_tools).
function random_blob_geometry(; seed::String, nx=325, ny=325,
        n=5, min_radius_frac=0.02, max_radius_frac=0.12, center_bias=0.07)
    rng = MersenneTwister(seed)
    model = CartesianDiscreteModel(Gridap.Point(0.0, 0.0), Gridap.Point(1.0, 1.0), (nx - 1, ny - 1))

    min_r = max(min_radius_frac, 2.0 / min(nx - 1, ny - 1))
    max_r = max(max_radius_frac, min_r)

    blobs = NTuple{3,Float64}[]
    for _ in 1:n
        r = min_r + (max_r - min_r) * rand(rng)
        bx = by = 0.0
        placed = false
        for _ in 1:50
            bx = clamp(center_bias * randn(rng) + 0.5, r, 1.0 - r)
            by = clamp(center_bias * randn(rng) + 0.5, r, 1.0 - r)
            if !any((bx - px)^2 + (by - py)^2 < (r + pr)^2 for (px, py, pr) in blobs)
                placed = true
                break
            end
        end
        placed || continue  # couldn't place without overlap — skip this blob
        push!(blobs, (bx, by, r))
    end

    ls = fill(-1.0, nx * ny)
    @inbounds for j in 1:ny
        y = (j - 1) / (ny - 1)
        for i in 1:nx
            x = (i - 1) / (nx - 1)
            idx = (j - 1) * nx + i
            for (bx, by, r) in blobs
                d = r - hypot(x - bx, y - by)
                d > ls[idx] && (ls[idx] = d)
            end
        end
    end

    return EvolvingDiscreteGeometry(ls, model)
end

function initialize_geometry_from_format(grid_path::String, levelset_path::String)
    grid_info = Helpers.read_grid(grid_path)
    levelset_data = Helpers.read_field(levelset_path)

    # Grid file now stores node counts directly
    nx_nodes, ny_nodes = grid_info["nx"], grid_info["ny"]
    xmin, xmax = grid_info["xmin"], grid_info["xmax"]
    ymin, ymax = grid_info["ymin"], grid_info["ymax"]

    # Partition is cells = nodes - 1
    nx_cells = nx_nodes - 1
    ny_cells = ny_nodes - 1

    # Create Cartesian grid with cell partition
    pmin = Gridap.Point(xmin, ymin)
    pmax = Gridap.Point(xmax, ymax)
    partition = (nx_cells, ny_cells)
    model = CartesianDiscreteModel(pmin, pmax, partition)

    # Get level set values (node-centered)
    ls_values = levelset_data["values"]

    # Validate levelset data size matches node count
    expected_size = nx_nodes * ny_nodes
    actual_size = length(ls_values)
    if actual_size != expected_size
        error("Levelset data size ($actual_size) doesn't match grid node dimensions ($nx_nodes × $ny_nodes = $expected_size). " *
              "Check that your grid file and levelset file correspond to the same simulation.")
    end

    # Create evolving geometry
    geom = EvolvingDiscreteGeometry(ls_values, model)

    return geom
end
