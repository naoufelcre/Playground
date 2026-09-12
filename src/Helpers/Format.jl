using FileIO
using Images
using ImageMorphology
using ImageFiltering
include("GeometrySmoothing.jl")

# ============================================================================
# Core I/O Functions
# ============================================================================

function write_grid(path, sim_id, nx, ny, dx, dy, xmin, xmax, ymin, ymax)
    open(path, "w") do f
        println(f, sim_id)
        println(f, "grid")
        println(f, "$nx $ny")
        println(f, "$dx $dy")
        println(f, "$xmin $xmax")
        println(f, "$ymin $ymax")
    end
end

function read_grid(path)
    lines = readlines(path)
    nx, ny = parse.(Int, split(lines[3]))
    dx, dy = parse.(Float64, split(lines[4]))
    xmin, xmax = parse.(Float64, split(lines[5]))
    ymin, ymax = parse.(Float64, split(lines[6]))
    return Dict(
        "sim_id" => lines[1],
        "nx" => nx, "ny" => ny,
        "dx" => dx, "dy" => dy,
        "xmin" => xmin, "xmax" => xmax,
        "ymin" => ymin, "ymax" => ymax,
    )
end

function write_field(path, sim_id, field_type, values, nx, ny, dx, dy, xmin, ymin; compact::Bool=false)
    if compact
        write_field_compact(path, sim_id, field_type, values, nx, ny)
    else
        write_field_human(path, sim_id, field_type, values, nx, ny, dx, dy, xmin, ymin)
    end
end

function write_field_human(path, sim_id, field_type, values, nx, ny, dx, dy, xmin, ymin)
    open(path, "w") do f
        println(f, sim_id)
        println(f, field_type)
        for j in 1:ny
            for i in 1:nx
                x = xmin + (i - 1) * dx
                y = ymin + (j - 1) * dy
                idx = (j - 1) * nx + i
                v = values[idx]
                if isa(v, Real)
                    println(f, "$x $y $v")
                else
                    println(f, "$x $y $(v[1]) $(v[2])")
                end
            end
        end
    end
end

const COMPACT_MAGIC = b"CSF1"

function write_field_compact(path, sim_id, field_type, values, nx, ny)
    # Change extension to .bin
    bin_path = replace(path, r"\.dat$" => ".bin")
    n_pts = nx * ny
    function _is_vector_value(v)
        try
            return v[1] isa Number && v[2] isa Number
        catch
            return false
        end
    end
    is_vector = n_pts > 0 ? _is_vector_value(values[1]) : false

    open(bin_path, "w") do f
        write(f, COMPACT_MAGIC)
        write(f, Int64(length(sim_id)))
        write(f, sim_id)
        write(f, Int64(length(field_type)))
        write(f, field_type)
        write(f, Int64(nx))
        write(f, Int64(ny))
        # dtype: 3=scalar Float32, 4=vector Float32
        write(f, Int64(is_vector ? 4 : 3))

        if is_vector
            for idx in 1:n_pts
                v = values[idx]
                write(f, Float32(v[1]))
                write(f, Float32(v[2]))
            end
        else
            for idx in 1:n_pts
                write(f, Float32(values[idx]))
            end
        end
    end
end

function read_field(path)
    # Auto-detect: .bin extension or .dat with binary magic header
    if endswith(path, ".bin")
        return read_field_compact(path)
    end

    # Check if .dat file is actually binary (e.g. renamed)
    open(path, "r") do f
        magic = read(f, 4)
        if magic == COMPACT_MAGIC
            return read_field_compact(path)
        end
    end

    return read_field_human(path)
end

function read_field_human(path)
    lines = readlines(path)
    sim_id = lines[1]
    field_type = lines[2]
    n_data = length(lines) - 2
    n_data == 0 && return Dict("sim_id" => sim_id, "field_type" => field_type, "values" => Float64[])

    first_data = lines[3]
    parts_check = split(first_data)
    is_vector = length(parts_check) >= 4
    data_lines = @view lines[3:end]

    if is_vector
        values = Vector{Vector{Float64}}(undef, n_data)
        for (k, line) in enumerate(data_lines)
            parts = split(line)
            vx = parse(Float64, parts[3])
            vy = parse(Float64, parts[4])
            values[k] = [vx, vy]
        end
    else
        values = Vector{Float64}(undef, n_data)
        for (k, line) in enumerate(data_lines)
            parts = split(line)
            values[k] = parse(Float64, parts[3])
        end
    end

    return Dict("sim_id" => sim_id, "field_type" => field_type, "values" => values)
end

function read_field_compact(path)
    open(path, "r") do f
        magic = read(f, 4)
        @assert magic == COMPACT_MAGIC "Invalid compact file magic header"

        sim_id_len = read(f, Int64)
        sim_id = String(read(f, sim_id_len))

        field_type_len = read(f, Int64)
        field_type = String(read(f, field_type_len))

        nx = read(f, Int64)
        ny = read(f, Int64)
        dtype = read(f, Int64)
        n_pts = nx * ny

        if dtype == 4  # vector Float32
            values = Vector{Vector{Float64}}(undef, n_pts)
            for idx in 1:n_pts
                vx = Float64(read(f, Float32))
                vy = Float64(read(f, Float32))
                values[idx] = [vx, vy]
            end
        elseif dtype == 3  # scalar Float32
            values = Vector{Float64}(undef, n_pts)
            for idx in 1:n_pts
                values[idx] = Float64(read(f, Float32))
            end
        elseif dtype == 2  # vector Float64 (legacy)
            values = Vector{Vector{Float64}}(undef, n_pts)
            for idx in 1:n_pts
                vx = read(f, Float64)
                vy = read(f, Float64)
                values[idx] = [vx, vy]
            end
        else  # scalar Float64 (legacy, dtype == 1)
            values = Vector{Float64}(undef, n_pts)
            for idx in 1:n_pts
                values[idx] = read(f, Float64)
            end
        end

        return Dict("sim_id" => sim_id, "field_type" => field_type, "values" => values)
    end
end

# ============================================================================
# Geometry Processing
# ============================================================================

"""
Trace 8-connected boundary contours from binary mask.
Returns closed contours as vectors of (x, y) pixel coordinates.
"""
function trace_contours(mask::BitMatrix)
    ny, nx = size(mask)
    eroded = erode(mask)
    boundary = mask .& (.!eroded)
    visited = falses(ny, nx)
    contours = Vector{Vector{Tuple{Float64,Float64}}}()
    nbrs = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    
    for start_r in 1:ny, start_c in 1:nx
        (!boundary[start_r, start_c] || visited[start_r, start_c]) && continue
        pts = Tuple{Float64,Float64}[]
        r, c = start_r, start_c
        while true
            visited[r, c] = true
            push!(pts, (Float64(c), Float64(ny - r + 1)))
            found = false
            for (dr, dc) in nbrs
                nr, nc = r + dr, c + dc
                (1 <= nr <= ny && 1 <= nc <= nx) || continue
                (!boundary[nr, nc] || visited[nr, nc]) && continue
                r, c = nr, nc
                found = true
                break
            end
            found || break
        end
        # Ensure closed loop
        if pts[1] != pts[end]
            push!(pts, pts[1])
        end
        length(pts) >= 8 && push!(contours, pts)
    end
    return contours
end

"""
Remove small connected components from mask.
"""
function filter_components(mask::BitMatrix, min_area::Int)
    min_area <= 0 && return mask
    ny, nx = size(mask)
    visited = falses(ny, nx)
    cleaned = falses(ny, nx)
    nbrs = [(-1,0), (1,0), (0,-1), (0,1)]
    
    for start_r in 1:ny, start_c in 1:nx
        visited[start_r, start_c] && continue
        component = Tuple{Int,Int}[]
        queue = [(start_r, start_c)]
        visited[start_r, start_c] = true
        
        while !isempty(queue)
            r, c = pop!(queue)
            push!(component, (r, c))
            for (dr, dc) in nbrs
                nr, nc = r + dr, c + dc
                if 1 <= nr <= ny && 1 <= nc <= nx && !visited[nr, nc] && mask[nr, nc] == mask[start_r, start_c]
                    visited[nr, nc] = true
                    push!(queue, (nr, nc))
                end
            end
        end
        
        if length(component) >= min_area
            for (r, c) in component
                cleaned[r, c] = mask[r, c]
            end
        end
    end
    return cleaned
end

"""
Exact minimum distance from point to polygon edge.
Computes distance to each line segment, returns minimum.
"""
function point_to_polygon_distance(px::Float64, py::Float64, 
                                     xs::Vector{Float64}, ys::Vector{Float64})
    n = length(xs)
    d_min_sq = Inf
    
    @inbounds for i in 1:(n-1)
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i+1], ys[i+1]
        
        # Vector from segment start to point
        dx = x2 - x1
        dy = y2 - y1
        dpx = px - x1
        dpy = py - y1
        
        seg_len_sq = dx^2 + dy^2
        
        if seg_len_sq < 1e-14
            # Degenerate segment
            d_sq = dpx^2 + dpy^2
        else
            # Project onto segment
            t = clamp((dpx*dx + dpy*dy) / seg_len_sq, 0.0, 1.0)
            proj_x = x1 + t*dx
            proj_y = y1 + t*dy
            d_sq = (px - proj_x)^2 + (py - proj_y)^2
        end
        
        d_sq < d_min_sq && (d_min_sq = d_sq)
    end
    
    return sqrt(d_min_sq)
end

@inline function polygon_area(xs::Vector{Float64}, ys::Vector{Float64})
    n = length(xs)
    n < 3 && return 0.0
    a = 0.0
    @inbounds for i in 1:(n-1)
        a += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    end
    return abs(a) * 0.5
end

@inline function point_in_polygon(px::Float64, py::Float64,
                                  xs::Vector{Float64}, ys::Vector{Float64})
    inside = false
    n = length(xs)
    n < 3 && return false
    j = n
    @inbounds for i in 1:n
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        # Ray casting test with half-open rule for robustness
        crosses = ((yi > py) != (yj > py))
        if crosses
            xint = (xj - xi) * (py - yi) / (yj - yi + 1e-16) + xi
            if px < xint
                inside = !inside
            end
        end
        j = i
    end
    return inside
end

# ============================================================================
# Main Interface
# ============================================================================

"""
    mask_to_levelset_format(mask_path, output_prefix; ...)

Convert PNG mask to level-set using polygon-based approach:
1. Gaussian blur to smooth
2. Remove small components
3. Trace closed contours
4. Simplify with Douglas-Peucker (closed-loop aware)
5. Smooth polygon with Chaikin corner-cutting (optional)
6. Compute exact distance to polygon edges
6. Sign from original mask

Parameters:
- smooth_sigma: Gaussian blur radius (default 2.0)
- min_component_area: Min pixels per component (default 25)
- dp_epsilon: Douglas-Peucker tolerance in pixels (default 2.0)
- polygon_smooth_iters: number of Chaikin smoothing iterations (default 2)
- polygon_smooth_alpha: Chaikin alpha in (0,0.5), default 0.25
"""
function mask_to_levelset_format(mask_path::String, output_prefix::String;
                                  domain::NTuple{4,Float64}=(0.0, 1.0, 0.0, 1.0),
                                  sim_id::String="from_mask",
                                  smooth_sigma::Float64=2.0,
                                  min_component_area::Int=25,
                                  dp_epsilon::Float64=2.0,
                                  polygon_smooth_iters::Int=2,
                                  polygon_smooth_alpha::Float64=0.25,
                                  laplace_smooth_iters::Int=6,
                                  laplace_smooth_lambda::Float64=0.35,
                                  max_points_per_polygon::Int=300,
                                  min_polygon_area::Float64=25.0,
                                  max_polygons::Int=20,
                                  return_polygons::Bool=false)
    xmin, xmax, ymin, ymax = domain
    
    # Load and preprocess mask
    img = load(mask_path)
    gray_img = Gray.(img)
    
    if smooth_sigma > 0
        gray_img = imfilter(gray_img, Kernel.gaussian(smooth_sigma))
    end
    
    mask_raw = BitMatrix(gray_img .> 0.5)
    mask = filter_components(mask_raw, min_component_area)
    
    ny_cells, nx_cells = size(mask)

    # Normalize domain so the longest image side has length 1
    if nx_cells >= ny_cells
        xmax = xmin + 1.0
        ymax = ymin + (ny_cells / nx_cells)
    else
        ymax = ymin + 1.0
        xmax = xmin + (nx_cells / ny_cells)
    end

    # Fixed 250 steps per unit length for coarsened grid
    x_size, y_size = xmax - xmin, ymax - ymin
    target_steps = 250
    dx = dy = 1.0 / target_steps
    nx_nodes = Int(round(x_size * target_steps)) + 1
    ny_nodes = Int(round(y_size * target_steps)) + 1
    
    # Trace contours
    contours = trace_contours(mask)
    isempty(contours) && error("No contours found")
    println("  Found $(length(contours)) contour(s)")
    
    # Simplify polygons
    polygon_data = Vector{Tuple{Vector{Float64},Vector{Float64},Float64}}()
    for (ci, pts) in enumerate(contours)
        n_raw = length(pts)
        
        # Douglas-Peucker simplification
        if dp_epsilon > 0 && n_raw > 6
            pts = simplify_polygon(pts, dp_epsilon)
        end
        
        # Optional tractable smoothing on polygon (no spline overshoot)
        pts = smooth_polygon_chaikin(pts, polygon_smooth_iters, polygon_smooth_alpha)
        pts = smooth_polygon_laplacian(pts, laplace_smooth_iters, laplace_smooth_lambda)
        pts = cap_polygon_points(pts, max_points_per_polygon)

        n_pts = length(pts)
        if n_pts < 4
            println("  Contour $ci: skipped (too few points: $n_pts)")
            continue
        end
        
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        area = polygon_area(xs, ys)
        if area < min_polygon_area
            println("  Contour $ci: skipped (area=$(round(area, sigdigits=4)) < $min_polygon_area)")
            continue
        end
        push!(polygon_data, (xs, ys, area))
        println("  Contour $ci: $n_raw pts → $n_pts pts (simplified), area=$(round(area, sigdigits=4))")
    end

    isempty(polygon_data) && error("All contours degenerate")

    sort!(polygon_data, by = p -> p[3], rev = true)
    if max_polygons > 0 && length(polygon_data) > max_polygons
        polygon_data = polygon_data[1:max_polygons]
    end
    polygons = [(p[1], p[2]) for p in polygon_data]

    # Convert polygons from pixel coordinates to physical domain coordinates
    sx = x_size / nx_cells
    sy = y_size / ny_cells
    polygons_phys = [(
        [xmin + (x - 0.5) * sx for x in xs],
        [ymin + (y - 0.5) * sy for y in ys]
    ) for (xs, ys) in polygons]
    bboxes = [(minimum(p[1]), maximum(p[1]), minimum(p[2]), maximum(p[2])) for p in polygons_phys]
    println("  Kept $(length(polygons)) polygon(s) after area filtering/limit")
    
    # Compute SDF
    println("  Computing SDF on $(nx_nodes)×$(ny_nodes) nodes...")
    values = Vector{Float64}(undef, nx_nodes * ny_nodes)
    
    Threads.@threads for j in 1:ny_nodes
        py = ymin + (Float64(j) - 1.0) * dy
        for i in 1:nx_nodes
            px = xmin + (Float64(i) - 1.0) * dx
            
            # Min distance to any polygon
            d_min = Inf
            inside = false
            @inbounds for k in eachindex(polygons_phys)
                xs, ys = polygons_phys[k]
                d = point_to_polygon_distance(px, py, xs, ys)
                d < d_min && (d_min = d)

                # Sign from simplified geometry, not raster mask
                xmin_p, xmax_p, ymin_p, ymax_p = bboxes[k]
                if (xmin_p <= px <= xmax_p) && (ymin_p <= py <= ymax_p)
                    inside = inside || point_in_polygon(px, py, xs, ys)
                end
            end

            signed_d = inside ? d_min : -d_min
            
            values[(j-1)*nx_nodes + i] = signed_d
        end
    end
    
    println("  SDF range: [$(round(minimum(values),sigdigits=4)), $(round(maximum(values),sigdigits=4))]")
    
    # Write outputs
    grid_path = output_prefix * ".grid"
    write_grid(grid_path, sim_id, nx_nodes, ny_nodes, dx, dy, xmin, xmax, ymin, ymax)
    
    levelset_path = output_prefix * "_levelset.dat"
    write_field(levelset_path, sim_id, "levelset", values, nx_nodes, ny_nodes, dx, dy, xmin, ymin)
    
    if return_polygons
        return (grid_path, levelset_path, polygons)
    else
        return (grid_path, levelset_path)
    end
end
