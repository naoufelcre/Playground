# Export utilities - Generic visualization from Format.jl data
using CairoMakie
using PlotUtils
using JSON
using Dates
using Random

"""
Export density field as a heatmap from Format.jl serialized data.
"""
function export_density_field(grid_dict, ρ_dict, levelset_dict, current_t, output_dir, i)
    # Create figure for density with colorbar
    fig_d = Makie.Figure(size=(900, 800))
    ax_d = Makie.Axis(fig_d[1, 1], aspect=Makie.DataAspect())

    # Extract grid info
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    dx = grid_dict["dx"]
    dy = grid_dict["dy"]
    xmin = grid_dict["xmin"]
    ymin = grid_dict["ymin"]

    # Build coordinate ranges
    xs = range(xmin, step=dx, length=nx)
    ys = range(ymin, step=dy, length=ny)

    # Reshape values to 2D
    data = reshape(ρ_dict["values"], (nx, ny))
    display_data = clamp.(data, 0.0, 1.0)
    levelset_data = reshape(levelset_dict["values"], (nx, ny))

    hm = Makie.heatmap!(
        ax_d,
        xs,
        ys,
        display_data,
        colormap=cgrad([:white, :black], 12, categorical=true),
        colorrange=(0.0, 1.0)
    )
    Makie.contour!(ax_d, xs, ys, levelset_data, levels=[0.0], color=:black, linewidth=1.0)
    Makie.Colorbar(fig_d[1, 2], hm, label="Density")

    ax_d.title = "Density (t = $(round(current_t, digits=3)))"
    out_subdir = joinpath(output_dir, "density")
    mkpath(out_subdir)
    Makie.save(joinpath(out_subdir, "density_$(lpad(i, 4, '0')).png"), fig_d)
end

"""
Export strain norm as a heatmap from Format.jl serialized data.
"""
function export_strain_norm(grid_dict, ε_dicts, current_t, output_dir, i)
    # Create figure for strain norm
    fig_e = Makie.Figure(size=(900, 800))
    ax_e = Makie.Axis(fig_e[1, 1], aspect=Makie.DataAspect())

    # Extract grid info
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    dx = grid_dict["dx"]
    dy = grid_dict["dy"]
    xmin = grid_dict["xmin"]
    ymin = grid_dict["ymin"]

    # Build coordinate ranges
    xs = range(xmin, step=dx, length=nx)
    ys = range(ymin, step=dy, length=ny)

    # ε is [ε_xx, ε_yy, ε_xy]
    # Norm = sqrt(ε_xx^2 + ε_yy^2 + 2*ε_xy^2)
    ε_xx = ε_dicts[1]["values"]
    ε_yy = ε_dicts[2]["values"]
    ε_xy = ε_dicts[3]["values"]

    norm_data = sqrt.(ε_xx .^ 2 .+ ε_yy .^ 2 .+ 2 .* ε_xy .^ 2)
    data = reshape(norm_data, (nx, ny))

    hm = Makie.heatmap!(ax_e, xs, ys, data, colormap=:magma)
    Makie.Colorbar(fig_e[1, 2], hm, label="Strain Norm")

    ax_e.title = "Strain Norm (t = $(round(current_t, digits=3)))"
    out_subdir = joinpath(output_dir, "strain")
    mkpath(out_subdir)
    Makie.save(joinpath(out_subdir, "strain_$(lpad(i, 4, '0')).png"), fig_e)
end

"""
Export velocity field as a quiver plot from Format.jl serialized data.
Assumes velocity is stored as a vector of tuples/vectors [vx, vy] in values.
"""
function export_velocity_field(grid_dict, v_dict, current_t, output_dir, i; stride=8)
    fig = Makie.Figure(size=(900, 800))
    ax = Makie.Axis(fig[1, 1], aspect=Makie.DataAspect())

    # Extract grid info
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    dx = grid_dict["dx"]
    dy = grid_dict["dy"]
    xmin = grid_dict["xmin"]
    ymin = grid_dict["ymin"]

    # Subsample for better visualization
    ixs = 1:stride:nx
    iys = 1:stride:ny

    px = Float64[]
    py = Float64[]
    ux = Float64[]
    uy = Float64[]

    values = v_dict["values"]

    for j in iys, i in ixs
        idx = i + (j - 1) * nx
        v = values[idx]
        # v should be a VectorValue or tuple (vx, vy)
        push!(px, xmin + (i - 1) * dx)
        push!(py, ymin + (j - 1) * dy)
        push!(ux, v[1])
        push!(uy, v[2])
    end

    if !isempty(px)
        Makie.arrows!(ax, px, py, ux, uy; color=:blue, lengthscale=0.1, tipwidth=8, tiplength=12)
    end

    ax.title = "Velocity (t = $(round(current_t, digits=3)))"
    out_subdir = joinpath(output_dir, "velocity")
    mkpath(out_subdir)
    Makie.save(joinpath(out_subdir, "velocity_$(lpad(i, 4, '0')).png"), fig)
end

"""
Generate a readable simulation ID based on physics parameters, grid resolution,
geometry type, timestamp, and random suffix for parallel safety.

Format: sim_A{X}_m{M}_xi{Y}_nu{Z}_{NX}x{NY}_{geo}_{timestamp}_{rand}
"""
function generate_simulation_id(p, info, horizon, steps, reinit_freq, geometry_tag="multicircle")
    # Extract physics parameters
    A = p.A
    m = p.m
    xi = p.ξ
    nu = p.ν

    # Extract grid dimensions
    nx, ny = info.dims

    # Timestamp
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

    # Random suffix (6 hex chars for parallel safety)
    rand_suffix = randstring("abcdefghijklmnopqrstuvwxyz0123456789", 6)

    # Build ID
    sim_id = "sim$(geometry_tag)_$(rand_suffix)_A$(A)_m$(m)_xi$(xi)_nu$(nu)_$(nx)x$(ny)_$(timestamp)"

    return sim_id
end

"""
Export simulation metadata to JSON file containing complete provenance.
"""
function export_metadata_json(p, info, horizon, steps, reinit_freq, sim_id, output_dir)
    dt = horizon / steps

    metadata = Dict(
        "sim_id" => sim_id,
        "timestamp" => string(now()),
        "zeta_function" => "m/(2*(m-1)) * (1-ρ^(m-1))",
        "physics" => Dict(
            "A" => p.A,
            "m" => p.m,
            "xi" => p.ξ,
            "nu" => p.ν
        ),
        "numerics" => Dict(
            "nx" => info.dims[1],
            "ny" => info.dims[2],
            "horizon" => horizon,
            "steps" => steps,
            "dt" => dt,
            "reinit_freq" => reinit_freq
        ),
        "geometry" => Dict(
            "type" => "multicircle",
            "description" => "5 circles of varying radii embedded in unit square",
            "circles" => [
                Dict("center" => [0.43, 0.55], "radius" => 0.05),
                Dict("center" => [0.57, 0.45], "radius" => 0.05),
                Dict("center" => [0.37, 0.35], "radius" => 0.025),
                Dict("center" => [0.62, 0.41], "radius" => 0.02),
                Dict("center" => [0.52, 0.66], "radius" => 0.04)
            ]
        ),
        "versioning" => Dict(
            "git_commit" => "unknown",  # Can be populated if git is available
            "julia_version" => string(VERSION)
        )
    )

    # Write JSON file
    metadata_path = joinpath(output_dir, "metadata.json")
    open(metadata_path, "w") do f
        JSON.print(f, metadata, 2)  # 2-space indentation
    end

    return metadata_path
end

"""
Export cell classification visualization from geometry.
"""
function export_classification(fig, ax, geom, current_t, output_dir, i)
    empty!(ax)
    Visualization.plot_cell_classification!(ax, geom)
    ax.title = "(t = $(round(current_t, digits=3)))"
    out_subdir = joinpath(output_dir, "classification")
    mkpath(out_subdir)
    save(joinpath(out_subdir, "classification_$(lpad(i, 4, '0')).png"), fig)
end
