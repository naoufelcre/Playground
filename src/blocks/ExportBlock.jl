# =========== EXPORT BLOCK ============
function export_blocks!(i, save_stride, output_dir, sim_id, info,
                        compact, export_images, current_t, geom, ρ, ε, v_grid)
    save_stride > 0 && i % save_stride == 0 || return nothing

    nx, ny = info.dims
    dx, dy = info.spacing
    xmin, ymin = info.origin
    xmax, ymax = xmin + (nx - 1) * dx, ymin + (ny - 1) * dy
    mkpath(output_dir)

    grid_path = joinpath(output_dir, "grid.dat")
    isfile(grid_path) || write_grid(
        grid_path, sim_id, nx, ny, dx, dy, xmin, xmax, ymin, ymax)

    frame = lpad(i, 4, '0')
    fields = (
        ("levelset", geom.levelset),
        ("density", ρ.data),
        ("strain_xx", ε[1].data),
        ("strain_yy", ε[2].data),
        ("strain_xy", ε[3].data),
        ("velocity", v_grid.data),
    )

    for (name, values) in fields
        data_dir = joinpath(output_dir, name)
        mkpath(data_dir)
        write_field(joinpath(data_dir, "$(name)_$(frame).dat"), sim_id, name,
                    values, nx, ny, dx, dy, xmin, ymin; compact)
    end

    if export_images
        extension = compact ? "bin" : "dat"
        density = read_field(joinpath(output_dir, "density", "density_$(frame).$(extension)"))
        levelset = read_field(joinpath(output_dir, "levelset", "levelset_$(frame).$(extension)"))
        export_density_field(read_grid(grid_path), density, levelset,
                             current_t, output_dir, i)
    end

    return nothing
end
# =========== FIN EXPORT BLOCK ============
