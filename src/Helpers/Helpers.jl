module Helpers

# Low-level I/O from Format.jl
include("Format.jl")
export write_grid, read_grid, write_field, read_field, write_field_compact, read_field_compact
export mask_to_levelset_format

# Visualization utilities
include("Visualization.jl")
using .Visualization
export plot_cell_classification!, plot_raymaps!, get_raymap_bounding_box

# Visualization from Export.jl
include("Export.jl")
export generate_simulation_id, export_metadata_json
export export_density_field, export_strain_norm, export_velocity_field, export_classification


end
