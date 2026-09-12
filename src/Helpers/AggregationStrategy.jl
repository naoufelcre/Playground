# Threshold-1 specialization of GridapEmbedded aggregation.
#
# Purpose:
#   Keep the aggregation failure diagnostic, while avoiding the expensive
#   cut-cell measure path for the threshold-1 strategy used by the simulation.
#
# Changes vs upstream (GridapEmbedded/src/AgFEM/CellAggregation.jl):
#   1. threshold-1 aggregation skips cut-cell measure construction
#   2. @assert all_aggregated -> descriptive error(...) with cell counts
#   3. an all-CUT component uses its largest physical cell as a root
#
# Usage:
#   include("Aggregation/AggregationStrategy.jl")  # after `using GridapEmbedded`
#
# The existing call site:
#   strategy = AggregateCutCellsByThreshold(1.0)
#   aggregates = aggregate(strategy, cutgeo)
# dispatches to the specialized two-argument `aggregate` below.

import GridapEmbedded.AgFEM: aggregate,
                             _find_best_neighbor,
                             _touch_aggregated_cells!
using Gridap.Arrays: array_cache, getindex!
using GridapEmbedded.Interfaces: EmbeddedDiscretization, IN, CUT
using Gridap.Geometry: get_cell_coordinates, get_grid_topology, get_faces,
                       get_triangulation, num_cell_dims

# AggregateCutCellsByThreshold(1.0) is the only strategy used by the
# simulation.  At this threshold the stored cell classification already gives
# the exact root mask, so no cut-cell measure or temporary triangulation is
# needed.
function aggregate(strategy::GridapEmbedded.AgFEM.AggregateCutCellsByThreshold,
                   cutgeo::EmbeddedDiscretization)
  @assert strategy.threshold == 1.0
  raw_status = cutgeo.ls_to_bgcell_to_inoutcut
  cell_to_inoutcut = eltype(raw_status) <: AbstractVector ? raw_status[1] : raw_status
  facet_to_inoutcut = GridapEmbedded.compute_bgfacet_to_inoutcut(
    cutgeo.bgmodel, cutgeo.geo)
  bgtrian = get_triangulation(cutgeo.bgmodel)
  topo = get_grid_topology(cutgeo.bgmodel)
  D = num_cell_dims(cutgeo.bgmodel)
  cell_to_faces = get_faces(topo, D, D - 1)
  face_to_cells = get_faces(topo, D - 1, D)
  cell_to_unit_cut_meas = Float64.(cell_to_inoutcut .== IN)
  _aggregate_by_threshold_barrier_fast(
    strategy.threshold, cell_to_unit_cut_meas, facet_to_inoutcut,
    cell_to_inoutcut, IN, get_cell_coordinates(bgtrian), cell_to_faces,
    face_to_cells; fallback_root_scores=() -> _cut_cell_areas(cutgeo, length(cell_to_inoutcut)))
end

function _cut_cell_areas(cutgeo, n_cells)
  areas = zeros(Float64, n_cells)
  subcells = cutgeo.subcells
  connectivity = subcells.cell_to_points
  subcell_states = only(cutgeo.ls_to_subcell_to_inout)
  for subcell in eachindex(subcells.cell_to_bgcell)
    subcell_states[subcell] == IN || continue
    first_point = Int(connectivity.ptrs[subcell])
    next_point = Int(connectivity.ptrs[subcell + 1])
    next_point == first_point + 3 ||
      error("threshold-1 aggregation requires triangular cut subcells")
    p1 = subcells.point_to_coords[connectivity.data[first_point]]
    p2 = subcells.point_to_coords[connectivity.data[first_point + 1]]
    p3 = subcells.point_to_coords[connectivity.data[first_point + 2]]
    areas[subcells.cell_to_bgcell[subcell]] += abs(
      (p2[1] - p1[1]) * (p3[2] - p1[2]) -
      (p3[1] - p1[1]) * (p2[2] - p1[2])) / 2
  end
  areas
end

function _aggregate_by_threshold_barrier_fast(
  threshold, cell_to_unit_cut_meas, facet_to_inoutcut, cell_to_inoutcut,
  loc, cell_to_coords, cell_to_faces, face_to_cells;
  fallback_root_scores=nothing)

  n_cells = length(cell_to_unit_cut_meas)
  cell_to_cellin = zeros(Int32, n_cells)
  cell_to_touched = fill(false, n_cells)

  for cell in 1:n_cells
    if cell_to_unit_cut_meas[cell] >= threshold
      cell_to_cellin[cell] = cell
      cell_to_touched[cell] = true
    end
  end

  c1 = array_cache(cell_to_faces)
  c2 = array_cache(face_to_cells)
  c3 = array_cache(cell_to_coords)
  c4 = array_cache(cell_to_coords)

  fallback_roots = Int[]
  root_scores = nothing
  for _ in 1:n_cells
    unaggregated = 0
    made_progress = false
    for cell in 1:n_cells
      if !cell_to_touched[cell] && cell_to_inoutcut[cell] == CUT
        neigh_cell = _find_best_neighbor(
          c1, c2, c3, c4, cell,
          cell_to_faces,
          face_to_cells,
          cell_to_coords,
          cell_to_touched,
          cell_to_cellin,
          facet_to_inoutcut,
          loc)
        if neigh_cell > 0
          cellin = cell_to_cellin[neigh_cell]
          cell_to_cellin[cell] = cellin
          made_progress = true
        else
          unaggregated += 1
        end
      end
    end
    if unaggregated == 0
      break
    end
    _touch_aggregated_cells!(cell_to_touched, cell_to_cellin)

    if !made_progress
      isnothing(fallback_root_scores) && break
      isnothing(root_scores) && (root_scores = fallback_root_scores())
      root = 0
      best_score = 0.0
      for cell in 1:n_cells
        if !cell_to_touched[cell] && cell_to_inoutcut[cell] == CUT &&
           root_scores[cell] > best_score
          root = cell
          best_score = root_scores[cell]
        end
      end
      root > 0 || break
      cell_to_cellin[root] = root
      cell_to_touched[root] = true
      push!(fallback_roots, root)
    end
  end

  unagg_cells = findall(
    i -> cell_to_cellin[i] == 0 && cell_to_inoutcut[i] == CUT,
    1:n_cells)
  if !isempty(unagg_cells)
    n_cut = count(==(CUT), cell_to_inoutcut)
    error(
      "Aggregation failed: $(length(unagg_cells)) / $n_cut CUT cells remain unaggregated " *
      "(n_cells=$n_cells). " *
      "Unaggregated cell IDs: $(join(unagg_cells[1:min(10, length(unagg_cells))], ", "))" *
      (length(unagg_cells) > 10 ? "..." : "") *
      ". Inspect the geometry near pinch-off regions."
    )
  end

  isempty(fallback_roots) || @warn(
    "AgFEM component has no interior cell; using its largest cut cell as root",
    roots=fallback_roots, maxlog=3)

  cell_to_cellin
end
