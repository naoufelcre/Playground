module Patterns

using Gridap
using Gridap.TensorValues
using Gridap.Geometry: get_node_coordinates
using Base.Threads
using EvolvingDomains
using EvolvingDomains.Geometric
using Random

export AbstractDensityPattern, UniformTissuePattern, PatchDensityPattern,
       initialize_density
export AbstractStiffnessPattern, UniformStiffnessPattern, RadialStiffnessPattern,
       PatchStiffnessPattern, stiffness_field

abstract type AbstractDensityPattern end

# Homogeneous tissue: constant ρ in the tissue (negative phase), zero in the
# lesion/void (positive phase). Ported from AMD_tools UniformTissuePattern.
struct UniformTissuePattern{T<:Real} <: AbstractDensityPattern
    density::T
end

function UniformTissuePattern(; density=1.0)
    density isa Real && !(density isa Bool) && isfinite(density) && density > 0 ||
        throw(ArgumentError("density must be finite and greater than zero"))
    return UniformTissuePattern(Float64(density))
end

function initialize_density(info, pattern::UniformTissuePattern, geom)
    data = zeros(Float64, prod(info.dims))
    @inbounds for idx in eachindex(data)
        if geom.levelset[idx] < 0.0
            data[idx] = pattern.density
        end
    end
    return CartesianMeshField(data, info)
end

struct PatchDensityPattern <: AbstractDensityPattern
    seed::String
    n_patches::Int
    minimum_density::Float64
end

function PatchDensityPattern(; seed::String, n_patches=6, minimum_density=0.8)
    0.2 < minimum_density <= 1.0 || throw(ArgumentError("minimum density must be in (0.2, 1]"))
    return PatchDensityPattern(seed, n_patches, Float64(minimum_density))
end

# ponytail: shared gaussian-patch core for density + stiffness, split if they diverge
_make_patches(seed::String, n_patches) =
    let rng = MersenneTwister(seed)
        [(rand(rng), rand(rng), 0.05 + 0.10rand(rng), 0.35 + 0.45rand(rng))
         for _ in 1:n_patches]
    end
_patch_value(saturation, minimum) = inv(1.0 + (inv(minimum) - 1.0) * saturation)

function initialize_density(info, pattern::PatchDensityPattern, geom)
    patches = _make_patches(pattern.seed, pattern.n_patches)
    nx, ny = info.dims
    dx, dy = info.spacing
    ox, oy = info.origin
    data = zeros(Float64, nx * ny)
    @inbounds for j in 1:ny, i in 1:nx
        idx = (j - 1) * nx + i
        geom.levelset[idx] < 0.0 || continue
        x, y = ox + (i - 1) * dx, oy + (j - 1) * dy
        # Build round-cell area first, then use ρ = 1/area. This keeps the
        # universal reference density at one without encoding area in strain.
        expansion = sum(depth * exp(-((x - cx)^2 + (y - cy)^2) / (2width^2))
                        for (cx, cy, width, depth) in patches)
        data[idx] = _patch_value(-expm1(-expansion), pattern.minimum_density)
    end
    return CartesianMeshField(data, info)
end

# =============================================================================
# Substrate stiffness α(x): static field; Φ(α) is the substrate friction
# multiplying the first-order velocity term v⋅w in the Kelvin-Voigt weak form.
# Radial stiffness is larger at the centre of the [0,1]² box and decreases
# toward its boundary; the formula is ported from AMD_tools
# (VarForms.friction_coefficient, historically a friction).
# =============================================================================

abstract type AbstractStiffnessPattern end

struct UniformStiffnessPattern <: AbstractStiffnessPattern end

struct RadialStiffnessPattern <: AbstractStiffnessPattern
    center_coefficient::Float64
    boundary_coefficient::Float64
end

RadialStiffnessPattern(; center_coefficient=1.0, boundary_coefficient=0.25) =
    RadialStiffnessPattern(Float64(center_coefficient), Float64(boundary_coefficient))

function stiffness_field(info, ::UniformStiffnessPattern)
    data = ones(Float64, prod(info.dims))
    return CartesianMeshField(data, info)
end

function stiffness_field(info, pattern::RadialStiffnessPattern)
    nx, ny = info.dims
    dx, dy = info.spacing
    ox, oy = info.origin
    data = Vector{Float64}(undef, nx * ny)
    @inbounds for j in 1:ny
        y = oy + (j - 1) * dy
        for i in 1:nx
            x = ox + (i - 1) * dx
            r = ((x - 0.5)^2 + (y - 0.5)^2) / 0.5
            data[(j - 1) * nx + i] =
                pattern.center_coefficient +
                (pattern.boundary_coefficient - pattern.center_coefficient) * r
        end
    end
    return CartesianMeshField(data, info)
end

struct PatchStiffnessPattern <: AbstractStiffnessPattern
    seed::String
    n_patches::Int
    minimum_stiffness::Float64
end

function PatchStiffnessPattern(; seed::String, n_patches=6, minimum_stiffness=0.25)
    0.0 < minimum_stiffness <= 1.0 ||
        throw(ArgumentError("minimum stiffness must be in (0, 1]"))
    return PatchStiffnessPattern(seed, n_patches, Float64(minimum_stiffness))
end

function stiffness_field(info, pattern::PatchStiffnessPattern)
    patches = _make_patches(pattern.seed, pattern.n_patches)
    nx, ny = info.dims
    dx, dy = info.spacing
    ox, oy = info.origin
    data = Vector{Float64}(undef, nx * ny)
    @inbounds for j in 1:ny, i in 1:nx
        x, y = ox + (i - 1) * dx, oy + (j - 1) * dy
        expansion = sum(depth * exp(-((x - cx)^2 + (y - cy)^2) / (2width^2))
                        for (cx, cy, width, depth) in patches)
        data[(j - 1) * nx + i] = _patch_value(-expm1(-expansion), pattern.minimum_stiffness)
    end
    return CartesianMeshField(data, info)
end

end
