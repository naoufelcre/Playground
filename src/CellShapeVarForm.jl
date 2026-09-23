module CellShapeVarForm

using Base.Threads
using EvolvingDomains
using EvolvingDomains.Geometric: CartesianMeshField, get_interpolator
using StaticAssembler
import DensityModel

const SF = DensityModel.StaticForms
const MIN_CONSTITUTIVE_DENSITY = 0.2

export augment_form, evolve!, observables!, stress_norm_squared, stress_field!,
    von_mises_stress

"""
    augment_form(base, parameters, β)

Replace the base form's infinitesimal area strain by `1/ρ - 1` and add the
directional active stress `2Aβρ dev(ε)`. Substrate stiffness is inherited
unchanged from the base form: `Φ(α)` friction in the bilinear term and
`A ζ_α(α)` contraction via `base_source`. `α` (coefficient 5) is forwarded
as-is; only `ρ` is clamped. `β` stays independent of `α`.
"""
function augment_form(base, parameters, β::Real)
    β = Float64(β)
    isfinite(β) && β >= 0 ||
        throw(ArgumentError("β must be finite and nonnegative"))
    base.ncoefficients == 5 ||
        throw(ArgumentError("density-strain augmentation requires five coefficients (ρ, ε×3, α)"))
    length(base.rhs) == 1 && base.rhs[1].test_operator isa SF.SymGradOp ||
        throw(ArgumentError("density-strain augmentation requires one symmetric-stress RHS"))

    stress = base.rhs[1]
    source = let base_source=stress.source, activity=Float64(parameters.A), β=β
        function (c)
            ρ = max(c[1], MIN_CONSTITUTIVE_DENSITY)
            εxx, εyy, εxy = c[2], c[3], c[4]
            α = c[5]
            trace = εxx + εyy
            deviatoric_xx = 0.5 * (εxx - εyy)
            area_correction = 0.5 * (inv(ρ) - 1.0 - trace)
            directional = 2.0 * activity * β * ρ
            coefficients = (ρ, εxx, εyy, εxy, α)
            return base_source(coefficients) + SF.symtensor(
                area_correction + directional * deviatoric_xx,
                area_correction - directional * deviatoric_xx,
                directional * εxy,
            )
        end
    end
    combined_rhs = RhsSlot(stress.test_operator, source, stress.scale)
    return ElementVariationalForm(
        base.test_space,
        base.trial_space,
        base.bilinear,
        (combined_rhs,),
        base.ncoefficients,
    )
end

"""
    evolve!(state, map, static_geometry, v_extended)

Advance geometry, transport density conservatively, and transport/update the
traceless strain intensively with no local relaxation term.
"""
function evolve!(state, map, static_geometry, v_extended)
    geom = state.geom
    velocity = DensityModel.FieldVelocity(get_interpolator(v_extended))
    EvolvingDomains.advance!(geom, v_extended.data, state.Δt)

    map = if isnothing(map)
        DensityModel.FastTransportMap(geom, velocity, state.Δt, static_geometry)
    else
        DensityModel.update_fast_transport_map!(
            map, geom, velocity, state.Δt, static_geometry)
    end

    DensityModel.advect_fields!([state.ρ_target], [state.ρ], map)
    DensityModel.advect_intensive!(state.ε_target, state.ε, map)
    DensityModel.update_strain!(state.ε_target, v_extended, state.Δt,
                              state.info, geom.levelset; deviatoric=true)
    DensityModel.swap_buffers!(state)
    return map
end

function observables!(nematic_xx, nematic_xy, density_order, density, ε)
    εxx, εyy, εxy = ε[1].data, ε[2].data, ε[3].data
    ρ = density isa CartesianMeshField ? density.data : density
    length(nematic_xx) == length(nematic_xy) == length(density_order) ==
        length(ρ) == length(εxx) == length(εyy) == length(εxy) ||
        throw(DimensionMismatch("density, strain, and observable dimensions differ"))
    @inbounds @simd for i in eachindex(density_order)
        difference = εxx[i] - εyy[i]
        nematic_xx[i] = 0.5 * difference
        nematic_xy[i] = εxy[i]
        density_order[i] = ρ[i] * hypot(difference, 2.0 * εxy[i])
    end
    return nematic_xx, nematic_xy, density_order
end

"""
    stress_norm_squared(state, p, β)

`∫Ω |σ|²` (Frobenius) with the augmented stress from `augment_form`: base
`E(ε,ν) + A (ζ(ρ) + ζ_α(α))I` at clamped density plus area correction and
`2Aβρ dev(ε)`. Node quadrature masked by the level set.
"""
function stress_norm_squared(state, p, β::Real)
    β = Float64(β)
    ρ = state.ρ.data
    εxx, εyy, εxy = state.ε[1].data, state.ε[2].data, state.ε[3].data
    α = state.α.data
    ls = state.geom.levelset
    dA = state.info.spacing[1] * state.info.spacing[2]
    elastic = 1.0 - 2.0 * p.ν
    s = 0.0
    @inbounds for i in eachindex(ρ, α, ls)
        ls[i] < 0 || continue
        ρc = max(ρ[i], MIN_CONSTITUTIVE_DENSITY)
        trace = εxx[i] + εyy[i]
        pressure = Float64(p.A) * (DensityModel.ζ(ρc, p.m) + DensityModel.ζ_α(α[i], p.m))
        area = 0.5 * (inv(ρc) - 1.0 - trace)
        directional = 2.0 * Float64(p.A) * β * ρc
        deviatoric_xx = 0.5 * (εxx[i] - εyy[i])
        σxx = elastic * εxx[i] + p.ν * trace + pressure + area +
              directional * deviatoric_xx
        σyy = elastic * εyy[i] + p.ν * trace + pressure + area -
              directional * deviatoric_xx
        σxy = elastic * εxy[i] + directional * εxy[i]
        s += σxx^2 + σyy^2 + 2.0 * σxy^2
    end
    return s * dA
end

"""
    stress_field!(out, state, p, β)

Per-node Frobenius norm `|σ|` with the same augmented `σ` as
`stress_norm_squared`. Nodes with `ls >= 0` are set to `NaN` so
terminal-plot autoscaling only sees the tissue.
"""
function stress_field!(out, state, p, β::Real)
    β = Float64(β)
    ρ = state.ρ.data
    εxx, εyy, εxy = state.ε[1].data, state.ε[2].data, state.ε[3].data
    α = state.α.data
    ls = state.geom.levelset
    elastic = 1.0 - 2.0 * p.ν
    @inbounds for i in eachindex(out, ρ, α, ls)
        if ls[i] < 0
            ρc = max(ρ[i], MIN_CONSTITUTIVE_DENSITY)
            trace = εxx[i] + εyy[i]
            pressure = Float64(p.A) * (DensityModel.ζ(ρc, p.m) + DensityModel.ζ_α(α[i], p.m))
            area = 0.5 * (inv(ρc) - 1.0 - trace)
            directional = 2.0 * Float64(p.A) * β * ρc
            deviatoric_xx = 0.5 * (εxx[i] - εyy[i])
            σxx = elastic * εxx[i] + p.ν * trace + pressure + area +
                  directional * deviatoric_xx
            σyy = elastic * εyy[i] + p.ν * trace + pressure + area -
                  directional * deviatoric_xx
            σxy = elastic * εxy[i] + directional * εxy[i]
            out[i] = sqrt(σxx^2 + σyy^2 + 2.0 * σxy^2)
        else
            out[i] = NaN
        end
    end
    return out
end

"""
    von_mises_stress(state, p, β)

`∫Ω σ_vm` with the same augmented `σ` as `stress_norm_squared` and the
plane von Mises stress `σ_vm² = σxx² - σxx*σyy + σyy² + 3σxy²`.
Node quadrature masked by the level set.
"""
function von_mises_stress(state, p, β::Real)
    β = Float64(β)
    ρ = state.ρ.data
    εxx, εyy, εxy = state.ε[1].data, state.ε[2].data, state.ε[3].data
    α = state.α.data
    ls = state.geom.levelset
    dA = state.info.spacing[1] * state.info.spacing[2]
    elastic = 1.0 - 2.0 * p.ν
    s = 0.0
    @inbounds for i in eachindex(ρ, α, ls)
        ls[i] < 0 || continue
        ρc = max(ρ[i], MIN_CONSTITUTIVE_DENSITY)
        trace = εxx[i] + εyy[i]
        pressure = Float64(p.A) * (DensityModel.ζ(ρc, p.m) + DensityModel.ζ_α(α[i], p.m))
        area = 0.5 * (inv(ρc) - 1.0 - trace)
        directional = 2.0 * Float64(p.A) * β * ρc
        deviatoric_xx = 0.5 * (εxx[i] - εyy[i])
        σxx = elastic * εxx[i] + p.ν * trace + pressure + area +
              directional * deviatoric_xx
        σyy = elastic * εyy[i] + p.ν * trace + pressure + area -
              directional * deviatoric_xx
        σxy = elastic * εxy[i] + directional * εxy[i]
        s += sqrt(max(σxx^2 - σxx * σyy + σyy^2 + 3.0 * σxy^2, 0.0))
    end
    return s * dA
end

end
