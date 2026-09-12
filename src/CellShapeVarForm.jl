module CellShapeVarForm

using Base.Threads
using EvolvingDomains
using EvolvingDomains.Geometric: CartesianMeshField, get_interpolator
using StaticAssembler
import DensityModel

const SF = DensityModel.StaticForms
const MIN_CONSTITUTIVE_DENSITY = 0.2

export augment_form, evolve!, observables!

"""
    augment_form(base, parameters, β)

Replace the base form's infinitesimal area strain by `1/ρ - 1` and add the
directional active stress `2Aβρ dev(ε)`. No coefficient fields are added.
"""
function augment_form(base, parameters, β::Real)
    β = Float64(β)
    isfinite(β) && β >= 0 ||
        throw(ArgumentError("β must be finite and nonnegative"))
    base.ncoefficients == 5 ||
        throw(ArgumentError("density-strain augmentation requires five coefficients"))
    length(base.rhs) == 1 && base.rhs[1].test_operator isa SF.SymGradOp ||
        throw(ArgumentError("density-strain augmentation requires one symmetric-stress RHS"))

    stress = base.rhs[1]
    source = let base_source=stress.source, activity=Float64(parameters.A), β=β
        function (c)
            ρ = max(c[1], MIN_CONSTITUTIVE_DENSITY)
            εxx, εyy, εxy = c[2], c[3], c[4]
            trace = εxx + εyy
            deviatoric_xx = 0.5 * (εxx - εyy)
            area_correction = 0.5 * (inv(ρ) - 1.0 - trace)
            directional = 2.0 * activity * β * ρ
            coefficients = (ρ, εxx, εyy, εxy, c[5])
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

end
