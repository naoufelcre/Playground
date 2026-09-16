# This file contains the code to initialize parameters with the proper scaling
# ... (Theory comments remain the same) ...

const MIN_TIMESTEP = 1e-4

Base.@kwdef struct AdimensionalParameters
    ξ::Float64 # Internal Friction Number
    A::Float64 # Activity Number
    ν::Float64 # Poisson Ratio
    m::Int = 2 # Perthame pressure exponent
end

Base.@kwdef struct PhysicalBasis
    E_ref::Float64
    T_ref::Float64
    L_ref::Float64
end

"""
    get_V_ref(basis::PhysicalBasis)
Compute the reference velocity basis V_ref = L_ref / T_ref.
"""
get_V_ref(basis::PhysicalBasis) = basis.L_ref / basis.T_ref

"""
    physical_parameters(; Y, ν, α, η, ζ, m=2, L_ref=2.0)

Compute adimensionalized parameters from physical values.
- Y: Young Modulus [Pa]
- ν: Poisson Ratio [-] (Protected against ν=0.5)
- α: Substrate stiffness [Pa] (Φ(α) is the substrate friction [Pa.s/m²])
- η: Internal viscosity [Pa.s]
- ζ: Active prestress [Pa]
- m: Perthame pressure exponent (> 1)
- L_ref: Reference length scale [m]. Default 2.0 to match the [-1, 1] domain.
"""
function physical_parameters(; Y, ν, α, η, ζ, m::Int=2, L_ref=2.0)
    # Protection against incompressibility singularity
    if ν >= 0.5
        @warn "Poisson ratio ν >= 0.5 detected. Clipping to 0.499 to avoid singularity in E_ref."
        ν = 0.499
    end

    # E_ref corresponds to the plane strain scaling factor
    E_ref = Y / ((1.0 + ν) * (1.0 - 2.0 * ν))

    # T_ref eliminates the substrate friction Φ(α): T_ref = Φ(α) L² / E_ref
    T_ref = Φ(α) * L_ref^2 / E_ref

    ξ = η / (T_ref * E_ref)
    A = ζ / E_ref

    params = AdimensionalParameters(ξ=ξ, A=A, ν=ν, m=m)
    basis = PhysicalBasis(E_ref=E_ref, T_ref=T_ref, L_ref=L_ref)

    return params, basis
end
