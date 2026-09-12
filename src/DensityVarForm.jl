"""
    E(ε, ν)

Scaled elasticity tensor `Ẽ = (1-2ν)ε + ν tr(ε)I`.
"""
E(ε, ν) = (1.0 - 2.0ν) * ε + ν * tr(ε) * I

"""
Density-dependent active pressure from the centered Perthame family,
`ζₘ(ρ) = m/(2(m-1)) * (1-ρ^(m-1))`. The factor `1/2` preserves the existing
calibration `ζ₂(ρ)=1-ρ`; `m>1` controls the crowding nonlinearity.
"""
function ζ(ρ, m::Integer=2)
    m > 1 || throw(ArgumentError("pressure exponent m must be greater than 1"))
    return m / (2.0 * (m - 1)) * (1.0 - ρ^(m - 1))
end

"""
    kelvin_voigt_form(p)

Static form of
`a(u,v) = ∫Ω α u⋅v + ξ D(u)⊙D(v)` and
`l(v) = -∫Ω (E(ε,ν) + A ζ(ρ)I)⊙D(v)`.
"""
function kelvin_voigt_form(p)
    space = StaticAssembler.FiniteElement(
        StaticAssembler.Q1ReferenceQuadrilateral(), 2)

    return @static_form space (u, v) fields=(
        ρ,
        ε = symtensor(ε₁₁, ε₂₂, ε₁₂),
        α,
    ) begin
        bilinear(α * (u ⋅ v))
        bilinear(p.ξ * (D(u) ⊙ D(v)))
        rhs(-(E(ε, p.ν) + p.A * ζ(ρ, p.m) * I) ⊙ D(v))
    end
end
