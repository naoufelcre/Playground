"""
    E(ε, ν)

Scaled elasticity tensor `Ẽ = (1-2ν)ε + ν tr(ε)I`.
"""
E(ε, ν) = (1.0 - 2.0ν) * ε + ν * tr(ε) * I

"""
Hill function helper
"""
function hill(x,amplitude,mode,threshold)
	return amplitude*(x^mode/(threshold^mode + x^mode ))
end

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
    Substrate stiffness dependant contraction, the heuristic is that heterogeneous substrate stiffness induce the heteorgeneity

    now what we can observe is that α is the modulus and thus we can have a saturating form; Φ(α) is then the substrate friction
"""
function ζ_α(α,m::Integer=2)
    return 1.0 - α^(m - 1)
end

"""
    Substrate friction from stiffness: Φ(α)
"""
function Φ(α)
    return hill(α,1.0,1.0,0.5)
end


"""
    kelvin_voigt_form(p)

Static form of
`a(u,v) = ∫Ω Φ(α) u⋅v + ξ D(u)⊙D(v)` and
`l(v) = -∫Ω (E(ε,ν) + A (ζ(ρ) + ζ_α(α))I)⊙D(v)`.
"""
function kelvin_voigt_form(p)
    space = StaticAssembler.FiniteElement(
        StaticAssembler.Q1ReferenceQuadrilateral(), 2)

    return @static_form space (u, v) fields=(
        ρ,
        ε = symtensor(ε₁₁, ε₂₂, ε₁₂),
        α,
    ) begin
        bilinear(Φ(α) * (u ⋅ v))
        bilinear(p.ξ * (D(u) ⊙ D(v)))
        rhs(-(E(ε, p.ν) + p.A * (ζ(ρ, p.m) + ζ_α(α, p.m) )* I) ⊙ D(v))
    end
end

"""
    stress_norm_squared(state)

`∫Ω |σ|²` with `σ = E(ε,ν) + A (ζ(ρ) + ζ_α(α))I` (Frobenius norm).
Node quadrature on the extended grid masked by the level set (`ls < 0`).
"""
function stress_norm_squared(state)
    ρ = state.ρ.data
    εxx, εyy, εxy = state.ε[1].data, state.ε[2].data, state.ε[3].data
    α = state.α.data
    ls = state.geom.levelset
    p = state.p
    dA = state.info.spacing[1] * state.info.spacing[2]
    elastic = 1.0 - 2.0 * p.ν
    s = 0.0
    @inbounds for i in eachindex(ρ, α, ls)
        ls[i] < 0 || continue
        trace = εxx[i] + εyy[i]
        pressure = p.A * (ζ(ρ[i], p.m) + ζ_α(α[i], p.m))
        σxx = elastic * εxx[i] + p.ν * trace + pressure
        σyy = elastic * εyy[i] + p.ν * trace + pressure
        σxy = elastic * εxy[i]
        s += σxx^2 + σyy^2 + 2.0 * σxy^2
    end
    return s * dA
end

"""
    stress_field!(out, state)

Per-node Frobenius norm `|σ|` with the same `σ` as `stress_norm_squared`.
Nodes with `ls >= 0` are set to `NaN` so terminal-plot autoscaling only
sees the tissue.
"""
function stress_field!(out, state)
    ρ = state.ρ.data
    εxx, εyy, εxy = state.ε[1].data, state.ε[2].data, state.ε[3].data
    α = state.α.data
    ls = state.geom.levelset
    p = state.p
    elastic = 1.0 - 2.0 * p.ν
    @inbounds for i in eachindex(out, ρ, α, ls)
        if ls[i] < 0
            trace = εxx[i] + εyy[i]
            pressure = p.A * (ζ(ρ[i], p.m) + ζ_α(α[i], p.m))
            σxx = elastic * εxx[i] + p.ν * trace + pressure
            σyy = elastic * εyy[i] + p.ν * trace + pressure
            σxy = elastic * εxy[i]
            out[i] = sqrt(σxx^2 + σyy^2 + 2.0 * σxy^2)
        else
            out[i] = NaN
        end
    end
    return out
end
