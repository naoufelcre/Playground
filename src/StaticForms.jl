module StaticForms

using LinearAlgebra: UniformScaling
import LinearAlgebra: tr
using StaticAssembler

export @static_form, SymTensor2, symtensor

"""A small immutable symmetric 2D tensor stored as `(11, 22, 12)`."""
struct SymTensor2{T}
    xx::T
    yy::T
    xy::T
end

@inline function symtensor(xx, yy, xy)
    xx, yy, xy = promote(xx, yy, xy)
    return SymTensor2(xx, yy, xy)
end

@inline Base.:+(a::SymTensor2, b::SymTensor2) =
    symtensor(a.xx + b.xx, a.yy + b.yy, a.xy + b.xy)
@inline Base.:-(a::SymTensor2) = symtensor(-a.xx, -a.yy, -a.xy)
@inline Base.:*(a::Number, b::SymTensor2) =
    symtensor(a * b.xx, a * b.yy, a * b.xy)
@inline Base.:*(a::SymTensor2, b::Number) = b * a
@inline Base.:+(a::SymTensor2, b::UniformScaling) =
    symtensor(a.xx + b.λ, a.yy + b.λ, a.xy)
@inline Base.:+(a::UniformScaling, b::SymTensor2) = b + a
@inline tr(a::SymTensor2) = a.xx + a.yy

@inline function Base.getindex(a::SymTensor2, i::Int, j::Int)
    i == 1 && j == 1 && return a.xx
    i == 2 && j == 2 && return a.yy
    ((i == 1 && j == 2) || (i == 2 && j == 1)) && return a.xy
    throw(BoundsError(a, (i, j)))
end

# Standard 2D operators used by the readable form. The local vector DOF order
# is component-blocked: all x-component nodes, then all y-component nodes.
struct IdentityOp <: StaticAssembler.AbstractDifferentialOperator end
struct SymGradOp <: StaticAssembler.AbstractDifferentialOperator end

@inline _row_node(row::Int, nnode::Int) = row <= nnode ? row : row - nnode
@inline _row_comp(row::Int, nnode::Int) = row <= nnode ? 1 : 2

@inline function StaticAssembler._op_pair(::IdentityOp, ::IdentityOp, row::Int,
                                          col::Int, sh_t, gx_t, gy_t,
                                          sh_c, gx_c, gy_c)
    nnode = length(sh_t)
    _row_comp(row, nnode) == _row_comp(col, nnode) || return zero(first(sh_t))
    return sh_t[_row_node(row, nnode)] * sh_c[_row_node(col, nnode)]
end

@inline function StaticAssembler._op_pair(::SymGradOp, ::SymGradOp, row::Int,
                                          col::Int, sh_t, gx_t, gy_t,
                                          sh_c, gx_c, gy_c)
    nnode = length(sh_t)
    rnode = _row_node(row, nnode)
    cnode = _row_node(col, nnode)
    if row <= nnode
        if col <= nnode
            return gx_t[rnode] * gx_c[cnode] + gy_t[rnode] * gy_c[cnode] / 2
        else
            return gy_t[rnode] * gx_c[cnode] / 2
        end
    elseif col <= nnode
        return gx_t[rnode] * gy_c[cnode] / 2
    else
        return gy_t[rnode] * gy_c[cnode] + gx_t[rnode] * gx_c[cnode] / 2
    end
end

@inline function StaticAssembler._rhs_contract!(fe, ::IdentityOp, slot, shape,
                                                gx, gy, c, w, offset::Int)
    source = slot.source(c)
    nnode = length(shape)
    @inbounds for row in 1:2nnode
        node = _row_node(row, nnode)
        value = row <= nnode ? source[1] : source[2]
        fe[offset + row] += slot.scale * value * shape[node] * w
    end
    return nothing
end

@inline function StaticAssembler._rhs_contract!(fe, ::SymGradOp, slot, shape,
                                                gx, gy, c, w, offset::Int)
    σ = slot.source(c)
    σ11, σ22, σ12 = σ[1, 1], σ[2, 2], σ[1, 2]
    nnode = length(shape)
    @inbounds for row in 1:2nnode
        node = _row_node(row, nnode)
        if row <= nnode
            fe[offset + row] += slot.scale * (σ11 * gx[node] + σ12 * gy[node]) * w
        else
            fe[offset + row] += slot.scale * (σ22 * gy[node] + σ12 * gx[node]) * w
        end
    end
    return nothing
end

const _SELF = @__MODULE__

_form_error(message, expression=nothing) = throw(ArgumentError(
    isnothing(expression) ? message : "$message: $(sprint(show, expression))"))

_is_call(expression, name::Symbol) = expression isa Expr &&
    expression.head === :call && expression.args[1] === name

function _split_product(expression)
    _is_call(expression, :*) || return Any[expression]
    factors = Any[]
    for argument in expression.args[2:end]
        append!(factors, _split_product(argument))
    end
    return factors
end

function _product(factors)
    isempty(factors) && return 1.0
    length(factors) == 1 && return only(factors)
    return Expr(:call, :*, factors...)
end

function _contains_symbol(expression, symbols::Set{Symbol})
    expression isa Symbol && return expression in symbols
    expression isa Expr || return false
    return any(argument -> _contains_symbol(argument, symbols), expression.args)
end

function _parse_fields(argument)
    argument isa Expr && argument.head === :(=) && argument.args[1] === :fields ||
        _form_error("expected fields=(...)", argument)
    entries = argument.args[2]
    entries isa Expr && entries.head === :tuple ||
        _form_error("fields must be a tuple", entries)

    scalar = Dict{Symbol,Int}()
    tensors = Dict{Symbol,NTuple{3,Int}}()
    next_index = 1

    function register(name)
        name isa Symbol || _form_error("field names must be symbols", name)
        (haskey(scalar, name) || haskey(tensors, name)) &&
            _form_error("duplicate field $name")
        scalar[name] = next_index
        next_index += 1
    end

    for entry in entries.args
        if entry isa Symbol
            register(entry)
            continue
        end
        entry isa Expr && entry.head === :(=) && entry.args[1] isa Symbol ||
            _form_error("expected a scalar field or tensor alias", entry)
        alias = entry.args[1]
        value = entry.args[2]
        _is_call(value, :symtensor) && length(value.args) == 4 ||
            _form_error("tensor fields must use symtensor(ε₁₁, ε₂₂, ε₁₂)", value)
        (haskey(scalar, alias) || haskey(tensors, alias)) &&
            _form_error("duplicate field $alias")
        components = value.args[2:4]
        all(component -> component isa Symbol, components) ||
            _form_error("tensor component names must be symbols", value)
        alias in components && _form_error("tensor alias must differ from its components", value)
        indices = ntuple(3) do component
            register(components[component])
            scalar[components[component]]
        end
        tensors[alias] = indices
    end

    return scalar, tensors, next_index - 1
end

function _field_expression(name::Symbol, coefficient, scalar, tensors)
    if haskey(scalar, name)
        return Expr(:ref, coefficient, scalar[name])
    end
    indices = tensors[name]
    return Expr(:call, GlobalRef(_SELF, :symtensor),
                (Expr(:ref, coefficient, index) for index in indices)...)
end

function _rewrite_fields(expression, coefficient, scalar, tensors)
    if expression isa Symbol && (haskey(scalar, expression) || haskey(tensors, expression))
        return _field_expression(expression, coefficient, scalar, tensors)
    end
    expression isa Expr || return expression
    return Expr(expression.head,
                (_rewrite_fields(argument, coefficient, scalar, tensors)
                 for argument in expression.args)...)
end

function _operator_operand(expression, variable::Symbol)
    factors = _split_product(expression)
    matches = Int[]
    kinds = Symbol[]
    for (index, factor) in pairs(factors)
        if factor === variable
            push!(matches, index)
            push!(kinds, :identity)
        elseif _is_call(factor, :D) && length(factor.args) == 2 &&
               factor.args[2] === variable
            push!(matches, index)
            push!(kinds, :symgrad)
        end
    end
    length(matches) == 1 ||
        _form_error("expected exactly one occurrence of $variable or D($variable)", expression)
    extras = Any[factor for (index, factor) in pairs(factors) if index != only(matches)]
    return only(kinds), extras
end

function _pairing(expression)
    factors = _split_product(expression)
    matches = findall(factor -> _is_call(factor, :⋅) || _is_call(factor, :⊙), factors)
    length(matches) == 1 || _form_error("expected one operator pairing", expression)
    index = only(matches)
    pairing = factors[index]
    length(pairing.args) == 3 || _form_error("operator pairing must have two operands", pairing)
    outside = Any[factor for (i, factor) in pairs(factors) if i != index]
    return pairing.args[1], pairing.args[2], pairing.args[3], outside
end

function _operator_expression(kind::Symbol)
    name = kind === :identity ? :IdentityOp : :SymGradOp
    return Expr(:call, GlobalRef(_SELF, name))
end

function _coefficient_expression(expression, scalar, tensors)
    if expression isa Symbol && haskey(scalar, expression)
        return Expr(:call, GlobalRef(StaticAssembler, :FieldCoefficient), scalar[expression])
    end
    expression isa Symbol && haskey(tensors, expression) &&
        _form_error("a tensor field cannot be a scalar bilinear coefficient", expression)

    names = Set{Symbol}((keys(scalar)..., keys(tensors)...))
    if _contains_symbol(expression, names)
        coefficient = gensym(:coefficients)
        value = _rewrite_fields(expression, coefficient, scalar, tensors)
        closure = Expr(:->, coefficient, value)
        return Expr(:call, GlobalRef(StaticAssembler, :StateFunction), closure)
    end
    return Expr(:call, GlobalRef(StaticAssembler, :ConstantCoefficient), expression)
end

function _bilinear_slot(expression, trial::Symbol, test::Symbol, scalar, tensors)
    contraction, left, right, outside = _pairing(expression)
    trial_operator, left_factors = _operator_operand(left, trial)
    test_operator, right_factors = _operator_operand(right, test)
    if contraction === :⋅
        trial_operator === test_operator === :identity ||
            _form_error("⋅ requires value operators on trial and test fields", expression)
    else
        trial_operator === test_operator === :symgrad ||
            _form_error("⊙ currently supports D(trial) and D(test)", expression)
    end

    coefficient = _product(vcat(outside, left_factors, right_factors))
    _contains_symbol(coefficient, Set((trial, test))) &&
        _form_error("bilinear coefficient cannot depend on trial or test fields", coefficient)
    coefficient = _coefficient_expression(coefficient, scalar, tensors)
    return Expr(:call, GlobalRef(StaticAssembler, :BilinearSlot),
                _operator_expression(test_operator),
                _operator_expression(trial_operator), coefficient)
end

function _peel_minus(expression, scale)
    while _is_call(expression, :-) && length(expression.args) == 2
        scale = -scale
        expression = expression.args[2]
    end
    return expression, scale
end

function _rhs_slot(expression, trial::Symbol, test::Symbol, scalar, tensors)
    expression, scale = _peel_minus(expression, 1.0)
    contraction, source, test_expression, outside = _pairing(expression)
    source, scale = _peel_minus(source, scale)
    test_operator, test_factors = _operator_operand(test_expression, test)
    if contraction === :⋅
        test_operator === :identity ||
            _form_error("⋅ requires the test field value", expression)
    else
        test_operator === :symgrad ||
            _form_error("⊙ currently requires D(test)", expression)
    end

    source = _product(vcat(outside, test_factors, Any[source]))
    _contains_symbol(source, Set((trial, test))) &&
        _form_error("right-hand-side source cannot depend on trial or test fields", source)
    coefficient = gensym(:coefficients)
    value = _rewrite_fields(source, coefficient, scalar, tensors)
    closure = Expr(:->, coefficient, value)
    return Expr(:call, GlobalRef(StaticAssembler, :RhsSlot),
                _operator_expression(test_operator), closure, scale)
end

"""
    @static_form space (trial, test) fields=(...) begin
        bilinear(...)
        rhs(...)
    end

Lower volume-integral terms to a concrete `StaticAssembler.ElementVariationalForm`.
Supported pairings are `trial ⋅ test`, `D(trial) ⊙ D(test)`, and their matching
right-hand-side contractions. A declaration such as
`ε=symtensor(ε₁₁, ε₂₂, ε₁₂)` consumes three consecutive scalar fields.
"""
macro static_form(space, variables, fields, body)
    variables isa Expr && variables.head === :tuple && length(variables.args) == 2 &&
        all(variable -> variable isa Symbol, variables.args) ||
        _form_error("expected (trial, test)", variables)
    trial, test = variables.args
    scalar, tensors, ncoefficients = _parse_fields(fields)

    statements = body isa Expr && body.head === :block ? body.args : Any[body]
    bilinear = Any[]
    rhs = Any[]
    for statement in statements
        statement isa LineNumberNode && continue
        if _is_call(statement, :bilinear) && length(statement.args) == 2
            push!(bilinear, _bilinear_slot(statement.args[2], trial, test, scalar, tensors))
        elseif _is_call(statement, :rhs) && length(statement.args) == 2
            push!(rhs, _rhs_slot(statement.args[2], trial, test, scalar, tensors))
        else
            _form_error("expected bilinear(...) or rhs(...)", statement)
        end
    end
    isempty(bilinear) && _form_error("a form needs at least one bilinear term")

    form = Expr(:call, GlobalRef(StaticAssembler, :ElementVariationalForm), space,
                Expr(:tuple, bilinear...), Expr(:tuple, rhs...), ncoefficients)
    return esc(form)
end

end
