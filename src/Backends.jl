# Projected assembly backend: the master-only operator `B = EᵀAE` over the
# host's Kelvin-Voigt kernel. Phases are explicit so the simulation loop
# reads as discretize → restrict → assemble → solve → recover.

const LINEAR_SOLVER_RELATIVE_TOLERANCE = 1e-6

Base.@kwdef mutable struct ProjectedBackend
    plan = nothing              # StaticAssembler.StaticSystem
    form = nothing              # host kelvin_voigt_form(p)
    guess = nothing             # master-DOF vector (n_m)
    velocity = nothing          # full-background vector (n_bg)
    cg_state = nothing
    Pc = nothing                # incomplete LDL, diagonal only on factorization failure
    pc_structure = nothing      # projection instance used to build Pc
    box_structure = nothing     # projection instance used to identify wall DOFs
    box_dofs = BitVector()
end

coefficients(state) = (state.ρ.data, state.ε[1].data, state.ε[2].data, state.ε[3].data, state.α.data)

# Keep the reusable vectors synchronized with the current projected space.
function _sync_buffers!(b::ProjectedBackend)
    E = b.plan.refresh.E
    n_bg, n_m = size(E)
    if isnothing(b.guess) || length(b.guess) != n_m
        b.guess = zeros(Float64, n_m)
        b.cg_state = IterativeSolvers.CGStateVariables(
            similar(b.guess), similar(b.guess), similar(b.guess))
    end
    if isnothing(b.velocity) || length(b.velocity) != n_bg
        b.velocity = zeros(Float64, n_bg)
    end
    if b.box_structure !== E
        nx, ny = b.plan.nx + 1, b.plan.ny + 1
        resize!(b.box_dofs, n_m)
        for dof in eachindex(b.box_dofs)
            row = b.plan.refresh.master_rows[dof]
            node = (row + 1) ÷ 2
            i = (node - 1) % nx + 1
            j = (node - 1) ÷ nx + 1
            b.box_dofs[dof] = isodd(row) ? (i == 1 || i == nx) : (j == 1 || j == ny)
        end
        b.box_structure = E
    end
    return b
end

function discretize!(b::ProjectedBackend, state)
    cutgeo = ensure_cut!(state.geom)
    aggregates = aggregate_cut_cells(cutgeo)
    if isnothing(b.plan)
        b.plan = StaticAssembler.StaticSystem(
            state.geom.grid, cutgeo, aggregates,
            state.info.dims[1] - 1, state.info.dims[2] - 1;
            form=isnothing(b.form) ? kelvin_voigt_form(state.p) : b.form)
    else
        StaticAssembler.refresh_geometry!(b.plan, cutgeo, aggregates)
    end
    return _sync_buffers!(b)
end

# Restrict the previous velocity to master DOFs as the CG initial guess, and
# hand the current background coefficient fields to assembly.
function restrict_fields!(b::ProjectedBackend, state)
    StaticAssembler.restrict_solution!(b.guess, b.plan, b.velocity)
    return coefficients(state)
end

function assemble_system!(b::ProjectedBackend, coefficients)
    operator, rhs = StaticAssembler.assemble!(b.plan, coefficients)
    _apply_closed_box!(operator, rhs, b.box_dofs)
    return operator, rhs
end

function _apply_closed_box!(operator, rhs, constrained)
    @inbounds for column in axes(operator, 2)
        diagonal_found = false
        for entry in nzrange(operator, column)
            row = operator.rowval[entry]
            if constrained[column] || constrained[row]
                diagonal = row == column
                operator.nzval[entry] = diagonal ? 1.0 : 0.0
                diagonal_found |= diagonal
            end
        end
        if constrained[column]
            diagonal_found || error("closed-box DOF has no matrix diagonal")
            rhs[column] = 0.0
        end
    end
    return nothing
end

function _update_preconditioner!(b::ProjectedBackend, operator)
    # StaticSystem replaces E whenever it rebuilds the projected sparse structure.
    Pc = if b.Pc isa Preconditioners.LLDL.LimitedLDLFactorization &&
       b.Pc.n == size(operator, 1) &&
       b.pc_structure === b.plan.refresh.E
        Preconditioners.LLDL.lldl(operator; memory=20, P=b.Pc.P)
    else
        Preconditioners.LLDL.lldl(operator; memory=20)
    end
    Preconditioners.LLDL.factorized(Pc) || error("incomplete LDL factorization failed")
    b.Pc = Pc
    b.pc_structure = b.plan.refresh.E
    return Pc
end

function solve_velocity!(b::ProjectedBackend, operator, rhs)
    try
        _update_preconditioner!(b, operator)
    catch e
        e isa InterruptException && rethrow()
        @warn "Incomplete LDL failed; using diagonal preconditioning" exception=e
        b.Pc = DiagonalPreconditioner(operator)
    end
    tolerance = LINEAR_SOLVER_RELATIVE_TOLERANCE * norm(rhs)
    if b.Pc isa DiagonalPreconditioner || all(>(0), b.Pc.D)
        return IterativeSolvers.cg!(
            b.guess, operator, rhs, Pl=b.Pc, statevars=b.cg_state,
            abstol=tolerance, reltol=0.0)
    end
    return IterativeSolvers.gmres!(
        b.guess, operator, rhs, Pr=b.Pc, restart=20,
        abstol=tolerance, reltol=0.0)
end

function recover_velocity!(b::ProjectedBackend, state, x)
    mul!(b.velocity, b.plan.refresh.E, x)
    v = state.v_grid
    @inbounds for node in eachindex(v.data)
        v.data[node] = VectorValue(b.velocity[2node - 1], b.velocity[2node])
    end
    return v
end
