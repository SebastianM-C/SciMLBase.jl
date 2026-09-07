using SciMLBase
using Enzyme
using Test

import Enzyme: EnzymeRules

# Regression test for SciMLSensitivity.jl#1499.
#
# Solver algorithms are pure configuration and are never differentiated with
# respect to, so the whole `AbstractDEAlgorithm` hierarchy must be declared
# `EnzymeRules.inactive_type`. Without this, on Julia 1.12+ a nested
# polyalgorithm (e.g. the `CompositeAlgorithm` behind `DefaultODEAlgorithm`)
# gets split by the "inline-roots" calling convention and trips Enzyme's
# `roots_activep != activep` assertion inside the `solve_up` custom rule.
struct Issue1499ODEAlg <: SciMLBase.AbstractODEAlgorithm end
struct Issue1499SDEAlg <: SciMLBase.AbstractSDEAlgorithm end
struct Issue1499DAEAlg <: SciMLBase.AbstractDAEAlgorithm end

@testset "AbstractDEAlgorithm is Enzyme-inactive (SciMLSensitivity#1499)" begin
    @test EnzymeRules.inactive_type(Issue1499ODEAlg)
    @test EnzymeRules.inactive_type(Issue1499SDEAlg)
    @test EnzymeRules.inactive_type(Issue1499DAEAlg)
    @test EnzymeRules.inactive_type(SciMLBase.AbstractDEAlgorithm)
end

# `ODEFunction(::NonlinearFunction)` used to install a closure over the whole
# `NonlinearFunction` as the rhs. Enzyme cannot prove such a container free of
# differentiable state (its `sys`/`observed`/`initialization_data` may hold
# numbers behind abstract fields), so `Duplicated` on it meant allocating and
# re-zeroing a shadow of everything per call. The `NonlinearFunctionRHS` adapter
# holds only the residual, which Enzyme proves constant.
@testset "ODEFunction(::NonlinearFunction) rhs carries no differentiable state" begin
    resid(du, u, p) = (du .= u .^ 2 .- p; nothing)
    # a container field Enzyme cannot see through
    nlf = NonlinearFunction{true}(resid; sys = Dict{Any, Any}(:k => 1.0))
    ode = ODEFunction{true}(nlf)

    @test ode.f isa SciMLBase.NonlinearFunctionRHS
    @test Enzyme.make_zero(ode.f) === ode.f
    vf = SciMLBase.Void(ode.f)
    @test Enzyme.make_zero(vf) === vf
    # whereas a shadow of the container is a fresh object
    vnlf = SciMLBase.Void(nlf)
    @test Enzyme.make_zero(vnlf) !== vnlf

    # and the adapter differentiates exactly like the residual, passed `Const`
    u = [1.0, 2.0]
    du, ddu, dλ = zeros(2), ones(2), zeros(2)
    Enzyme.autodiff(
        Enzyme.Reverse, Enzyme.Const(vf), Enzyme.Const,
        Enzyme.Duplicated(du, ddu), Enzyme.Duplicated(copy(u), dλ),
        Enzyme.Const(1.0), Enzyme.Const(0.0)
    )
    @test dλ == 2 .* u
end

if isdefined(Base, :ispublic)
    @testset "Sensitivity algorithm supertypes are public" begin
        for name in (
                :AbstractSensitivityAlgorithm,
                :AbstractOverloadingSensitivityAlgorithm,
                :AbstractForwardSensitivityAlgorithm,
                :AbstractAdjointSensitivityAlgorithm,
                :AbstractSecondOrderSensitivityAlgorithm,
                :AbstractShadowingSensitivityAlgorithm,
            )
            @test Base.ispublic(SciMLBase, name)
        end
    end
end
