
using Printf, Test

include("../../hybmc/simulations/McSimulation.jl")
include("../../hybmc/simulations/Payoffs.jl")

@testset "test_payoffSetup" begin
    F = Fixed(1.0)
    P = Pay(F, 1.0)
    A = Asset(5.0,"EUR")
    Z = ZeroBond(5.0,10.0)
    #
    E1 = A - Z
    E2 = Z + 2
    E3 = 1.0 / A
    println(E1)
    println(E2)
    println(E3)
    P = E1 % 2.4
    println(P)
    println(obsTime(P))
end

