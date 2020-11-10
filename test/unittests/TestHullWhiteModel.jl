
using Printf, Test

include("../../hybmc/termstructures/YieldCurve.jl")
include("../../hybmc/models/HullWhiteModel.jl")
include("../../hybmc/simulations/McSimulation.jl")
include("../../hybmc/simulations/Payoffs.jl")


function setup()
    curve = YieldCurve(0.03)
    mean = 0.03
    times = [ 1.0,    2.0,    5.0,    10.0     ]
    vols  = [ 0.0060, 0.0070, 0.0080,  0.0100  ]
    model = HullWhiteModel(curve,mean,times,vols)
    return model
end


@testset "test_zeroBondPrice" begin
    model = setup()
    obsTime = 10.0
    matTime = 20.0
    states  = [ -0.10, -0.05, 0.00, 0.03, 0.07, 0.12 ]
    refResult = [  # reference results checked against QuantLib
        1.7178994273756056,
        1.1153102854629025,
        0.7240918839816156,
        0.5587692049384843,
        0.39550396436404667,
        0.25677267968500767]
    for (x,res) in zip(states,refResult)
        pv = zeroBondPrice(model,obsTime,matTime,x)
        @test isapprox(pv,res,atol=1.0e-16, rtol=0.0)
        #println(pv)        
    end
end

@testset "test_couponBondOption" begin
    model = setup()
    excTime = 10.0
    payTimes  = [ 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 20.0 ]
    cashFlows = cat(0.025*ones(10), [1.0], dims=1) # bond cash flows
    callOrPut = 1.0 # call
    strikes   = [ 0.8, 0.9, 1.0, 1.1, 1.2 ]
    refResult = [  # reference results checked against QuantLib
    0.12572604970665557,
    0.07466919528750032,
    0.039969719617809936,
    0.019503405494683164,
    0.008804423283331253]
    for (strike,res) in zip(strikes,refResult)
        pv = couponBondOption(model,excTime,payTimes,cashFlows,strike,callOrPut)
        @test isapprox(pv,res,atol=1.0e-8, rtol=0.0)
        #println(pv)
    end
    return nothing
end


@testset "test_monteCarloSimulation" begin
    model = setup()
    times = Array(range(0.0, stop=10.0, length=11))
    nPaths = 2^10
    seed = 1234
    # risk-neutral simulation
    mcSim = McSimulation(model,times,nPaths,seed)
    discZeroBonds = [
        (1.0 / numeraire(model,times[i],mcSim.X[j,i,:]))
        for i = 1:size(times)[1], j = 1:nPaths ]    
    mcZeroBondsRiskNeutral = mean(discZeroBonds, dims=2)
    zeroBonds = [ discount(model.yieldCurve,t) for t in times ]
    # @printf("  T     ZeroRate    RiskNeutral\n")
    for k = 2:size(times)[1]
        t           = times[k]
        zeroRate    = -log(zeroBonds[k])/t
        riskNeutral = -log(mcZeroBondsRiskNeutral[k])/t
        # @printf(" %4.1f   %8.6lf    %8.6lf\n", t, zeroRate, riskNeutral)
        @test isapprox(zeroRate,riskNeutral,atol=1.0e-8, rtol=2.0e-2)
    end
    return nothing
end

@testset "test_liborPayoff" begin
    model = setup()
    # Libor Rate payoff
    liborTimes = Array(range(0.0, stop=10.0, length=11))
    dT0 = 2.0 / 365
    dT1 = 0.5
    cfs = [ Pay(LiborRate(t,t+dT0,t+dT0+dT1),t+dT0+dT1) for t in liborTimes]
    # times = set.union(*[ p.observationTimes() for p in cfs ])
    # times = np.array(sorted(list(times)))
    times = 
    [ 0., 0.50547945, 1., 1.50547945, 2., 2.50547945, 3., 3.50547945,
      4., 4.50547945, 5., 5.50547945, 6., 6.50547945, 7., 7.50547945,
      8., 8.50547945, 9., 9.50547945, 10., 10.50547945]
    times =  Array(range(0.0, stop=10.0, length=101))
    #
    nPaths = 2^13
    seed = 4321
    # risk-neutral simulation
    mcSim = McSimulation(model,times,nPaths,seed)
    cfPVs = [ discountedAt(payoff, path_) for payoff in cfs, path_ in paths(mcSim) ]
    cfPVs = mean(cfPVs, dims=2)
    discounts0 = [ discount(mcSim.model.yieldCurve,t+dT0) for t in liborTimes ]
    discounts1 = [ discount(mcSim.model.yieldCurve,t+dT0+dT1) for t in liborTimes ]
    liborsCv = (discounts0./discounts1 .- 1.0)./dT1
    liborsMC = cfPVs ./ discounts1
    println("  T     LiborRate  MnteCarlo")
    for (k,t) in enumerate(liborTimes)
        @printf(" %4.1f   %7.5f     %7.5f\n", t, liborsCv[k], liborsMC[k])
    end
end