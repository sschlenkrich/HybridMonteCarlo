
using Printf, Test, Plots, LinearAlgebra

include("../../hybmc/mathutils/Helpers.jl")

include("../../hybmc/termstructures/YieldCurve.jl")
include("../../hybmc/models/HullWhiteModel.jl")
include("../../hybmc/models/AssetModel.jl")
include("../../hybmc/models/DeterministicModel.jl")
include("../../hybmc/models/QuasiGaussianModel.jl")
include("../../hybmc/models/HybridModel.jl")

include("../../hybmc/simulations/McSimulation.jl")
include("../../hybmc/simulations/Payoffs.jl")


using LinearAlgebra

function HwModel(rate=0.01,vol=0.0050,mean=0.03)
    curve = YieldCurve(rate)
    times = [ 10.0 ]
    vols  = [ vol  ]
    return HullWhiteModel(curve, mean, times, vols)
end

function fwd(mcSim::McSimulation,p::Payoff)
    samples = discountedAt(p,mcSim)
    fwd = mean(samples) /
        discount(mcSim.model.domRatesModel.yieldCurve,obsTime(p))
    err = std(samples) / sqrt(size(samples)[1]) /
        discount(mcSim.model.domRatesModel.yieldCurve,obsTime(p))
    return (fwd, err)
end

function setUp()
    ### full smile/skew model
    # domestic rates
    domAlias = "EUR"
    eurCurve = YieldCurve(0.03)
    d     = 2
    times = [  10.0     ]
    sigma = [ 0.0060 ;
              0.0040 ]
    slope = [ 0.10  ;
              0.15  ]
    curve = [ 0.05  ;
              0.10  ]
    delta = [ 1.0, 10.0  ]
    chi   = [ 0.01, 0.15 ]
    Gamma = [ 1.0   0.6  ;
              0.6   1.0  ]
    eurRatesModel = QuasiGaussianModel(eurCurve,d,times,sigma,slope,curve,delta,chi,Gamma)
    # 
    forAliases = [ "USD", "GBP" ]
    spotS0     = [   1.0,   2.0 ]
    spotVol    = [   0.3,   0.2 ]
    forAssetModels = [
        AssetModel(S0, vol) for (S0, vol) in zip(spotS0,spotVol) ]
    # USD rates
    usdCurve = YieldCurve(0.02)
    d     = 3
    times = [  10.0   ]
    sigma = [  0.0060 ;
               0.0050 ;
               0.0040 ]
    slope = [  0.10   ;
               0.20   ;
               0.30   ]
    curve = [  0.05   ;
               0.10   ;
               0.20   ]
    delta = [ 1.0, 5.0, 20.0 ]
    chi   = [ 0.01, 0.05, 0.15 ]
    Gamma = [ 1.0  0.8  0.6 ;
              0.8  1.0  0.8 ;
              0.6  0.8  1.0 ]
    usdRatesModel = QuasiGaussianModel(usdCurve,d,times,sigma,slope,curve,delta,chi,Gamma)
    #
    gbpRatesModel = HwModel()
    #
           # 'EUR_x_0', 'EUR_x_1', 'USD_logS', 'USD_x_0', 'USD_x_1', 'USD_x_2', 'GBP_logS', 'GBP_x'
    corr = [  1.0        0.0         0.5       -0.5        0.0        0.0        -0.5       -0.5   ;   # EUR_x_0
              0.0        1.0         0.0        0.0       -0.5        0.0        -0.5        0.0   ;   # EUR_x_1
              0.5        0.0         1.0       -0.5       -0.5       -0.5         0.0        0.0   ;   # USD_logS
             -0.5        0.0        -0.5        1.0        0.0        0.0         0.0        0.0   ;   # USD_x_0
              0.0       -0.5        -0.5        0.0        1.0        0.0         0.0        0.0   ;   # USD_x_1
              0.0        0.0        -0.5        0.0        0.0        1.0         0.0        0.0   ;   # USD_x_2
             -0.5       -0.5         0.0        0.0        0.0        0.0         1.0        0.5   ;   # GBP_logS
             -0.5        0.0         0.0        0.0        0.0        0.0         0.5        1.0   ]   # GBP_x
    model = HybridModel(domAlias,eurRatesModel,forAliases,forAssetModels,[usdRatesModel,gbpRatesModel],corr)
    return model
end

@testset "test_ModelSetup" begin
    model = setUp()
    @test stateSize(model) == 24
    @test factors(model) == 8
end

@testset "test_HybridSimulation" begin
    model = setUp()
    times = append!(Array(range(0,stop=10.0,length=11)), [10.5])
    nPaths = 2^13
    seed = 314159265359
    # risk-neutral simulation
    mcSim = McSimulation(model,times,nPaths,seed,false)
    # 
    T = 10.0
    P = Pay(Fixed(1.0),T)
    (fw, err) = fwd(mcSim,P)
    # domestic numeraire
    @printf("1.0   @ %4.1lfy %8.6lf - mc_err = %8.6lf\n", T,fw,err)
    @test abs(fw - 1.0) < 1.5*err
    # foreign assets
    for (k, alias) in enumerate(model.forAliases)
        p = Asset(T,alias)
        xT = model.forAssetModels[k].X0 *
            discount(model.forRatesModels[k].yieldCurve,T) /
            discount(model.domRatesModel.yieldCurve,T)
        (fw, err) = fwd(mcSim,p)
        @printf("%s   @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", alias,T,fw,xT,err)
        @test abs(fw - xT) < 2.0*err
    end
    # domestic Libor rate
    Tstart = 10.0
    Tend = 10.5
    L = Pay(LiborRate(T,Tstart,Tend,alias="EUR"),Tend)
    (fw, err) = fwd(mcSim,L)
    Lref = (discount(mcSim.model.domRatesModel.yieldCurve,Tstart) /
            discount(mcSim.model.domRatesModel.yieldCurve, Tend) - 1) /
            (Tend - Tstart) 
    @printf("L_EUR @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", T,fw,Lref,err)
    @test abs(fw - Lref) < 2.0*err
    # foreign Lbor rates
    for (k, alias) in enumerate(model.forAliases)
        L = Pay(LiborRate(T,Tstart,Tend,alias=alias)*Asset(Tend,alias),Tend)
        (fw, err) = fwd(mcSim,L)
        fw *= discount(mcSim.model.domRatesModel.yieldCurve, Tend) /
              discount(mcSim.model.forRatesModels[k].yieldCurve, Tend) /
              mcSim.model.forAssetModels[k].X0
        err *= discount(mcSim.model.domRatesModel.yieldCurve, Tend) /
               discount(mcSim.model.forRatesModels[k].yieldCurve, Tend) /
               mcSim.model.forAssetModels[k].X0
        Lref = (discount(mcSim.model.forRatesModels[k].yieldCurve,Tstart) /
                discount(mcSim.model.forRatesModels[k].yieldCurve, Tend) - 1) /
                (Tend - Tstart) 
        @printf("L_%s @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", alias,T,fw,Lref,err)
        @test abs(fw - Lref) < 2.0*err
    end
end

@testset "test_HybridVolAdjusterCalculation" begin
    model = setUp()
    hybAdjTimes = Array(range(0.0, stop=20.0, length=21))
    hybVolAdj = hybridVolAdjuster(model,hybAdjTimes)
    model = HybridModel(model,hybAdjTimes,hybVolAdj)
    # plot(hybAdjTimes, transpose(hybVolAdj))
    # savefig("hybVolAdj.png")
    times = Array(range(0.0, stop=10.0, length=11))
    nPaths = 2^13
    seed = 314159265359
    # risk-neutral simulation
    mcSim = McSimulation(model,times,nPaths,seed,false)
    # 
    T = 10.0
    for (k, alias) in enumerate(model.forAliases)
        # ATM forward
        xT = model.forAssetModels[k].X0 *
             discount(model.forRatesModels[k].yieldCurve, T) /
             discount(model.domRatesModel.yieldCurve, T)
        K = Fixed(xT)
        Z = Fixed(0.0)
        C = Pay(Max(Asset(T,alias)-K,Z),T)
        (fw, err) = fwd(mcSim,C)
        vol = BlackImpliedVol(fw,xT,xT,T,1.0)
        vega = BlackVega(xT,xT,vol,T)
        err /= vega
        volRef = model.forAssetModels[k].sigma
        @printf("C_%s @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", alias,T,vol,volRef,err)
        @test abs(vol - volRef) < 5.0*err
        P = Pay(Max(K-Asset(T,alias),Z),T)
        (fw, err) = fwd(mcSim,P)
        vol = BlackImpliedVol(fw,xT,xT,T,-1.0)
        vega = BlackVega(xT,xT,vol,T)
        err /= vega
        volRef = model.forAssetModels[k].sigma
        @printf("P_%s @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", alias,T,vol,volRef,err)
        @test abs(vol - volRef) < 5.0*err
    end
end

