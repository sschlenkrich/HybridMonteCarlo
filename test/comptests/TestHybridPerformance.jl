
using Printf, Test, Plots, LinearAlgebra, BenchmarkTools

include("../../hybmc/mathutils/Helpers.jl")

include("../../hybmc/termstructures/YieldCurve.jl")
include("../../hybmc/models/HullWhiteModel.jl")
include("../../hybmc/models/AssetModel.jl")
include("../../hybmc/models/DeterministicModel.jl")
include("../../hybmc/models/QuasiGaussianModel.jl")
include("../../hybmc/models/HybridModel.jl")

include("../../hybmc/simulations/McSimulation.jl")
include("../../hybmc/simulations/Payoffs.jl")


function HwModel(rate=0.01,vol=0.0050,mean=0.03)
    curve = YieldCurve(rate)
    times = [ 10.0 ]
    vols  = [ vol  ]
    return HullWhiteModel(curve, mean, times, vols)
end

function HybridHullWhiteModel4Ccy()
    domAlias = "EUR"
    domModel = HwModel(0.01, 0.0050, 0.03)
    forAliases = [ "USD", "GBP", "JPY" ]
    spotS0     = [   1.0,   2.0,   3.0 ]
    spotVol    = [   0.3,   0.3,   0.3 ]
    rates      = [  0.02,  0.03,  0.04 ]
    ratesVols  = [ 0.006, 0.007, 0.008 ]
    mean       = [  0.01,  0.01,  0.01 ]
    #
    forAssetModels = [
        AssetModel(S0, vol) for (S0, vol) in zip(spotS0,spotVol) ]
    forRatesModels = [
        HwModel(r,v,m) for (r,v,m) in zip(rates,ratesVols,mean) ]
    #
    nCorrs = 2 * size(forAliases)[1] + 1
    corr = Matrix{Float64}(I, nCorrs, nCorrs)
    corr[2,3] = 0.6  # USD quanto
    corr[3,2] = 0.6
    corr[4,5] = 0.8  # GBP quanto
    corr[5,4] = 0.8
    corr[6,7] = -0.7  # JPY quanto
    corr[7,6] = -0.7
    #
    return HybridModel(domAlias,domModel,forAliases,forAssetModels,forRatesModels,corr)
end

function HybridQuasiaussianModel3Ccy()
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

function HybridHwDcfModel2Ccy()
    curve0 = YieldCurve(0.03)
    hwModel = HwModel(0.01,0.0050,0.03)
    asModel = AssetModel(1.0,0.30)
    dcfModel0 = DcfModel(curve0,"USD")
    corr =  Matrix{Float64}(I, 2, 2)
    model = HybridModel("EUR",hwModel,["USD"],[asModel],[dcfModel0],corr)
    return model
end


function fwd(mcSim::McSimulation,p::Payoff)
    samples = discountedAt(p,mcSim)
    fwd = mean(samples) /
        discount(mcSim.model.domRatesModel.yieldCurve,obsTime(p))
    err = std(samples) / sqrt(size(samples)[1]) /
        discount(mcSim.model.domRatesModel.yieldCurve,obsTime(p))
    return (fwd, err)
end

function npv(mcSim::McSimulation,leg)
    samples = transpose(discountedAt(leg,mcSim))
    npv_ = mean(sum(samples, dims=2), dims=1)
    err = std(sum(samples, dims=2), dims=1)
    return (npv_,err)
end

yieldCurve(model) = model.yieldCurve

yieldCurve(model::DeterministicModel) = model.domCurve

yieldCurve(model::HybridModel) = yieldCurve(model.domRatesModel)

yieldCurve(sim::McSimulation) = yieldCurve(sim.model)

# We run the test

function run_test(model, modelName)
    times = append!(Array(range(0,stop=10.0,length=11)), [10.5])
    nPaths = 2^13
    seed = 141592653593
    @printf("\n")
    @printf("Run performance test for model %s: nPaths = %d  seed = %d\n", modelName, nPaths, seed)
    sim = McSimulation(model,times,2,seed,true)  #  warm-up
    @printf("Simulation:\n")
    @time sim = McSimulation(model,times,nPaths,seed,true)
    # domestic numeraire
    T = 10.0
    P = Pay(Fixed(1.0),T)
    (fw, err) = fwd(sim,P)
    @printf("1.0   @ %4.1lfy %8.6lf - mc_err = %8.6lf\n", T,fw,err)
    @time fwd(sim,P)
    # foreign asset
    alias = "USD"
    k = 1
    p = Asset(T,alias)
    xT = model.forAssetModels[k].X0 *
        discount(yieldCurve(model.forRatesModels[k]),T) /
        discount(yieldCurve(model.domRatesModel),T)
    (fw, err) = fwd(sim,p)
    @printf("%s   @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", alias,T,fw,xT,err)
    @time fwd(sim,p)
    # domestic Libor rate
    Tstart = 10.0
    Tend = 10.5
    L = Pay(LiborRate(T,Tstart,Tend,alias="EUR"),Tend)
    (fw, err) = fwd(sim,L)
    Lref = (discount(yieldCurve(sim),Tstart) /
            discount(yieldCurve(sim), Tend) - 1) /
            (Tend - Tstart) 
    @printf("L_EUR @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", T,fw,Lref,err)
    @time fwd(sim,L)
    # foreign Lbor rates
    alias = "USD"
    k = 1
    L = Pay(LiborRate(T,Tstart,Tend,alias=alias)*Asset(Tend,alias),Tend)
    (fw, err) = fwd(sim,L)
    fw *= discount(yieldCurve(sim.model.domRatesModel), Tend) /
          discount(yieldCurve(sim.model.forRatesModels[k]), Tend) /
          sim.model.forAssetModels[k].X0
    err *= discount(yieldCurve(sim.model.domRatesModel), Tend) /
           discount(yieldCurve(sim.model.forRatesModels[k]), Tend) /
           sim.model.forAssetModels[k].X0
    Lref = (discount(yieldCurve(sim.model.forRatesModels[k]),Tstart) /
            discount(yieldCurve(sim.model.forRatesModels[k]), Tend) - 1) /
            (Tend - Tstart) 
    @printf("L_%s @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", alias,T,fw,Lref,err)
    @time fwd(sim,L)
    # Libor leg
    alias = "EUR"
    schedule = Array(range(0,stop=T,length=21))
    Leg = [ (0.5 * LiborRate(Tstart,Tstart,Tend,alias="EUR")) % Tend
            for (Tstart,Tend) in zip(schedule[begin:end-1], schedule[begin+1:end]) ]
    (npv_,err) = npv(sim,Leg)
    npv_ref = 1.0 - discount(yieldCurve(sim), T)
    @printf("F_%s @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", alias,T,npv_[1,1],npv_ref,err[1,1])
    @time npv(sim,Leg)
end

# actually run the tests...

run_test(HybridHwDcfModel2Ccy(), "HybridHwDcfModel2Ccy")

run_test(HybridHullWhiteModel4Ccy(), "HybridHullWhiteModel4Ccy")

run_test(HybridQuasiaussianModel3Ccy(), "HybridQuasiaussianModel3Ccy")
