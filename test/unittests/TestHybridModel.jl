
using Printf, Test

include("../../hybmc/termstructures/YieldCurve.jl")
include("../../hybmc/models/HullWhiteModel.jl")
include("../../hybmc/models/AssetModel.jl")
include("../../hybmc/models/HybridModel.jl")

include("../../hybmc/simulations/McSimulation.jl")

using LinearAlgebra

function HwModel(rate=0.01,vol=0.0050,mean=0.03)
    curve = YieldCurve(rate)
    times = [ 10.0 ]
    vols  = [ vol  ]
    return HullWhiteModel(curve, mean, times, vols)
end

function setUp()
    domAlias = "EUR"
    domModel = HwModel(0.01, 0.0050, 0.03)
    forAliases = [ "USD", "GBP", "JPY" ]
    spotS0     = [   1.0,   1.0,   1.0 ]
    spotVol    = [   0.3,   0.3,   0.3 ]
    rates      = [  0.01,  0.01,  0.01 ]
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

@testset "test_modelSetup" begin
    #x = setUp()
    #println(x._size)
    #println(x._factors)
    #print(initialValues(x))
end

@testset "test_hybridSimulation" begin
    model = setUp()
    times = [0.0, 1.0]
    nPaths = 1
    seed = 1234
    # risk-neutral simulation
    cSim = McSimulation(model,times,nPaths,seed)
    #println(factors(model.domRatesModel))
end
