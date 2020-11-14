
using Printf, Test, Plots

include("../../hybmc/mathutils/Helpers.jl")

include("../../hybmc/termstructures/YieldCurve.jl")
include("../../hybmc/models/HullWhiteModel.jl")
include("../../hybmc/models/AssetModel.jl")
include("../../hybmc/models/DeterministicModel.jl")
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

function setUp()
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

@testset "test_ModelSetup" begin
    model = setUp()
    @test stateSize(model) == 11
    @test factors(model)  == 7
    @test size(initialValues(model))[1] == stateSize(model)
    @test model.index == Dict([ ("USD",1), ("GBP",2), ("JPY",3) ])
    @test model.modelsStartIdx == [2, 3, 5, 6, 8, 9]
    C = model.L * transpose(model.L)
    @test norm(C-model.correlations) == 0.0
end

@testset "test_HybridModel" begin
    model = setUp()
    x = initialValues(model)
    @test asset(model,0.0,x,"USD") == model.forAssetModels[1].X0
    @test asset(model,0.0,x,"GBP") == model.forAssetModels[2].X0
    @test asset(model,0.0,x,"JPY") == model.forAssetModels[3].X0
    #
    @test zeroBond(model,0.0,5.0,x,"EUR") == discount(model.domRatesModel.yieldCurve,5.0)
    @test zeroBond(model,0.0,5.0,x,"USD") == discount(model.forRatesModels[1].yieldCurve,5.0)
    @test zeroBond(model,0.0,5.0,x,"GBP") == discount(model.forRatesModels[2].yieldCurve,5.0)
    @test zeroBond(model,0.0,5.0,x,"JPY") == discount(model.forRatesModels[3].yieldCurve,5.0)
end

@testset "test_HybridModelEvolution" begin
    model = setUp()
    times = [0.0]
    nPaths = 1
    seed = 314159265359
    # risk-neutral simulation
    mcSim = McSimulation(model,times,nPaths,seed,false)
    p = Path(mcSim,1)
    #
    @test asset(p,0.0,"USD") == model.forAssetModels[1].X0
    @test asset(p,0.0,"GBP") == model.forAssetModels[2].X0
    @test asset(p,0.0,"JPY") == model.forAssetModels[3].X0
    #
    @test zeroBond(p,0.0,5.0,"EUR") == discount(model.domRatesModel.yieldCurve,5.0)
    @test zeroBond(p,0.0,5.0,"USD") == discount(model.forRatesModels[1].yieldCurve,5.0)
    @test zeroBond(p,0.0,5.0,"GBP") == discount(model.forRatesModels[2].yieldCurve,5.0)
    @test zeroBond(p,0.0,5.0,"JPY") == discount(model.forRatesModels[3].yieldCurve,5.0)
end

@testset "test_HybridModelWithDeterministicRates" begin
    curve0 = YieldCurve(0.03)
    dcfModel0 = DcfModel(curve0,"EUR")
    #
    times = [0.0, 1.0]
    nPaths = 1
    seed = 314159265359
    # hybrid adjuster
    hybAdjTimes = [0.0, 1.0, 2.0]
    # simulate deterministic model only
    mcSim = McSimulation(dcfModel0,times,nPaths,seed,false)
    @test zeroBond(Path(mcSim,1),1.0,10.0,nothing) == discount(curve0,10.0)/discount(curve0,1.0)
    # simulate deterministic domestic model
    hwModel = HwModel(0.01,0.0050,0.03)
    asModel = AssetModel(1.0,0.30)
    corr =  Matrix{Float64}(I, 2, 2)
    model = HybridModel("EUR",dcfModel0,["USD"],[asModel],[hwModel],corr)
    hybVolAdj = hybridVolAdjuster(model,hybAdjTimes)
    model = HybridModel(model,hybAdjTimes,hybVolAdj)
    mcSim = McSimulation(model,times,nPaths,seed,false)
    p = Path(mcSim,1)
    @test asset(p,0.0,"USD") == model.forAssetModels[1].X0
    @test zeroBond(p,0.0,5.0,"EUR") == discount(curve0,5.0)
    @test zeroBond(p,0.0,5.0,"USD") == discount(model.forRatesModels[1].yieldCurve,5.0)
    # simulate deterministic foreign model
    dcfModel0 = DcfModel(curve0,"USD")
    model = HybridModel("EUR",hwModel,["USD"],[asModel],[dcfModel0],corr)
    hybVolAdj = hybridVolAdjuster(model,hybAdjTimes)
    model = HybridModel(model,hybAdjTimes,hybVolAdj)
    mcSim = McSimulation(model,times,nPaths,seed,false)
    p = Path(mcSim,1)
    @test asset(p,0.0,"USD") == model.forAssetModels[1].X0
    @test zeroBond(p,0.0,5.0,"EUR") == discount(model.domRatesModel.yieldCurve,5.0)
    @test zeroBond(p,0.0,5.0,"USD") == discount(curve0,5.0)
        # simulate deterministic domestic and foreign curve
    curve1 = YieldCurve(0.05)
    dcfModel0 = DcfModel(curve0,"EUR")
    dcfModel1 = DcfModel(curve1,"USD")
    corr =  Matrix{Float64}(I, 1, 1)
    model = HybridModel("EUR",dcfModel0,["USD"],[asModel],[dcfModel1],corr)
    hybVolAdj = hybridVolAdjuster(model,hybAdjTimes)
    model = HybridModel(model,hybAdjTimes,hybVolAdj)
    mcSim = McSimulation(model,times,nPaths,seed,false)
    p = Path(mcSim,1)
    @test asset(p,0.0,"USD") == model.forAssetModels[1].X0
    @test zeroBond(p,0.0,5.0,"EUR") == discount(curve0,5.0)
    @test zeroBond(p,0.0,5.0,"USD") == discount(curve1,5.0)
end


function fwd(mcSim::McSimulation,p::Payoff)
    samples = [ discountedAt(p,path_) for path_ in paths(mcSim) ]
    fwd = mean(samples) /
        discount(mcSim.model.domRatesModel.yieldCurve,obsTime(p))
    err = std(samples) / sqrt(size(samples)[1]) /
        discount(mcSim.model.domRatesModel.yieldCurve,obsTime(p))
    return (fwd, err)
end

@testset "test_hybridSimulation" begin
    model = setUp()
    times = append!(Array(range(0,stop=10.0,length=10)), [10.5])
    nPaths = 2^13
    seed = 141592653593
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
    # we set up a hybrid model consistent to QuantLib
    domAlias = "EUR"
    domModel = HwModel(0.01, 0.0050, 0.01)
    forAliases = [ "USD", "GBP" ]
    forAssetModels = [
        AssetModel(1.0, 0.30),
        AssetModel(2.0, 0.15) ]
    forRatesModels = [
        HwModel(0.02, 0.0060, 0.02), 
        HwModel(0.02, 0.0070, 0.03) ]
    nCorrs = 2 * size(forAliases)[1] + 1
    corr = Matrix{Float64}(I, nCorrs, nCorrs)
    # [ EUR, USD-EUR, USD, GBP-EUR, GBP ] - +1 for Julia indeces
    #   0    1        2    3        4
    # USD-EUR - EUR
    corr[0+1,1+1] = 0.5
    corr[1+1,0+1] = 0.5
    # USD-EUR - USD
    corr[1+1,2+1] = -0.5
    corr[2+1,1+1] = -0.5
    # EUR - USD
    corr[0+1,2+1] = -0.5
    corr[2+1,0+1] = -0.5
    # GBP-EUR - EUR
    corr[0+1,3+1] = -0.5
    corr[3+1,0+1] = -0.5
    # GBP-EUR - GBP
    corr[3+1,4+1] = 0.5
    corr[4+1,3+1] = 0.5
    # EUR - GBP
    corr[0+1,4+1] = -0.8
    corr[4+1,0+1] = -0.8
    # USD - GBP
    corr[2+1,4+1] = 0.0
    corr[4+1,2+1] = 0.0
    model = HybridModel(domAlias,domModel,forAliases,forAssetModels,forRatesModels,corr)
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
        @test abs(vol - volRef) < 3.0*err
        P = Pay(Max(K-Asset(T,alias),Z),T)
        (fw, err) = fwd(mcSim,P)
        vol = BlackImpliedVol(fw,xT,xT,T,-1.0)
        vega = BlackVega(xT,xT,vol,T)
        err /= vega
        volRef = model.forAssetModels[k].sigma
        @printf("P_%s @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf\n", alias,T,vol,volRef,err)
        @test abs(vol - volRef) < 3.0*err
    end
end

