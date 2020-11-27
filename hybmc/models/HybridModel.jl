
include("../models/StochasticProcess.jl")
include("../models/AssetModel.jl")

using LinearAlgebra, Interpolations

struct HybridModel{T<:AbstractFloat,D<:StochasticProcess,A<:AssetModel,F<:StochasticProcess} <: StochasticProcess
    domAlias::String                          #  name of our domestic (numeraire) currency
    domRatesModel::D                          #  domestic rates model specifies numeraire
    forAliases::Array{String}                 #  list of foreign currencies (all relative to dom currency)
    forAssetModels::Array{A}                  #  list of foreign asset models
    forRatesModels::Array{F}                  #  list of foreign rates models
    correlations::Array{T,2}                  #  2d array of instantanous correlations
    # additional data
    index::Dict{String,Int64}                 # a dictionary from foreign alias to index
    _size::Int64                              # sum of model states
    _factors::Int64                           # sum of model _factors
    modelsStartIdx::Array{Int64}              # were do foreign models start in X
    L::LowerTriangular{Float64,Array{Float64,2}}   # L L^T = correlations
    hybAdjTimes::Array{T}                    # array of reference times for adjuster
    hybVolAdj::Array{T}                      # array of the additive volatility adjuster
end

# constructor

function HybridModel(domAlias, domRatesModel,
    forAliases, forAssetModels, forRatesModels, correlations)
    # add sanity checks here
    # we need to know the model index for a given alias
    index = Dict([ (forAliases[k],k) for k = 1:size(forAliases)[1] ])
    # manage model indices in state variable
    _size = stateSize(domRatesModel)
    _factors = factors(domRatesModel)
    modelsStartIdx = Vector{Int64}()
    lastModelIdx = stateSize(domRatesModel)
    for (assetModel, ratesModel) in zip(forAssetModels,forRatesModels)
        _size += (stateSize(assetModel) + stateSize(ratesModel))
        _factors += (factors(assetModel) + factors(ratesModel))
        push!(modelsStartIdx,lastModelIdx)
        push!(modelsStartIdx,lastModelIdx + stateSize(assetModel))
        lastModelIdx += (stateSize(assetModel) + stateSize(ratesModel))
    end
    if  !isnothing(correlations)
        L = cholesky(correlations).L
    else
        L = nothing
    end
    hybAdjTimes = zeros(0)
    hybVolAdj = zeros(0)
    return HybridModel(domAlias,domRatesModel,forAliases,forAssetModels,forRatesModels,
        correlations,index,_size,_factors,modelsStartIdx,L,hybAdjTimes,hybVolAdj)
end

HybridModel(model::HybridModel,hybAdjTimes,hybVolAdj) =
    HybridModel(model.domAlias, model.domRatesModel, model.forAliases, model.forAssetModels,
        model.forRatesModels, model.correlations, model.index, model._size, model._factors,
        model.modelsStartIdx, model.L, hybAdjTimes, hybVolAdj)

# stochastic process interface
    
function stateSize(self::HybridModel)   # dimension of X(t)
    return self._size
end

function factors(self::HybridModel)   # dimension of W(t)
    return self._factors
end

function initialValues(self::HybridModel)
    X0 = initialValues(self.domRatesModel)
    for (assetModel, ratesModel) in zip(self.forAssetModels,self.forRatesModels)
        X0 = append!(X0, initialValues(assetModel))
        X0 = append!(X0, initialValues(ratesModel))
    end
    return X0
end


# evolve X(t0) -> X(t0+dt) using independent Brownian increments dW
# t0, dt are assumed float, X0, X1, dW are np.array
@views function evolve(self::HybridModel, t0, X0, dt, dW, X1)
    if !isnothing(self.L)
        dZ = self.L * dW
    else
        dZ = dW
    end
    if isnothing(self.forAliases)  # take a short cut
        evolve(self.domRatesModel,t0, X0, dt, dZ, X1)
        return nothing
    end
    # evolve domestic rates
    domSize = stateSize(self.domRatesModel)  # shorten expressions
    evolve(self.domRatesModel, t0, X0[begin:domSize], dt, dZ[begin:factors(self.domRatesModel)], X1[begin:domSize])
    # we need the domestic drift r_d
    r_d = shortRateOverPeriod(self.domRatesModel, t0, dt, X0[begin:domSize], X1[begin:domSize])
    # now we iterate over foreign models
    corrStartIdx = factors(self.domRatesModel)  #  we need to keep track of our sub-correlations
    for (k, alias) in enumerate(self.forAliases)
        # carefully collect Brownian increments
        startIdx = corrStartIdx + 1
        endIdx   = startIdx + factors(self.forAssetModels[k]) - 1
        dw_asset = dZ[ startIdx : endIdx ]
        #
        startIdx = endIdx + 1
        endIdx   = startIdx + factors(self.forRatesModels[k]) - 1
        dw_rates = dZ[ startIdx : endIdx ]
        # we need the starting point states for evolution, y0 (asset), x0 (rates)
        startIdx = self.modelsStartIdx[2 * k - 1] + 1
        endIdx   = startIdx + stateSize(self.forAssetModels[k]) - 1
        y0 = X0[ startIdx : endIdx ]
        y1 = X1[ startIdx : endIdx ]
        #
        startIdx = endIdx + 1
        endIdx   = startIdx + stateSize(self.forRatesModels[k]) - 1
        x0 = X0[ startIdx : endIdx ]
        x1 = X1[ startIdx : endIdx ]
        #
        # Quanto adjustment
        # we use the model-independent implementation to allow for credit hybrid components
        # todo: capture case self.correlations = None
        colIdx   = corrStartIdx + 1
        startIdx = colIdx   + factors(self.forAssetModels[k])
        endIdx   = startIdx + factors(self.forRatesModels[k]) - 1
        qAdj = self.correlations[colIdx, startIdx : endIdx]
        # we want to modify qAdj vector w/o changing the correlation
        qAdj_ = Array(qAdj)
        # we need to extend the input state for our asset mode to account for drift and adjuster
        y0_ = append!(Array(y0), [0.0, 0.0])
        y0_[end] = hybridVolAdjuster(self, k, t0)  # implement adjuster!
        assetVol = volatility(self.forAssetModels[k], t0, y0_)
        qAdj_ *= (assetVol*sqrt(dt))
        dw_rates_ = dw_rates - qAdj_  #  create a new vector
        # evolve foreign rates
        evolve(self.forRatesModels[k], t0, x0, dt, dw_rates_, x1)
        # calculate FX drift, volAdjuster and extend input state
        r_f = shortRateOverPeriod(self.forRatesModels[k], t0, dt, x0, x1)
        y0_[end-1] = r_d - r_f   # FX drift
        # finally we can evolve FX
        evolve(self.forAssetModels[k], t0, y0_, dt, dw_asset, y1)
        # no need to copy results coz we work on views of X1
        # but we need to update stochastic factor index
        corrStartIdx += (factors(self.forAssetModels[k]) + factors(self.forRatesModels[k]))
    end
    return nothing
end


# interface for payoff calculation

@views function numeraire(self::HybridModel, t, X)
    return numeraire(self.domRatesModel,t,X[begin:stateSize(self.domRatesModel)])
end
    
@views function asset(self::HybridModel, t, X, alias)
    if isnothing(alias)
        return 1.0
    end
    if alias==self.domAlias
        return 1.0
    end
    k = self.index[alias]  # this should throw an exception if alias is unknown
    startIdx = self.modelsStartIdx[2 * k - 1] + 1
    endIdx   = startIdx + stateSize(self.forAssetModels[k]) - 1
    y = X[ startIdx : endIdx ]
    return asset(self.forAssetModels[k], t, y, alias)
end

@views function zeroBond(self::HybridModel, t, T, X, alias)
    if isnothing(alias) || alias==self.domAlias
        x = X[begin:stateSize(self.domRatesModel)]
        return zeroBond(self.domRatesModel, t, T, x, alias)
    end
    k = self.index[alias]  # this should throw an exception if alias is unknown
    startIdx = self.modelsStartIdx[2 * k] + 1
    endIdx   = startIdx + stateSize(self.forRatesModels[k]) - 1
    x = X[ startIdx : endIdx ]
    return zeroBond(self.forRatesModels[k], t, T, x, alias)
end

# adjuster methodology for stochastic rates and FX volatility

function hybridVolAdjuster(self::HybridModel, forIdx, t)
    if size(self.hybAdjTimes)==(0,)
        return 0.0  # default
    end
    # linear interpolation with constant extrapolation
    # maybe better use scipy interpolation with linear extraplation
    interp = LinearInterpolation(self.hybAdjTimes,@view self.hybVolAdj[forIdx,:])
    return interp(t)
end


function hybridVolAdjuster(self::HybridModel, hybAdjTimes::Array)
    if !(size(hybAdjTimes)[1]>1)
        throw(ArgumentError("size(hybAdjTimes)[1]>1 required."))
    end
    if !(hybAdjTimes[1]==0.0)
        throw(ArgumentError("hybAdjTimes[0]==0.0 required."))
    end
    for k = 2:size(hybAdjTimes)[1]
        if !(hybAdjTimes[k]>hybAdjTimes[k-1])
            throw(ArgumentError("hybAdjTimes[k]>hybAdjTimes[k-1] required."))
        end
    end
    # initialise 
    localVol = zeros((size(self.forAliases)[1],size(hybAdjTimes)[1]))
    hybrdVol = ones((size(self.forAliases)[1],size(hybAdjTimes)[1])) #  1.0 is required for p calculation
    hybVolAdj = zeros((size(self.forAliases)[1],size(hybAdjTimes)[1]))
    S0 = [ asset(m, 0.0, initialValues(m), nothing) for m in self.forAssetModels ]
    #
    # calculate vols at zero
    for i = 1:size(S0)[1]
        y0 = append!(initialValues(self.forAssetModels[i]), [0.0, 0.0])
        localVol[i,1] = volatility(self.forAssetModels[i], 0.0, y0)
        hybrdVol[i,1] = localVol[i,1]
    end
    # now we start with the actual methodology...
    corrStartIdx = factors(self.domRatesModel)        
    for i = 1:size(S0)[1]
        # we collect all relevant correlations
        # recall, 
        #   Y0 is domestic rates model
        #   X1 is asset (or FX) model
        #   Y1 is forign rates model
        #
        colIdx   = corrStartIdx + 1
        # domestic rates vs FX, ASSUME vol-FX correlation is zero
        rhoY0X1 = self.correlations[ begin:factors(self.domRatesModel), colIdx]   #  
        # foreign rates vs FX, ASSUME vol-FX correlation is zero
        startIdx = colIdx   + factors(self.forAssetModels[i])
        endIdx   = startIdx + factors(self.forRatesModels[i]) - 1
        # rhoX1Y1 = self.correlations[colIdx, startIdx : endIdx]
        rhoY1X1 = self.correlations[startIdx : endIdx, colIdx]  # use col-vector to avoid transpose
        # rates vs rates, ASSUME all vol-... correlation are zero
        rhoY0Y1 = self.correlations[ begin:factors(self.domRatesModel), startIdx : endIdx ]
        # update stochastic factor index
        corrStartIdx += (factors(self.forAssetModels[i]) + factors(self.forRatesModels[i]))
        # bootstrap over adjuster times
        for (k, T) in [ e for e in enumerate(hybAdjTimes) ][2:end]
            # ATM forward and effective local volatility
            dfDom = zeroBond(self.domRatesModel, 0.0, T, initialValues(self.domRatesModel), nothing)
            dfFor = zeroBond(self.forRatesModels[i], 0.0, T, initialValues(self.forRatesModels[i]), nothing)
            S = S0[i] * dfFor / dfDom  # maybe it's worth to save S for debugging
            y = zeros(stateSize(self.forAssetModels[i]) + 2)  # this is asset model-dependent
            y[1] = log(S / S0[i])
            localVol[i,k] = volatility(self.forAssetModels[i],hybAdjTimes[k], y)
            # calculate derivative of hybrid variance
            hPrime = zeros(k)
            for (j, t) in enumerate(hybAdjTimes[begin:k])
                sigmaP0      = zeroBondVolatility(self.domRatesModel, t, T)
                sigmaP0Prime = zeroBondVolatilityPrime(self.domRatesModel, t, T)
                sigmaP1      = zeroBondVolatility(self.forRatesModels[i], t, T)
                sigmaP1Prime = zeroBondVolatilityPrime(self.forRatesModels[i], t, T)
                #
                sigma0 = sigmaP0 - rhoY0Y1*sigmaP1 + hybrdVol[i,j]*rhoY0X1  # bootstrapping enters here
                sum0 = transpose(sigmaP0Prime) * sigma0
                #
                sigma1 = sigmaP1 - transpose(rhoY0Y1)*sigmaP0 - hybrdVol[i,j]*rhoY1X1  # bootstrapping enters here
                sum1 = transpose(sigma1) * sigmaP1Prime
                # collect terms and finish
                hPrime[j] = 2.0*(sum0 + sum1)  # this will likely fail for deterministic models
            end
            p = 0.5 * hPrime[k] * (hybAdjTimes[k] - hybAdjTimes[k - 1])
            q = 0.5 * hPrime[k-1] * (hybAdjTimes[k] - hybAdjTimes[k - 1])
            for j = 2:k-1
                q += 0.5 * (hPrime[j - 1] + hPrime[j]) * (hybAdjTimes[j] - hybAdjTimes[j - 1])
            end
            # let's see if this works...
            root2 = p*p / 4.0 - q + localVol[i,k] * localVol[i,k]
            if !(root2>=0.0)
                throw(ArgumentError("root2>=0.0 required."))
            end
            hybrdVol[i,k] = -p / 2.0 + sqrt(root2)
            if !(hybrdVol[i,k]>0.0)
                throw(ArgumentError("hybrdVol[i,k]>0.0 required."))
            end
            # maybe we should add some more safety checks here...
            hybVolAdj[i,k] = hybrdVol[i,k] - localVol[i,k]
        end
    end
    #
    return hybVolAdj
end
