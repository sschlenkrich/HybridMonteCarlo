
include("../models/StochasticProcess.jl")

struct DeterministicModel <: StochasticProcess
    domAlias
    domCurve
    forAliases   
    forAssetSpots
    forCurves
    index
end

function DeterministicModel(domAlias,domCurve,forAliases,forAssetSpots,forCurves)
    index = Dict([ (forAliases[k],k) for k = 1:size(forAliases)[1] ])
    return DeterministicModel(domAlias,domCurve,forAliases,forAssetSpots,forCurves,index)
end

stateSize(self::DeterministicModel) = 0

factors(self::DeterministicModel) = 0

initialValues(self::DeterministicModel) = zeros(0)

evolve(self::DeterministicModel, t0, X0, dt, dW, X1) = nothing

function shortRateOverPeriod(self::DeterministicModel, t0, dt, X0, X1)
    B_d = discount(self.domCurve,t0) / discount(self.domCurve,t0 + dt)
    return log(B_d) / dt
end

zeroBondVolatility(self::DeterministicModel, t, T) = zeros(0)

zeroBondVolatilityPrime(self::DeterministicModel, t, T) = zeros(0)

numeraire(self::DeterministicModel, t, X) = 1.0 / discount(self.domCurve,t)

function asset(self::DeterministicModel, t, X, alias)
    k = self.index[alias]  # this should throw an exception if alias is unknown
    return self.forAssetSpots[k] * discount(self.forCurves[k],t) / discount(self.domCurve,t)
end

function zeroBond(self::DeterministicModel, t, T, X, alias)
    if isnothing(alias) || alias==self.domAlias
        return discount(self.domCurve,T) / discount(self.domCurve,t)
    end
    k = self.index[alias]  # this should throw an exception if alias is unknown
    return discount(self.forCurves[k],T) / discount(self.forCurves[k],t)
end

DcfModel(curve) = DeterministicModel(nothing,curve,nothing,nothing,nothing)

DcfModel(curve, alias) = DeterministicModel(alias,curve,nothing,nothing,nothing)

