
include("../models/StochasticProcess.jl")

using Random, Distributions, Distributed

struct McSimulation{T<:AbstractFloat}
    model::StochasticProcess
    times::Array{T}
    nPaths::Int64
    seed::Int64
    timeInterpolation::Bool
    dW::Array{T}
    X::Array{T}
end

function McSimulation(model,times,nPaths,seed=123,timeInterpolation=true)
    Random.seed!(seed) # Setting the seed
    norm = Normal()
    dW = rand( norm, (nPaths,size(times)[1]-1,factors(model)) )
    dW = permutedims(dW, [3, 2, 1])  # we do not want to break unit tests
    X = zeros((stateSize(model), size(times)[1], nPaths))
    @views @inbounds Threads.@threads for i = 1:nPaths
        X[:,1,i] = initialValues(model)
        for j = 1:size(times)[1]-1
            evolve(model,times[j],X[:,j,i],times[j+1]-times[j],dW[:,j,i],X[:,j+1,i])
        end
    end
    return McSimulation(model,times,nPaths,seed,timeInterpolation,dW,X)
end

function McSimulationWithBrownians(model,times, dW, timeInterpolation=true)
    nPaths = size(dW)[3]  # dW = (factors,times,paths)
    seed = 0  #  this should better be nothing but we must provide Int64
    # X = zeros((nPaths,size(times)[1],stateSize(model)))
    X = zeros((stateSize(model), size(times)[1], nPaths))
    @views for i = 1:nPaths
        X[:,1,i] = initialValues(model)
        for j = 1:size(times)[1]-1
            evolve(model,times[j],X[:,j,i],times[j+1]-times[j],dW[:,j,i],X[:,j+1,i])
        end
    end
    return McSimulation(model,times,nPaths,seed,timeInterpolation,dW,X)
end

function state(self::McSimulation, idx, t)
    if idx>self.nPaths
        throw(ArgumentError("idx<=self.nPaths required."))
    end
    tIdx = searchsortedfirst(self.times,t)
    tIdx = min(tIdx,size(self.times)[1])  # use the last element
    if abs(self.times[tIdx]-t)<0.5/365  # we give some tolerance of half a day
        return @view self.X[:,tIdx,idx]
    end
    if !self.timeInterpolation
        throw(ArgumentError("timeInterpolation required for input time."))
    end
    if t >= self.times[tIdx] || tIdx==0  # extrapolation
        return self.X[:,tIdx,idx]
    end
    # linear interpolation
    rho = (self.times[tIdx] - t) / (self.times[tIdx] - self.times[tIdx-1])
    return rho * self.X[:,tIdx-1,idx] + (1.0-rho) * self.X[:,tIdx,idx]  # better use view?
end

struct Path
    simulation::McSimulation
    idx::Int64
end

function paths(self::McSimulation)
    return [ Path(self,k) for k=1:self.nPaths ]
end

# stochastic process interface for payoffs

# the numeraire in the domestic currency used for discounting future payoffs
function numeraire(self::Path, t)
    # we may add numeraire adjuster here...
    return numeraire(self.simulation.model, t, state(self.simulation,self.idx,t))
end
    
# a domestic/foreign currency zero coupon bond
function zeroBond(self::Path, t, T, alias)
        # we may add zcb adjuster here
    return zeroBond(self.simulation.model, t, T, state(self.simulation,self.idx,t), alias)
end

# an asset price for a given currency alias
function asset(self::Path, t, alias)
    # we may add asset adjuster here
    return asset(self.simulation.model, t, state(self.simulation,self.idx,t), alias)
end

# Cumulated intensity; probability of tau > t, conditional on F_t
function hazardProcess(self::Path, t, alias)
    return hazardProcess(self.simulation.model, t, self.simulation.state(self.idx,t), alias)
end

# instantanous probability of default
function hazardRate(self::Path, t, alias)
    return hazardRate(self.simulation.model, t, state(self.simulation,self.idx,t), alias)
end

# probavility of survival consitional on information at t
function survivalProb(self::Path, t, T, alias)
    return survivalProb(self.simulation.model, t, T, state(self.simulation,self.idx,t), alias)
end
