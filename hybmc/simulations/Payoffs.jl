
include("../simulations/McSimulation.jl")

abstract type Payoff end
abstract type Leaf <: Payoff end
abstract type BinaryNode <: Payoff end

function at(self::Payoff, p::Path)
    throw(ArgumentError("Implementation of method at() required."))
end

function obsTime(self::Payoff)
    throw(ArgumentError("Implementation of method obsTime() required."))
end

function discountedAt(self::Payoff, p::Path)
    return at(self,p) / numeraire(p,obsTime(self))
end

function discountedAt(self::Payoff, paths::Array{Path,1})
    res = zeros(size(paths,1))
    for j = 1:size(res,1)
        res[j] = at(self,paths[j]) / numeraire(paths[j],obsTime(self))
    end
    return res
end

discountedAt(self::Payoff, sim::McSimulation) = discountedAt(self, paths(sim))

function discountedAt(payoffs::Array, paths::Array{Path,1})
    res = zeros( (size(payoffs,1), size(paths,1)) )
    for j = 1:size(paths,1)
        for i = 1:size(payoffs,1)
            res[i,j] = at(payoffs[i],paths[j]) / numeraire(paths[j],obsTime(payoffs[i]))
        end
    end
    return res
end

discountedAt(payoffs::Array{}, sim::McSimulation) = discountedAt(payoffs, paths(sim))

function observationTimes(self::Payoff)
    throw(ArgumentError("Implementation of method observationTimes() required."))
end

function observationTimes(self::Leaf)
    return Set([ obsTime(self) ])
end

function observationTimes(self::BinaryNode)
    return union(observationTimes(self.x),observationTimes(self.y))
end

function observationTimes(list::Array)
    times = Set{Float64}()
    for item in list
        times = union(times,observationTimes(item))
    end
    return sort([ t for t in times ])
end

# simplify payoff scripting

import Base.+ 
(+)(x::Payoff,y::Payoff) = Axpy(1.0,x,y)
(+)(x::Payoff,y) = Axpy(1.0,x,Fixed(y))
(+)(x,y::Payoff) = Axpy(1.0,Fixed(x),y)
#
import Base.-
(-)(x::Payoff,y::Payoff) = Axpy(-1.0,y,x)
(-)(x::Payoff,y) = Axpy(-1.0,Fixed(y),x)
(-)(x,y::Payoff) = Axpy(-1.0,y,Fixed(x))
#
import Base.*
(*)(x::Payoff,y::Payoff) = Mult(x,y)
(*)(x::Payoff,y) = Mult(x,Fixed(y))
(*)(x,y::Payoff) = Mult(Fixed(x),y)
#
import Base./
(/)(x::Payoff,y::Payoff) = Div(x,y)
(/)(x::Payoff,y) = Div(x,Fixed(y))
(/)(x,y::Payoff) = Div(Fixed(x),y)
#
import Base.%
(%)(x::Payoff,t) = Pay(x,t)
#
import Base.<
(<)(x::Payoff,y::Payoff) = Logical(x,y,(a,b)->Float64(a<b))
(<)(x::Payoff,y) = Logical(x,Fixed(y),(a,b)->Float64(a<b))
(<)(x,y::Payoff) = Logical(Fixed(x),y,(a,b)->Float64(a<b))
import Base.<=
(<=)(x::Payoff,y::Payoff) = Logical(x,y,(a,b)->Float64(a<=b))
(<=)(x::Payoff,y) = Logical(x,Fixed(y),(a,b)->Float64(a<=b))
(<=)(x,y::Payoff) = Logical(Fixed(x),y,(a,b)->Float64(a<=b))
import Base.==
(==)(x::Payoff,y::Payoff) = Logical(x,y,(a,b)->Float64(a==b))
(==)(x::Payoff,y) = Logical(x,Fixed(y),(a,b)->Float64(a==b))
(==)(x,y::Payoff) = Logical(Fixed(x),y,(a,b)->Float64(a==b))
import Base.>=
(>=)(x::Payoff,y::Payoff) = Logical(x,y,(a,b)->Float64(a>=b))
(>=)(x::Payoff,y) = Logical(x,Fixed(y),(a,b)->Float64(a>=b))
(>=)(x,y::Payoff) = Logical(Fixed(x),y,(a,b)->Float64(a>=b))
import Base.>
(>)(x::Payoff,y::Payoff) = Logical(x,y,(a,b)->Float64(a>b))
(>)(x::Payoff,y) = Logical(x,Fixed(y),(a,b)->Float64(a>b))
(>)(x,y::Payoff) = Logical(Fixed(x),y,(a,b)->Float64(a>b))


# basic payoffs

struct Fixed{T<:AbstractFloat} <: Leaf
    x::T
end

at(self::Fixed, p::Path) = self.x
obsTime(self::Fixed) = 0.0

#

struct Pay{T<:AbstractFloat} <: Payoff
    x::Payoff
    payTime::T
end

at(self::Pay, p::Path) = at(self.x, p)
obsTime(self::Pay) = self.payTime
function observationTimes(self::Pay)
    return union(observationTimes(self.x),[self.payTime])
end

#

struct Asset{T<:AbstractFloat} <: Leaf
    obsTime::T
    alias::String
end

Asset(obsTime) = Asset(obsTime,nothing)
at(self::Asset, p::Path) = asset(p,self.obsTime,self.alias)
obsTime(self::Asset) = self.obsTime

#

# basic rates payoffs

struct ZeroBond{T<:AbstractFloat} <: Leaf
    obsTime::T
    payTime::T
    alias::String
end

ZeroBond(obsTime,payTime) = ZeroBond(obsTime,payTime,"")
at(self::ZeroBond, p::Path) = zeroBond(p,self.obsTime,self.payTime,self.alias)
obsTime(self::ZeroBond) = self.obsTime

#

struct LiborRate{T<:AbstractFloat} <: Leaf
    obsTime::T
    startTime::T
    endTime::T
    yearFraction::T
    tenorBasis::T
    alias::String
    _dummy_::Bool  # avoid stack overflow during constructor
end

function LiborRate(obsTime,startTime,endTime;yearFraction=nothing,tenorBasis=1.0,alias=nothing)
    if isnothing(yearFraction)
        yearFraction = endTime - startTime
    end
    if isnothing(alias)
        alias = ""
    end
    return LiborRate(obsTime,startTime,endTime,yearFraction,tenorBasis,alias,true)
end

function at(self::LiborRate, p::Path)
    pStart = zeroBond(p, self.obsTime, self.startTime, self.alias)
    pEnd   = zeroBond(p, self.obsTime, self.endTime, self.alias)
    return (pStart/pEnd*self.tenorBasis - 1.0)/self.yearFraction
end
obsTime(self::LiborRate) = self.obsTime

#

struct SwapRate{T<:AbstractFloat} <: Leaf
    obsTime::T
    floatTimes::T
    floatWeights::T
    fixedTimes::T
    fixedWeights::T
    alias::String
end

SwapRate(obsTime,floatTimes,floatWeights,fixedTimes,fixedWeights) =
    SwapRate(obsTime,floatTimes,floatWeights,fixedTimes,fixedWeights,"")
function at(self::SwapRate, p::Path)
    num = sum([ w*zeroBond(p,self.obsTime,T,self.alias) for (w,T) in zip(self.floatWeights,self.floatTimes) ])
    den = sum([ w*zeroBond(p,self.obsTime,T,self.alias) for (w,T) in zip(self.fixedWeights,self.fixedTimes) ])
    return num / den
end
obsTime(self::SwapRate) = self.obsTime

#

struct FixedLeg{T<:AbstractFloat} <: Leaf
    obsTime::T
    payTimes::T
    payWeights::T
    alias::String
end

FixedLeg(obsTime,payTime,payWeights) = FixedLeg(obsTime,payTime,payWeights,"")
function at(self::FixedLeg, p::Path)
    return sum([ w*zeroBond(p,self.obsTime,T,self.alias) for (w,T) in zip(self.payWeights,self.payTimes) ])
end
obsTime(self::FixedLeg) = self.obsTime

# arithmetic operations

struct Axpy{T<:AbstractFloat} <: BinaryNode
    a::T
    x::Payoff
    y::Payoff
end

at(self::Axpy, p::Path) = self.a * at(self.x,p) + at(self.y,p)
obsTime(self::Axpy) = 0.0

#

struct Mult <: BinaryNode
    x::Payoff
    y::Payoff
end

at(self::Mult, p::Path) = at(self.x,p) * at(self.y,p)
obsTime(self::Mult) = 0.0

#

struct Div <: BinaryNode
    x::Payoff
    y::Payoff
end

at(self::Div, p::Path) = at(self.x,p) / at(self.y,p)
obsTime(self::Div) = 0.0

#

struct Max <: BinaryNode
    x::Payoff
    y::Payoff
end

at(self::Max, p::Path) = max(at(self.x,p),at(self.y,p))
obsTime(self::Div) = 0.0

#

struct Min <: BinaryNode
    x::Payoff
    y::Payoff
end

at(self::Min, p::Path) = min(at(self.x,p),at(self.y,p))
obsTime(self::Div) = 0.0

#

struct Logical <: BinaryNode
    x::Payoff
    y::Payoff
    op
end

at(self::Logical, p::Path) = self.op(at(self.x,p),at(self.y,p))
obsTime(self::Div) = 0.0

