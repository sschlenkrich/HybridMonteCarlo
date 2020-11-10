
include("../simulations/McSimulation.jl")

abstract type Payoff end

function at(self::Payoff, p::Path)
    throw(ArgumentError("Implementation of method at() required."))
end

function obsTime(self::Payoff)
    throw(ArgumentError("Implementation of method obsTime() required."))
end

function discountedAt(self::Payoff, p::Path)
    return at(self,p) / numeraire(p,obsTime(self))
end

# simplify payoff scripting

import Base.+ 
(+)(x::Payoff,y::Payoff) = Axpy(1.0,x,y)
(+)(x::Payoff,y) = Axpy(1.0,x,Fixed(y))
(+)(x,y::Payoff) = Axpy(1.0,Fixed(x),y)
#
import Base.-
(-)(x::Payoff,y::Payoff) = Axpy(-1.0,y,x)
(-)(x::Payoff,y) = Axpy(-1.0,Fixed(y),9x)
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

# basic payoffs

struct Fixed <: Payoff
    x
end

at(self::Fixed, p::Path) = self.x
obsTime(self::Fixed) = 0.0

#

struct Pay <: Payoff
    x::Payoff
    payTime
end

at(self::Pay, p::Path) = at(self.x, p)
obsTime(self::Pay) = self.payTime

#

struct Asset <: Payoff
    obsTime
    alias
end

Asset(obsTime) = Asset(obsTime,nothing)
at(self::Asset, p::Path) = asset(p,self.t,self.alias)
obsTime(self::Asset) = self.obsTime

#

# basic rates payoffs

struct ZeroBond <: Payoff
    obsTime
    payTime
    alias
end

ZeroBond(obsTime,payTime) = ZeroBond(obsTime,payTime,nothing)
at(self::ZeroBond, p::Path) = zeroBond(p,self.obsTime,self.payTime,self.alias)
obsTime(self::ZeroBond) = self.obsTime

#

struct LiborRate <: Payoff
    obsTime
    startTime
    endTime
    yearFraction
    tenorBasis
    alias
    _dummy_::Bool  # avoid stack overflow during constructor
end

function LiborRate(obsTime,startTime,endTime,yearFraction=nothing,tenorBasis=1.0,alias=nothing)
    if isnothing(yearFraction)
        yearFraction = endTime - startTime
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

struct SwapRate <: Payoff
    obsTime
    floatTimes
    floatWeights
    fixedTimes
    fixedWeights
    alias
end

SwapRate(obsTime,floatTimes,floatWeights,fixedTimes,fixedWeights) =
    SwapRate(obsTime,floatTimes,floatWeights,fixedTimes,fixedWeights,nothing)
function at(self::SwapRate, p::Path)
    num = sum([ w*zeroBond(p,self.obsTime,T,self.alias) for (w,T) in zip(self.floatWeights,self.floatTimes) ])
    den = sum([ w*zeroBond(p,self.obsTime,T,self.alias) for (w,T) in zip(self.fixedWeights,self.fixedTimes) ])
    return num / den
end
obsTime(self::SwapRate) = self.obsTime

#

struct FixedLeg <: Payoff
    obsTime
    payTimes
    payWeights
    alias
end

FixedLeg(obsTime,payTime,payWeights) = FixedLeg(obsTime,payTime,payWeights,nothing)
function at(self::FixedLeg, p::Path)
    return sum([ w*zeroBond(p,self.obsTime,T,self.alias) for (w,T) in zip(self.payWeights,self.payTimes) ])
end
obsTime(self::FixedLeg) = self.obsTime

# arithmetic operations

struct Axpy <: Payoff
    a
    x::Payoff
    y::Payoff
end

at(self::Axpy, p::Path) = self.a * at(self.x,p) + at(self.y,p)
obsTime(self::Axpy) = 0.0

#

struct Mult <: Payoff
    x::Payoff
    y::Payoff
end

at(self::Mult, p::Path) = at(self.x,p) * at(self.y,p)
obsTime(self::Mult) = 0.0

struct Div <: Payoff
    x::Payoff
    y::Payoff
end

at(self::Div, p::Path) = at(self.x,p) / at(self.y,p)
obsTime(self::Div) = 0.0

