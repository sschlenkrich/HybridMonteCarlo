#!/usr/bin/python

# create Julia payoffs from Python payoffs

from julia import Main
Main.include('./hybmc/simulations/Payoffs.jl')

import sys
sys.path.append('./')

from hybmc.simulations.Payoffs import Payoff, Pay, Fixed, ZeroBond, LiborRate, Asset, Axpy, Mult, Div, Max


def JuliaPayoff(p):
    #
    if isinstance(p,Pay):
        x = JuliaPayoff(p.x)
        return Main.Pay(x,p.obsTime)
    #
    if isinstance(p,Fixed):
        return Main.Fixed(p.x)
    #
    if isinstance(p,ZeroBond):
        return Main.ZeroBond(p.obsTime,p.payTime,p.alias)
    #
    if isinstance(p,LiborRate):
        return Main.LiborRate(p.obsTime,p.startTime,p.endTime,p.yearFraction,p.tenorBasis,p.alias,False)
    #
    if isinstance(p,Asset):
        return Main.Asset(p.obsTime,p.alias)
    #
    if isinstance(p,Axpy):        
        x = JuliaPayoff(p.x)
        if p.y is None: p.y = Fixed(0.0)
        y = JuliaPayoff(p.y)
        return Main.Axpy(p.a,x,y)
    #
    if isinstance(p,Mult):
        x = JuliaPayoff(p.x)
        y = JuliaPayoff(p.y)
        return Main.Mult(x,y)
    if isinstance(p,Div):
        x = JuliaPayoff(p.x)
        y = JuliaPayoff(p.y)
        return Main.Div(x,y)
    else:
        raise NotImplementedError('Implementation of JuliaPayoff for %s required.' % str(type(p)))
    return

def JuliaPayoffs(payoffList):
    return [ JuliaPayoff(p) for p in payoffList ]

