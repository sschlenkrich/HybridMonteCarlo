#!/usr/bin/python

# create QuantLib payoffs from Python payoffs

try:
    import QuantLib as ql
except ModuleNotFoundError as e:
    print('Error: Module QuantLibPayoffs requires a (custom) QuantLib installation.')
    raise e

try:
    from QuantLib import RealMCPay, RealMCFixedAmount, RealMCZeroBond, RealMCAsset, \
        RealMCAxpy, RealMCMult, RealMCDivision
except ImportError as e:
    print('Error: Module QuantLibPayoffs requires a custom QuantLib installation.')
    print('It seems you do have a QuantLib installed but maybe not the custom version')
    print('with payoff scripting. Checkout https://github.com/sschlenkrich/QuantLib')
    raise e

import sys
sys.path.append('./')

from hybmc.simulations.Payoffs import Payoff, Pay, Fixed, ZeroBond, LiborRate, Asset, Axpy, Mult, Div, Max


def QuantLibPayoff(p):
    #
    if isinstance(p,Pay):
        x = QuantLibPayoff(p.x)
        return RealMCPay(x,p.obsTime)
    #
    if isinstance(p,Fixed):
        return RealMCFixedAmount(p.x)
    #
    if isinstance(p,ZeroBond):
        alias = p.alias
        if alias is None: alias = ''
        return RealMCZeroBond(p.obsTime,p.payTime,alias)
    #
    if isinstance(p,LiborRate):
        alias = p.alias
        if alias is None: alias = ''
        P0 = RealMCZeroBond(p.obsTime,p.startTime,alias)
        P1 = RealMCZeroBond(p.obsTime,p.endTime,alias)
        P0_over_P1 = RealMCDivision(P0,P1)
        D_over_tau = p.tenorBasis / p.yearFraction
        minus_one_over_tau = RealMCFixedAmount(-1.0/p.yearFraction)
        return RealMCAxpy(D_over_tau,P0_over_P1,minus_one_over_tau)
    #
    if isinstance(p,Asset):
        return RealMCAsset(p.obsTime,p.alias)
    #
    if isinstance(p,Axpy):        
        x = QuantLibPayoff(p.x)
        if p.y is None: p.y = Fixed(0.0)
        y = QuantLibPayoff(p.y)
        return RealMCAxpy(p.a,x,y)
    #
    if isinstance(p,Mult):
        x = QuantLibPayoff(p.x)
        y = QuantLibPayoff(p.y)
        return RealMCMult(x,y)
    #
    if isinstance(p,Div):
        x = QuantLibPayoff(p.x)
        y = QuantLibPayoff(p.y)
        return RealMCDivision(x,y)
    else:
        raise NotImplementedError('Implementation of QuantLibPayoff for %s required.' % str(type(p)))
    return

def QuantLibPayoffs(payoffList):
    return [ QuantLibPayoff(p) for p in payoffList ]

