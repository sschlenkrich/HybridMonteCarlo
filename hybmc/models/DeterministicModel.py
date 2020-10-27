#!/usr/bin/python

import numpy as np
from hybmc.models.StochasticProcess import StochasticProcess

class DeterministicModel(StochasticProcess):

    # Python constructor
    def __init__(self,             
                 domAlias,         #  name of our domestic (numeraire) currency
                 domCurve,         #  domestic (discounting) yield curve
                 forAliases,       #  list of foreign currencies (all relative to dom currency)
                 forAssetSpots,    #  list of foreign asset initial values
                 forCurves ):      #  list of foreign (discounting) yield curves
        #
        self.domAlias      = domAlias 
        self.domCurve      = domCurve
        self.forAliases    = forAliases   
        self.forAssetSpots = forAssetSpots 
        self.forCurves     = forCurves
        #
        # we need to know the model index for a given alias
        if self.forAliases is not None:
            self.index = { self.forAliases[k] : k for k in range(len(self.forAliases)) }
        else:
            self.index = None

    def size(self):
        return 0

    def factors(self):
        return 0

    def initialValues(self):
        return np.array([])

    def evolve(self, t0, X0, dt, dW, X1):
        # there is nothing to be done
        return

    # the short rate over an integration time period
    # this is required for drift calculation in multi-asset and hybrid models
    def shortRateOverPeriod(self, t0, dt, X0, X1):
        B_d = self.domCurve.discount(t0) / self.domCurve.discount(t0 + dt)  # deterministic drift part for r_d
        return np.log(B_d) / dt

    # bond volatility is used in hybrid model vol adjuster

    def zeroBondVolatility(self, t, T):
        # we wrap scalar bond volatility into array to allow
        # for generalisation to multi-factor models
        return np.array([  ])

    def zeroBondVolatilityPrime(self, t, T):
        # we wrap scalar bond volatility into array to allow
        # for generalisation to multi-factor models
        return np.array([  ])

    # interface for payoff calculation

    def numeraire(self, t, X):        
        return 1.0 / self.domCurve.discount(t)
    
    def asset(self, t, X, alias):
        k = self.index[alias]  # this should throw an exception if alias is unknown
        return self.forAssetSpots[k] * self.forCurves[k].discount(t) / self.domCurve.discount(t)

    def zeroBond(self, t, T, X, alias):
        if alias is None or alias==self.domAlias:            
            return self.domCurve.discount(T) / self.domCurve.discount(t)
        k = self.index[alias]  # this should throw an exception if alias is unknown
        return self.forCurves[k].discount(T) / self.forCurves[k].discount(t)

    def path(self):
        return self.Path(self)

    # for actual payoffs we need a path object
    class Path():
        # Python constructor
        def __init__(self, model):         
            self.model = model        
        # the numeraire in the domestic currency used for discounting future payoffs
        def numeraire(self, t):
            # we may add numeraire adjuster here...
            return self.model.numeraire(t,None)        
        # a domestic/foreign currency zero coupon bond
        def zeroBond(self, t, T, alias):
            # we may add zcb adjuster here
            return self.model.zeroBond(t, T, None, alias)        
        # an asset price for a given currency alias
        def asset(self, t, alias):
            # we may add asset adjuster here
            return self.model.asset(t, None, alias)

    # there are no components to keep track of

    def stateAliases(self):
        return []

    def factorAliases(self):
        return []

# some easy to use functions

def DcfModel(curve):
    return DeterministicModel(None,curve,None,None,None)
