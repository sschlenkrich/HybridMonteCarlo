#!/usr/bin/python

import numpy as np
from hybmc.models.StochasticProcess import StochasticProcess


class SpreadModel(StochasticProcess):

    # Python constructor
    def __init__(self,             
                 baseModel,      #  rates or hybrid model
                 sprdModel,      #  rates model for spread simulation
                 correlations ): #  np.array of instantanous correlations
        #
        self.baseModel     = baseModel 
        self.sprdModel     = sprdModel
        self.correlations  = correlations
        # add sanity checks here
        self._size = baseModel.size() + sprdModel.size()
        self._factors = baseModel.factors() + sprdModel.factors()
        # we need to check and factor the hybrid correlation matrix
        if self.correlations is not None:  # None is interpreted as identity
            self.L = np.linalg.cholesky(self.correlations)
        else:
            self.L = None

    def size(self):
        return self._size

    def factors(self):
        return self._factors

    def initialValues(self):
        return np.concatenate([
            self.baseModel.initialValues(), self.sprdModel.initialValues() ])

    def evolve(self, t0, X0, dt, dW, X1):
        if self.L is not None:
            dZ = self.L.dot(dW)
        else:
            dZ = dW
        self.baseModel.evolve( t0,
            X0[ :self.baseModel.size()    ], dt,
            dZ[ :self.baseModel.factors() ],
            X1[ :self.baseModel.size()    ] )
        self.sprdModel.evolve( t0,
            X0[ self.baseModel.size():    ], dt,
            dZ[ self.baseModel.factors(): ],
            X1[ self.baseModel.size():    ] )
        return

    # the numeraire is credit-risky bank account
    def numeraire(self, t, X):
        return self.baseModel.numeraire(t, X[ :self.baseModel.size()  ]) * \
               self.sprdModel.numeraire(t, X[  self.baseModel.size(): ])
    
    # asset calculation is delegated to base model
    def asset(self, t, X, alias):
        return self.baseModel.asset(t, X[:self.baseModel.size()], alias)

    # zero bond wwith and without credit risk
    def zeroBond(self, t, T, X, alias):
        if alias is None:  # a credit-risky zero coupon bond
            return self.baseModel.zeroBond(t, T, X[ :self.baseModel.size()  ], None) * \
                   self.sprdModel.zeroBond(t, T, X[  self.baseModel.size(): ], None)
        # a foreign or domestic currency zero coupon bond is delegated to base model
        return self.baseModel.zeroBond(t, T, X[ :self.baseModel.size()  ], alias)

    # Cumulated intensity; probability of tau > t, conditional on F_t
    def hazardProcess(self, t, X, alias):
        return self.baseModel.hazardProcess(t, X[:self.baseModel.size()], alias)
    
    # instantanous probability of default
    def hazardRate(self, t, X, alias):
        return self.baseModel.hazardRate(t, X[:self.baseModel.size()], alias)
    
    # probavility of survival consitional on information at t
    def survivalProb(self, t, T, X, alias):
        return self.baseModel.survivalProb(t, T, X[:self.baseModel.size()], alias)

    # the short rate over an integration time period
    # this is required for drift calculation in multi-asset and hybrid models
    def shortRateOverPeriod(self, t0, dt, X0, X1):
        return self.baseModel.shortRate(t0, dt, X0[:self.baseModel.size()], X1[:self.baseModel.size()]) + \
               self.sprdModel.shortRate(t0, dt, X0[self.baseModel.size():], X1[self.baseModel.size():])

    # keep track of components in spread model

    def stateAliases(self):
        return [ 'base_' + alias for alias in self.baseModel.stateAliases() ] + \
               [ 'sprd_' + alias for alias in self.sprdModel.stateAliases() ]
        
    def factorAliases(self):
        return [ 'base_' + alias for alias in self.baseModel.factorAliases() ] + \
               [ 'sprd_' + alias for alias in self.sprdModel.factorAliases() ]
