#!/usr/bin/python

import numpy as np
from hybmc.models.StochasticProcess import StochasticProcess


class CreditModel(StochasticProcess):

    # Python constructor
    def __init__(self,             
                 baseModel,      #  rates or hybrid model
                 creditAliases,  #  list of credit names (all relative to dom/base currency)
                 creditModels,   #  rates model for spread simulation
                 correlations ): #  np.array of instantanous correlations
        #
        self.baseModel     = baseModel 
        self.creditAliases = creditAliases
        self.creditModels  = creditModels
        self.correlations  = correlations
        # we need to know the model index for a given alias
        self.index = { self.creditAliases[k] : k for k in range(len(self.creditAliases)) }
        # add sanity checks here
        self._size = baseModel.size()
        self._factors = baseModel.factors()
        self.modelStartIdx = [ baseModel.size() ]        # N+1 elements
        self.brownianStartIdx = [ baseModel.factors() ]  # N+1 elements
        for model in self.creditModels:
            self._size    += model.size()
            self._factors += model.factors()
            self.modelStartIdx.append(self.modelStartIdx[-1] + model.size())
            self.brownianStartIdx.append(self.brownianStartIdx[-1] + model.factors())
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
        initialValSet = [ self.baseModel.initialValues() ]
        initialValSet += [ model.initialValues()
            for model in self.creditModels ]
        return np.concatenate(initialValSet)

    def evolve(self, t0, X0, dt, dW, X1):
        if self.L is not None:
            dZ = self.L.dot(dW)
        else:
            dZ = dW
        # first we evolve base model
        self.baseModel.evolve( t0,
            X0[ :self.baseModel.size()    ], dt,
            dZ[ :self.baseModel.factors() ],
            X1[ :self.baseModel.size()    ] )
        # now we evolve all the spread models
        for k in range(len(self.creditModels)):
            self.creditModels[k].evolve( t0,
                X0[ self.modelStartIdx[k]    : self.modelStartIdx[k+1]    ], dt,
                dZ[ self.brownianStartIdx[k] : self.brownianStartIdx[k+1] ],
                X1[ self.modelStartIdx[k]    : self.modelStartIdx[k+1]    ] )
        return

    # the numeraire is delegated to base model
    def numeraire(self, t, X):
        return self.baseModel.numeraire(t, X[ :self.baseModel.size()  ])
    
    # asset calculation is delegated to base model
    def asset(self, t, X, alias):
        return self.baseModel.asset(t, X[:self.baseModel.size()], alias)

    # zero bond is also delegated to base model
    def zeroBond(self, t, T, X, alias):
        return self.baseModel.zeroBond(t, T, X[ :self.baseModel.size()  ], alias)

    # crutial part are credit functions for which we use rates methodologies

    # Cumulated intensity; probability of tau > t, conditional on F_t
    def hazardProcess(self, t, X, alias):
        k = self.index[alias]  # this should throw an exception if alias is unknown
        x = X[ self.modelStartIdx[k] : self.modelStartIdx[k+1] ]
        return 1.0 / self.creditModels[k].numeraire(t, x )
    
    # instantanous probability of default
    def hazardRate(self, t, X, alias):
        k = self.index[alias]  # this should throw an exception if alias is unknown
        x = X[ self.modelStartIdx[k] : self.modelStartIdx[k+1] ]
        dt = 1.0/365  # one day as a proxy
        return self.creditModels[k].shortRateOverPeriod(t, dt, x, x)
    
    # probavility of survival consitional on information at t
    def survivalProb(self, t, T, X, alias):
        k = self.index[alias]  # this should throw an exception if alias is unknown
        x = X[ self.modelStartIdx[k] : self.modelStartIdx[k+1] ]
        return self.creditModels[k].zeroBond(t, T, x, alias)

    # keep track of components in hybrid model

    def stateAliases(self):
        aliases = self.baseModel.stateAliases()
        for k, alias in enumerate(self.creditAliases):
            aliases += [ alias + '_' + stateAlias for stateAlias in self.creditModels[k].stateAliases() ]
        return aliases

    def factorAliases(self):
        aliases = self.baseModel.factorAliases()
        for k, alias in enumerate(self.creditAliases):
            aliases += [ alias + '_' + factorAlias for factorAlias in self.creditModels[k].factorAliases() ]
        return aliases
    
