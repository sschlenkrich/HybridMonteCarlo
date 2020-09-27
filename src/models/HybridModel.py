#!/usr/bin/python

import numpy as np
from src.models.StochasticProcess import StochasticProcess


class HybridModel(StochasticProcess):

    # Python constructor
    def __init__(self,             
                 domAlias,         #  name of our domestic (numeraire) currency
                 domRatesModel,    #  domestic rates model specifies numeraire
                 forAliases,       #  list of foreign currencies (all relative to dom currency)
                 forAssetModels,   #  list of foreign asset models
                 forRatesModels,   #  list of foreign rates models
                 correlations ):   #  np.array of instantanous correlations
        #
        self.domAlias       = domAlias 
        self.domRatesModel  = domRatesModel
        self.forAliases     = forAliases   
        self.forAssetModels = forAssetModels 
        self.forRatesModels = forRatesModels 
        self.correlations   = correlations
        # add sanity checks here
        # we need to know the model index for a given alias
        self.index = { self.forAliases[k] : k for k in range(len(self.forAliases)) }
        # manage model indices in state variable
        self._size = self.domRatesModel.size()
        self._factors = self.domRatesModel.factors()
        self.modelsStartIdx = [ ]
        lastModelIdx = domRatesModel.size()
        for assetModel, ratesModel in zip(self.forAssetModels,self.forRatesModels):
            self._size += (assetModel.size() + ratesModel.size())
            self._factors += (assetModel.factors() + ratesModel.factors())
            self.modelsStartIdx.append(lastModelIdx)
            self.modelsStartIdx.append(lastModelIdx + assetModel.size())
            lastModelIdx += (assetModel.size() + ratesModel.size())
        # check correlation matrix properties here
        if self.correlations is not None:  # None is interpreted as identity
            self.L = np.linalg.cholesky(self.correlations)
        else:
            self.L = None

    def size(self):
        return self._size

    def factors(self):
        return self._factors

    def initialValues(self):
        initialValueList = [ self.domRatesModel.initialValues() ]
        for assetModel, ratesModel in zip(self.forAssetModels,self.forRatesModels):
            initialValueList.append(assetModel.initialValues())
            initialValueList.append(ratesModel.initialValues())
        return np.concatenate(initialValueList)

    def evolve(self, t0, X0, dt, dW, X1):
        if self.L is not None:
            dZ = self.L.dot(dW)
        else:
            dZ = dW
        if not self.forAliases:  # take a short cut
            self.domRatesModel.evolve(t0, X0, dt, dZ, X1)
            return
        # evolve domestic rates
        domSize = self.domRatesModel.size()  # shorten expressions
        self.domRatesModel.evolve(t0, X0[:domSize], dt, dZ[:self.domRatesModel.factors()], X1[:domSize])
        # we need the domestic drift r_d
        r_d = self.domRatesModel.shortRateOverPeriod(t0, dt, X0[:domSize], X1[:domSize])
        # now we iterate over foreign models
        corrStartIdx = self.domRatesModel.factors()  #  we need to keep track of our sub-correlations
        for k, alias in enumerate(self.forAliases):
            # carefully collect Brownian increments
            dw_asset = dZ[corrStartIdx : \
                          corrStartIdx + self.forAssetModels[k].factors()]
            dw_rates = dZ[corrStartIdx + self.forAssetModels[k].factors() : \
                          corrStartIdx + self.forAssetModels[k].factors() + self.forRatesModels[k].factors()]
            # we need the starting point states for evolution, y0 (asset), x0 (rates)
            y0 = X0[ self.modelsStartIdx[2 * k] : \
                     self.modelsStartIdx[2 * k] + self.forAssetModels[k].size() ]
            x0 = X0[ self.modelsStartIdx[2 * k + 1] : \
                     self.modelsStartIdx[2 * k + 1] + self.forRatesModels[k].size() ]
            # Quanto adjustment
            # we use the model-independent implementation to allow for credit hybrid components
            # todo: capture case self.correlations = None
            qAdj = self.correlations[corrStartIdx,
                                     corrStartIdx + self.forAssetModels[k].factors() : 
                                     corrStartIdx + self.forAssetModels[k].factors() + self.forRatesModels[k].factors()]
            # we want to modify qAdj vector w/o changing the correlation
            qAdj = np.array(qAdj)
            # we need to extend the input state for our asset mode to account for drift and adjuster
            y0 = np.concatenate([ y0, np.array([0.0, 0.0]) ])  # maybe better use append here, but check view/copy
            y0[-1] = 0.0  #  we need to ADD HYBRID VOL ADJUSTER HERE
            assetVol = self.forAssetModels[k].volatility(t0, y0)
            qAdj *= (assetVol*np.sqrt(dt))
            dw_rates = dw_rates - qAdj  #  create a new vector
            # evolve foreign rates
            x1 = X1[ self.modelsStartIdx[2 * k + 1] : \
                     self.modelsStartIdx[2 * k + 1] + self.forRatesModels[k].size() ]
            self.forRatesModels[k].evolve(t0, x0, dt, dw_rates, x1)
            # calculate FX drift, volAdjuster and extend input state
            r_f = self.forRatesModels[k].shortRateOverPeriod(t0, dt, x0, x1)
            y0[-2] = r_d - r_f   # FX drift
            # finally we can evolve FX
            y1 = X1[ self.modelsStartIdx[2 * k] : \
                     self.modelsStartIdx[2 * k] + self.forAssetModels[k].size() ]
            self.forAssetModels[k].evolve(t0, y0, dt, dw_asset, y1)
            # no need to copy results coz we work on views of X1
            # but we need to update stochastic factor index
            corrStartIdx += (self.forAssetModels[k].factors() + self.forRatesModels[k].factors())
        return

    # interface for payoff calculation

    def numeraire(self, t, X):
        return self.domRatesModel.numeraire(t,X[:self.domRatesModel.size()])
    
    def asset(self, t, X, alias):
        k = self.index[alias]  # this should throw an exception if alias is unknown
        y = X[ self.modelsStartIdx[2 * k] : self.modelsStartIdx[2 * k] + self.forAssetModels[k].size() ]
        return self.forAssetModels[k].asset(t, y, alias)

    def zeroBond(self, t, T, X, alias):
        if alias is None or alias==self.domAlias:
            x = X[:self.domRatesModel.size()]
            return self.domRatesModel.zeroBond(t, T, x)
        k = self.index[alias]  # this should throw an exception if alias is unknown
        x = X[ self.modelsStartIdx[2 * k + 1] : \
               self.modelsStartIdx[2 * k + 1] + self.forRatesModels[k].size() ]
        return self.forRatesModels[k].zeroBond(t, T, x)

    # keep track of components in hybrid model

    def stateAliases(self):
        aliases = [ self.domAlias + '_' + stateAlias for stateAlias in self.domRatesModel.stateAliases() ]
        for k, alias in enumerate(self.forAliases):
            aliases += [ alias + '_' + stateAlias for stateAlias in self.forAssetModels[k].stateAliases() ]
            aliases += [ alias + '_' + stateAlias for stateAlias in self.forRatesModels[k].stateAliases() ]
        return aliases

    def factorAliases(self):
        aliases = [ self.domAlias + '_' + factorAlias for factorAlias in self.domRatesModel.factorAliases() ]
        for k, alias in enumerate(self.forAliases):
            aliases += [ alias + '_' + factorAlias for factorAlias in self.forAssetModels[k].factorAliases() ]
            aliases += [ alias + '_' + factorAlias for factorAlias in self.forRatesModels[k].factorAliases() ]
        return aliases
    
    # add hybrid vol adjuster methodology here