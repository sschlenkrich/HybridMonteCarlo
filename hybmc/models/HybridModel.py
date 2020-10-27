#!/usr/bin/python

import numpy as np
from hybmc.models.StochasticProcess import StochasticProcess


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
        # per default we do not apply hybrid adjuster
        self.hybAdjTimes = None

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
            y0[-1] = self.hybridVolAdjuster(k, t0)
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
            return self.domRatesModel.zeroBond(t, T, x, alias)
        k = self.index[alias]  # this should throw an exception if alias is unknown
        x = X[ self.modelsStartIdx[2 * k + 1] : \
               self.modelsStartIdx[2 * k + 1] + self.forRatesModels[k].size() ]
        return self.forRatesModels[k].zeroBond(t, T, x, alias)

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
    # we calculate term structures of local volatility, correspondingeffective hybrid volatility
    # and an adjuster based on an additive shift

    def hybridVolAdjuster(self, forIdx, t):
        if self.hybAdjTimes is None:
            return 0.0  # default
        # linear interpolation with constant extrapolation
        # maybe better use scipy interpolation with linear extraplation
        return np.interp(t,self.hybAdjTimes,self.hybVolAdj[forIdx],self.hybVolAdj[forIdx,0],self.hybVolAdj[forIdx,-1])

    def recalculateHybridVolAdjuster(self, hybAdjTimes = None):
        if hybAdjTimes is not None: # if we don't supply time grid we want to keep the current grid
                self.hybAdjTimes = hybAdjTimes
        if self.hybAdjTimes.shape[0]==0: # patological case, do nothing
            return
        # we need to check for consistent times again
        if not self.hybAdjTimes[0]==0.0:
            raise ValueError('HybridModel: hybAdjTimes_[0]==0.0 required.')
        for k in range(1,self.hybAdjTimes.shape[0]):
            if not self.hybAdjTimes[k] > self.hybAdjTimes[k-1]:
                raise ValueError('HybridModel: hybAdjTimes_[k]>hybAdjTimes_[k-1] required.')
        # initialise 
        self.localVol = np.zeros([len(self.forAliases),len(self.hybAdjTimes)])
        self.hybrdVol = np.ones([len(self.forAliases),len(self.hybAdjTimes)])  #  1.0 is required for p calculation
        self.hybVolAdj = np.zeros([len(self.forAliases),len(self.hybAdjTimes)])
        S0 = np.array([ m.asset(0.0, m.initialValues(), None) for m in self.forAssetModels ])
        # calculate vols at zero
        for i in range(len(S0)):
            y0 = np.concatenate([ self.forAssetModels[i].initialValues(), [0.0, 0.0]])  # this is asset model-dependent
            self.localVol[i,0] = self.forAssetModels[i].volatility(0.0, y0)
            self.hybrdVol[i,0] = self.localVol[i,0]
        if len(self.hybAdjTimes)==1:  # nothing else to do
            return 
        # now we start with the actual methodology...
        corrStartIdx = self.domRatesModel.factors()        
        for i in range(len(S0)):
            # we collect all relevant correlations
            # recall, 
            #   Y0 is domestic rates model
            #   X1 is asset (or FX) model
            #   Y1 is forign rates model
            #
            # domestic rates vs FX, ASSUME vol-FX correlation is zero
            rhoY0X1 = self.correlations[ :self.domRatesModel.factors(), corrStartIdx]   #  
            # foreign rates vs FX, ASSUME vol-FX correlation is zero
            rhoX1Y1 = self.correlations[ corrStartIdx,
                                         corrStartIdx + self.forAssetModels[i].factors() :
                                         corrStartIdx + self.forAssetModels[i].factors() + self.forRatesModels[i].factors() ]
            # rates vs rates, ASSUME all vol-... correlation are zero
            rhoY0Y1 = self.correlations[ :self.domRatesModel.factors(),
                                         corrStartIdx + self.forAssetModels[i].factors() :
                                         corrStartIdx + self.forAssetModels[i].factors() + self.forRatesModels[i].factors() ]
            # update stochastic factor index
            corrStartIdx += (self.forAssetModels[i].factors() + self.forRatesModels[i].factors())
            # bootstrap over adjuster times
            for k, T in list(enumerate(self.hybAdjTimes))[1:]:
                # ATM forward and effective local volatility
                dfDom = self.domRatesModel.zeroBond(0.0, T, self.domRatesModel.initialValues(), None)
                dfFor = self.forRatesModels[i].zeroBond(0.0, T, self.forRatesModels[i].initialValues(), None)
                S = S0[i] * dfFor / dfDom  # maybe it's worth to save S for debugging
                y = np.zeros(self.forAssetModels[i].size() + 2)  # this is asset model-dependent
                y[0] = np.log(S / S0[i])
                self.localVol[i,k] = self.forAssetModels[i].volatility(self.hybAdjTimes[k], y)
                # calculate derivative of hybrid variance
                hPrime = np.zeros(k+1)
                for j, t in enumerate(self.hybAdjTimes[:k+1]):
                    sigmaP0      = self.domRatesModel.zeroBondVolatility(t, T)
                    sigmaP0Prime = self.domRatesModel.zeroBondVolatilityPrime(t, T)
                    sigmaP1      = self.forRatesModels[i].zeroBondVolatility(t, T)
                    sigmaP1Prime = self.forRatesModels[i].zeroBondVolatilityPrime(t, T)
                    #
                    sigma0 = sigmaP0 - rhoY0Y1.dot(sigmaP1) + self.hybrdVol[i,j]*rhoY0X1  # bootstrapping enters here
                    sum0 = sigmaP0Prime.dot(sigma0)
                    #
                    sigma1 = sigmaP1 - sigmaP0.dot(rhoY0Y1) - self.hybrdVol[i,j]*rhoX1Y1  # bootstrapping enters here
                    sum1 = sigma1.dot(sigmaP1Prime)
                    # collect terms and finish
                    hPrime[j] = 2.0*(sum0 + sum1)
                p = 0.5 * hPrime[k] * (self.hybAdjTimes[k] - self.hybAdjTimes[k - 1])
                q = 0.5 * hPrime[k-1] * (self.hybAdjTimes[k] - self.hybAdjTimes[k - 1])
                for j in range(1,k):
                    q += 0.5 * (hPrime[j - 1] + hPrime[j]) * (self.hybAdjTimes[j] - self.hybAdjTimes[j - 1])
                # let's see if this works...
                root2 = p*p / 4.0 - q + self.localVol[i,k] * self.localVol[i,k]
                if not root2 >= 0.0:
                    raise ValueError('HybridModel: root2>=0.0 required.')
                self.hybrdVol[i,k] = -p / 2.0 + np.sqrt(root2)
                if not self.hybrdVol[i,k] > 0.0:
                    raise ValueError('HybridModel: hybrdVol[i,k]>0.0 required.')
                # maybe we should add some more safety checks here...
                self.hybVolAdj[i,k] = self.hybrdVol[i,k] - self.localVol[i,k]
