#!/usr/bin/python


import numpy as np

from hybmc.models.StochasticProcess import StochasticProcess


# We implement a general deterministic shift extension
# following Brigo/Mercurio 2007, sec. 3.8

#  r(t)  = [f(0,t) - f^x(0,t)] + x(t)
#  dx(t) = [ base model dynamics ]


class ShiftedRatesModel(StochasticProcess):

    # Python constructor
    def __init__(self, yieldCurve, baseModel):
        self.yieldCurve  = yieldCurve
        self.baseModel   = baseModel
        # we cache initial value for ZCB's
        self.initialValues_ = self.baseModel.initialValues()

    # forward rate shift translates into spread discount factor
    def PhiStar(self,t,T,alias):
        dfCurve = self.yieldCurve.discount(T) / self.yieldCurve.discount(t)
        dfModel = self.baseModel.zeroBond(0.0,T,self.initialValues_,alias) / \
                  self.baseModel.zeroBond(0.0,t,self.initialValues_,alias)
        return dfCurve / dfModel

    # stochastic process interface
    
    def size(self):   # dimension of X(t)
        return self.baseModel.size()

    def factors(self):   # dimension of W(t)
        return self.baseModel.factors()

    def initialValues(self):
        return self.initialValues_
    
    def zeroBond(self, t, T, X, alias):
        return self.PhiStar(t,T,alias) * self.baseModel.zeroBond(t,T,X,alias)

    def numeraire(self, t, X):
        dfCurve = self.yieldCurve.discount(t)
        dfModel = self.baseModel.zeroBond(0.0,t,self.initialValues_,None)
        return dfModel / dfCurve * self.baseModel.numeraire(t,X)

    # evolve X(t0) -> X(t0+dt) using independent Brownian increments dW
    # t0, dt are assumed float, X0, X1, dW are np.array
    def evolve(self, t0, X0, dt, dW, X1):
        self.baseModel.evolve(t0,X0,dt,dW,X1)

    # the short rate over an integration time period
    # this is required for drift calculation in multi-asset and hybrid models
    def shortRateOverPeriod(self, t0, dt, X0, X1):
        B_d = 1.0 / self.PhiStar(t0,t0+dt,None)
        return np.log(B_d) / dt + self.baseModel.shortRateOverPeriod(t0,dt,X0,X1)

    # keep track of components in hybrid model

    def stateAliases(self):
        return self.baseModel.stateAliases()

    def factorAliases(self):
        return self.baseModel.factorAliases()
    
