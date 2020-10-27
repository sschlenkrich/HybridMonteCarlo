#!/usr/bin/python


#   X(t) = X0 * exp{x(t)}
#   
#   We use extended state variables Y = [ x, (z), mu, volAdj ],
#
#   dx(t)     = [mu - 0.5*sigma^2]dt + sigma dW   
#   dr_d(t)   = 0 dt (domestic rates)
#   dr_f(t)   = 0 dt (foreign rates)
#   dz(t)     = 0 dt (stochastic volatility, currently not implemented)
#   mu        = r_d - r_f (rates differential, provided exogenously)
#   sigma     = sigmaLV + volAdj (hybrid volatility adjuster, provided exogenously)

import numpy as np
from hybmc.models.StochasticProcess import StochasticProcess


class AssetModel(StochasticProcess):

    # Python constructor
    def __init__(self, X0, sigma):
        self.X0 = X0
        self.sigma = sigma

    def size(self):
        return 1
        
    def factors(self):
        return 1

    def initialValues(self):
        return np.array([ 0.0 ])

    # calculate volatility taking into account exogenous hybrid adjuster
    def volatility(self, t, Y):
        return self.sigma + Y[2]

    def evolve(self, t0, Y0, dt, dW, Y1):
        sigma = self.volatility(t0,Y0)
        Y1[0] = Y0[0] + (Y0[1] - 0.5*sigma*sigma)*dt + sigma*dW[0]*np.sqrt(dt)

    def asset(self, t, Y, alias):
        return self.X0 * np.exp(Y[0])

    # keep track of components in hybrid model

    def stateAliases(self):
        return [ 'logS' ]

    def factorAliases(self):
        return [ 'logS' ]
        