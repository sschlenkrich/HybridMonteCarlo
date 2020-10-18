#!/usr/bin/python

import numpy as np
from src.models.StochasticProcess import StochasticProcess


class QuasiGaussianModel(StochasticProcess):

    # Python constructor
    def __init__(self, yieldCurve, d, times, sigma, slope, curve, delta, chi, Gamma):
        self.yieldCurve       = yieldCurve    # initial yield term structure 
        self._d               = d             # specify d-dimensional Brownian motion for x(t)
                                              # we do not use stoch vol z(t) for now
        self._times           = times         # time-grid of left-constant model parameter values
        self._sigma           = sigma         # volatility
        self._slope           = slope         # skew
        self._curve           = curve         # smile
        self._delta           = delta         # maturity of benchmark rates f(t,t+delta_i)
        self._chi             = chi           # mean reversion
        self._Gamma           = Gamma         # (benchmark rate) correlation matrix
        #
        # run some thorough checks on the input parameters...
        #
        self._DfT = np.linalg.cholesky(self._Gamma)
        # [ H^f H^{-1} ] = [ exp{-chi_j*delta_i} ]
        Hf_H_inv = np.exp(np.array([ [ -chi * delta for chi in self._chi ] for delta in self._delta ]))
        self._HHfInv = np.linalg.inv(Hf_H_inv)

    # time-dependent model parameters are assumed (left-) piece-wise constant
    def _idx(self,t):
        return min(np.searchsorted(self._times,t),len(self._times)-1)

    def sigma(self, i, t):
        return self._sigma[i,self._idx(t)]

    def slope(self, i, t):
        return self._slope[i,self._idx(t)]

    def curve(self, i, t):
        return self._curve[i,self._idx(t)]

    # additional model functions

    def G(self, i, t, T):
        return (1.0-np.exp(-self._chi[i]*(T-t)))/self._chi[i]

    def GPrime(self, i, t, T):
        return np.exp(-self._chi[i]*(T-t))

    # path/payoff interfaces

    def zeroBond(self, t, T, X, alias, G=None):
        if t==T:
            return 1.0  # short cut
        if t==0:
            return self.yieldCurve.discount(T)  # short cut, assume x=0, y=0 initial values
        if G is None:
            G = np.array([ self.G(i, t, T) for i in range(self._d) ])
        DF1 = self.yieldCurve.discount(t)
        DF2 = self.yieldCurve.discount(T)
        Gx = 0.0
        for i in range(self._d):  # maybe better use numpy's dot methods and remove state
            Gx += X[i] * G[i]   # s.x(i) * G[i]
        GyG = 0.0
        for i in range(self._d):
            tmp = 0.0
            for j in range(self._d):
                tmp += X[self._d + i*self._d + j] * G[j]   # s.y(i,j) * G[j]
            GyG += G[i]*tmp
        ZCB = DF2 / DF1 * np.exp(-Gx - 0.5*GyG)
        return ZCB


    def numeraire(self, t, X):
        return np.exp(X[-1]) / self.yieldCurve.discount(t)  # depends on stoch process spec

    # process simulation interface
    # X = [ x, y, s ] (y row-wise)

    def size(self):   # dimension of X(t) = [ x, y, s ]
        return self._d + self._d**2 + 1

    def factors(self):   # dimension of W(t)
        return self._d

    def initialValues(self):
        return np.zeros(self.size())
    
    # volatility specification is key to model properties

    # f - f0 for local volatility calculation
    def deltaForwardRate(self, t, T, X, GPrime=None, G=None):
        # we allow GPrime, G to be provided by user to avoid costly exponentials
        if GPrime is None:
            GPrime = np.array([ self.GPrime(i, t, T) for i in range(self._d) ])
        if G is None:
            G = (1.0 - GPrime) / self._chi
        df = 0.0
        for i in range(self._d):   # maybe better use numpy's dot methods and remove state
            tmp = X[i]  # s.x(i)
            for j in range(self._d):
                tmp += X[self._d + i*self._d + j] * G[j]   # s.y(i, j) * G[j]
            df += GPrime[i] * tmp
        return df

    # local volatiity spec, interpreted as diagonal matrix
    def sigma_f(self, t, X):
        res = np.zeros(self._d)
        eps = 1.0e-4   # 1bp lower cutoff for volatility
        maxFactor = 10.0  # 10 sigma upper cut-off
        for k in range(self._d):
            df = self.deltaForwardRate(t, t + self._delta[k], X)
            sigma_ = self.sigma(k,t)
            slope_ = self.slope(k,t)
            curve_ = self.curve(k,t)
            vol = sigma_ + slope_*df + curve_ * (df if (df>0.0) else -df)
            vol = min(max(vol,eps),maxFactor*sigma_)
            res[k] = vol
        return res

    def sigma_xT(self, t, X):
        # tmp = H*Hf^-1 * sigma_f  ... broadcast second dimension of sigma_f
        # res = tmp * Df^T  ... matrix multiplication
        return (self._HHfInv * self.sigma_f(t,X)) @ self._DfT

    # simulate Quasi-Gaussian model as Gaussian model with frozen vol matrix
    def evolve(self, t0, X0, dt, dW, X1):
        # first we may simulate stochastic vol via lognormal approximation (or otherwise)
        # next we need V = z * sigmaxT sigmax
        sigmaxT = self.sigma_xT(t0, X0)
        V = sigmaxT @ sigmaxT.T   # we need V later for x simulation
        # we also calculate intermediate variables; these do strictly depend on X and could be cached as well
        GPrime = np.array([ self.GPrime(i,0,dt) for i in range(self._d) ])
        G      = (1.0 - GPrime) / self._chi
        # now we can calculate y
        chi_i_p_chi_j = np.array([[ chi_i + chi_j for chi_i in self._chi ] for chi_j in self._chi])
        b = V / chi_i_p_chi_j
        y0 = X0[self._d : self._d + self._d**2]  # create a view
        y0.shape = (self._d,self._d)  # re-shape as matrix
        a = y0 - b
        # y[i,j] = a[i,j] exp{-(chi_i + chi_j)(T-t)} + b[i,j]
        for i in range(self._d):
            for j in range(self._d):
                X1[self._d + i*self._d + j] = a[i,j] * GPrime[i] * GPrime[j] + b[i,j]
        # finally we calculate x
        for i in range(self._d):  # E[ x1 | x0 ]
            X1[i] = X0[i]
            for j in range(self._d):
                X1[i] += a[i,j] * G[j]
            X1[i] *= GPrime[i]
            sumB = np.sum(b[i,:])
            X1[i] += sumB *G[i]
        # we overwrite V by the variance
        for i in range(self._d):
            for j in range(self._d):
                V[i,j] *= (1.0 - GPrime[i] * GPrime[j]) / (self._chi[i] + self._chi[j])
        L = np.linalg.cholesky(V)  # maybe better do this in-place
        for i in range(self._d):
            for j in range(self._d):
                X1[i] += L[i,j] * dW[j]  # maybe also exploit that L is lower triangular
        # finally, we need to update s as well
        x0 = np.sum(X0[:self._d])
        x1 = np.sum(X1[:self._d])
        X1[-1] = X0[-1] + 0.5 * (x0 + x1) * dt
        # all done
        return
    
    # the short rate over an integration time period
    # this is required for drift calculation in multi-asset and hybrid models
    def shortRateOverPeriod(self, t0, dt, X0, X1):
        B_d = self.yieldCurve.discount(t0) / self.yieldCurve.discount(t0 + dt)  # deterministic drift part for r_d
        x_av = 0.5 * (np.sum(X0[:self._d]) + np.sum(X1[:self._d]))
        return np.log(B_d) / dt + x_av

    # bond volatility is used in hybrid model vol adjuster

    def zeroBondVolatility(self, t, T):
        # sigma_r^T = sigma_x^T sqrt(z) 
        # sigma_P^T = G(t,T)^T sigma_x^T sqrt(z)
        G = np.array([ self.G(i, t, T) for i in range(self._d) ])
        sigmaxT = self.sigma_xT(t, self.initialValues())  # we approximate at x=0
        return G @ sigmaxT


    def zeroBondVolatilityPrime(self, t, T):
        # we wrap scalar bond volatility into array to allow
        # for generalisation to multi-factor models
        GPrime = np.array([ self.GPrime(i, t, T) for i in range(self._d) ])
        sigmaxT = self.sigma_xT(t, self.initialValues())  # we approximate at x=0
        return GPrime @ sigmaxT

    def stateAliases(self):
        aliases = [ 'x_%d' % i for i in range(self._d) ]
        for i in range(self._d):
            aliases += [ 'y_%d_%d' % (i,j) for j in range(self._d) ]
        aliases += [ 's' ]
        return aliases

    def factorAliases(self):
        aliases = [ 'x_%d' % i for i in range(self._d) ]
        return aliases











