#!/usr/bin/python


import numpy as np
from hybmc.models.StochasticProcess import StochasticProcess


class AndersenFutureModel(StochasticProcess):

    # Python constructor
    def __init__(self, futuresCurve, kappa, sigma_0, sigma_infty, rho_infty):
        self._futuresCurve = futuresCurve
        self._kappa        = kappa
        self._sigma_0      = sigma_0
        self._sigma_infty  = sigma_infty
        self._rho_infty    = rho_infty
        #

    # model parameter functions

    def h1(self, t):
        return -self._sigma_infty + self._rho_infty * self._sigma_0

    def h2(self, t):
        return self._sigma_0 * np.sqrt(1.0 - self._rho_infty**2)

    def hInfty(self):
        return self._sigma_infty

    def covariance(self, t, T):
        h1_ = self.h1(0.5*(t+T))
        h2_ = self.h2(0.5*(t+T))
        hi_ = self.hInfty()
        m11 = (1.0 - np.exp(-2*self._kappa*(T-t)))/2.0/self._kappa * (h1_*h1_ + h2_*h2_)
        m12 = (1.0 - np.exp(-  self._kappa*(T-t)))/self._kappa * h1_ * hi_
        m22 = (T-t) * hi_ * hi_
        return np.array([ [m11, m12], [m12, m22] ])

    def y(self, t):
        # simple if no time-dependence
        return self.covariance(0.0, t)

    # path/payoff interfaces

    def futurePrice(self, t, T, X, alias):
        # F(t,T) = F(0,T) * exp{ ... }
        # [...]  = h(t,T)^T [z(t) - 0.5 y(t) h(t,T) ]  ... this is wrong
        h = np.array([ np.exp(-self._kappa*(T-t)), 1.0 ])
        yt = self.y(t)
        x2 = X - 0.5 * (yt @ h)
        XtT = np.dot(h,x2)
        FtT = np.exp(XtT) # do not forget F(0,T)
        # FtT *= futuresCurve.future(T)
        return FtT

    # process simulation interface
    # X = [ z1, z2 ] (y calculated, no numeraire)

    def size(self):   # dimension of X(t) = [ x ]
        return 2

    def factors(self):   # dimension of W(t)
        return 2

    def initialValues(self):
        return np.zeros(2)

    def evolve(self, t0, X0, dt, dW, X1):
        cov = self.covariance(t0,t0+dt)
        L = np.linalg.cholesky(cov)
        H = np.array([ np.exp(-self._kappa*dt), 1.0 ])
        G = np.array([ (1.0 - np.exp(-self._kappa*dt))/self._kappa, dt ])
        theta = np.zeros(2)  # incorporate quanto adjustment here
        # H(t0,t1) x0 + G(t0,t1) theta
        for i in range(2):
            X1[i] = H[i] * X0[i] + G[i] * theta[i]
        # add diffusion term
        for i in range(2):
            for j in range(2):
                X1[i] += L[i,j] * dW[j]  # maybe also exploit that L is lower triangular
        # done
