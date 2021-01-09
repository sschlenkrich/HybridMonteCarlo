#!/usr/bin/python

import numpy as np
from hybmc.models.StochasticProcess import StochasticProcess

class MarkovFutureModel(StochasticProcess):
#
# A futures model based on Andersen 2008 (http://ssrn.com/abstract=1138782),
# equ. (10), (11), (12)
#
# We keep notation as close as possible to Quasi-Gaussian model.

    # Python constructor
    def __init__(self, futuresCurve, d, times, sigmaT, chi):
        self.futuresCurve     = futuresCurve  # initial futures term structure; providing a method future(T) representing F(0,T) 
        self._d               = d             # specify d-dimensional Brownian motion for x(t)
        self._times           = times         # time-grid of left-constant model (i.e. volatility) parameter values
        self._sigmaT          = sigmaT        # volatility of x
        self._chi             = chi           # mean reversion
        self._y               = np.zeros([self._times.shape[0],self._d,self._d])
        for k, t1 in enumerate(self._times):
            self._y[k] = self.y(t1)

    # time-dependent model parameters are assumed (left-) piece-wise constant
    def _idx(self,t):
        return min(np.searchsorted(self._times,t),len(self._times)-1)

    def sigmaT(self, t):
        return self._sigmaT[self._idx(t)]

    def covariance(self, t, T):
        sigmaT = self.sigmaT(0.5*(t+T)) # assume sigma^T constant on (t,T)
        M = sigmaT @ sigmaT.T
        h = np.exp(-self._chi*(T-t))
        #display(M,h)
        for i in range(self._d):
            for j in range(self._d):
                chi_ij = self._chi[i] + self._chi[j]
                if np.abs(chi_ij) < 1.0e-6:  # avoid division by zero
                    a = chi_ij * (T - t)
                    delta = (T-t) * (1.0 - a/2.0*(1.0 - a/3.0))
                else:
                    delta = (1.0 - h[i]*h[j]) / chi_ij
                M[i,j] *= delta
        return M, h
        
    def y(self,t):
        # find idx s.t. t[idx-1] < t <= t[idx]
        idx = np.searchsorted(self._times,t)
        t0 = 0.0                         if idx==0 else self._times[idx-1]
        y0 = np.zeros([self._d,self._d]) if idx==0 else self._y[idx-1]
        #
        M, h = self.covariance(t0,t)
        y1 = np.zeros([self._d,self._d])
        for i in range(self._d):
            for j in range(self._d):
                y1[i,j] = h[i] * y0[i,j] * h[j] + M[i,j]
        return y1

    # path/payoff interfaces

    def futurePrice(self, t, T, X, alias):
        # F(t,T) = F(0,T) * exp{ ... }
        # [...]  = h(t,T)^T [0.5 y(t) (1-h(t,T)) + x(t) ]
        h = np.exp(-self._chi*(T-t))
        yt = self.y(t)
        x2 = 0.5 * (yt @ (1.0 - h)) + X
        XtT = np.dot(h,x2)
        FtT = np.exp(XtT) # do not forget F(0,T)
        # FtT *= futuresCurve.future(T)
        return FtT


    # process simulation interface
    # X = [ x ] (y calculated, no numeraire)

    def size(self):   # dimension of X(t) = [ x ]
        return self._d

    def factors(self):   # dimension of W(t)
        return self._d

    def initialValues(self):
        return np.zeros(self.size())

    # simulate Markov model state variable
    def evolve(self, t0, X0, dt, dW, X1):
        h = np.exp(-self._chi*dt)
        G = np.zeros(self._d)
        for i in range(self._d):
            if np.abs(self._chi[i]) < 1.0e-6:  # avoid division by zero
                a = self._chi[i] * dt
                G[i] = dt * (1.0 - a/2.0*(1.0 - a/3.0))
            else:
                G[i] = (1.0 - h[i])/self._chi[i]
        y0 = self.y(t0)
        cov = self.y(t0+dt)
        for i in range(self._d):
            for j in range(self._d):
                cov[i,j] -= h[i] * y0[i,j] * h[j]
        L = np.linalg.cholesky(cov)  # maybe better do this in-place                
        # co-variance matrix could be cached
        # E[ x1 | x0 ] = H(t0,t1) x0 + \int_t0^t1 H(u,t1)\theta(u)du
        #
        # H(t0,t1) x0
        for i in range(self._d):
            X1[i] = h[i] * X0[i]
        # \int_t0^t1 H(u,t1) y(u)
        sigmaT = self.sigmaT(t0+0.5*dt) # assume sigma constant on (t,T)
        V = sigmaT @ sigmaT.T
        # E = H y G + [G B - H B G]
        # H y G
        E0 = np.zeros([self._d,self._d])
        for j in range(self._d):
            for i in range(self._d):
                E0[i,j] = h[i] * y0[i,j] * G[j]
        # G B - H B G
        C = np.zeros([self._d,self._d])
        for i in range(self._d):
            for j in range(self._d):
                if i==j:
                    C[i,j] = 0.5 * G[i]*G[j]
                else:
                    C[i,j] = (G[i] - h[i] * G[j]) / (self._chi[i] + self._chi[j])  # this fails if chi_i = -chi_j
                C[i,j] *= V[i,j]
        #
        E = E0 + C
        # \int_t0^t1 H(u,t1)\theta(u)du = 0.5 * [ E \chi 1 - G V 1 ]
        I = 0.5 * ( E @ self._chi - G * np.sum(V,axis=1) )
        # E[ x1 | x0 ] = H(t0,t1) x0 + I
        for i in range(self._d):
            X1[i] += I[i]
        #
        # add diffusion term        
        for i in range(self._d):
            for j in range(self._d):
                X1[i] += L[i,j] * dW[j]  # maybe also exploit that L is lower triangular
        # done
        







