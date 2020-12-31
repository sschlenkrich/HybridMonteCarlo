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
    def __init__(self, futuresCurve, d, times, sigma, chi):
        self.futuresCurve     = futuresCurve  # initial futures term structure; providing a method future(T) representing F(0,T) 
        self._d               = d             # specify d-dimensional Brownian motion for x(t)
        self._times           = times         # time-grid of left-constant model (i.e. volatility) parameter values
        self._sigma           = sigma         # volatility of x
        self._chi             = chi           # mean reversion
        self._y               = np.zeros([self._times.shape[0],self._d,self._d])
        # t0 = 0.0
        # y0 = np.zeros([self._d,self._d])
        for k, t1 in enumerate(self._times):
            # h = np.exp(-self._chi*(t1-t0))
            # V = self._sigma[k].T @ self._sigma[k]
            # # isplay(V,h)
            # for i in range(self._d):
            #     for j in range(self._d):
            #         chi_ij = self._chi[i] + self._chi[j]
            #         if np.abs(chi_ij) < 1.0e-6:
            #             dt = t1 - t0
            #             delta = dt - 0.5*chi_ij*dt**2 + 1.0/6.0*chi_ij*dt**3
            #         else:
            #             delta = (1.0 - h[i]*h[j]) / chi_ij
            #         self._y[k,i,j] = h[i] * y0[i,j] * h[j] + V[i,j] * delta
            # t0 = t1
            # y0 = self._y[k]
            self._y[k] = self.y(t1)

    # auxilliary methods

    #def G(self, i, t, T):
    #    return (1.0 - np.exp(-self._chi[i]*(T-t))) / self._chi[i]
    
    #def GPrime(self, i, t, T):
    #    return np.exp(-self._chi[i]*(T-t))

    # time-dependent model parameters are assumed (left-) piece-wise constant
    def _idx(self,t):
        return min(np.searchsorted(self._times,t),len(self._times)-1)

    def sigma(self, t):
        return self._sigma[self._idx(t)]    

    def covariance(self, t, T):
        sigma = self.sigma(0.5*(t+T)) # assume sigma constant on (t,T)
        M = sigma.T @ sigma  
        h = np.exp(-self._chi*(T-t))
        #display(M,h)
        for i in range(self._d):
            for j in range(self._d):
                chi_ij = self._chi[i] + self._chi[j]
                if np.abs(chi_ij) < 1.0e-6:
                    dt = T - t
                    delta = dt - 0.5*chi_ij*dt**2 + 1.0/6.0*chi_ij*dt**3
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
        sigma = self.sigma(t0+0.5*dt) # assume sigma constant on (t,T)
        V = sigma.T @ sigma
        chi_i_p_chi_j = np.array([[ chi_i + chi_j for chi_i in self._chi ] for chi_j in self._chi])
        B = V / chi_i_p_chi_j  #  this fails if chi_i + chi_j == 0 (or small)
        A = y0 - B
        # E = H A G + G B
        E = np.zeros([self._d,self._d])
        for i in range(self._d):
            for j in range(self._d):
                E[i,j] = h[i] * A[i,j] * (1.0 - h[j])/self._chi[j] + (1.0 - h[i])/self._chi[i] * B[i,j]
        # \int_t0^t1 H(u,t1)\theta(u)du = 0.5 * [ E \chi 1 - G V 1 ]
        I = 0.5 * ( E @ self._chi - ((1.0 - h)/self._chi) * np.sum(V,axis=1) )
        # E[ x1 | x0 ] = H(t0,t1) x0 + I
        for i in range(self._d):
            X1[i] += I[i]
        #
        # add diffusion term        
        for i in range(self._d):
            for j in range(self._d):
                X1[i] += L[i,j] * dW[j]  # maybe also exploit that L is lower triangular
        # done
        







