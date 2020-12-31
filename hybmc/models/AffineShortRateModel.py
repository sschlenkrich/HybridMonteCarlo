#!/usr/bin/python


import numpy as np
from scipy.integrate import ode
from scipy.stats import norm
from hybmc.models.StochasticProcess import StochasticProcess


# We implement three different approaches to evolve a sqrt-process.
# Details follow Andersen/Piterbarg 2010, sec. 9.5

# Andersen/Piterbarg 2010, Corollary 8.3.3; BUT beware the notation
def cirMoments(r0,dt,chi,theta,sigma):
    # E[r1 | r0]
    m  = theta + (r0 - theta) * np.exp(-chi*dt)
    # Var[r1 |r0]
    tmp0 = np.exp(-chi*dt)
    tmp1 = 1.0 - tmp0
    tmp2 = sigma * sigma / chi
    s2   = r0*tmp2*tmp0*tmp1 + theta*tmp2/2.0*tmp1*tmp1
    return (m,s2)        

# Andersen/Piterbarg 2010, sec. 9.5.2.1
def fullTruncation(r0, m, s2, dw):
    if r0<0.0:
        return m
    return m + np.sqrt(s2)*dw

# Andersen/Piterbarg 2010, sec. 9.5.3.1
def lognormalApproximation(r0, m, s2, dw):
    si2 = np.log(1.0 + s2 / m / m)
    si  = np.sqrt(si2)
    # mu = np.log(m) - si*si / 2.0
    r1 = m * np.exp(-si2/2.0 + si*dw)
    return r1

# Andersen/Piterbarg 2010, sec. 9.5.3.4
class quadraticExponential:
    def __init__(self, psi_c = 1.5):
        self.psi_c = psi_c
    def __call__(self, r0, m, s2, dw):
        psi = s2/m/m
        if psi <= self.psi_c:
            b2 = 2.0/psi - 1.0 + np.sqrt(2.0/psi) * np.sqrt(2.0/psi-1)
            a  = m / (1.0 + b2)
            b  = np.sqrt(b2)
            return a * (b + dw)**2
        else:
            p = (psi-1.0) / (psi + 1.0)
            beta = (1.0 - p) / m
            U = norm.cdf(dw)
            if U <= p:
                return 0.0
            else:
                return np.log((1.0-p)/(1.0-U)) / beta
        return 0.0 # this should never be reached



class AffineShortRateModel(StochasticProcess):
#
# dr(t) = chi(t) [theta(t) - r(t)] dt + sigma(t) sqrt{ alpha(t) + beta(t)r(t) }
#
# See Andersen/Piterbarg 2010, sec. 10.2.1.4
#

    # Python constructor
    def __init__(self, r0, modelTimes, chi, theta, sigma, alpha, beta, sqrtEvolve=lognormalApproximation):
        self.r0           = r0
        self.modelTimes_  = modelTimes    # assume positive and ascending
        self.chi_         = chi
        self.theta_       = theta
        self.sigma_       = sigma
        self.alpha_       = alpha
        self.beta_        = beta
        # check dimensions
        if not self.modelTimes_.shape == self.chi_.shape:
            raise IndexError('AffineShortRateModel: Dimension mismatch.')
        if not self.modelTimes_.shape == self.theta_.shape:
            raise IndexError('AffineShortRateModel: Dimension mismatch.')
        if not self.modelTimes_.shape == self.sigma_.shape:
            raise IndexError('AffineShortRateModel: Dimension mismatch.')
        if not self.modelTimes_.shape == self.alpha_.shape:
            raise IndexError('AffineShortRateModel: Dimension mismatch.')
        if not self.modelTimes_.shape == self.beta_.shape:
            raise IndexError('AffineShortRateModel: Dimension mismatch.')
        # select sqrt-process evolution method
        self.sqrtEvolve = sqrtEvolve

    # piece-wise constant model functions

    def idx(self, t):
        return min(np.searchsorted(self.modelTimes_,t), self.modelTimes_.shape[0] - 1)

    def chi(self, t):
        return self.chi_[self.idx(t)]

    def theta(self, t):
        return self.theta_[self.idx(t)]

    def sigma(self, t):
        return self.sigma_[self.idx(t)]

    def alpha(self, t):
        return self.alpha_[self.idx(t)]

    def beta(self, t):
        return self.beta_[self.idx(t)]
    
    # model functions

    def ricattiAB(self, t, T, c1, c2):
    # solve the Ricatti ODE system for A(t,T) and B(t,T)
    # we set Y = [A,B]
        f = lambda t,Y : np.array([                                                            \
            self.chi(t)*self.theta(t)*Y[1] - 0.5*(self.sigma(t)**2)*self.alpha(t)*Y[1]**2,     \
            self.chi(t)              *Y[1] + 0.5*(self.sigma(t)**2)*self.beta(t) *Y[1]**2 - c2 ])
        J = lambda t,Y : np.array([                                                            \
            [ 0.0, self.chi(t)*self.theta(t) - (self.sigma(t)**2)*self.alpha(t)*Y[1] ],        \
            [ 0.0, self.chi(t)               + (self.sigma(t)**2)*self.beta(t) *Y[1] ]         ])
        ricatti = ode(f,J)
        ricatti.set_initial_value(np.array([0,c1]),T)
        AB = ricatti.integrate(t)
        if not ricatti.successful():
            print('ODE error code %s' % str(ricatti.get_return_code()))
        return AB

    def extendedTransform(self, t, rt, T, c1, c2):
        AB = self.ricattiAB(t, T, c1, c2)
        return np.exp(AB[0] - AB[1]*rt)

    def zeroBondPrice(self, t, T, rt):
        return self.extendedTransform(t,rt,T,0.0,1.0)

    # stochastic process interface
    
    def size(self):   # dimension of X(t)
        return 2      # [r, s], we also need for numeraire s = \int_0^t r dt

    def factors(self):   # dimension of W(t)
        return 1

    def initialValues(self):
        return np.array([ self.r0, 0.0 ])
    
    def zeroBond(self, t, T, X, alias):
        return self.zeroBondPrice(t, T, X[0])

    def numeraire(self, t, X):
        return np.exp(X[1])

    # evolve X(t0) -> X(t0+dt) using independent Brownian increments dW
    # t0, dt are assumed float, X0, X1, dW are np.array
    def evolve(self, t0, X0, dt, dW, X1):
        # firt apply transformation y = alpha + beta r
        # then
        #   dy = chi [(alpha + beta theta) - y] dt + (beta sigma) sqrt{y} dW
        t = t0 + 0.5*dt  # calculate parameters via 'mid-point rule' 
        y0    = self.alpha(t) + self.beta(t) * X0[0]
        chi   = self.chi(t)
        theta = self.alpha(t) + self.beta(t) * self.theta(t)
        sigma = self.beta(t) * self.sigma(t)
        # evolve is based on first and second moments
        (expY, varY) = cirMoments(y0,dt,chi,theta,sigma)
        y1 = self.sqrtEvolve(y0,expY,varY,dW[0])
        # inverse transformation
        X1[0] = (y1 - self.alpha(t)) / self.beta(t)
        X1[1] = X0[1] + (X0[0] + X1[0]) * dt / 2
        return

    # the short rate over an integration time period
    # this is required for drift calculation in multi-asset and hybrid models
    def shortRateOverPeriod(self, t0, dt, X0, X1):
        return 0.5 * (X0[0] + X1[0])

    # keep track of components in hybrid model

    def stateAliases(self):
        return [ 'r', 's' ]

    def factorAliases(self):
        return [ 'r' ]

class CoxIngersollRossModel(AffineShortRateModel):
#
# dr(t) = chi [theta - r(t)] dt + sigma sqrt{ r(t) }
#
# See Andersen/Piterbarg 2010, sec. 10.2.1.4
#

    # Python constructor
    def __init__(self, r0, chi, theta, sigma, sqrtEvolve=lognormalApproximation):
        super().__init__(r0,np.array([0.0]),np.array([chi]),np.array([theta]),np.array([sigma]),np.array([0.0]),np.array([1.0]),sqrtEvolve)

    def ricattiAB(self, t, T, c1, c2):
    # solve the Ricatti ODE system for A(t,T) and B(t,T)
    # exact solution from Prop. 10.2.4 (with typo fixed)
        chi   = self.chi(0.0)
        theta = self.theta(0.0)
        sigma = self.sigma(0.0)
        dt    = T - t
        # we need gamma for A_CIR and B_CIR
        gamma = np.sqrt((chi*chi + 2.0 * sigma*sigma * c2))
        # A_CIR
        t1 = chi*theta / sigma / sigma * (chi + gamma)*dt
        t2 = 1.0 + (chi + gamma + c1 * sigma*sigma) * (np.exp(gamma * dt) - 1.0) / 2.0 / gamma
        if not t2 > 0.0:
            raise ArithmeticError('t2 > 0.0 required')
        A_CIR = t1 - 2.0*chi*theta / sigma / sigma*np.log(t2)
        # B_CIR
        emGdt = np.exp(-gamma * dt)
        numer = (2.0*c2 - chi*c1)*(1.0 - emGdt) + gamma*c1*(1.0 + emGdt)
        denum = (chi + gamma + c1*sigma*sigma) * (1.0 - emGdt) + 2.0*gamma*emGdt
        B_CIR = numer / denum
        return np.array([A_CIR, B_CIR])
