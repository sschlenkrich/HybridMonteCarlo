#!/usr/bin/python

import numpy as np

from src.simulations.Payoffs import Payoff
from src.mathutils.Regression import Regression


class AmcPayoff(Payoff):
    """
    Approximate conditional expectation via linear regression.
    We calculate E[ f(X,Y) | t=obsTime ] using a regression
    operator R[X,Y](Z).

    X, Y are random variables calculated as sum of discounted
    payoffs
      X = B(t) sum(x_i/B_i)  and  Y = B(t) sum(y_i/B_i).
    Y may be None.
    
    Z is a list of (un-discounted) payoffs used as control
    variables for regression. Z should only incorporate
    information available at observation time t.

    Function f() is implemented in derived classes to.
    Important instances are
      f(X,Y) = max(X,Y) = 1_{X>Y} X + (1-1_{X>Y}) Y
      f(X,Y) = min(X,Y) = 1_{X>Y} Y + (1-1_{X>Y}) X
      f(X,Y) = 1_{X>Y}
      f(X)   = X

    """

    # Python constructor
    def __init__(self, obsTime, x, y, z, simulation, maxPolynDegree):
        Payoff.__init__(self, obsTime)   # the time at which we aim to calculate the conditional expecttion
        self.x = x                       # a list of payoffs, first argument of f() 
        self.y = y                       # a list of payoffs, second argument of f()
        self.z = z                       # a list of payoffs used as control variables
        self.simulation = simulation           # a monte carlo simulation for regression calibration
        self.maxPolynDegree = maxPolynDegree   # maximum degree of monomials used for regression
        self.regression = None           # initial state, we use lazy calibration

    def f(self, X, Y, trigger, p):
        # a function calculating f(X,Y;T), this is defined in derived classes
        raise ValueError('Implementation of payoff funtion f() required.')

    def calibrateRegression(self):
        if self.regression is None and \
           self.z is not None and      \
           self.simulation is not None :  # only in this case we calculate the regression
            T = np.zeros(self.simulation.nPaths)  # the actual trigger used for regression
            Z = np.zeros([self.simulation.nPaths,len(self.z)])
            for k in range(self.simulation.nPaths):
                p = self.simulation.path(k)
                numeraire = p.numeraire(self.obsTime)
                for x in self.x:
                    T[k] += x.discountedAt(p)
                if self.y is not None:  # we want to allow calibration with single argument
                    for y in self.y:
                        T[k] -= y.discountedAt(p)
                T[k] *= numeraire
                for i, z in enumerate(self.z):
                    Z[k,i] = z.at(p)
            self.regression = Regression(Z,T,self.maxPolynDegree)
        # now we come to the actual payoff calculation

    def at(self, p):
        self.calibrateRegression()  # lazy call
        if self.regression is not None:
            Z = np.array([ z.at(p) for z in self.z ])
            trigger = self.regression.value(Z)  # estimate for X-Y
            return self.f(None,None,trigger,p)
        # if there is no regression we look into the future
        X = 0.0
        Y = 0.0
        numeraire = p.numeraire(self.obsTime)
        for x in self.x:
            X += x.discountedAt(p)
        X *= numeraire
        if self.y is not None:  # we want to allow payoff function with single argument
            for y in self.y:
                Y += y.discountedAt(p)
            Y *= numeraire
        trigger = X - Y
        return self.f(X,Y,trigger,None)

    def __str__(self):
        text = '%.2f,[' % self.obsTime
        for x in self.x:
            text += str(x) + ','
        text = text[:-1] + ']'
        if self.y is not None:
            text += ',['
            for y in self.y:
                text += str(y) + ','
        text = text[:-1] + ']'
        if self.regression is None and \
           self.z is not None and      \
           self.simulation is not None :  # only in this case we calculate the regression
            text += ';['
            for z in self.z:
                text += str(z) + ','
        text = text[:-1] + ']'
        return text


class AmcMax(AmcPayoff):

    # Python constructor
    def __init__(self, obsTime, x, y, z, simulation, maxPolynDegree):
        AmcPayoff.__init__(self, obsTime, x, y, z, simulation, maxPolynDegree)

    def f(self, X, Y, trigger, p):
        if p is not None:  # we need to calculate from path
            X = 0.0  # this shadows input X
            Y = 0.0  # this shadows input Y
            if trigger>0.0: # we we only calculate one branch
                for x in self.x:
                    X += x.discountedAt(p)
                X *= p.numeraire(self.obsTime)
            else:
                for y in self.y:
                    Y += y.discountedAt(p)
                Y *= p.numeraire(self.obsTime)
        return X if trigger>0.0 else Y

    def __str__(self):
        return 'AmcMax(%s)' % super().__str__()


class AmcMin(AmcPayoff):

    # Python constructor
    def __init__(self, obsTime, x, y, z, simulation, maxPolynDegree):
        AmcPayoff.__init__(self, obsTime, x, y, z, simulation, maxPolynDegree)

    def f(self, X, Y, trigger, p):
        if p is not None:  # we need to calculate from path
            X = 0.0  # this shadows input X
            Y = 0.0  # this shadows input Y
            if trigger<0.0: # we we only calculate one branch
                for x in self.x:
                    X += x.discountedAt(p)
                X *= p.numeraire(self.obsTime)
            else:
                for y in self.y:
                    Y += y.discountedAt(p)
                Y *= p.numeraire(self.obsTime)
        return X if trigger<0.0 else Y

    def __str__(self):
        return 'AmcMin(%s)' % super().__str__()


class AmcOne(AmcPayoff):

    # Python constructor
    def __init__(self, obsTime, x, y, z, simulation, maxPolynDegree):
        AmcPayoff.__init__(self, obsTime, x, y, z, simulation, maxPolynDegree)

    def f(self, X, Y, trigger, p):
        # no need to calculate any payoffs
        return 1.0 if trigger>0.0 else 0.0

    def __str__(self):
        return 'AmcOne(%s)' % super().__str__()


class AmcSum(AmcPayoff):

    # Python constructor
    def __init__(self, obsTime, x, z, simulation, maxPolynDegree):
        AmcPayoff.__init__(self, obsTime, x, None, z, simulation, maxPolynDegree)

    def f(self, X, Y, trigger, p):
        return trigger   # trigger is (estimate of) X if y in None

    def __str__(self):
        return 'AmcSum(%s)' % super().__str__()
