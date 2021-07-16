#!/usr/bin/python

import numpy as np

from hybmc.simulations.Payoffs import Payoff, Cache
from hybmc.mathutils.Regression import Regression


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
        #
        self.withTrigger = False   # we want to allow accessing result AND exercise decision trigger

    def f(self, X, Y, trigger, p):
        # a function calculating f(X,Y;T), this is defined in derived classes
        raise NotImplementedError('Implementation of payoff funtion f() required.')

    def __useRegression(self):  # we need this condition several times
        return self.regression is None and \
           self.z is not None and          \
           self.simulation is not None

    def calibrateRegression(self):
        if self.__useRegression() :  # only in this case we calculate the regression
            idx = np.array([ True for k in range(self.simulation.nPaths) ])  # reference all paths
            p = self.simulation.path(idx)  # all paths
            # first calculate trigger
            T = 0.0
            for x in self.x:
                T += x.discountedAt(p)
            if self.y is not None:  # we want to allow calibration with single argument
                for y in self.y:
                    T -= y.discountedAt(p)
            T *= p.numeraire(self.obsTime)
            Z = np.zeros([len(self.z), self.simulation.nPaths])
            for i, z in enumerate(self.z):
                Z[i] = z.at(p)
            self.regression = Regression(Z.transpose(),T,self.maxPolynDegree)
        # now we come to the actual payoff calculation

    def at(self, p):
        self.calibrateRegression()  # lazy call
        if self.regression is not None:
            Z = np.array([ z.at(p) for z in self.z ])
            trigger = self.regression.value(Z)  # estimate for X-Y
            res = self.f(None,None,trigger,p)
            if self.withTrigger:
                return (res, trigger)
            return res
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
        res = self.f(X,Y,trigger,None)
        if self.withTrigger:
            return (res, trigger)
        return res

    def observationTimes(self):
        obsTimes = { self.obsTime }
        for x in self.x:
            obsTimes = obsTimes.union(x.observationTimes())
        if self.y is not None:  # we want to allow payoff function with single argument
            for y in self.y:
                obsTimes = obsTimes.union(y.observationTimes())
        if self.__useRegression():
            for z in self.z:
                obsTimes = obsTimes.union(z.observationTimes())
        return obsTimes

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
        if self.__useRegression():  # only in this case we calculate the regression
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
            if (not isinstance(trigger, float)) or (trigger>0.0): # we we only calculate one branch
                for x in self.x:
                    X += x.discountedAt(p)
                X *= p.numeraire(self.obsTime)
            if (not isinstance(trigger, float)) or (trigger<=0.0): # we we only calculate one branch
                for y in self.y:
                    Y += y.discountedAt(p)
                Y *= p.numeraire(self.obsTime)
        #return X if trigger>0.0 else Y
        rho = (trigger>0.0)
        return rho * X + (1.0 - rho) * Y

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
            if (not isinstance(trigger, float)) or (trigger<0.0): # we we only calculate one branch
                for x in self.x:
                    X += x.discountedAt(p)
                X *= p.numeraire(self.obsTime)
            if (not isinstance(trigger, float)) or (trigger>=0.0): # we we only calculate one branch
                for y in self.y:
                    Y += y.discountedAt(p)
                Y *= p.numeraire(self.obsTime)
        #return X if trigger<0.0 else Y
        rho = (trigger<0.0)
        return rho * X + (1.0 - rho) * Y

    def __str__(self):
        return 'AmcMin(%s)' % super().__str__()


class AmcOne(AmcPayoff):

    # Python constructor
    def __init__(self, obsTime, x, y, z, simulation, maxPolynDegree):
        AmcPayoff.__init__(self, obsTime, x, y, z, simulation, maxPolynDegree)

    def f(self, X, Y, trigger, p):
        # no need to calculate any payoffs
        #return 1.0 if trigger>0.0 else 0.0
        return 1.0 * (trigger>0.0)

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


class AmcElement(Payoff):
    """
    Extract the value or trigger from an AmcPayoff.

    This class requires a cached AMC payoff as input.
    """

    # Python constructor
    def __init__(self, x, element='VALUE'):
        Payoff.__init__(self, x.obsTime)
        assert isinstance(x, Cache), 'Input payoff must be Cache.'
        assert isinstance(x.x, AmcPayoff), 'Cached input must be AmcPayoff'
        x.x.withTrigger = True  # make sure we do calculate value AND trigger
        self.x = x
        #
        assert isinstance(element, str), 'Element string required'
        self.idx = -1
        if element.upper()=='VALUE':
            self.idx = 0
        if element.upper()=='TRIGGER':
            self.idx = 1
        assert self.idx >= 0, 'Unknown element specifier.'

    def at(self, p):
        return self.x.at(p)[self.idx]

    def __str__(self):
        if self.idx == 0: # pass on value 
            return self.x.__str__()
        if self.idx == 1:
            return 'AmcTrigger(%s)' % self.x.__str__()
        return 'Unknown'