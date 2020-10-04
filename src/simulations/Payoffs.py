#!/usr/bin/python

import numpy as np

class Payoff:

    # Python constructor
    def __init__(self, obsTime):
        self.obsTime = obsTime

    def at(self, p):
        raise NotImplementedError('Implementation of method at() required.')

    def discountedAt(self, p):
        return self.at(p) / p.numeraire(self.obsTime)
        
    def observationTimes(self):
        return { self.obsTime }  # default implementation

    # simplify payoff scripting

    # normal operators

    def __add__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Axpy(1.0,self,other)

    def __sub__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Axpy(-1.0,other,self)
    
    def __mul__(self, other):
        if isinstance(other,(int,float)):
            return Axpy(other,self,None)
        return Mult(self,other)
    
    def __truediv__(self, other):
        if isinstance(other,(int,float)):
            return Axpy(1.0/other,self,None)
        return Div(self,other)

    # reflective operators

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Axpy(-1.0,self,other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Div(other,self)

    # logical operators

    def __lt__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Logical(self,other,'<')

    def __le__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Logical(self,other,'<=')

    def __eq__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Logical(self,other,'==')

    def __ne__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Logical(self,other,'!=')

    def __ge__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Logical(self,other,'>=')

    def __gt__(self, other):
        if isinstance(other,(int,float)):
            other = Fixed(other)
        return Logical(self,other,'>')

    # set pay time

    def __matmul__(self, other):
        # better check that other is a float
        return Pay(self,other)

    # use payoff as function

    def __call__(self, obsTime):
        raise NotImplementedError('Implementation of call operator () required.')


# basic payoffs

class Fixed(Payoff):
    def __init__(self, x):
        Payoff.__init__(self, 0.0)
        self.x = x
    def at(self, p):
        return self.x
    def __str__(self):
        return '%.4f' % self.x

class Pay(Payoff):
    def __init__(self, x, payTime):
        Payoff.__init__(self, payTime)
        self.x = x
    def at(self, p):
        return self.x.at(p)
    def observationTimes(self):
        return self.x.observationTimes().union({ self.obsTime })
    def __str__(self):
        return '(%s @ %.2f)' % (self.x,self.obsTime)

class Asset(Payoff):
    def __init__(self, t, alias=None):
        Payoff.__init__(self, t)
        self.alias = alias
    def at(self, p):
        return p.asset(self.obsTime,self.alias)
    def __str__(self):
        return '%s(%.2f)' % (self.alias,self.obsTime)

# basic rates payoffs

class ZeroBond(Payoff):
    def __init__(self, t, T, alias=None):
        Payoff.__init__(self, t)
        self.payTime = T
        self.alias = alias
    def at(self, p):
        return p.zeroBond(self.obsTime,self.payTime,self.alias)
    def __str__(self):
        return 'P_%s(%.2f,%.2f)' % (self.alias,self.obsTime,self.payTime)

class LiborRate(Payoff):
    def __init__(self, obsTime, startTime, endTime, yearFraction=None, tenorBasis = 1.0, alias=None):
        Payoff.__init__(self, obsTime)
        self.startTime = startTime
        self.endTime = endTime
        self.yearFraction = yearFraction if yearFraction is not None else (self.endTime - self.startTime)
        self.tenorBasis = tenorBasis
        self.alias = alias
    def at(self, p):
        pStart = p.zeroBond(self.obsTime,self.startTime,self.alias)
        pEnd   = p.zeroBond(self.obsTime,self.endTime,self.alias)
        return (pStart/pEnd*self.tenorBasis - 1.0)/self.yearFraction
    def __str__(self):
        return 'L_%s(%.2f;%.2f,%.2f)' % (self.alias,self.obsTime,self.startTime,self.endTime)

class SwapRate(Payoff):
    def __init__(self, obsTime, floatTimes, floatWeights, fixedTimes, fixedWeights, alias=None):
        Payoff.__init__(self, obsTime)
        self.floatTimes = floatTimes
        self.floatWeights = floatWeights
        self.fixedTimes = fixedTimes
        self.fixedWeights = fixedWeights
        self.alias = alias
    def at(self, p):
        num = sum([ w*p.zeroBond(self.obsTime,T,self.alias) for w,T in zip(self.floatWeights,self.floatTimes) ])
        den = sum([ w*p.zeroBond(self.obsTime,T,self.alias) for w,T in zip(self.fixedWeights,self.fixedTimes) ])
        return num / den
    def __str__(self):
        return 'S_%s(%.2f;%.2f,%.2f)' % (self.alias,self.obsTime,self.floatTimes[0],self.floatTimes[-1])

# arithmetic operations

class Axpy(Payoff):
    def __init__(self, a, x, y=None):
        Payoff.__init__(self, 0.0)
        self.a = a
        self.x = x
        self.y = y
    def at(self, p):
        res = self.a * self.x.at(p)
        if self.y:
            res += self.y.at(p)
        return res
    def observationTimes(self):
        if self.y:
            return self.x.observationTimes().union(self.y.observationTimes())
        return self.x.observationTimes()
    def __str__(self):
        if self.y is None:
            return '%.4f %s' % (self.a,self.x)
        if self.a == 1.0:
            return '(%s + %s)' % (self.x,self.y)
        if self.a == -1.0:
            return '(%s - %s)' % (self.y,self.x)
        return '(%.4f %s + %s)' % (self.a,self.x,self.y)
    
class Mult(Payoff):
    def __init__(self, x, y):
        Payoff.__init__(self, 0.0)
        self.x = x
        self.y = y
    def at(self, p):
        return self.x.at(p) * self.y.at(p)
    def observationTimes(self):
        return self.x.observationTimes().union(self.y.observationTimes())
    def __str__(self):
        return '%s %s' % (self.x,self.y)
    
class Div(Payoff):
    def __init__(self, x, y):
        Payoff.__init__(self, 0.0)
        self.x = x
        self.y = y
    def at(self, p):
        return self.x.at(p) / self.y.at(p)
    def observationTimes(self):
        return self.x.observationTimes().union(self.y.observationTimes())
    def __str__(self):
        return '%s / %s' % (self.x,self.y)

class Max(Payoff):
    def __init__(self, x, y):
        Payoff.__init__(self, 0.0)
        self.x = x
        self.y = y
    def at(self, p):
        return max(self.x.at(p), self.y.at(p))
    def observationTimes(self):
        return self.x.observationTimes().union(self.y.observationTimes())
    def __str__(self):
        return 'Max(%s, %s)' % (self.x,self.y)

class Min(Payoff):
    def __init__(self, x, y):
        Payoff.__init__(self, 0.0)
        self.x = x
        self.y = y
    def at(self, p):
        return min(self.x.at(p), self.y.at(p))
    def observationTimes(self):
        return self.x.observationTimes().union(self.y.observationTimes())
    def __str__(self):
        return 'Min(%s, %s)' % (self.x,self.y)

class Logical(Payoff):
    def __init__(self, x, y, opString):
        Payoff.__init__(self, 0.0)
        self.x = x
        self.y = y
        self.opString = opString
        self.op = None
        if opString == '<':
            self.op = lambda p : float(self.x.at(p)<self.y.at(p))
        if opString == '<=':
            self.op = lambda p : float(self.x.at(p)<=self.y.at(p))
        if opString == '==':
            self.op = lambda p : float(self.x.at(p)==self.y.at(p))
        if opString == '!=':
            self.op = lambda p : float(self.x.at(p)!=self.y.at(p))
        if opString == '>=':
            self.op = lambda p : float(self.x.at(p)>=self.y.at(p))
        if opString == '>':
            self.op = lambda p : float(self.x.at(p)>self.y.at(p))
        if not self.op:
            raise ValueError('Unknown operator: %s.' % opString)
    def at(self, p):
        return self.op(p)
    def observationTimes(self):
        return self.x.observationTimes().union(self.y.observationTimes())
    def __str__(self):
        return '(%s %s %s)' % (self.x,self.opString,self.y)



# Add further payoffs here
#   - exp, log, sqrt
#   - logicals
#   - basket

