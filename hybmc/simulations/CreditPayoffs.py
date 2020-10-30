#!/usr/bin/python

import numpy as np

from hybmc.simulations.Payoffs import Payoff


class HazardProcess(Payoff):
    def __init__(self, t, alias):
        Payoff.__init__(self, t)
        self.alias = alias
    def at(self, p):
        return p.hazardProcess(self.obsTime,self.alias)
    def __str__(self):
        return 'Q_%s(%.2f)' % (self.alias,self.obsTime)

class HazardRate(Payoff):
    def __init__(self, t, alias):
        Payoff.__init__(self, t)
        self.alias = alias
    def at(self, p):
        return p.hazardRate(self.obsTime,self.alias)
    def __str__(self):
        return 'q_%s(%.2f)' % (self.alias,self.obsTime)

class SurvivalProb(Payoff):
    def __init__(self, t, T, alias):
        Payoff.__init__(self, t)
        self.maturity = T
        self.alias = alias
    def at(self, p):
        return p.survivalProb(self.obsTime,self.maturity,self.alias)
    def __str__(self):
        return 'Q_%s(%.2f,%.2f)' % (self.alias,self.obsTime,self.maturity)

def zetaScenarios(alias,obstime,sim,normalized=True):
    timeLine = [ HazardProcess(t,alias) * HazardRate(t,alias) for t in obstime ]
    zetas = np.array([ [payoff.at(path) for payoff in timeLine]
        for path in sim.paths() ])
    if normalized:
        zetas /= np.average(zetas,axis=0)
    return zetas