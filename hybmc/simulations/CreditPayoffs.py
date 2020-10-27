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
        return p.HazardRate(self.obsTime,self.alias)
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
