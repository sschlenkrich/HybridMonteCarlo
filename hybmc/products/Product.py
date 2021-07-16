#!/usr/bin/python

import numpy as np

class Product:

    # We specify a common interface for our concrete products

    # return a list of payoffs V with payTime > obsTime
    # written as future present value observed at obsTime
    def cashFlows(self, obsTime):
        raise NotImplementedError('Implementation of method cashFlows required.')

    # return a time line of future cash flows per observation times
    def timeLine(self, obsTimes):
        return { t : self.cashFlows(t) for t in obsTimes }

    def scenarios(self, obsTimes, sim):
        tl = self.timeLine(obsTimes)
        idx = np.array([ True for k in range(sim.nPaths) ])  # reference all paths
        path = sim.path(idx)  # all paths
        zero = np.zeros(sim.nPaths)  # make sure dimensions match up
        scen = np.array([ zero + sum([
            payoff.discountedAt(path)  for payoff in tl[t] ]) for t in tl ])
        return scen.transpose()


# implement operations on time lines like join, observation times, ...