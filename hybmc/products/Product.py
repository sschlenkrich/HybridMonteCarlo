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
        return np.array([ 
            [ sum([ p.discountedAt(sim.path(k)) for p in tl[t] ]) for t in tl ]
            for k in range(sim.nPaths) ])

# implement operations on time lines like join, observation times, ...