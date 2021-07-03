#!/usr/bin/python

import numpy as np

class McSimulation:

    # Python constructor
    def __init__(self, model, times, nPaths, seed=123, timeInterpolation=True, showProgress=True):
        if showProgress: print('Start MC Simulation:', end='', flush=True)
        self.model  = model   # an object implementing stochastic process interface
        self.times  = times   # simulation times [0, ..., T], np.array
        self.nPaths = nPaths  # number of paths, long
        self.seed   = seed    # the random seed
        self.timeInterpolation = timeInterpolation  # allow state calculation in between simulated states
        # random number generator
        if showProgress: print(' |dW\'s', end='', flush=True)
        rg = np.random.Generator(np.random.PCG64(self.seed))
        self.dW = rg.standard_normal([self.nPaths,len(self.times)-1,model.factors()])
        if showProgress: print('|', end='', flush=True)
        # simulate states
        self.X = np.zeros([self.nPaths,len(self.times),model.size()])
        for i in range(self.nPaths):
            if showProgress and i % max(int(self.nPaths/10),1) == 0 : print('s', end='', flush=True)
            self.X[i,0] = self.model.initialValues()
            for j in range(len(self.times)-1):
                model.evolve(self.times[j],self.X[i,j],times[j+1]-times[j],self.dW[i,j],self.X[i,j+1])
        if showProgress: print('| Finished.', end='\n', flush=True)

    def state(self, idx, t):
        if not idx<self.nPaths:
            raise IndexError('idx < nPaths required.')
        tIdx = np.searchsorted(self.times,t)
        tIdx = min(tIdx,len(self.times)-1)  # use the last element
        if np.abs(self.times[tIdx]-t)<0.5/365:  # we give some tolerance of half a day
            return self.X[idx,tIdx]
        if not self.timeInterpolation:
            raise ValueError('timeInterpolation required for input time %lf' % (t))
        if t >= self.times[tIdx] or tIdx==0:  # extrapolation
            return self.X[idx,tIdx]
        # linear interpolation
        rho = (self.times[tIdx] - t) / (self.times[tIdx] - self.times[tIdx-1])
        return rho * self.X[idx,tIdx-1] + (1.0-rho) * self.X[idx,tIdx]

    def states(self, t):
        """
        Return a 2-dim array of all states of the simulation at a given time.
        Result is of shape (nPaths,model.factors())
        """
        tIdx = np.searchsorted(self.times,t)
        tIdx = min(tIdx,len(self.times)-1)  # use the last element
        if np.abs(self.times[tIdx]-t)<0.5/365:  # we give some tolerance of half a day
            return self.X[idx,tIdx]
        if not self.timeInterpolation:
            raise ValueError('timeInterpolation required for input time %lf' % (t))
        if t >= self.times[tIdx] or tIdx==0:  # extrapolation
            return self.X[idx,tIdx]
        # linear interpolation
        rho = (self.times[tIdx] - t) / (self.times[tIdx] - self.times[tIdx-1])
        return rho * self.X[idx,tIdx-1] + (1.0-rho) * self.X[idx,tIdx]

    def path(self, idx):
        return Path(self, idx)

    def paths(self):
        return [ self.path(k) for k in range(self.nPaths) ]

class Path:

    # Python constructor
    def __init__(self, simulation, idx):
        self.simulation = simulation
        self.idx = idx
        # maybe add some checks here...

    # stochastic process interface for payoffs

    # the numeraire in the domestic currency used for discounting future payoffs
    def numeraire(self, t):
        # we may add numeraire adjuster here...
        return self.simulation.model.numeraire(t, self.simulation.state(self.idx,t))
    
    # a domestic/foreign currency zero coupon bond
    def zeroBond(self, t, T, alias):
        # we may add zcb adjuster here
        return self.simulation.model.zeroBond(t, T, self.simulation.state(self.idx,t), alias)
    
    # an asset price for a given currency alias
    def asset(self, t, alias):
        # we may add asset adjuster here
        return self.simulation.model.asset(t, self.simulation.state(self.idx,t), alias)

    # Cumulated intensity; probability of tau > t, conditional on F_t
    def hazardProcess(self, t, alias):
        return self.simulation.model.hazardProcess(t, self.simulation.state(self.idx,t), alias)
    
    # instantanous probability of default
    def hazardRate(self, t, alias):
        return self.simulation.model.hazardRate(t, self.simulation.state(self.idx,t), alias)
    
    # probavility of survival consitional on information at t
    def survivalProb(self, t, T, alias):
        return self.simulation.model.survivalProb(t, T, self.simulation.state(self.idx,t), alias)
