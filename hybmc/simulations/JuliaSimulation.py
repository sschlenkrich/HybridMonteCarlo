#!/usr/bin/python

# create Julia simulation from Python McSimulation

from julia import Main
Main.include('./hybmc/simulations/Payoffs.jl')
Main.include('./hybmc/simulations/McSimulation.jl')

import sys
sys.path.append('./')

from hybmc.simulations.McSimulation import McSimulation
from hybmc.models.JuliaModel import JuliaModel


def JuliaSimulation(sim, simulate=False, useBrownians=False, times=None, nPaths=None, seed=None, timeInterpolation=None):
    if not isinstance(sim,McSimulation):
        raise NotImplementedError('Implementation of JuliaSimulation for %s required.' % str(type(sim)))
    model = JuliaModel(sim.model)
    # check for manual parameters
    if times is None: times = sim.times
    if nPaths is None: nPaths = sim.nPaths
    if seed is None: seed = sim.seed
    if timeInterpolation is None: timeInterpolation = sim.timeInterpolation
    #
    if simulate:
        if useBrownians:
            return Main.McSimulationWithBrownians(model,times,sim.dW,timeInterpolation)
        else:
            return Main.McSimulation(model,times,nPaths,seed,timeInterpolation)
    else:  # do not use this with manual inputs
        return Main.McSimulation(model,times,nPaths,seed,timeInterpolation,sim.dW,sim.X)
    return None

def JuliaDiscountedAt(jSim, jPayoffs):
    Main.sim = jSim
    Main.payoffs = jPayoffs
    scenarios = Main.eval("[ discountedAt(payoff,path) for payoff in payoffs, path in paths(sim) ]")
    return scenarios

