#!/usr/bin/python

# create QuantLib models from Python models

try:
    import QuantLib as ql
except ModuleNotFoundError as e:
    print('Error: Module QuantLibPayoffs requires a (custom) QuantLib installation.')
    raise e

try:
    from QuantLib import RealMCSimulation, RealMCPayoffPricer_discountedAt
except ImportError as e:
    print('Error: Module QuantLibPayoffs requires a custom QuantLib installation.')
    print('It seems you do have a QuantLib installed but maybe not the custom version')
    print('with payoff scripting. Checkout https://github.com/sschlenkrich/QuantLib')
    raise e

import numpy as np

import sys
sys.path.append('./')

from hybmc.simulations.McSimulation import McSimulation
from hybmc.wrappers.QuantLibModels import QuantLibModel


def QuantLibSimulation(sim, times=None, nPaths=None, seed=None, timeInterpolation=None, storeBrownians=True):
    if not isinstance(sim,McSimulation):
        raise NotImplementedError('Implementation of JuliaSimulation for %s required.' % str(type(sim)))
    model = QuantLibModel(sim.model)
    # check for manual parameters
    if times is None: times = sim.times
    if nPaths is None: nPaths = sim.nPaths
    if seed is None: seed = sim.seed
    if timeInterpolation is None: timeInterpolation = sim.timeInterpolation
    # a few default parameters
    obsTimes = times
    richardsonExtrapolation = False
    #
    # make sure seed is not too big for Size
    sim = RealMCSimulation(model,times,obsTimes,nPaths,seed,
        richardsonExtrapolation,timeInterpolation,storeBrownians)
    # don't forget to simulate in QuantLib
    sim.simulate()
    return sim


def QuantLibDiscountedAt(sim, payoffs):
    return np.array([ RealMCPayoffPricer_discountedAt(p,sim) for p in payoffs ])
