#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np

from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.HullWhiteModel import HullWhiteModel
from hybmc.models.SpreadModel import SpreadModel

from hybmc.simulations.McSimulation import McSimulation
from hybmc.simulations.Payoffs import Fixed, Pay

# a quick way to get a model
def HWModel(rate=0.01, vol=0.0050, mean=0.03):
    curve = YieldCurve(rate)
    times = np.array([ 10.0 ])
    vols  = np.array([ vol  ])
    return HullWhiteModel(curve, mean, times, vols)

def fwd(mcSim,p):
    DF = mcSim.model.baseModel.yieldCurve.discount(p.obsTime) * \
         mcSim.model.sprdModel.yieldCurve.discount(p.obsTime)
    samples = np.array([
        p.discountedAt(mcSim.path(k)) for k in range(mcSim.nPaths) ])
    fwd = np.average(samples) / DF
    err = np.std(samples) / np.sqrt(samples.shape[0]) / DF
    return fwd, err

class TestSpreadModel(unittest.TestCase):

    # set up the stage for testing the models
    def setUp(self):
        baseModel = HWModel(0.02, 0.0070, 0.03)
        sprdModel = HWModel(0.01, 0.0030, 0.01)
        #
        corr = np.identity(2)
        corr[0,1] = 0.0  # credit-rates corr
        corr[1,0] = 0.0
        #
        self.model = SpreadModel(baseModel,sprdModel,corr)

    def test_SpreadModelSetup(self):
        # we check against known values
        print('')
        print(self.model.stateAliases())
        print(self.model.factorAliases())

    def test_SpreadSimulation(self):
        times = np.linspace(0.0, 10.0, 11)
        nPaths = 2**10
        seed = 1234
        # risk-neutral simulation
        print('')
        mcSim = McSimulation(self.model,times,nPaths,seed,False)
        # 
        T = 10.0
        P = Pay(Fixed(1.0),T)
        pv, err = fwd(mcSim,P)
        # domestic numeraire
        print('1.0 @ %4.1lfy %8.6lf - mc_err = %8.6lf' % (T,pv,err))



if __name__ == '__main__':
    unittest.main()
    

