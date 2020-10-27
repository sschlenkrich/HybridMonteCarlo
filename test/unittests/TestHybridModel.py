#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np


from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.HullWhiteModel import HullWhiteModel, HullWhiteModelWithDiscreteNumeraire
from hybmc.models.AssetModel import AssetModel
from hybmc.models.DeterministicModel import DcfModel
from hybmc.models.HybridModel import HybridModel

from hybmc.simulations.McSimulation import McSimulation

# a quick way to get a model
def HWModel(rate=0.01, vol=0.0050, mean=0.03):
    curve = YieldCurve(rate)
    times = np.array([ 10.0 ])
    vols  = np.array([ vol  ])
    return HullWhiteModel(curve, mean, times, vols)

class TestHybridModel(unittest.TestCase):

    # set up the stage for testing the models
    def setUp(self):
        domAlias = 'EUR'
        domModel = HWModel(0.01, 0.0050, 0.03)
        forAliases = [ 'USD', 'GBP', 'JPY' ]
        spotS0     = [   1.0,   1.0,   1.0 ]
        spotVol    = [   0.3,   0.3,   0.3 ]
        rates      = [  0.01,  0.01,  0.01 ]
        ratesVols  = [ 0.006, 0.007, 0.008 ]
        mean       = [  0.01,  0.01,  0.01 ]
        #
        forAssetModels = [
            AssetModel(S0, vol) for S0, vol in zip(spotS0,spotVol) ]
        forRatesModels = [
            HWModel(r,v,m) for r,v,m in zip(rates,ratesVols,mean) ]
        #
        corr = np.identity(2 * len(forAliases) + 1)
        corr[1,2] = 0.6  # USD quanto
        corr[2,1] = 0.6
        corr[3,4] = 0.8  # GBP quanto
        corr[4,3] = 0.8
        corr[5,6] = -0.7  # JPY quanto
        corr[6,5] = -0.7
        #
        self.model = HybridModel(domAlias,domModel,forAliases,forAssetModels,forRatesModels,corr)

    def test_HybridModelSetup(self):
        # we check against known values
        self.assertListEqual(self.model.modelsStartIdx, [2, 3, 5, 6, 8, 9])
        self.assertDictEqual(self.model.index, {'USD': 0, 'GBP': 1, 'JPY': 2})
        self.assertEqual(self.model.size(), len(self.model.stateAliases()))
        self.assertEqual(self.model.factors(), len(self.model.factorAliases()))
        self.assertListEqual(self.model.stateAliases(), \
            [            'EUR_x', 'EUR_s', 
             'USD_logS', 'USD_x', 'USD_s',
             'GBP_logS', 'GBP_x', 'GBP_s',
             'JPY_logS', 'JPY_x', 'JPY_s' ])
        self.assertListEqual(self.model.factorAliases(), \
            ['EUR_x', 'USD_logS', 'USD_x', 'GBP_logS', 'GBP_x', 'JPY_logS', 'JPY_x'])
        # check correlation
        C = self.model.L.dot(self.model.L.transpose())
        C = np.abs(C - self.model.correlations)
        self.assertLess(np.max(C), 1.0e-15)

    def test_HybridModelEvolution(self):
        times = np.array([0.0])
        nPaths = 1
        seed = 314159265359
        # risk-neutral simulation
        mcSim = McSimulation(self.model,times,nPaths,seed,False)
        p = mcSim.path(0)
        #
        self.assertEqual(p.asset(0.0,'USD'),self.model.forAssetModels[0].X0)
        self.assertEqual(p.asset(0.0,'GBP'),self.model.forAssetModels[1].X0)
        self.assertEqual(p.asset(0.0,'JPY'),self.model.forAssetModels[2].X0)
        #
        self.assertEqual(p.zeroBond(0.0,5.0,'EUR'),self.model.domRatesModel.yieldCurve.discount(5.0))
        self.assertEqual(p.zeroBond(0.0,5.0,'USD'),self.model.forRatesModels[0].yieldCurve.discount(5.0))
        self.assertEqual(p.zeroBond(0.0,5.0,'GBP'),self.model.forRatesModels[1].yieldCurve.discount(5.0))
        self.assertEqual(p.zeroBond(0.0,5.0,'JPY'),self.model.forRatesModels[2].yieldCurve.discount(5.0))

    def test_HybridModelWithDeterministicRates(self):
        curve0 = YieldCurve(0.03)
        dcfModel0 = DcfModel(curve0)
        #
        times = np.array([0.0, 1.0])
        nPaths = 1
        seed = 314159265359
        # hybrid adjuster
        hybAdjTimes = np.array([0.0, 1.0, 2.0])
        # simulate deterministic model only
        mcSim = McSimulation(dcfModel0,times,nPaths,seed,False)
        self.assertEqual(mcSim.path(0).zeroBond(1.0,10.0,None),curve0.discount(10.0)/curve0.discount(1.0))
        # simulate deterministic domestic model
        hwModel = HWModel(0.01,0.0050,0.03)
        asModel = AssetModel(1.0,0.30)
        corr = np.identity(2)
        model = HybridModel('EUR',dcfModel0,['USD'],[asModel],[hwModel],corr)
        model.recalculateHybridVolAdjuster(hybAdjTimes)
        mcSim = McSimulation(model,times,nPaths,seed,False)
        dcfModel0.domAlias = 'EUR'
        p = mcSim.path(0)
        self.assertEqual(p.asset(0.0,'USD'),self.model.forAssetModels[0].X0)
        self.assertEqual(p.zeroBond(0.0,5.0,'EUR'),curve0.discount(5.0))
        self.assertEqual(p.zeroBond(0.0,5.0,'USD'),model.forRatesModels[0].yieldCurve.discount(5.0))
        # simulate deterministic foreign model
        model = HybridModel('EUR',hwModel,['USD'],[asModel],[dcfModel0],corr)
        model.recalculateHybridVolAdjuster(hybAdjTimes)
        mcSim = McSimulation(model,times,nPaths,seed,False)
        dcfModel0.domAlias = 'USD'
        p = mcSim.path(0)
        self.assertEqual(p.asset(0.0,'USD'),self.model.forAssetModels[0].X0)
        self.assertEqual(p.zeroBond(0.0,5.0,'EUR'),model.domRatesModel.yieldCurve.discount(5.0))
        self.assertEqual(p.zeroBond(0.0,5.0,'USD'),curve0.discount(5.0))
        # simulate deterministic domestic and foreign curve
        curve1 = YieldCurve(0.05)
        dcfModel1 = DcfModel(curve1)
        corr = np.identity(1)
        model = HybridModel('EUR',dcfModel0,['USD'],[asModel],[dcfModel1],corr)
        model.recalculateHybridVolAdjuster(hybAdjTimes)
        mcSim = McSimulation(model,times,nPaths,seed,False)
        dcfModel0.domAlias = 'EUR'
        dcfModel1.domAlias = 'USD'
        p = mcSim.path(0)
        self.assertEqual(p.asset(0.0,'USD'),self.model.forAssetModels[0].X0)
        self.assertEqual(p.zeroBond(0.0,5.0,'EUR'),curve0.discount(5.0))
        self.assertEqual(p.zeroBond(0.0,5.0,'USD'),curve1.discount(5.0))







if __name__ == '__main__':
    unittest.main()
    
