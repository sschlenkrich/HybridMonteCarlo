#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np

from src.termstructures.YieldCurve import YieldCurve
from src.models.HullWhiteModel import HullWhiteModel, HullWhiteModelWithDiscreteNumeraire
from src.models.AssetModel import AssetModel
from src.models.HybridModel import HybridModel

from src.simulations.MCSimulation import MCSimulation
from src.simulations.Payoffs import Fixed, Pay, Asset, LiborRate

import matplotlib.pyplot as plt

# a quick way to get a model
def HWModel(rate=0.01, vol=0.0050, mean=0.03):
    curve = YieldCurve(rate)
    times = np.array([ 10.0 ])
    vols  = np.array([ vol  ])
    return HullWhiteModel(curve, mean, times, vols)

def fwd(mcSim,p):
    samples = np.array([
        p.discountedAt(mcSim.path(k)) for k in range(mcSim.nPaths) ])
    fwd = np.average(samples) / \
        mcSim.model.domRatesModel.yieldCurve.discount(p.obsTime)
    err = np.std(samples) / np.sqrt(samples.shape[0]) / \
        mcSim.model.domRatesModel.yieldCurve.discount(p.obsTime)
    return fwd, err

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

    def test_HybridSimulation(self):
        times = np.linspace(0.0, 10.0, 3)
        nPaths = 2**10
        seed = 1234
        # risk-neutral simulation
        mcSim = MCSimulation(self.model,times,nPaths,seed,False)
        # 
        T = 10.0
        P = Pay(Fixed(1.0),T)
        pv, err = fwd(mcSim,P)
        # domestic numeraire
        print('1.0 @ %4.1lfy %8.6lf - mc_err = %8.6lf' % (T,pv,err))
        # foreign asset e
        for k, alias in enumerate(self.model.forAliases):
            p = Asset(T,alias)
            xT = self.model.forAssetModels[k].X0 * \
                self.model.forRatesModels[k].yieldCurve.discount(T) / \
                self.model.domRatesModel.yieldCurve.discount(T)
            pv, err = fwd(mcSim,p)
            print(alias + ' @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf' % (T,pv,xT,err))

    def test_HybridVolAdjusterCalculation(self):
        # we set up a hybrid model consistent to QuantLib
        domAlias = 'EUR'
        domModel = HWModel(0.01, 0.0050, 0.01)
        forAliases = [ 'USD', 'GBP' ]
        forAssetModels = [
            AssetModel(1.0, 0.30),
            AssetModel(2.0, 0.15) ]
        forRatesModels = [
            HWModel(0.02, 0.0060, 0.02), 
            HWModel(0.02, 0.0070, 0.03) 
        ]
        corr = np.identity(2 * len(forAliases) + 1)
        # [ EUR, USD-EUR, USD, GBP-EUR, GBP ]
        #   0    1        2    3        4
        # USD-EUR - EUR
        corr[0,1] = 0.5
        corr[1,0] = 0.5
        # USD-EUR - USD
        corr[1,2] = -0.5
        corr[2,1] = -0.5
        # EUR - USD
        corr[0,2] = -0.5
        corr[2,0] = -0.5
        # GBP-EUR - EUR
        corr[0,3] = -0.5
        corr[3,0] = -0.5
        # GBP-EUR - GBP
        corr[3,4] = 0.5
        corr[4,3] = 0.5
        # EUR - GBP
        corr[0,4] = -0.8
        corr[4,0] = -0.8
        # USD - GBP
        corr[2,4] = 0.0
        corr[4,2] = 0.0
        print(corr)
        model = HybridModel(domAlias,domModel,forAliases,forAssetModels,forRatesModels,corr)
        hybVolAdjTimes = np.linspace(0.0, 20.0, 21)
        model.recalculateHybridVolAdjuster(hybVolAdjTimes)
        plt.plot(model.hybAdjTimes,model.hybVolAdj[0], 'r*')
        plt.plot(model.hybAdjTimes,model.hybVolAdj[1], 'b*')
        #
        times = np.linspace(0.0,20.0,101)
        plt.plot(times,[ model.hybridVolAdjuster(0,t) for t in times ]  , 'r-')
        plt.plot(times,[ model.hybridVolAdjuster(1,t) for t in times ]  , 'b-')
        plt.show()


if __name__ == '__main__':
    unittest.main()
    
