#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
import matplotlib.pyplot as plt

from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.HullWhiteModel import HullWhiteModel
from hybmc.models.QuasiGaussianModel import QuasiGaussianModel
from hybmc.models.AssetModel import AssetModel
from hybmc.models.HybridModel import HybridModel


from hybmc.simulations.McSimulation import McSimulation
from hybmc.simulations.Payoffs import Payoff, Fixed, LiborRate
from hybmc.mathutils.Regression import Regression
from hybmc.simulations.AmcPayoffs import AmcMax, AmcMin, AmcOne, AmcSum

# a quick way to get a model
def HWModel(rate=0.01, vol=0.0050, mean=0.03):
    curve = YieldCurve(rate)
    times = np.array([ 10.0 ])
    vols  = np.array([ vol  ])
    return HullWhiteModel(curve, mean, times, vols)


class TestMultiPath(unittest.TestCase):


    def test_staticPayoffs(self):
        a  = 0.5
        px = np.array([ 1.0, 1.0 ])
        py = np.array([ 2.0, 2.0 ])
        x = Fixed(px)
        y = Fixed(py)
        # 
        np.testing.assert_array_equal((x+y).at(None), px+py)
        np.testing.assert_array_equal((x-y).at(None), px-py)
        np.testing.assert_array_equal((x*y).at(None), px*py)
        np.testing.assert_array_equal((x/y).at(None), px/py)
        # 
        np.testing.assert_array_equal((x+a).at(None), px+a)
        np.testing.assert_array_equal((x-a).at(None), px-a)
        np.testing.assert_array_equal((x*a).at(None), px*a)
        np.testing.assert_array_equal((x/a).at(None), px/a)
        # 
        np.testing.assert_array_equal((a+y).at(None), a+py)
        np.testing.assert_array_equal((a-y).at(None), a-py)
        np.testing.assert_array_equal((a*y).at(None), a*py)
        np.testing.assert_array_equal((a/y).at(None), a/py)
        #
        z = a*x + a*x*y - y
        np.testing.assert_array_equal(z.at(None), a*px + a*px*py - py)
        np.testing.assert_array_equal((~z).at(not None), a*px + a*px*py - py)  # we can't use None here
        #
        np.testing.assert_array_equal( (x<y).at(None),1.0*(px<py))
        np.testing.assert_array_equal((x<=y).at(None),1.0*(px<=py))
        np.testing.assert_array_equal((x==y).at(None),1.0*(px==py))
        np.testing.assert_array_equal((x>=y).at(None),1.0*(px>=py))
        np.testing.assert_array_equal( (x>y).at(None),1.0*(px>py))
        #
        np.testing.assert_array_equal( (x<a).at(None),1.0*(px<a))
        np.testing.assert_array_equal((x<=a).at(None),1.0*(px<=a))
        np.testing.assert_array_equal((x==a).at(None),1.0*(px==a))
        np.testing.assert_array_equal((x>=a).at(None),1.0*(px>=a))
        np.testing.assert_array_equal( (x>a).at(None),1.0*(px>a))
        #
        np.testing.assert_array_equal( (a<y).at(None),1.0*(a<py))
        np.testing.assert_array_equal((a<=y).at(None),1.0*(a<=py))
        np.testing.assert_array_equal((a==y).at(None),1.0*(a==py))
        np.testing.assert_array_equal((a>=y).at(None),1.0*(a>=py))
        np.testing.assert_array_equal( (a>y).at(None),1.0*(a>py))
        #
        np.testing.assert_array_equal((x+y).obsTime, 0.0)
        np.testing.assert_array_equal(((x+y)@1.0).obsTime, 1.0)


    def test_HullWhite(self):
        curve = YieldCurve(0.00)
        mean  = 0.01
        times = np.array([ 10.0   ])
        vols  = np.array([ 0.0050 ])
        model = HullWhiteModel(curve, mean, times, vols)
        X = np.random.normal(0.0, 0.01, [model.size(), 5])
        # zero bond
        zcb1 = model.zeroBond(1.0, 3.0, X, None)
        zcb2 = np.array([ model.zeroBond(1.0, 3.0, x, None)
            for x in X.transpose() ])
        np.testing.assert_array_equal(zcb1, zcb2)
        # numeraire
        num1 = model.numeraire(5.0, X)
        num2 = np.array([ model.numeraire(5.0, x)        
            for x in X.transpose() ])
        np.testing.assert_array_equal(num1, num2)


    def test_QuasiGaussian(self):
        yieldCurve = YieldCurve(0.03)
        d     = 3
        times = np.array([ 1.0,    2.0,    5.0,    10.0     ])
        sigma = np.array([ [ 0.0060, 0.0070, 0.0080,  0.0100  ],
                           [ 0.0030, 0.0035, 0.0040,  0.0050  ],
                           [ 0.0025, 0.0025, 0.0025,  0.0025  ] ])
        slope = np.array([ [ 0.10,   0.15,   0.20  ,  0.25    ],
                           [ 0.20,   0.35,   0.40  ,  0.55    ],
                           [ 0.10,   0.10,   0.10  ,  0.10    ] ])
        curve = np.array([ [ 0.05,   0.05,   0.05  ,  0.05    ],
                           [ 0.10,   0.10,   0.10  ,  0.10    ],
                           [ 0.20,   0.15,   0.10  ,  0.00    ] ])
        delta = np.array([ 1.0, 5.0, 20.0 ])
        chi   = np.array([ 0.01, 0.05, 0.15 ])
        Gamma = np.array([ [1.0, 0.8, 0.6],
                           [0.8, 1.0, 0.8],
                           [0.6, 0.8, 1.0] ])

        # test matricies
        model = QuasiGaussianModel(yieldCurve,d,times,sigma,slope,curve,delta,chi,Gamma)
        X = np.random.normal(0.0, 0.01, [model.size(), 5])
        # zero bond
        zcb1 = model.zeroBond(1.0, 3.0, X, None)
        zcb2 = np.array([ model.zeroBond(1.0, 3.0, x, None)
            for x in X.transpose() ])
        np.testing.assert_array_equal(zcb1, zcb2)
        # numeraire
        num1 = model.numeraire(5.0, X)
        num2 = np.array([ model.numeraire(5.0, x)        
            for x in X.transpose() ])
        np.testing.assert_array_equal(num1, num2)


    def test_HybridModel(self):
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
        model = HybridModel(domAlias,domModel,forAliases,forAssetModels,forRatesModels,corr)
        #
        X = np.random.normal(0.0, 0.01, [model.size(), 5])
        # zero bond
        zcb1 = model.zeroBond(1.0, 3.0, X, None)
        zcb2 = np.array([ model.zeroBond(1.0, 3.0, x, None)
            for x in X.transpose() ])
        np.testing.assert_array_equal(zcb1, zcb2)
        # numeraire
        num1 = model.numeraire(5.0, X)
        num2 = np.array([ model.numeraire(5.0, x)        
            for x in X.transpose() ])
        np.testing.assert_array_equal(num1, num2)
        # asset
        ass1 = model.asset(2.5, X, 'JPY')
        ass2 = np.array([ model.asset(2.5, x, 'JPY')        
            for x in X.transpose() ])
        np.testing.assert_array_equal(ass1, ass2)


    def test_SimulationState(self):
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
        model = HybridModel(domAlias,domModel,forAliases,forAssetModels,forRatesModels,corr)
        #
        times = np.linspace(0.0, 10.0, 11)
        nPaths = 2**3
        seed = 1234
        # risk-neutral simulation
        mcSim = McSimulation(model,times,nPaths,seed,showProgress=False)
        idx = [ True for k in range(mcSim.nPaths) ]
        state = mcSim.state(idx, 4.5)
        print(state.shape)

if __name__ == '__main__':
    unittest.main()
    
