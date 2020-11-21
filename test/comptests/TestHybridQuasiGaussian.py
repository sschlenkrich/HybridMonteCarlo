#!/usr/bin/python

import sys
sys.path.append('./')

import unittest
import copy

import numpy as np

from hybmc.mathutils.Helpers import BlackImpliedVol, BlackVega
from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.AssetModel import AssetModel
from hybmc.models.HybridModel import HybridModel
from hybmc.models.HullWhiteModel import HullWhiteModel
from hybmc.models.QuasiGaussianModel import QuasiGaussianModel

from hybmc.simulations.McSimulation import McSimulation
from hybmc.simulations.Payoffs import Fixed, Pay, Asset, LiborRate, Max

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


class TestHybridQuasiGaussian(unittest.TestCase):

    # set up the stage for testing the models
    def setUp(self):
        ### full smile/skew model
        # domestic rates
        domAlias = 'EUR'
        eurCurve = YieldCurve(0.03)
        d     = 2
        times = np.array([  10.0     ])
        sigma = np.array([ [ 0.0060 ],
                           [ 0.0040 ] ])
        slope = np.array([ [ 0.10  ],
                           [ 0.15  ] ])
        curve = np.array([ [ 0.05  ],
                           [ 0.10  ] ])
        delta = np.array([ 1.0, 10.0 ])
        chi   = np.array([ 0.01, 0.15 ])
        Gamma = np.array([ [1.0, 0.6],
                           [0.6, 1.0] ])
        eurRatesModel = QuasiGaussianModel(eurCurve,d,times,sigma,slope,curve,delta,chi,Gamma)
        # assets
        forAliases = [ 'USD', 'GBP' ]
        spotS0     = [   1.0,   2.0 ]
        spotVol    = [   0.3,   0.2 ]
        forAssetModels = [
            AssetModel(S0, vol) for S0, vol in zip(spotS0,spotVol) ]
        # USD rates
        usdCurve = YieldCurve(0.02)
        d     = 3
        times = np.array([  10.0     ])
        sigma = np.array([ [ 0.0060 ],
                           [ 0.0050 ],
                           [ 0.0040 ] ])
        slope = np.array([ [ 0.10  ],
                           [ 0.20  ],
                           [ 0.30  ] ])
        curve = np.array([ [ 0.05  ],
                           [ 0.10  ],
                           [ 0.20  ] ])
        delta = np.array([ 1.0, 5.0, 20.0 ])
        chi   = np.array([ 0.01, 0.05, 0.15 ])
        Gamma = np.array([ [1.0, 0.8, 0.6],
                           [0.8, 1.0, 0.8],
                           [0.6, 0.8, 1.0] ])
        usdRatesModel = QuasiGaussianModel(usdCurve,d,times,sigma,slope,curve,delta,chi,Gamma)
        #
        gbpRatesModel = HWModel()
        #
        # 'EUR_x_0', 'EUR_x_1', 'USD_logS', 'USD_x_0', 'USD_x_1', 'USD_x_2', 'GBP_logS', 'GBP_x'
        corr = np.array([
            [   1.0,       0.0,        0.5,      -0.5,       0.0,       0.0,       -0.5,   -0.5   ],   # EUR_x_0
            [   0.0,       1.0,        0.0,       0.0,      -0.5,       0.0,       -0.5,    0.0   ],   # EUR_x_1
            [   0.5,       0.0,        1.0,      -0.5,      -0.5,      -0.5,        0.0,    0.0   ],   # USD_logS
            [  -0.5,       0.0,       -0.5,       1.0,       0.0,       0.0,        0.0,    0.0   ],   # USD_x_0
            [   0.0,      -0.5,       -0.5,       0.0,       1.0,       0.0,        0.0,    0.0   ],   # USD_x_1
            [   0.0,       0.0,       -0.5,       0.0,       0.0,       1.0,        0.0,    0.0   ],   # USD_x_2
            [  -0.5,      -0.5,        0.0,       0.0,       0.0,       0.0,        1.0,    0.5   ],   # GBP_logS
            [  -0.5,       0.0,        0.0,       0.0,       0.0,       0.0,        0.5,    1.0   ],   # GBP_x
        ])
        # 
        # corr = np.identity(2 + 1 + 3 + 1 + 1 )  # overwrite
        # 
        self.model = HybridModel(domAlias,eurRatesModel,forAliases,forAssetModels,[usdRatesModel,gbpRatesModel],corr)
        ### Gaussian model
        # domestic rates
        domAlias = 'EUR'
        eurCurve = YieldCurve(0.03)
        d     = 2
        times = np.array([  10.0     ])
        sigma = np.array([ [ 0.0060 ],
                           [ 0.0040 ] ])
        slope = np.array([ [ 0.00  ],
                           [ 0.00  ] ])
        curve = np.array([ [ 0.00  ],
                           [ 0.00  ] ])
        delta = np.array([ 1.0, 10.0 ])
        chi   = np.array([ 0.01, 0.15 ])
        Gamma = np.array([ [1.0, 0.6],
                           [0.6, 1.0] ])
        eurRatesModel = QuasiGaussianModel(eurCurve,d,times,sigma,slope,curve,delta,chi,Gamma)
        # assets
        forAliases = [ 'USD', 'GBP' ]
        spotS0     = [   1.0,   2.0 ]
        spotVol    = [   0.3,   0.2 ]
        forAssetModels = [
            AssetModel(S0, vol) for S0, vol in zip(spotS0,spotVol) ]
        # USD rates
        usdCurve = YieldCurve(0.02)
        d     = 3
        times = np.array([  10.0     ])
        sigma = np.array([ [ 0.0060 ],
                           [ 0.0050 ],
                           [ 0.0040 ] ])
        slope = np.array([ [ 0.10  ],
                           [ 0.20  ],
                           [ 0.30  ] ])
        curve = np.array([ [ 0.05  ],
                           [ 0.10  ],
                           [ 0.20  ] ])
        delta = np.array([ 1.0, 5.0, 20.0 ])
        chi   = np.array([ 0.01, 0.05, 0.15 ])
        Gamma = np.array([ [1.0, 0.8, 0.6],
                           [0.8, 1.0, 0.8],
                           [0.6, 0.8, 1.0] ])
        self.gaussianModel = HybridModel(domAlias,eurRatesModel,forAliases,forAssetModels,[usdRatesModel,gbpRatesModel],corr)


    def test_ModelSetup(self):
        self.assertListEqual(self.model.stateAliases(),
            ['EUR_x_0', 'EUR_x_1',
             'EUR_y_0_0', 'EUR_y_0_1',
             'EUR_y_1_0', 'EUR_y_1_1',
             'EUR_s',
             'USD_logS', 'USD_x_0', 'USD_x_1', 'USD_x_2',
             'USD_y_0_0', 'USD_y_0_1', 'USD_y_0_2',
             'USD_y_1_0', 'USD_y_1_1', 'USD_y_1_2',
             'USD_y_2_0', 'USD_y_2_1', 'USD_y_2_2',
             'USD_s',
             'GBP_logS', 'GBP_x', 'GBP_s'])
        self.assertListEqual(self.model.factorAliases(),
            ['EUR_x_0', 'EUR_x_1',
             'USD_logS', 'USD_x_0', 'USD_x_1', 'USD_x_2',
             'GBP_logS', 'GBP_x'])        


    # @unittest.skip('Too time consuming')
    def test_HybridSimulation(self):
        times = np.concatenate([ np.linspace(0.0, 10.0, 11), [10.5] ])
        nPaths = 2**13
        seed = 314159265359
        # risk-neutral simulation
        print('')
        mcSim = McSimulation(self.model,times,nPaths,seed,False)
        # 
        T = 10.0
        P = Pay(Fixed(1.0),T)
        fw, err = fwd(mcSim,P)
        # domestic numeraire
        print('1.0   @ %4.1lfy %8.6lf - mc_err = %8.6lf' % (T,fw,err))
        # foreign assets
        for k, alias in enumerate(self.model.forAliases):
            p = Asset(T,alias)
            xT = self.model.forAssetModels[k].X0 * \
                self.model.forRatesModels[k].yieldCurve.discount(T) / \
                self.model.domRatesModel.yieldCurve.discount(T)
            fw, err = fwd(mcSim,p)
            print(alias + '   @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf' % (T,fw,xT,err))
        # domestic Libor rate
        Tstart = 10.0
        Tend = 10.5
        L = Pay(LiborRate(T,Tstart,Tend,alias='EUR'),Tend)
        fw, err = fwd(mcSim,L)
        Lref = (mcSim.model.domRatesModel.yieldCurve.discount(Tstart) /    \
                mcSim.model.domRatesModel.yieldCurve.discount(Tend) - 1) / \
               (Tend - Tstart) 
        print('L_EUR @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf' % (T,fw,Lref,err))
        # foreign Lbor rates
        for k, alias in enumerate(self.model.forAliases):
            L = Pay(LiborRate(T,Tstart,Tend,alias=alias)*Asset(Tend,alias),Tend)
            fw, err = fwd(mcSim,L)
            fw *= mcSim.model.domRatesModel.yieldCurve.discount(Tend) / \
                  mcSim.model.forRatesModels[k].yieldCurve.discount(Tend) / \
                  mcSim.model.forAssetModels[k].X0
            err *= mcSim.model.domRatesModel.yieldCurve.discount(Tend) / \
                   mcSim.model.forRatesModels[k].yieldCurve.discount(Tend) / \
                   mcSim.model.forAssetModels[k].X0
            Lref = (mcSim.model.forRatesModels[k].yieldCurve.discount(Tstart) /    \
                    mcSim.model.forRatesModels[k].yieldCurve.discount(Tend) - 1) / \
                   (Tend - Tstart) 
            print('L_%s @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf' % (alias,T,fw,Lref,err))


    def test_HybridVolAdjusterCalculation(self):
        model = copy.deepcopy(self.model)
        # model = copy.deepcopy(self.gaussianModel)
        hybVolAdjTimes = np.linspace(0.0, 20.0, 21)
        model.recalculateHybridVolAdjuster(hybVolAdjTimes)
        plt.plot(model.hybAdjTimes,model.hybVolAdj[0], 'r*', label='USD')
        plt.plot(model.hybAdjTimes,model.hybVolAdj[1], 'b*', label='GBP')
        plt.legend()
        #
        times = np.linspace(0.0,20.0,101)
        plt.plot(times,[ model.hybridVolAdjuster(0,t) for t in times ]  , 'r-')
        plt.plot(times,[ model.hybridVolAdjuster(1,t) for t in times ]  , 'b-')
        plt.show()
        #
        # return
        times = np.linspace(0.0, 10.0, 11)
        nPaths = 2**13
        seed = 314159265359
        # risk-neutral simulation
        print('')
        mcSim = McSimulation(model,times,nPaths,seed,False)
        # 
        T = 10.0
        for k, alias in enumerate(model.forAliases):
            # ATM forward
            xT = model.forAssetModels[k].X0 * \
                 model.forRatesModels[k].yieldCurve.discount(T) / \
                 model.domRatesModel.yieldCurve.discount(T)
            K = Fixed(xT)
            Z = Fixed(0.0)
            C = Pay(Max(Asset(T,alias)-K,Z),T)
            fw, err = fwd(mcSim,C)
            vol = BlackImpliedVol(fw,xT,xT,T,1.0)
            vega = BlackVega(xT,xT,vol,T)
            err /= vega
            volRef = model.forAssetModels[k].sigma
            print('C_%s @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf' % (alias,T,vol,volRef,err))
            P = Pay(Max(K-Asset(T,alias),Z),T)
            fw, err = fwd(mcSim,P)
            vol = BlackImpliedVol(fw,xT,xT,T,-1.0)
            vega = BlackVega(xT,xT,vol,T)
            err /= vega
            volRef = model.forAssetModels[k].sigma
            print('P_%s @ %4.1lfy %8.6lf vs %8.6lf (curve) - mc_err = %8.6lf' % (alias,T,vol,volRef,err))



if __name__ == '__main__':
    unittest.main()
    
