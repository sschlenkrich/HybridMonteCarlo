#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import QuantLib as ql

import numpy as np
import matplotlib.pyplot as plt

from src.termstructures.YieldCurve import YieldCurve
from src.models.HullWhiteModel import HullWhiteModel
from src.simulations.MCSimulation import MCSimulation

from src.products.Swap import Swap, AmcSwap

# a quick way to get a model
def HWModel(rate=0.01, vol=0.0050, mean=0.03):
    curve = YieldCurve(rate)
    times = np.array([ 10.0 ])
    vols  = np.array([ vol  ])
    return HullWhiteModel(curve, mean, times, vols)


class TestSwapProduct(unittest.TestCase):

    def setUp(self):
        # we set up a QuantLib VanillaSwap as basic test object
        today     = ql.Date(5,ql.October,2020)
        ql.Settings.instance().evaluationDate = today
        # we need curves for the float leg
        discYtsH = ql.YieldTermStructureHandle(
            ql.FlatForward(today,0.01,ql.Actual365Fixed()))
        projYtsH = ql.YieldTermStructureHandle(
            ql.FlatForward(today,0.02,ql.Actual365Fixed()))
        index = ql.Euribor6M(projYtsH)
        # we set start in the future to avoid the need of index fixings
        startDate  = ql.Date(12,ql.October,2020)
        endDate    = ql.Date(12,ql.October,2030)
        calendar   = ql.TARGET()
        fixedTenor = ql.Period('1y')
        floatTenor = ql.Period('6m')
        fixedSchedule = ql.MakeSchedule(startDate,endDate,tenor=fixedTenor,calendar=calendar)
        floatSchedule = ql.MakeSchedule(startDate,endDate,tenor=floatTenor,calendar=calendar)
        swapType = ql.VanillaSwap.Payer
        vanillaSwap = ql.VanillaSwap(ql.VanillaSwap.Receiver,1.0,fixedSchedule,0.025,ql.Thirty360(),floatSchedule,index,0.0050,ql.Actual360())
        # we have all the incredients to price the swap - but not really needed for payoffs
        engine = ql.DiscountingSwapEngine(discYtsH)
        vanillaSwap.setPricingEngine(engine)
        # It is easier to work with legs instead of Swap intruments
        self.legs = [vanillaSwap.fixedLeg(), vanillaSwap.floatingLeg()]
        self.pors = [1.0, -1.0] if swapType==ql.VanillaSwap.Receiver else [-1.0, 1.0]
        self.discYtsH = discYtsH

    def test_SwapSetup(self):
        swap = Swap(self.legs,self.pors,self.discYtsH)
        #swap.cashFlows(8.1)
        timeLine = swap.timeLine([1.0, 2.0, 3.0])
        #
        for t in timeLine:
            print('ObsTime: %.2f' % t)
            for p in timeLine[t]:
                print(p)

    @unittest.skip('Too time consuming')
    def test_SwapSimulation(self):
        model = HWModel()
        obsTimes = np.linspace(0.0,10.0,121)
        nPaths = 2**7
        seed = 314159265359
        mcSim = MCSimulation(model,obsTimes,nPaths,seed,True)
        swap = Swap(self.legs,self.pors,self.discYtsH)
        scen = swap.scenarios(obsTimes,mcSim)
        epe = np.average(np.maximum(scen,0.0),axis=0)
        plt.plot(obsTimes,epe)
        plt.show()

    def test_AmcSwapSetup(self):
        model = HWModel()
        obsTimes = np.array([0.0])
        nPaths = 1
        seed = 1
        mcSim = MCSimulation(model,obsTimes,nPaths,seed,True)
        swap = AmcSwap(self.legs,self.pors,mcSim,2,self.discYtsH)
        #swap.cashFlows(8.1)
        timeLine = swap.timeLine([1.0, 2.0, 3.0])
        #
        for t in timeLine:
            print('ObsTime: %.2f' % t)
            for p in timeLine[t]:
                print(p)

    @unittest.skip('Too time consuming')
    def test_AmcSwapSimulation(self):
        model = HWModel()
        obsTimes = np.linspace(0.0,10.0,121)
        nPaths = 2**7
        # calibration
        seed0 = 314159265359
        mcSim0 = MCSimulation(model,obsTimes,nPaths,seed0,True)
        # simulation
        seed0 = 141592653593
        mcSim1 = MCSimulation(model,obsTimes,nPaths,seed0,True)
        swap = AmcSwap(self.legs,self.pors,mcSim0,2,self.discYtsH)
        scen = swap.scenarios(obsTimes,mcSim1)
        epe = np.average(np.maximum(scen,0.0),axis=0)
        plt.plot(obsTimes,epe)
        plt.show()


    def test_BothSwapSimulation(self):
        model = HWModel()
        obsTimes = np.linspace(0.0,10.0,121)
        nPaths = 2**7
        # calibration
        seed0 = 314159265359
        mcSim0 = MCSimulation(model,obsTimes,nPaths,seed0,True)
        # simulation
        seed0 = 141592653593
        mcSim1 = MCSimulation(model,obsTimes,nPaths,seed0,True)
        #
        swap = Swap(self.legs,self.pors,self.discYtsH)
        scen = swap.scenarios(obsTimes,mcSim1)
        epe = np.average(np.maximum(scen,0.0),axis=0)
        plt.plot(obsTimes,epe)
        #
        swap = AmcSwap(self.legs,self.pors,mcSim0,2,self.discYtsH)
        scen = swap.scenarios(obsTimes,mcSim1)
        epe = np.average(np.maximum(scen,0.0),axis=0)
        plt.plot(obsTimes,epe)
        #
        plt.show()


if __name__ == '__main__':
    unittest.main()
    
