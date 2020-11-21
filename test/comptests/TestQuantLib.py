#!/usr/bin/python

from julia import Main

import sys
sys.path.append('./')

import unittest

import numpy as np
import pandas as pd
import QuantLib as ql

from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.QuasiGaussianModel import QuasiGaussianModel
from hybmc.simulations.McSimulation import McSimulation
from hybmc.products.Swap import Swap

#from hybmc.simulations.JuliaPayoff import JuliaPayoff, JuliaPayoffs
#from hybmc.simulations.JuliaSimulation import JuliaSimulation, JuliaDiscountedAt

from hybmc.wrappers.QuantLibPayoffs import QuantLibPayoff, QuantLibPayoffs
from hybmc.wrappers.QuantLibSimulation import QuantLibSimulation, QuantLibDiscountedAt

class TestQuantLib(unittest.TestCase):

    def setUp(self):
        today     = ql.Date(5,ql.October,2020)
        ql.Settings.instance().evaluationDate = today
        #
        discYtsH   = ql.YieldTermStructureHandle(
                         ql.FlatForward(today,0.015,ql.Actual365Fixed()))
        projYtsH   = ql.YieldTermStructureHandle(
                         ql.FlatForward(today,0.020,ql.Actual365Fixed()))
        index      = ql.Euribor6M(projYtsH)
        startDate  = ql.Date(12,ql.October,2020)
        endDate    = ql.Date(12,ql.October,2030)
        calendar   = ql.TARGET()
        fixedTenor = ql.Period('1y')
        floatTenor = ql.Period('6m')
        fixedSchedule = ql.MakeSchedule(startDate,endDate,tenor=fixedTenor,calendar=calendar)
        floatSchedule = ql.MakeSchedule(startDate,endDate,tenor=floatTenor,calendar=calendar)
        couponDayCount = ql.Thirty360()
        notional   = 1.0
        fixedRate  = 0.02
        fixedLeg   = ql.FixedRateLeg(fixedSchedule,couponDayCount,[notional],[fixedRate])
        floatingLeg = ql.IborLeg([notional],floatSchedule,index)
        #
        swap = Swap([fixedLeg,floatingLeg],[1.0,-1.0],discYtsH)
        #
        observationTimes = np.linspace(0.0,10.0,11)
        self.timeline = swap.timeLine(observationTimes)
        #
        yc = YieldCurve(0.02)
        d = 2
        times = np.array(  [ 1.0,    5.0,   10.0    ])
        sigma = np.array([ [ 0.0050, 0.0060, 0.0070 ],
                           [ 0.0050, 0.0060, 0.0070 ] ])
        slope = np.array([ [ 0.0100, 0.0100, 0.0100 ],
                           [ 0.0200, 0.0200, 0.0200 ] ])
        curve = np.array([ [ 0.0000, 0.0000, 0.0000 ],
                           [ 0.0000, 0.0000, 0.0000 ] ])
        delta = np.array(  [ 1.0,  20.0 ])
        chi   = np.array(  [ 0.01, 0.15 ])
        Gamma = np.identity(2)        
        model = QuasiGaussianModel(yc,d,times,sigma,slope,curve,delta,chi,Gamma)
        #
        nPaths = 1
        simTimes = observationTimes
        seed = 314159265359
        timeInterpolation = True
        self.sim = McSimulation(model,simTimes,nPaths,seed,timeInterpolation,False)

    def test_QuantLibSimulation(self):
        model = self.sim.model
        simTimes = self.sim.times
        nPaths = 2**7
        seed = 314159
        timeInterpolation = True
        sim = McSimulation(model,simTimes,nPaths,seed,timeInterpolation,False)
        #
        qSim = QuantLibSimulation(sim)                                   # just copy
        #
        simTimes = np.array([0.0, 1.0, 2.0])
        nPaths = 2
        seed = 112
        timeInterpolation = False
        qSim = QuantLibSimulation(sim,nPaths=nPaths,times=simTimes,seed=seed,timeInterpolation=timeInterpolation)  # adjust simulation details
        self.assertEqual(qSim.nPaths(),nPaths)
        self.assertEqual(np.max(np.abs(qSim.simTimes() - simTimes)), 0.0)

 
    def test_QuantLibPayoff(self):
        model = self.sim.model
        simTimes = self.sim.times
        nPaths = 2**7
        seed = 3141592
        timeInterpolation = True
        sim = McSimulation(model,simTimes,nPaths,seed,timeInterpolation,False)
        qSim = QuantLibSimulation(sim) 
        #
        payoffs = self.timeline[0.0]
        scens = np.array([ [ payoff.discountedAt(path) for path in sim.paths() ]
                           for payoff in payoffs ])                           
        #
        qPayoffs = QuantLibPayoffs(payoffs)
        qScens = QuantLibDiscountedAt(qSim,qPayoffs)
        #
        self.assertAlmostEqual(np.mean(qScens),np.mean(scens),places=16)  # that's trivial at t=0



if __name__ == '__main__':
    unittest.main()
    



