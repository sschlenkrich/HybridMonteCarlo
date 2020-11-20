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

from hybmc.wrappers.JuliaPayoff import JuliaPayoff, JuliaPayoffs
from hybmc.wrappers.JuliaSimulation import JuliaSimulation, JuliaDiscountedAt

class TestJulia(unittest.TestCase):

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

    def test_JuliaSimulation(self):
        model = self.sim.model
        simTimes = self.sim.times
        nPaths = 2**7
        seed = 314159265359
        timeInterpolation = True
        sim = McSimulation(model,simTimes,nPaths,seed,timeInterpolation)
        #
        jSim = JuliaSimulation(sim)                                   # just copy
        self.assertEqual(np.max(np.abs(sim.X-jSim.X)), 0.0)
        jSim = JuliaSimulation(sim,simulate=True,useBrownians=True)   # only evolve
        self.assertLess(np.max(np.abs(sim.X-jSim.X)), 1.0e-15)
        jSim = JuliaSimulation(sim,simulate=True,useBrownians=False)  # simulate with new Brownians
        self.assertGreater(np.max(np.abs(sim.X-jSim.X)), 0.29)        # should not be close
        #
        simTimes = np.array([0.0, 1.0, 2.0])
        nPaths = 2
        seed = 112
        timeInterpolation = False
        jSim = JuliaSimulation(sim,simulate=True,nPaths=nPaths,times=simTimes,seed=seed,timeInterpolation=timeInterpolation)  # adjust simulation details
        self.assertEqual(jSim.nPaths,nPaths)
        self.assertEqual(jSim.seed,seed)
        self.assertEqual(jSim.timeInterpolation,timeInterpolation)
        self.assertEqual(np.max(np.abs(jSim.times - simTimes)), 0.0)

    def test_JuliaPayoff(self):
        model = self.sim.model
        simTimes = self.sim.times
        nPaths = 2**7
        seed = 314159265359
        timeInterpolation = True
        sim = McSimulation(model,simTimes,nPaths,seed,timeInterpolation)
        jSim = JuliaSimulation(sim,simulate=True,useBrownians=True)   # only evolve
        #
        payoffs = self.timeline[0.0]
        scens = np.array([ [ payoff.discountedAt(path) for path in sim.paths() ]
                           for payoff in payoffs ])
        jPayoffs = JuliaPayoffs(payoffs)
        jScens = JuliaDiscountedAt(jSim,jPayoffs)
        self.assertLess(np.max(np.abs(jScens - scens)),1.0e-15)



if __name__ == '__main__':
    unittest.main()
    



