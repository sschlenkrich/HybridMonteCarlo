#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.HullWhiteModel import HullWhiteModel, HullWhiteModelWithDiscreteNumeraire
from hybmc.simulations.McSimulation import McSimulation
from hybmc.simulations.Payoffs import Pay, LiborRate

class TestHullWhiteMonteCarlo(unittest.TestCase):

    # set up the stage for testing the models
    def setUp(self):
        self.curve = YieldCurve(0.03)
        mean  = 0.03
        times = np.array([ 1.0,    2.0,    5.0,    10.0     ])
        vols  = np.array([ 0.0060, 0.0070, 0.0080,  0.0100  ])
        self.modelRiskNeutral = HullWhiteModel(self.curve, mean, times, vols)
        self.modelDiscreteFwd = HullWhiteModelWithDiscreteNumeraire(self.curve, mean, times, vols)

                
    def test_monteCarloSimulation(self):
        times = np.linspace(0.0, 10.0, 11)
        nPaths = 2**11
        seed = 1234
        # risk-neutral simulation
        mcSim = McSimulation(self.modelRiskNeutral,times,nPaths,seed)
        discZeroBonds = np.array([
            [ 1.0 / self.modelRiskNeutral.numeraire(t,x) for t,x in zip(times,path) ]
            for path in mcSim.X ])
        mcZeroBondsRiskNeutral = np.average(discZeroBonds,axis=0)
        # discrete forward measure simulation
        mcSim = McSimulation(self.modelDiscreteFwd,times,nPaths,seed)
        discZeroBonds = np.array([
            [ 1.0 / self.modelDiscreteFwd.numeraire(t,x) for t,x in zip(times,path) ]
            for path in mcSim.X ])
        mcZeroBondsDiscreteFwd = np.average(discZeroBonds,axis=0)
        # curve
        zeroBonds = np.array([ self.curve.discount(t) for t in times ])
        print('  T     ZeroRate  RiskNeutral  DiscreteFwd')
        for k, t in enumerate(times[1:]):
            zeroRate    = -np.log(zeroBonds[k+1])/t
            riskNeutral = -np.log(mcZeroBondsRiskNeutral[k+1])/t
            discreteFwd = -np.log(mcZeroBondsDiscreteFwd[k+1])/t
            print(' %4.1f   %6.4lf    %6.4lf       %6.4lf' % (t, zeroRate, riskNeutral, discreteFwd) )
            self.assertAlmostEqual(zeroRate,riskNeutral,3)
            self.assertAlmostEqual(zeroRate,discreteFwd,3)

    def test_simulationStates(self):
        times = np.array([0.0, 2.0, 5.0, 10.0])
        nPaths = 2
        seed = 1234
        mcSim = McSimulation(self.modelRiskNeutral,times,nPaths,seed,False)
        # test exact values
        for k, t in enumerate(times):
            self.assertListEqual(list(mcSim.state(0,t)),list(mcSim.X[0][k]))
        # now we allow interpolation/extrapolation
        mcSim.timeInterpolation = True
        # test extrapolation
        self.assertListEqual(list(mcSim.state(1,-0.5)),list(mcSim.X[1][0]))
        self.assertListEqual(list(mcSim.state(1,12.5)),list(mcSim.X[1][3]))
        # test interpolation
        self.assertListEqual(list(mcSim.state(1,1.0)),list(0.5*(mcSim.X[1][0] + mcSim.X[1][1])))
        self.assertListEqual(list(mcSim.state(1,7.0)),list((0.6*mcSim.X[1][2] + 0.4*mcSim.X[1][3])))


    def test_pathGeneration(self):
        times = np.array([0.0, 2.0, 5.0, 10.0])
        nPaths = 2
        seed = 1234
        mcSim = McSimulation(self.modelRiskNeutral,times,nPaths,seed,True)
        idx = 0
        dT  = 5.7
        path = mcSim.path(idx)
        for t in np.linspace(-1.0, 11.0, 13):
            s = mcSim.state(idx,t)
            self.assertEqual(path.numeraire(t),mcSim.model.numeraire(t,s))
            self.assertEqual(path.zeroBond(t,t+dT,None),mcSim.model.zeroBond(t,t+dT,s,None))

    def test_liborPayoff(self):
        # Libor Rate payoff
        liborTimes = np.linspace(0.0, 10.0, 11)
        dT0 = 2.0 / 365
        dT1 = 0.5
        cfs = [ Pay(LiborRate(t,t+dT0,t+dT0+dT1),t+dT0+dT1) for t in liborTimes]
        times = set.union(*[ p.observationTimes() for p in cfs ])
        times = np.array(sorted(list(times)))
        #
        nPaths = 2**11
        seed = 1234
        # risk-neutral simulation
        mcSim = McSimulation(self.modelRiskNeutral,times,nPaths,seed)
        #
        cfPVs = np.array([
            [ p.discountedAt(mcSim.path(idx)) for p in cfs ] for idx in range(nPaths) ])
        cfPVs = np.average(cfPVs,axis=0)
        discounts0 = np.array([ mcSim.model.yieldCurve.discount(t+dT0) for t in liborTimes ])
        discounts1 = np.array([ mcSim.model.yieldCurve.discount(t+dT0+dT1) for t in liborTimes ])
        liborsCv = (discounts0/discounts1 - 1.0)/dT1
        liborsMC = cfPVs / discounts1
        print('  T     LiborRate  MnteCarlo')
        for k,t in enumerate(liborTimes):
            print(' %4.1f   %6.4lf     %6.4lf' % (t, liborsCv[k], liborsMC[k]) )
            self.assertAlmostEqual(liborsCv[k],liborsMC[k],2)


if __name__ == '__main__':
    unittest.main()
    
