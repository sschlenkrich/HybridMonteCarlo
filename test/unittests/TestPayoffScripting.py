#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
from src.termstructures.YieldCurve import YieldCurve
from src.models.HullWhiteModel import HullWhiteModel, HullWhiteModelWithDiscreteNumeraire
from src.simulations.MCSimulation import MCSimulation
from src.simulations.Payoffs import Payoff, Fixed, Pay, Asset, ZeroBond, LiborRate

class TestPayoffScripting(unittest.TestCase):

    # set up the stage for testing the models
    def setUp(self):
        self.curve = YieldCurve(0.03)
        mean  = 0.03
        times = np.array([ 1.0,    2.0,    5.0,    10.0     ])
        vols  = np.array([ 0.0060, 0.0070, 0.0080,  0.0100  ])
        self.modelRiskNeutral = HullWhiteModel(self.curve, mean, times, vols)
        self.modelDiscreteFwd = HullWhiteModelWithDiscreteNumeraire(self.curve, mean, times, vols)

    def test_staticPayoffs(self):
        a  = 0.5
        px = 1.0
        py = 2.0
        x = Fixed(px)
        y = Fixed(py)
        # 
        self.assertEqual((x+y).at(None), px+py)
        self.assertEqual((x-y).at(None), px-py)
        self.assertEqual((x*y).at(None), px*py)
        self.assertEqual((x/y).at(None), px/py)
        # 
        self.assertEqual((x+a).at(None), px+a)
        self.assertEqual((x-a).at(None), px-a)
        self.assertEqual((x*a).at(None), px*a)
        self.assertEqual((x/a).at(None), px/a)
        # 
        self.assertEqual((a+y).at(None), a+py)
        self.assertEqual((a-y).at(None), a-py)
        self.assertEqual((a*y).at(None), a*py)
        self.assertEqual((a/y).at(None), a/py)
        #
        z = a*x + a*x*y - y
        self.assertEqual(z.at(None), a*px + a*px*py - py)

    def test_payoffTimes(self):
        L1 = LiborRate(1.0, 1.0, 1.5)
        L2 = LiborRate(2.0, 3.0, 3.5)
        P  = Pay(L1 + L2 + 1.0, 5.0)
        self.assertListEqual(list(P.observationTimes()),[0.0, 1.0, 2.0, 5.0])

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
        mcSim = MCSimulation(self.modelRiskNeutral,times,nPaths,seed)
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
    
