#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
import matplotlib.pyplot as plt

from src.termstructures.YieldCurve import YieldCurve
from src.models.HullWhiteModel import HullWhiteModel
from src.simulations.MCSimulation import MCSimulation
from src.simulations.Payoffs import Payoff, Fixed, LiborRate
from src.mathutils.Regression import Regression
from src.simulations.AmcPayoffs import AmcMax, AmcMin, AmcOne, AmcSum


class TestAmcPayoffs(unittest.TestCase):

    # set up the stage for testing the models
    def setUp(self):
        self.curve = YieldCurve(0.00)
        mean  = 0.01
        times = np.array([ 10.0   ])
        vols  = np.array([ 0.0050 ])
        self.model = HullWhiteModel(self.curve, mean, times, vols)
        #
        times = np.linspace(0.0, 10.0, 11)
        # regression
        nPaths = 2**7
        seed = 314159265359
        self.mcSim0 = MCSimulation(self.model,times,nPaths,seed,False)
        # valuation
        nPaths = 2**7
        seed = 141592653593
        self.mcSim1 = MCSimulation(self.model,times,nPaths,seed,False)


    def test_AmcRegression(self):
        x = [ Fixed(1.0) @ 10.0 ]
        y = [ Fixed(1.0) @ 5.0  ]
        z = [ LiborRate(5.0, 5.0, 10.0) ]
        degree = 2
        # we manually set up a regression
        N = self.mcSim0.nPaths
        T = np.zeros(N)
        Z = np.zeros([N,1])
        for k in range(N):
            p = self.mcSim0.path(k)
            T[k] = p.numeraire(5.0)/p.numeraire(10.0) - 1.0
            Z[k,0] = z[0].at(p)
        refReg = Regression(Z,T)
        # visual check
        # plt.plot(Z[:,0],T,'r.')
        # plt.plot(Z[:,0],[refReg.value(z) for z in Z],'b.')
        # plt.show()
        refBeta = refReg.beta
        # and compare versus payoff
        aMax = AmcMax(5.0,x,y,z,self.mcSim0,degree)
        aMax.calibrateRegression()
        beta = aMax.regression.beta
        # check
        self.assertListEqual(list(beta.shape),list(refBeta.shape))
        self.assertListEqual(list(beta.round(12)),list(refBeta.round(12)))
        # check for consistency with min
        aMin = AmcMin(5.0,x,y,z,self.mcSim0,degree)
        aMin.calibrateRegression()
        self.assertListEqual(list(aMin.regression.beta),list(beta))
        # check for consistency with one
        aOne = AmcOne(5.0,x,y,z,self.mcSim0,degree)
        aOne.calibrateRegression()
        self.assertListEqual(list(aOne.regression.beta),list(beta))

    def test_AmcValuationViaRegression(self):
        x = [ Fixed(1.0) @ 10.0 ]
        y = [ Fixed(1.0) @ 5.0  ]
        z = [ LiborRate(5.0, 5.0, 10.0) ]
        degree = 2
        # we manually set up a regression
        N = self.mcSim0.nPaths
        T = np.zeros(N)
        Z = np.zeros([N,1])
        for k in range(N):
            p = self.mcSim0.path(k)
            T[k] = p.numeraire(5.0)/p.numeraire(10.0) - 1.0
            Z[k,0] = z[0].at(p)
        refReg = Regression(Z,T)
        #
        aMax = AmcMax(5.0,x,y,z,self.mcSim0,degree)
        aMin = AmcMin(5.0,x,y,z,self.mcSim0,degree)
        aOne = AmcOne(5.0,x,y,z,self.mcSim0,degree)
        # pricing
        aMax = np.array([ aMax.discountedAt(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        aMin = np.array([ aMin.discountedAt(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        aOne = np.array([ aOne.discountedAt(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        # reference results
        aMaxRef = np.zeros(self.mcSim1.nPaths)
        aMinRef = np.zeros(self.mcSim1.nPaths)
        aOneRef = np.zeros(self.mcSim1.nPaths)
        for k in range(self.mcSim1.nPaths):
            p = self.mcSim1.path(k)
            T = refReg.value(np.array([ [z[0].at(p)] ]))            
            aMaxRef[k] = 1.0/p.numeraire(10.0) if T>0.0 else 1.0/p.numeraire(5.0)
            aMinRef[k] = 1.0/p.numeraire(10.0) if T<0.0 else 1.0/p.numeraire(5.0)
            aOneRef[k] = 1.0/p.numeraire(5.0)  if T>0.0 else 0.0
        self.assertListEqual(list(aMax.round(14)),list(aMaxRef.round(14)))
        self.assertListEqual(list(aMin.round(14)),list(aMinRef.round(14)))
        self.assertListEqual(list(aOne.round(14)),list(aOneRef.round(14)))

    def test_AmcValuationWithoutRegression(self):
        x = [ Fixed(1.0) @ 10.0 ]
        y = [ Fixed(1.0) @ 5.0  ]
        #
        aMax = AmcMax(5.0,x,y,None,None,None)
        aMin = AmcMin(5.0,x,y,None,None,None)
        aOne = AmcOne(5.0,x,y,None,None,None)
        # pricing
        aMax = np.array([ aMax.discountedAt(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        aMin = np.array([ aMin.discountedAt(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        aOne = np.array([ aOne.discountedAt(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        # reference results
        aMaxRef = np.zeros(self.mcSim1.nPaths)
        aMinRef = np.zeros(self.mcSim1.nPaths)
        aOneRef = np.zeros(self.mcSim1.nPaths)
        for k in range(self.mcSim1.nPaths):
            p = self.mcSim1.path(k)
            T = 1.0/p.numeraire(10.0) - 1.0/p.numeraire(5.0)
            aMaxRef[k] = 1.0/p.numeraire(10.0) if T>0.0 else 1.0/p.numeraire(5.0)
            aMinRef[k] = 1.0/p.numeraire(10.0) if T<0.0 else 1.0/p.numeraire(5.0)
            aOneRef[k] = 1.0/p.numeraire(5.0)  if T>0.0 else 0.0
        self.assertListEqual(list(aMax.round(14)),list(aMaxRef.round(14)))
        self.assertListEqual(list(aMin.round(14)),list(aMinRef.round(14)))
        self.assertListEqual(list(aOne.round(14)),list(aOneRef.round(14)))

    def test_AmcFutureBias(self):
        x = [ Fixed(1.0) @ 10.0 ]
        y = [ Fixed(1.0) @ 5.0  ]
        z = [ LiborRate(5.0, 5.0, 10.0) ]
        degree = 2
        #
        aMax0 = AmcMax(5.0,x,y,z,self.mcSim0,degree)
        aMin0 = AmcMin(5.0,x,y,z,self.mcSim0,degree)
        #
        aMax0 = np.array([ aMax0.at(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        aMin0 = np.array([ aMin0.at(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        #
        aMax0 = np.average(aMax0)
        aMin0 = np.average(aMin0)
        #
        aMax1 = AmcMax(5.0,x,y,None,None,None)
        aMin1 = AmcMin(5.0,x,y,None,None,None)
        #
        aMax1 = np.array([ aMax1.at(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        aMin1 = np.array([ aMin1.at(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        #
        aMax1 = np.average(aMax1)
        aMin1 = np.average(aMin1)
        #
        self.assertLess(aMax0, aMax1)
        self.assertGreater(aMin0, aMin1)


    def test_AmcConditionalExpectation(self):
        # regress identity
        z = [ LiborRate(5.0, 5.0, 10.0) @ 5.0 ]
        degree = 2
        #
        aSum0 = AmcSum(5.0,z,z,self.mcSim0,degree)
        aSum1 = AmcSum(5.0,z,None,None,None)
        tmp = aSum0
        #
        aSum0 = np.array([ aSum0.discountedAt(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        aSum1 = np.array([ aSum1.discountedAt(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        self.assertListEqual(list(tmp.regression.beta.round(14)),[0.0, 1.0, 0.0])
        #
        aSum0 = np.average(aSum0)
        aSum1 = np.average(aSum1)
        self.assertAlmostEqual(aSum0,aSum1,14)
        #
        x = [ Fixed(1.0) @ 10.0 ]
        aSum0 = AmcSum(5.0,x,z,self.mcSim0,degree)
        aSum1 = AmcSum(5.0,x,None,None,None)
        tmp = aSum0
        #
        libs = np.array([ z[0].at(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ]) 
        aSum0 = np.array([ aSum0.at(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        aSum1 = np.array([ aSum1.at(self.mcSim1.path(k))
            for k in range(self.mcSim1.nPaths) ])
        # visual check
        # plt.plot(libs,aSum0,'r.')
        # plt.plot(libs,aSum1,'b.')
        # plt.show()
        #
        aSum0 = np.average(aSum0)
        aSum1 = np.average(aSum1)
        self.assertAlmostEqual(aSum0,aSum1,2)


    def test_AmcPayoffStrings(self):
        x = [ Fixed(1.0) @ 10.0 ]
        y = [ Fixed(1.0) @ 5.0  ]
        z = [ LiborRate(5.0, 5.0, 10.0) ]
        self.assertEqual(str(AmcMax(5.0,x,y,z,self.mcSim0,2)),'AmcMax(5.00,[(1.0000 @ 10.00)],[(1.0000 @ 5.00)];[L_None(5.00;5.00,10.00)])')
        self.assertEqual(str(AmcMin(5.0,x,y,z,self.mcSim0,2)),'AmcMin(5.00,[(1.0000 @ 10.00)],[(1.0000 @ 5.00)];[L_None(5.00;5.00,10.00)])')
        self.assertEqual(str(AmcOne(5.0,x,y,z,self.mcSim0,2)),'AmcOne(5.00,[(1.0000 @ 10.00)],[(1.0000 @ 5.00)];[L_None(5.00;5.00,10.00)])')
        self.assertEqual(str(AmcSum(5.0,x,z,self.mcSim0,2)),  'AmcSum(5.00,[(1.0000 @ 10.00)];[L_None(5.00;5.00,10.00)])')
        self.assertEqual(str(AmcMax(5.0,x,y,None,None,None)), 'AmcMax(5.00,[(1.0000 @ 10.00)],[(1.0000 @ 5.00)])')
        self.assertEqual(str(AmcMin(5.0,x,y,None,None,None)), 'AmcMin(5.00,[(1.0000 @ 10.00)],[(1.0000 @ 5.00)])')
        self.assertEqual(str(AmcOne(5.0,x,y,None,None,None)), 'AmcOne(5.00,[(1.0000 @ 10.00)],[(1.0000 @ 5.00)])')
        self.assertEqual(str(AmcSum(5.0,x,None,None,None)),   'AmcSum(5.00,[(1.0000 @ 10.00)])')


if __name__ == '__main__':
    unittest.main()
    
