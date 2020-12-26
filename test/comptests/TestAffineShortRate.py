#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.AffineShortRateModel import AffineShortRateModel, CoxIngersollRossModel, fullTruncation, lognormalApproximation, quadraticExponential
from hybmc.simulations.McSimulation import McSimulation

import matplotlib.pyplot as plt
import pandas as pd

class TestAffineShortRateModel(unittest.TestCase):

    def test_CirSimulation(self):
        r0         = 0.02
        chi_       = 0.07
        theta_     = 0.05
        sigma_     = 0.10
        modelFT = CoxIngersollRossModel(r0,chi_,theta_,sigma_,fullTruncation)
        modelLN = CoxIngersollRossModel(r0,chi_,theta_,sigma_,lognormalApproximation)
        modelQE = CoxIngersollRossModel(r0,chi_,theta_,sigma_,quadraticExponential(1.5))
        #
        dT = 5.0
        times = np.linspace(0.0, 10.0, 11)
        zcbs = np.array([ modelFT.zeroBondPrice(0.0,T + dT,r0) for T in times ])
        #
        nPaths = 2**13
        seed = 314159265359
        # risk-neutral simulation
        simFT = McSimulation(modelFT,times,nPaths,seed,showProgress=True)
        simLN = McSimulation(modelLN,times,nPaths,seed,showProgress=True)
        simQE = McSimulation(modelQE,times,nPaths,seed,showProgress=True)
        #
        zcbFT = np.mean(np.array([
            [ modelFT.zeroBond(times[t],times[t]+dT,simFT.X[p,t,:],None) / modelFT.numeraire(times[t],simFT.X[p,t,:]) for t in range(len(times)) ]
            for p in range(nPaths) ]), axis=0)
        zcbLN = np.mean(np.array([
            [ modelLN.zeroBond(times[t],times[t]+dT,simLN.X[p,t,:],None) / modelLN.numeraire(times[t],simLN.X[p,t,:]) for t in range(len(times)) ]
            for p in range(nPaths) ]), axis=0)
        zcbQE = np.mean(np.array([
            [ modelQE.zeroBond(times[t],times[t]+dT,simQE.X[p,t,:],None) / modelQE.numeraire(times[t],simQE.X[p,t,:]) for t in range(len(times)) ]
            for p in range(nPaths) ]), axis=0)
        #
        results = pd.DataFrame([ times, zcbs, zcbFT, zcbLN, zcbQE ]).T
        results.columns = ['times', 'zcbs', 'zcbFT', 'zcbLN', 'zcbQE']
        print(results)
            


if __name__ == '__main__':
    unittest.main()
    
