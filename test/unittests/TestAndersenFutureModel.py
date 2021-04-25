#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
from hybmc.models.AndersenFutureModel import AndersenFutureModel
from hybmc.models.MarkovFutureModel import MarkovFutureModel

from hybmc.simulations.McSimulation import McSimulation

import matplotlib.pyplot as plt

class TestAndersenFutureModel(unittest.TestCase):

    def test_ModelSetup(self):
        # Andersen example
        kappa       = 1.35
        sigma_0     = 0.50
        sigma_infty = 0.17
        rho_infty   = 0.50
        model = AndersenFutureModel(None,kappa,sigma_0,sigma_infty,rho_infty)
        #print(model.h1(0.0))
        #print(model.h2(0.0))
        #print(model.hInfty())
        self.assertAlmostEqual(model.h1(0.0),0.08,16)
        self.assertAlmostEqual(model.h2(0.0),0.4330127018922193,16)
        self.assertAlmostEqual(model.hInfty(),0.17,16)

    def test_MartingaleProperty(self):
        kappa       = 1.35
        sigma_0     = 0.50
        sigma_infty = 0.17
        rho_infty   = 0.50
        model = AndersenFutureModel(None,kappa,sigma_0,sigma_infty,rho_infty)
        #
        times = np.linspace(0.0, 5.0, 3)
        nPaths = 2**13
        seed = 14159265359
        sim = McSimulation(model,times,nPaths,seed,False,showProgress=False)
        #
        for idx in range(1,times.shape[0]):
            t = times[idx]
            for dT in [ 0.0, 1.0, 2.0, 5.0, 10.0]:
                T = t + dT
                F = np.array([ model.futurePrice(t,T,X,None) for X in sim.X[:,idx,:] ])
                Fav = np.mean(F)
                sigma = np.std(F) / np.sqrt(t)
                # print('t: %6.2f, T: %6.2f, F: %6.4f, sigma: %6.4f' % (t,T,Fav,sigma) )
                # print(np.abs(Fav - 1.0))
                self.assertLess(np.abs(Fav - 1.0),0.0026)

    def test_CompareWithMarkovModel(self):
        kappa       = 1.35
        sigma_0     = 0.50
        sigma_infty = 0.17
        rho_infty   = 0.50
        model0 = AndersenFutureModel(None,kappa,sigma_0,sigma_infty,rho_infty)
        #
        def sigmaT(sigma_0, sigma_infty, rho_infty):
            h1 = -sigma_infty + rho_infty * sigma_0
            h2 = sigma_0 * np.sqrt(1.0 - rho_infty**2)
            hi = sigma_infty
            return np.array([ [h1, h2], [hi, 0.0] ])
        #
        def chi(kappa):
            return np.array([ kappa, 0.0 ])
        #
        sigmaT_ = sigmaT(sigma_0,sigma_infty,rho_infty)
        chi_    = chi(kappa)
        #
        d = 2
        times = np.array([0.0])
        model1 = MarkovFutureModel(None,d,times,np.array([ sigmaT_ ]),chi_)
        #
        times = np.linspace(0.0, 5.0, 3)
        nPaths = 2**3
        seed = 14159265359
        #
        sim0 = McSimulation(model0,times,nPaths,seed,False,showProgress=False)
        sim1 = McSimulation(model1,times,nPaths,seed,False,showProgress=False)
        #
        for idx in range(1,times.shape[0]):
            t = times[idx]
            for dT in [ 0.0, 1.0, 2.0, 5.0, 10.0]:
                T = t + dT
                F0 = np.array([ model0.futurePrice(t,T,X,None) for X in sim0.X[:,idx,:] ])
                F1 = np.array([ model1.futurePrice(t,T,X,None) for X in sim1.X[:,idx,:] ])
                # print(F0-F1)
                self.assertLess(np.max(np.abs(F0-F1)),3.0e-16)


if __name__ == '__main__':
    unittest.main()

