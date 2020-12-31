#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
from hybmc.models.AffineShortRateModel import AffineShortRateModel, CoxIngersollRossModel, fullTruncation, lognormalApproximation, quadraticExponential
from hybmc.simulations.McSimulation import McSimulation

import matplotlib.pyplot as plt

class TestAffineShortRateModel(unittest.TestCase):

    def setUp(self):
        modelTimes = np.array([3.0, 6.0, 10.0])
        r0         = 0.05
        chi        = np.array([0.10, 0.15, 0.20])
        theta      = np.array([0.06, 0.07, 0.08])
        sigma      = np.array([0.15, 0.10, 0.08])
        alpha      = np.array([0.03, 0.02, 0.01])
        beta       = np.array([1.0,  1.0,  1.0 ])
        self.model = AffineShortRateModel(r0,modelTimes,chi,theta,sigma,alpha,beta)


    def test_ModelSetup(self):
        # print(self.model.stateAliases())
        # print(self.model.factorAliases())
        self.assertEqual(self.model.stateAliases(),['r', 's'])
        self.assertEqual(self.model.factorAliases(),['r'])

    def test_ModelEvolution(self):
        model = self.model
        #
        times = np.linspace(0.0, 10.0, 11)
        nPaths = 2**3
        seed = 314159265359
        # risk-neutral simulation
        mcSim = McSimulation(model,times,nPaths,seed,showProgress=False)
        #
        T = 10
        P10 = np.array([ 1.0 / model.numeraire(T,x) for x in mcSim.X[:,-1,:] ])
        #print(np.mean(P10))
        #print(model.zeroBondPrice(0.0,10.0,model.r0))
        self.assertAlmostEqual(np.mean(P10),0.6480670673701923, 12)
        self.assertAlmostEqual(model.zeroBondPrice(0.0,10.0,model.r0),0.5997174951900259, 12)
    
    def test_RicattiAB(self):
        # CIR parameters
        r0         = 0.02
        chi_       = 0.03
        theta_     = 0.05
        sigma_     = 0.10
        CirModel = CoxIngersollRossModel(r0,chi_,theta_,sigma_)
        # Affine Short Rate Parameters
        modelTimes = np.array([ 10.0   ])
        chi        = np.array([ chi_   ])
        theta      = np.array([ theta_ ])
        sigma      = np.array([ sigma_ ])
        alpha      = np.array([ 0.0 ])
        beta       = np.array([ 1.0 ])
        AsrModel = AffineShortRateModel(r0,modelTimes,chi,theta,sigma,alpha,beta)
        #
        t  = np.linspace(0.0, 10.0, 11)
        dt = np.linspace(0.0, 10.0, 11)
        #
        cirRicattiAB = np.array([ CirModel.ricattiAB(t_,t_+dt_,0.0,1.0) for t_ in t for dt_ in dt ])
        AsrRicattiAB = np.array([ AsrModel.ricattiAB(t_,t_+dt_,0.0,1.0) for t_ in t for dt_ in dt ])
        diff = AsrRicattiAB - cirRicattiAB        
        # print(diff)
        # print(np.max(np.abs(diff)))
        self.assertLess(np.max(np.abs(diff)),1.4e-6)


    def test_ModelYieldCurve(self):
        # CIR parameters
        r0         = 0.02
        chi_       = 0.07
        theta_     = 0.05
        sigma_     = 0.10
        CirModel = CoxIngersollRossModel(r0,chi_,theta_,sigma_)
        model = CirModel
        # model = self.model
        T = np.linspace(0.0,10.0,11)
        dt = 1.0/365.0
        f = np.array([ np.log( \
            model.zeroBondPrice(0.0,T_,model.r0) / model.zeroBondPrice(0.0,T_+dt,model.r0)) / dt \
            for T_ in T ])
        #plt.plot(T,f)
        #plt.show()

    def test_SqrtProcessEvolution(self):
        # CIR parameters
        r0         = 0.02
        chi_       = 0.03
        theta_     = 0.05
        sigma_     = 0.10
        model0 = CoxIngersollRossModel(r0,chi_,theta_,sigma_,fullTruncation)
        model1 = CoxIngersollRossModel(r0,chi_,theta_,sigma_,lognormalApproximation)
        model2 = CoxIngersollRossModel(r0,chi_,theta_,sigma_,quadraticExponential(1.5))
        models = [model0, model1, model2]
        # MC simulation params
        t  = 5.0
        dt = 1.0
        rt = np.linspace(0.0, 0.10,11)
        dw = np.linspace(-1.0,1.0,11)
        for rt_ in rt:
            results = []
            for dw_ in dw:
                res = np.zeros(3)
                for k, model in enumerate(models):
                    X1 = np.array([0.0,0.0])
                    model.evolve(t,np.array([rt_,0.0]),dt,np.array([dw_]),X1)
                    res[k] = X1[0]
                results.append(res)
            results = np.array(results)
            #plt.plot(dw,results[:,0],label='FT')
            #plt.plot(dw,results[:,1],label='LN')
            #plt.plot(dw,results[:,2],label='QE')
            #plt.legend()
            #plt.show()
            #print(np.max(np.abs(results[:,0]-results[:,1])))
            #print(np.max(np.abs(results[:,1]-results[:,2])))
            self.assertLess(np.max(np.abs(results[:,0]-results[:,1])),0.0048)
            self.assertLess(np.max(np.abs(results[:,1]-results[:,2])),0.0027)


if __name__ == '__main__':
    unittest.main()
    
