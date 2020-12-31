#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.MarkovFutureModel import MarkovFutureModel
from hybmc.models.QuasiGaussianModel import QuasiGaussianModel
from hybmc.simulations.McSimulation import McSimulation

import matplotlib.pyplot as plt

class TestMarkovFutureModel(unittest.TestCase):

    def test_SigmaFunction(self):
        d = 2
        times = np.array([1.0, 3.0])
        sigma = np.zeros([2,2,2])
        sigma[0,:,:] = np.array([[ 0.50, 0.60 ],
                                 [ 0.70, 0.80 ]])
        sigma[1,:,:] = np.array([[ 0.55, 0.65 ],
                                 [ 0.75, 0.85 ]])
        chi = np.array([0.1, 0.2])
        #
        model = MarkovFutureModel(None,d,times,sigma,chi)
        #
        T = np.linspace(-1.0, 5.0, 13)
        vols = np.array([ model.sigma(t) for t in T ])
        # plt.plot(T,vols[:,0,0],label='sigma[0,0]')
        # plt.plot(T,vols[:,0,1],label='sigma[0,1]')
        # plt.plot(T,vols[:,1,0],label='sigma[1,0]')
        # plt.plot(T,vols[:,1,1],label='sigma[1,1]')
        # plt.legend()
        # plt.show()
        refVols = np.array([
            [[0.5,  0.6 ], [0.7,  0.8 ]],
            [[0.5,  0.6 ], [0.7,  0.8 ]],
            [[0.5,  0.6 ], [0.7,  0.8 ]],
            [[0.5,  0.6 ], [0.7,  0.8 ]],
            [[0.5,  0.6 ], [0.7,  0.8 ]],
            [[0.55, 0.65], [0.75, 0.85]],
            [[0.55, 0.65], [0.75, 0.85]],
            [[0.55, 0.65], [0.75, 0.85]],
            [[0.55, 0.65], [0.75, 0.85]],
            [[0.55, 0.65], [0.75, 0.85]],
            [[0.55, 0.65], [0.75, 0.85]],
            [[0.55, 0.65], [0.75, 0.85]],
            [[0.55, 0.65], [0.75, 0.85]]])
        self.assertEqual(np.max(np.abs(vols-refVols)),0.0)
            
    def test_AuxilliaryState(self):
        d = 2
        times = np.array([0.0])
        sigma = np.zeros([1,2,2])
        sigma[0,:,:] = np.array([[ 0.50, 0.60 ],
                                 [ 0.70, 0.80 ]])
        chi = np.array([0.1, 0.2])
        #
        model0 = MarkovFutureModel(None,d,times,sigma,chi)
        self.assertEqual(np.max(np.abs(model0._y-np.zeros([1,2,2]))),0.0)
        #
        times = np.array([1.0, 3.0])
        sigma = np.zeros([2,2,2])
        sigma[0,:,:] = np.array([[ 0.50, 0.60 ],
                                 [ 0.70, 0.80 ]])
        sigma[1,:,:] = np.array([[ 0.50, 0.60 ],
                                 [ 0.70, 0.80 ]])
        #
        model1 = MarkovFutureModel(None,d,times,sigma,chi)
        T = np.linspace(0.0, 5.0, 6)
        y0 = np.array([ model0.y(t) for t in T ])
        y1 = np.array([ model1.y(t) for t in T ])
        self.assertLess(np.max(np.abs(y0 - y1)),1.0e-15)
        
    def test_AuxilliaryStateVsQuasiGaussian(self):
        curve = YieldCurve(0.03)
        d     = 3

        times = np.array([   1.0,    2.0,    5.0,    10.0     ])
        sigma = np.array([ [ 0.0060, 0.0070, 0.0080,  0.0100  ],
                           [ 0.0030, 0.0035, 0.0040,  0.0050  ],
                           [ 0.0025, 0.0025, 0.0025,  0.0025  ] ])
        slope = np.zeros([3,4])
        curve = np.zeros([3,4])

        delta = np.array([ 1.0, 5.0, 20.0 ])
        chi   = np.array([ 0.01, 0.05, 0.15 ])
      
        Gamma = np.array([ [1.0, 0.8, 0.6],
                           [0.8, 1.0, 0.8],
                           [0.6, 0.8, 1.0] ])
        qGmodel = QuasiGaussianModel(curve,d,times,sigma,slope,curve,delta,chi,Gamma)
        X0 = qGmodel.initialValues()
        #T = np.linspace(-1.0, 11.0, 121)
        #vols = np.array([ qGmodel.sigma_xT(t,X0) for t in T ])
        #plt.plot(T,vols[:,0,0],label='sigma[0,0]')
        vols = np.array([ qGmodel.sigma_xT(t,X0).T for t in times ])
        #plt.plot(times,vols[:,0,0],'*',label='sigma[0,0]')
        #plt.show()
        #
        mVmodel = MarkovFutureModel(None,d,times,vols,chi)
        # we need to evolve the Quasi Gaussian model to obtain y(t)
        times = np.linspace(0.0, 11.0, 121)
        sim = McSimulation(qGmodel,times,1,1,showProgress=False)
        Y0 = sim.X[0,:,d:d+d*d]
        Y0.shape = [121,3,3]
        Y1 = np.array([ mVmodel.y(t) for t in times ])
        # print(np.max(np.abs(Y1[1:]/Y0[1:] - 1.0)))
        self.assertLess(np.max(np.abs(Y1[1:]/Y0[1:] - 1.0)),0.01)
        # we need to fix sigma(t+0.5dt) in QuasiGaussian model to
        # expect a better consistency here
        
    def test_MartingalePropertyConstParameters(self):
        d = 2
        times = np.array([0.0])
        sigma = np.zeros([1,2,2])
        sigma[0,:,:] = np.array([[ 0.10, 0.15 ],
                                 [ 0.20, 0.25 ]])
        chi = np.array([0.1, 0.2])
        #
        model0 = MarkovFutureModel(None,d,times,sigma,chi)
        times = np.linspace(0.0, 5.0, 2)
        nPaths = 2**10
        seed = 314159265359
        sim = McSimulation(model0,times,nPaths,seed,False,showProgress=False)
        dT = np.linspace(0.0, 5.0, 6)
        for k, t_ in enumerate(times):
            for dT_ in dT:
                F = np.array([ model0.futurePrice(t_,t_+dT_,X,None) for X in sim.X[:,k,:] ])
                # print(np.mean(F))
                self.assertLess(np.abs(np.mean(F) - 1.0),0.07)
        #
        times = np.linspace(0.0, 5.0, 6)
        nPaths = 2**10
        sim = McSimulation(model0,times,nPaths,seed,False,showProgress=False)
        dT = np.linspace(0.0, 5.0, 3)
        for k, t_ in enumerate(times):
            for dT_ in dT:
                F = np.array([ model0.futurePrice(t_,t_+dT_,X,None) for X in sim.X[:,k,:] ])
                # print(np.abs(np.mean(F) - 1.0))
                self.assertLess(np.abs(np.mean(F) - 1.0),0.07)
        #

    def test_MartingalePropertyTimeDependentParameters(self):
        d = 2
        times = np.array([2.0, 4.0])
        sigma = np.zeros([2,2,2])
        sigma[0,:,:] = np.array([[ 0.10, 0.15 ],
                                 [ 0.20, 0.25 ]])
        sigma[1,:,:] = np.array([[ 0.15, 0.20 ],
                                 [ 0.25, 0.30 ]])
        chi = np.array([0.1, 0.2])
        #
        model0 = MarkovFutureModel(None,d,times,sigma,chi)
        #
        times = np.linspace(0.0, 5.0, 6)
        nPaths = 2**10
        seed = 314159265359
        sim = McSimulation(model0,times,nPaths,seed,False,showProgress=False)
        dT = np.linspace(0.0, 5.0, 3)
        for k, t_ in enumerate(times):
            for dT_ in dT:
                F = np.array([ model0.futurePrice(t_,t_+dT_,X,None) for X in sim.X[:,k,:] ])
                #print(np.abs(np.mean(F) - 1.0))
                self.assertLess(np.abs(np.mean(F) - 1.0),0.07)
        #




if __name__ == '__main__':
    unittest.main()
    
