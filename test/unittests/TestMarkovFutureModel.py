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
        sigmaT = np.zeros([2,2,2])
        sigmaT[0,:,:] = np.array([[ 0.50, 0.60 ],
                                  [ 0.70, 0.80 ]])
        sigmaT[1,:,:] = np.array([[ 0.55, 0.65 ],
                                  [ 0.75, 0.85 ]])
        chi = np.array([0.1, 0.2])
        #
        model = MarkovFutureModel(None,d,times,sigmaT,chi)
        #
        T = np.linspace(-1.0, 5.0, 13)
        vols = np.array([ model.sigmaT(t) for t in T ])
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
        sigmaT = np.zeros([1,2,2])
        sigmaT[0,:,:] = np.array([[ 0.50, 0.60 ],
                                 [ 0.70, 0.80 ]])
        chi = np.array([0.1, 0.2])
        #
        model0 = MarkovFutureModel(None,d,times,sigmaT,chi)
        self.assertEqual(np.max(np.abs(model0._y-np.zeros([1,2,2]))),0.0)
        #
        times = np.array([1.0, 3.0])
        sigmaT = np.zeros([2,2,2])
        sigmaT[0,:,:] = np.array([[ 0.50, 0.60 ],
                                  [ 0.70, 0.80 ]])
        sigmaT[1,:,:] = np.array([[ 0.50, 0.60 ],
                                  [ 0.70, 0.80 ]])
        #
        model1 = MarkovFutureModel(None,d,times,sigmaT,chi)
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
        vols = np.array([ qGmodel.sigma_xT(t,X0) for t in times ])
        #plt.plot(times,vols[:,0,0],'*',label='sigma[0,0]')
        #plt.show()
        #
        mVmodel = MarkovFutureModel(None,d,times,vols,chi)
        # we need to evolve the Quasi Gaussian model to obtain y(t)
        times = np.linspace(0.0, 11.0, 111)
        sim = McSimulation(qGmodel,times,1,1,showProgress=False)
        Y0 = sim.X[0,:,d:d+d*d]
        Y0.shape = [111,3,3]
        Y1 = np.array([ mVmodel.y(t) for t in times ])
        # print(np.max(np.abs(Y1[1:]/Y0[1:] - 1.0)))
        self.assertLess(np.max(np.abs(Y1[1:]/Y0[1:] - 1.0)),6.2e-14)
        # we need to fix sigma(t+0.5dt) in QuasiGaussian model to
        # expect a better consistency here
        
    # @unittest.skip('Too time consuming')
    def test_MartingalePropertyConstParameters(self):
        d = 2
        times = np.array([0.0])
        sigmaT = np.zeros([1,2,2])
        sigmaT[0,:,:] = np.array([[ 0.10, 0.15 ],
                                  [ 0.20, 0.25 ]])
        chi = np.array([0.1, 0.2])
        #
        model0 = MarkovFutureModel(None,d,times,sigmaT,chi)
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
                self.assertLess(np.abs(np.mean(F) - 1.0),0.05)
        #

    # @unittest.skip('Too time consuming')
    def test_MartingalePropertyTimeDependentParameters(self):
        d = 2
        times = np.array([2.0, 4.0])
        sigmaT = np.zeros([2,2,2])
        sigmaT[0,:,:] = np.array([[ 0.10, 0.15 ],
                                  [ 0.20, 0.25 ]])
        sigmaT[1,:,:] = np.array([[ 0.15, 0.20 ],
                                  [ 0.25, 0.30 ]])
        chi = np.array([0.1, 0.2])
        #
        model0 = MarkovFutureModel(None,d,times,sigmaT,chi)
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

    def test_KnownSimulationValues(self):
        d = 2
        times = np.array([2.0, 4.0])
        sigmaT = np.zeros([2,2,2])
        sigmaT[0,:,:] = np.array([[ 0.10, 0.15 ],
                                  [ 0.20, 0.25 ]]).T
        sigmaT[1,:,:] = np.array([[ 0.15, 0.20 ],
                                  [ 0.25, 0.30 ]]).T
        chi = np.array([0.1, 0.2])
        #
        model0 = MarkovFutureModel(None,d,times,sigmaT,chi)
        #
        times = np.linspace(0.0, 5.0, 6)
        nPaths = 2**3
        seed = 314159265359
        sim = McSimulation(model0,times,nPaths,seed,False,showProgress=False)
        # np.set_printoptions(precision=16)
        # print(sim.X[:,-1,:])
        Xref = np.array([
            [-0.6000110699011598, -0.6431768273428073],
            [ 0.138146279288649 ,  0.1270588210285978],
            [-0.2615455421299724, -0.3054110432018231],
            [-1.1042559026028758, -1.0151899131194109],
            [-0.934746072227638 , -1.1542393979975838],
            [-0.0771484387711384, -0.1328561062878627],
            [-0.5411246272833143, -0.4660671538747663],
            [ 0.6805897799894314,  0.7389950985079005] ])
        #print(np.max(np.abs(sim.X[:,-1,:]-Xref)))
        self.assertLess(np.max(np.abs(sim.X[:,-1,:]-Xref)), 1.2e-16)

    def test_ZeroMeanReversion(self):
        d = 2
        times = np.array([0.0])
        sigmaT = np.zeros([1,2,2])
        sigmaT[0,:,:] = np.array([[ 0.10, 0.15 ],
                                  [ 0.20, 0.25 ]]).T
        dt = 2.0
        dW = np.ones(2)
        for chi1_ in [ 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14, 0.0 ]:
            chi0_ = 0.1
            model0 = MarkovFutureModel(None,d,times,sigmaT,np.array([chi0_, chi1_]))
            model1 = MarkovFutureModel(None,d,times,sigmaT,np.array([chi0_, 0.1 * chi1_]))
            #
            X0 = model0.initialValues()
            X1_0 = model0.initialValues()
            X1_1 = model0.initialValues()
            #
            model0.evolve(0.0,X0,dt,dW,X1_0)
            model1.evolve(0.0,X0,dt,dW,X1_1)
            #
            # print(np.max(np.abs(X1_0 - X1_1)))
            self.assertLessEqual(np.max(np.abs(X1_0 - X1_1)), chi1_)


if __name__ == '__main__':
    unittest.main()
    
