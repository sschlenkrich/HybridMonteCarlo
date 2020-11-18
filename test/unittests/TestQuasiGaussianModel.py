#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.QuasiGaussianModel import QuasiGaussianModel

class TestQuasiGaussianModel(unittest.TestCase):

    # set up the stage for testing the models
    def test_ModelSetup(self):
        curve = YieldCurve(0.03)
        d     = 3

        times = np.array([ 1.0,    2.0,    5.0,    10.0     ])

        sigma = np.array([ [ 0.0060, 0.0070, 0.0080,  0.0100  ],
                           [ 0.0030, 0.0035, 0.0040,  0.0050  ],
                           [ 0.0025, 0.0025, 0.0025,  0.0025  ] ])

        slope = np.array([ [ 0.10,   0.15,   0.20  ,  0.25    ],
                           [ 0.20,   0.35,   0.40  ,  0.55    ],
                           [ 0.10,   0.10,   0.10  ,  0.10    ] ])

        curve = np.array([ [ 0.05,   0.05,   0.05  ,  0.05    ],
                           [ 0.10,   0.10,   0.10  ,  0.10    ],
                           [ 0.20,   0.15,   0.10  ,  0.00    ] ])

        delta = np.array([ 1.0, 5.0, 20.0 ])
        chi   = np.array([ 0.01, 0.05, 0.15 ])
      
        Gamma = np.array([ [1.0, 0.8, 0.6],
                           [0.8, 1.0, 0.8],
                           [0.6, 0.8, 1.0] ])

        # test matricies
        model = QuasiGaussianModel(curve,d,times,sigma,slope,curve,delta,chi,Gamma)
        self.assertTrue(np.allclose(model._DfT @ model._DfT.T,Gamma))
        Hf_H_inv = np.exp(np.array([ [ -chi_ * delta_ for chi_ in chi ] for delta_ in delta ]))
        self.assertTrue(np.allclose(Hf_H_inv @ model._HHfInv,np.eye(d)))
        aliases = model.stateAliases()
        aliasesRef = ['x_0', 'x_1', 'x_2', 'y_0_0', 'y_0_1', 'y_0_2', 'y_1_0', 'y_1_1', 'y_1_2', 'y_2_0', 'y_2_1', 'y_2_2', 's']
        self.assertListEqual(aliases,aliasesRef)
        aliases = model.factorAliases()
        aliasesRef = ['x_0', 'x_1', 'x_2']
        self.assertListEqual(aliases,aliasesRef)

        
        # test parameter functions
        times = np.linspace(0.0, 11.0, 23)
        #
        sigma = np.array([ [t] + [ model.sigma(i,t) for i in range(d) ]
                           for t in times ])
        sigmaRef = np.array([
            [0.00e+00, 6.00e-03, 3.00e-03, 2.50e-03],
            [5.00e-01, 6.00e-03, 3.00e-03, 2.50e-03],
            [1.00e+00, 6.00e-03, 3.00e-03, 2.50e-03],
            [1.50e+00, 7.00e-03, 3.50e-03, 2.50e-03],
            [2.00e+00, 7.00e-03, 3.50e-03, 2.50e-03],
            [2.50e+00, 8.00e-03, 4.00e-03, 2.50e-03],
            [3.00e+00, 8.00e-03, 4.00e-03, 2.50e-03],
            [3.50e+00, 8.00e-03, 4.00e-03, 2.50e-03],
            [4.00e+00, 8.00e-03, 4.00e-03, 2.50e-03],
            [4.50e+00, 8.00e-03, 4.00e-03, 2.50e-03],
            [5.00e+00, 8.00e-03, 4.00e-03, 2.50e-03],
            [5.50e+00, 1.00e-02, 5.00e-03, 2.50e-03],
            [6.00e+00, 1.00e-02, 5.00e-03, 2.50e-03],
            [6.50e+00, 1.00e-02, 5.00e-03, 2.50e-03],
            [7.00e+00, 1.00e-02, 5.00e-03, 2.50e-03],
            [7.50e+00, 1.00e-02, 5.00e-03, 2.50e-03],
            [8.00e+00, 1.00e-02, 5.00e-03, 2.50e-03],
            [8.50e+00, 1.00e-02, 5.00e-03, 2.50e-03],
            [9.00e+00, 1.00e-02, 5.00e-03, 2.50e-03],
            [9.50e+00, 1.00e-02, 5.00e-03, 2.50e-03],
            [1.00e+01, 1.00e-02, 5.00e-03, 2.50e-03],
            [1.05e+01, 1.00e-02, 5.00e-03, 2.50e-03],
            [1.10e+01, 1.00e-02, 5.00e-03, 2.50e-03] ])
        self.assertTrue(np.allclose(sigma,sigmaRef))
        #
        slope = np.array([ [t] + [ model.slope(i,t) for i in range(d) ]
                           for t in times ])
        slopeRef = np.array([
            [ 0. ,   0.1 ,  0.2 ,  0.1 ],
            [ 0.5,   0.1 ,  0.2 ,  0.1 ],
            [ 1. ,   0.1 ,  0.2 ,  0.1 ],
            [ 1.5,   0.15,  0.35,  0.1 ],
            [ 2. ,   0.15,  0.35,  0.1 ],
            [ 2.5,   0.2 ,  0.4 ,  0.1 ],
            [ 3. ,   0.2 ,  0.4 ,  0.1 ],
            [ 3.5,   0.2 ,  0.4 ,  0.1 ],
            [ 4. ,   0.2 ,  0.4 ,  0.1 ],
            [ 4.5,   0.2 ,  0.4 ,  0.1 ],
            [ 5. ,   0.2 ,  0.4 ,  0.1 ],
            [ 5.5,   0.25,  0.55,  0.1 ],
            [ 6. ,   0.25,  0.55,  0.1 ],
            [ 6.5,   0.25,  0.55,  0.1 ],
            [ 7. ,   0.25,  0.55,  0.1 ],
            [ 7.5,   0.25,  0.55,  0.1 ],
            [ 8. ,   0.25,  0.55,  0.1 ],
            [ 8.5,   0.25,  0.55,  0.1 ],
            [ 9. ,   0.25,  0.55,  0.1 ],
            [ 9.5,   0.25,  0.55,  0.1 ],
            [10. ,   0.25,  0.55,  0.1 ],
            [10.5,   0.25,  0.55,  0.1 ],
            [11. ,   0.25,  0.55,  0.1 ] ])
        self.assertTrue(np.allclose(slope,slopeRef))
        #
        curve = np.array([ [t] + [ model.curve(i,t) for i in range(d) ]
                           for t in times ])
        curveRef = np.array([
            [ 0. ,   0.05,  0.1,   0.2 ],
            [ 0.5,   0.05,  0.1,   0.2 ],
            [ 1. ,   0.05,  0.1,   0.2 ],
            [ 1.5,   0.05,  0.1,   0.15],
            [ 2. ,   0.05,  0.1,   0.15],
            [ 2.5,   0.05,  0.1,   0.1 ],
            [ 3. ,   0.05,  0.1,   0.1 ],
            [ 3.5,   0.05,  0.1,   0.1 ],
            [ 4. ,   0.05,  0.1,   0.1 ],
            [ 4.5,   0.05,  0.1,   0.1 ],
            [ 5. ,   0.05,  0.1,   0.1 ],
            [ 5.5,   0.05,  0.1,   0.  ],
            [ 6. ,   0.05,  0.1,   0.  ],
            [ 6.5,   0.05,  0.1,   0.  ],
            [ 7. ,   0.05,  0.1,   0.  ],
            [ 7.5,   0.05,  0.1,   0.  ],
            [ 8. ,   0.05,  0.1,   0.  ],
            [ 8.5,   0.05,  0.1,   0.  ],
            [ 9. ,   0.05,  0.1,   0.  ],
            [ 9.5,   0.05,  0.1,   0.  ],
            [10. ,   0.05,  0.1,   0.  ],
            [10.5,   0.05,  0.1,   0.  ],
            [11. ,   0.05,  0.1,   0.  ] ])
        self.assertTrue(np.allclose(curve,curveRef))
        
    def test_ZeroBond(self):
        yts = YieldCurve(0.03)
        d     = 3
        times = np.array([  10.0     ])
        sigma = np.array([ [ 0.0060 ],
                           [ 0.0050 ],
                           [ 0.0040 ] ])
        slope = np.array([ [ 0.10  ],
                           [ 0.20  ],
                           [ 0.30  ] ])
        curve = np.array([ [ 0.05  ],
                           [ 0.10  ],
                           [ 0.20  ] ])
        delta = np.array([ 1.0, 5.0, 20.0 ])
        chi   = np.array([ 0.01, 0.05, 0.15 ])
        Gamma = np.array([ [1.0, 0.8, 0.6],
                           [0.8, 1.0, 0.8],
                           [0.6, 0.8, 1.0] ])
        model = QuasiGaussianModel(yts,d,times,sigma,slope,curve,delta,chi,Gamma)
        #
        x = np.random.uniform(0.0,0.05,d)
        y = np.random.uniform(0.0,0.05,[d,d])
        y = y @ y.T
        t = 5.0
        T = 10.0
        #
        X = np.append(x,y.reshape((d*d,)))
        X = np.append(X,[0.0])
        zcb = model.zeroBond(t,T,X,None)
        #
        G = (1.0 - np.exp(-chi*(T-t)))/chi
        zcbRef = yts.discount(T) / yts.discount(t) * np.exp(-G.dot(x) - 0.5 * G.dot(y).dot(G) )
        self.assertAlmostEqual(zcb,zcbRef,14)


    def test_Sigma_xT(self):
        yts = YieldCurve(0.03)
        d     = 3
        times = np.array([  10.0     ])
        sigma = np.array([ [ 0.0060 ],
                           [ 0.0050 ],
                           [ 0.0040 ] ])
        slope = np.array([ [ 0.10  ],
                           [ 0.20  ],
                           [ 0.30  ] ])
        curve = np.array([ [ 0.05  ],
                           [ 0.10  ],
                           [ 0.20  ] ])
        delta = np.array([ 1.0, 5.0, 20.0 ])
        chi   = np.array([ 0.01, 0.05, 0.15 ])
        Gamma = np.array([ [1.0, 0.8, 0.6],
                           [0.8, 1.0, 0.8],
                           [0.6, 0.8, 1.0] ])
        model = QuasiGaussianModel(yts,d,times,sigma,slope,curve,delta,chi,Gamma)
        #
        x = np.array([0.00206853, 0.01093424, 0.00696041])  # np.random.uniform(0.0,0.05,d)
        y = np.array([[0.0298081,  0.00656372, 0.0041344 ],
                      [0.04282753, 0.02406725, 0.0001211 ],
                      [0.02906687, 0.01078535, 0.02328022]])  # np.random.uniform(0.0,0.05,[d,d])
        y = y @ y.T
        t = 5.0
        #
        X = np.append(x,y.reshape((d*d,)))
        X = np.append(X,[0.0])
        sigma_xT = model.sigma_xT(t,X)
        # reference result from QuantLib
        sigma_xT_Ref = np.array([
            [ 0.031658475910,   0.017375011335,   0.084573118461 ],
            [-0.007058576846,   0.021654204395,  -0.134352049767 ],
            [-0.016417240779,  -0.043917617245,   0.051199735933 ] ])
        self.assertTrue(np.allclose(sigma_xT,sigma_xT_Ref,rtol=0.0,atol=1.0e-12))


    def test_Evolve(self):
        # we set up a test case via QuantLib
        zeroRate = 0.03
        d     = 3
        times = np.array([  10.0     ])
        sigma = np.array([ [ 0.0060 ],
                           [ 0.0050 ],
                           [ 0.0040 ] ])
        slope = np.array([ [ 0.10  ],
                           [ 0.20  ],
                           [ 0.30  ] ])
        curve = np.array([ [ 0.05  ],
                           [ 0.10  ],
                           [ 0.20  ] ])
        delta = np.array([ 1.0, 5.0, 20.0 ])
        chi   = np.array([ 0.01, 0.05, 0.15 ])
        Gamma = np.array([ [1.0, 0.8, 0.6],
                           [0.8, 1.0, 0.8],
                           [0.6, 0.8, 1.0] ])
        # try QuantLib
        try:
            import QuantLib as ql
            ytsH = ql.YieldTermStructureHandle(ql.FlatForward(0,ql.NullCalendar(),zeroRate,ql.Actual365Fixed()))
            qg = ql.QuasiGaussianModel(ytsH,d,times,sigma,slope,curve,[ 0.0 ],delta,chi,Gamma,0.1)
            mc = ql.RealMCSimulation(qg,[0.0, 5.0, 10.0],[0.0, 5.0, 10.0],1,1234,False,False,True)
            mc.simulate()
            dW = np.array(mc.brownian(0))
            dW = dW[:,:3]
            #print(dW)
            X = np.array(mc.observedPath(0))
            Xref = np.delete(X,12,1)
            #print(Xref)
            qlzcb = ql.RealMCZeroBond(5.0,10.0,'')
            zcbRef = ql.RealMCPayoffPricer_at(qlzcb,mc)[0]
            #
            atol = 1.0e-14
            equalPlaces = 15
        except:
            # we save QL results in case we don't have QL available
            ytsH = YieldCurve(zeroRate)
            dW = np.array([
                [-0.8723107, -0.00585635, 0.31102388],
                [-0.15673275, 0.28482762, 0.79041947]])
            Xref = np.array([
                [ 0.        , 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                [-0.02182255, 0.0397456 , -0.01717905,  0.0006412 , -0.00116167,  0.00059747, -0.00116167,  0.00259077, -0.00141925,  0.00059747, -0.00141925,  0.00088815,  0.00185998],
                [-0.02476015, 0.04705765, -0.0083194 ,  0.00127306, -0.00232704,  0.00102049, -0.00232704,  0.00518216, -0.00243878,  0.00102049, -0.00243878,  0.00133699,  0.03866521]])
            #
            zcbRef = 0.851676232405887
            #
            atol = 1.0e-8
            equalPlaces = 9
        #
        model = QuasiGaussianModel(ytsH,d,times,sigma,slope,curve,delta,chi,Gamma)
        X = np.zeros([3,13])
        model.evolve(0.0,X[0],5.0,dW[0],X[1])
        model.evolve(5.0,X[1],5.0,dW[1],X[2])
        self.assertTrue(np.allclose(X,Xref,rtol=0.0,atol=atol))
        zcb = model.zeroBond(5.0,10.0,X[1],None)
        self.assertAlmostEqual(zcb,zcbRef,equalPlaces)




if __name__ == '__main__':
    unittest.main()
