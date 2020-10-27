#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np


from src.termstructures.YieldCurve import YieldCurve
from src.models.HullWhiteModel import HullWhiteModel
from src.models.DeterministicModel import DcfModel
from src.models.CreditModel import CreditModel


from src.simulations.McSimulation import McSimulation

# a quick way to get a model
def HWModel(rate=0.01, vol=0.0050, mean=0.03):
    curve = YieldCurve(rate)
    times = np.array([ 10.0 ])
    vols  = np.array([ vol  ])
    return HullWhiteModel(curve, mean, times, vols)

class TestCreditModel(unittest.TestCase):

    def test_ModelSetup(self):
        baseModel = HWModel()
        name1Model = HWModel(0.03, 0.0100, 0.10)
        name2Model = DcfModel(YieldCurve(0.02))
        name2Model.domAlias = 'Two'
        corr = np.identity(2)
        model = CreditModel(baseModel,['One', 'Two'],[name1Model,name2Model],corr)
        # basic tests
        self.assertEqual(model.size(),4)
        self.assertEqual(model.factors(),2)
        self.assertListEqual(model.stateAliases(),['x', 's', 'One_x', 'One_s'])  # no state for DCF model
        self.assertListEqual(model.factorAliases(),['x', 'One_x'])  # no state for DCF model
        #
        times = np.array([0.0, 5.0])
        nPaths = 1
        seed = 314159265359
        # risk-neutral simulation
        mcSim = McSimulation(model,times,nPaths,seed,False)
        p = mcSim.path(0)
        #
        self.assertEqual(p.zeroBond(0.0, 10.0, None), baseModel.yieldCurve.discount(10.0) )
        #
        self.assertEqual(p.hazardProcess(0.0,'One'), 1.0)
        self.assertAlmostEqual(p.hazardRate(0.0,'One'),0.03,13)
        self.assertEqual(p.survivalProb(0.0,10.0,'One'),name1Model.yieldCurve.discount(10.0))
        #
        self.assertEqual(p.hazardProcess(5.0,'Two'), name2Model.domCurve.discount(5.0))
        self.assertAlmostEqual(p.hazardRate(5.0,'Two'),0.02,13)
        self.assertEqual(p.survivalProb(5.0,10.0,'Two'),
            name2Model.domCurve.discount(10.0) / name2Model.domCurve.discount(5.0))
        # evolve manually
        X0 = baseModel.initialValues()
        X1 = np.array(X0)
        baseModel.evolve(0.0,X0,5.0,mcSim.dW[0,0,0:1],X1)
        self.assertTrue(np.allclose(X1,mcSim.X[0,1,:2],rtol=0.0,atol=1.0e-14))
        X0 = name1Model.initialValues()
        X1 = np.array(X0)
        name1Model.evolve(0.0,X0,5.0,mcSim.dW[0,0,1:2],X1)
        self.assertTrue(np.allclose(X1,mcSim.X[0,1,2:],rtol=0.0,atol=1.0e-14))



if __name__ == '__main__':
    unittest.main()
