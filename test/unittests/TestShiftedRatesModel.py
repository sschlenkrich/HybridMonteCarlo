#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.AffineShortRateModel import CoxIngersollRossModel
from hybmc.models.ShiftedRatesModel import ShiftedRatesModel

from hybmc.simulations.McSimulation import McSimulation

import matplotlib.pyplot as plt

class TestShiftedRatesModel(unittest.TestCase):

    def test_ModelCurve(self):
        # CIR parameters
        r0         = 0.02
        chi_       = 0.07
        theta_     = 0.05
        sigma_     = 0.10
        CirModel = CoxIngersollRossModel(r0,chi_,theta_,sigma_)
        model = CirModel
        T = np.linspace(0.0,10.0,11)
        dt = 1.0/365.0
        f = np.array([ np.log( \
            model.zeroBondPrice(0.0,T_,model.r0) / model.zeroBondPrice(0.0,T_+dt,model.r0)) / dt \
            for T_ in T ])
        # plt.plot(T,f,label='CIR')
        #
        curve = YieldCurve(0.03)
        model = ShiftedRatesModel(curve,CirModel)
        X0 = model.initialValues()
        f = np.array([ np.log( \
            model.zeroBond(0.0,T_,X0,None) / model.zeroBond(0.0,T_+dt,X0,None)) / dt \
            for T_ in T ])
        # plt.plot(T,f,label='Shifted')
        # plt.legend()
        # plt.show()
        # print(f - 0.03)
        self.assertLess(np.max(np.abs(f-0.03)),7.e-14)


if __name__ == '__main__':
    unittest.main()
    
