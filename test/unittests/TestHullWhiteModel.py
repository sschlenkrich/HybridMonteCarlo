#!/usr/bin/python

import sys
sys.path.append('../../src/')

import unittest

import numpy as np
from termstructures.YieldCurve import YieldCurve
from models.HullWhiteModel import HullWhiteModel

class TestHullWhiteModel(unittest.TestCase):

    # set up the stage for testing the models
    def setUp(self):
        self.curve = YieldCurve(0.03)
        mean  = 0.03
        times = np.array([ 1.0,    2.0,    5.0,    10.0     ])
        vols  = np.array([ 0.0060, 0.0070, 0.0080,  0.0100  ])
        self.model = HullWhiteModel(self.curve, mean, times, vols)
        
    def test_zeroBondPrice(self):
        obsTime = 10.0
        matTime = 20.0
        states  = [ -0.10, -0.05, 0.00, 0.03, 0.07, 0.12 ]
        refResult = [  # reference results checked against QuantLib
            1.7178994273756056,
            1.1153102854629025,
            0.7240918839816156,
            0.5587692049384843,
            0.39550396436404667,
            0.25677267968500767] 
        for x, res in zip(states,refResult):
            self.assertEqual(self.model.zeroBondPrice(obsTime,matTime,x),res)
        
    def test_couponBondOption(self):
        excTime = 10.0
        payTimes  = [ 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 20.0 ]
        cashFlows = [ 0.025] * 10 + [ 1.0 ]  # bond cash flows
        callOrPut = 1.0 # call
        strikes   = [ 0.8, 0.9, 1.0, 1.1, 1.2 ]
        refResult = [  # reference results checked against QuantLib
            0.12572604970665557,
            0.07466919528750032,
            0.039969719617809936,
            0.019503405494683164,
            0.008804423283331253]
        for strike, res in zip(strikes,refResult):
            self.assertEqual(self.model.couponBondOption(excTime,payTimes,cashFlows,strike,callOrPut),res)


if __name__ == '__main__':
    unittest.main()
    
