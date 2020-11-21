#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import numpy as np
from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.DeterministicModel import DcfModel
from hybmc.simulations.Payoffs import Payoff, Fixed, Pay, Asset, ZeroBond, LiborRate, SwapRate, Max, Min

class TestPayoffScripting(unittest.TestCase):

    def test_staticPayoffs(self):
        a  = 0.5
        px = 1.0
        py = 2.0
        x = Fixed(px)
        y = Fixed(py)
        # 
        self.assertEqual((x+y).at(None), px+py)
        self.assertEqual((x-y).at(None), px-py)
        self.assertEqual((x*y).at(None), px*py)
        self.assertEqual((x/y).at(None), px/py)
        # 
        self.assertEqual((x+a).at(None), px+a)
        self.assertEqual((x-a).at(None), px-a)
        self.assertEqual((x*a).at(None), px*a)
        self.assertEqual((x/a).at(None), px/a)
        # 
        self.assertEqual((a+y).at(None), a+py)
        self.assertEqual((a-y).at(None), a-py)
        self.assertEqual((a*y).at(None), a*py)
        self.assertEqual((a/y).at(None), a/py)
        #
        z = a*x + a*x*y - y
        self.assertEqual(z.at(None), a*px + a*px*py - py)
        self.assertEqual((~z).at(not None), a*px + a*px*py - py)  # we can't use None here
        #
        self.assertEqual( (x<y).at(None),float(px<py))
        self.assertEqual((x<=y).at(None),float(px<=py))
        self.assertEqual((x==y).at(None),float(px==py))
        self.assertEqual((x>=y).at(None),float(px>=py))
        self.assertEqual( (x>y).at(None),float(px>py))
        #
        self.assertEqual( (x<a).at(None),float(px<a))
        self.assertEqual((x<=a).at(None),float(px<=a))
        self.assertEqual((x==a).at(None),float(px==a))
        self.assertEqual((x>=a).at(None),float(px>=a))
        self.assertEqual( (x>a).at(None),float(px>a))
        #
        self.assertEqual( (a<y).at(None),float(a<py))
        self.assertEqual((a<=y).at(None),float(a<=py))
        self.assertEqual((a==y).at(None),float(a==py))
        self.assertEqual((a>=y).at(None),float(a>=py))
        self.assertEqual( (a>y).at(None),float(a>py))
        #
        self.assertEqual((x+y).obsTime, 0.0)
        self.assertEqual(((x+y)@1.0).obsTime, 1.0)

    def test_PayoffStrings(self):
        self.assertEqual(str(Fixed(1.0)@1.0),            '(1.0000 @ 1.00)')
        self.assertEqual(str(Asset(3.0,'EUR')@4.0),      '(EUR(3.00) @ 4.00)')
        self.assertEqual(str(ZeroBond(5.0,10.0)),        'P_None(5.00,10.00)')
        self.assertEqual(str(LiborRate(5.0,5.0,10.0,alias='USD')),
                                                    'L_USD(5.00;5.00,10.00)')
        self.assertEqual(str(SwapRate(2.0,[2.0, 5.0],[1.0, -1.0],[3.0, 4.0, 5.0],[1.0, 1.0, 1.0])),
                                                    'S_None(2.00;2.00,5.00)')
        y = Asset(2.0,'S')
        self.assertEqual(str(1.0+y),                     '(S(2.00) + 1.0000)')
        self.assertEqual(str(1.0-y),                     '(1.0000 - S(2.00))')
        self.assertEqual(str(0.5*y),                     '0.5000 S(2.00)')
        self.assertEqual(str(y/0.5 + 1.0),               '(2.0000 S(2.00) + 1.0000)')
        x = LiborRate(5.0,5.0,10.0,alias='USD')
        self.assertEqual(str(x*y),                       'L_USD(5.00;5.00,10.00) S(2.00)')
        self.assertEqual(str(x / y * Fixed(2.4)),        'L_USD(5.00;5.00,10.00) / S(2.00) 2.4000')
        self.assertEqual(str(Max(x,y)),                  'Max(L_USD(5.00;5.00,10.00), S(2.00))')
        self.assertEqual(str(Min(x,y)),                  'Min(L_USD(5.00;5.00,10.00), S(2.00))')
        self.assertEqual(str((x>y)*2.0),                 '2.0000 (L_USD(5.00;5.00,10.00) > S(2.00))')
        self.assertEqual(str(x==y),                      '(L_USD(5.00;5.00,10.00) == S(2.00))')
        self.assertEqual(str(~(0.5*y)*1.0),              '1.0000 ~{0.5000 S(2.00)}')

    def test_payoffTimes(self):
        L1 = LiborRate(1.0, 1.0, 1.5)
        L2 = LiborRate(2.0, 3.0, 3.5)
        P  = Pay(L1 + L2 + 1.0, 5.0)
        self.assertListEqual(list(P.observationTimes()),[0.0, 1.0, 2.0, 5.0])

    def test_liborPayoff(self):
        # Libor Rate payoff
        liborTimes = np.linspace(0.0, 10.0, 11)
        dT0 = 2.0 / 365
        dT1 = 0.5
        cfs = [ Pay(LiborRate(t,t+dT0,t+dT0+dT1),t+dT0+dT1) for t in liborTimes]

        curve = YieldCurve(0.03)
        path = DcfModel(curve).path()

        cfPVs = np.array([ p.discountedAt(path) for p in cfs ])
        discounts0 = np.array([ curve.discount(t+dT0) for t in liborTimes ])
        discounts1 = np.array([ curve.discount(t+dT0+dT1) for t in liborTimes ])
        liborsCv = (discounts0/discounts1 - 1.0)/dT1
        liborsMC = cfPVs / discounts1
        print('')
        print('  T     LiborRate  ModelLibor')
        for k,t in enumerate(liborTimes):
            print(' %4.1f   %6.4lf     %6.4lf' % (t, liborsCv[k], liborsMC[k]) )
            self.assertAlmostEqual(liborsCv[k],liborsMC[k],2)
        
    def test_cachePayoff(self):
        a = 1.0
        b = 1.0
        x = ~(Fixed(1.0) + 0.5)
        #print(x.at(a))
        #print(x.at(b))
        self.assertEqual(x._lastPath,None)
        x.at(a)
        self.assertEqual(x._lastPath,a)
        x.at(b)
        self.assertEqual(x._lastPath,b)
        

if __name__ == '__main__':
    unittest.main()
    
