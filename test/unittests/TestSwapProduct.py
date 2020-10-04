#!/usr/bin/python

import sys
sys.path.append('./')

import unittest

import QuantLib as ql

import numpy as np
from src.products.Swap import Swap

class TestSwapProduct(unittest.TestCase):

    def test_SwapSetup(self):
        # we set up a QuantLib VanillaSwap as basic test object
        today     = ql.Date(5,ql.October,2020)
        ql.Settings.instance().evaluationDate = today
        # we need curves for the float leg
        discYtsH = ql.YieldTermStructureHandle(
            ql.FlatForward(today,0.01,ql.Actual365Fixed()))
        projYtsH = ql.YieldTermStructureHandle(
            ql.FlatForward(today,0.02,ql.Actual365Fixed()))
        index = ql.Euribor6M(projYtsH)
        # we set start in the future to avoid the need of index fixings
        startDate  = ql.Date(12,ql.October,2020)
        endDate    = ql.Date(12,ql.October,2030)
        calendar   = ql.TARGET()
        fixedTenor = ql.Period('1y')
        floatTenor = ql.Period('6m')
        fixedSchedule = ql.MakeSchedule(startDate,endDate,tenor=fixedTenor,calendar=calendar)
        floatSchedule = ql.MakeSchedule(startDate,endDate,tenor=floatTenor,calendar=calendar)
        swapType = ql.VanillaSwap.Payer
        vanillaSwap = ql.VanillaSwap(ql.VanillaSwap.Receiver,1.0,fixedSchedule,0.025,ql.Thirty360(),floatSchedule,index,0.0050,ql.Actual360())
        # we have all the incredients to price the swap - but not really needed for payoffs
        engine = ql.DiscountingSwapEngine(discYtsH)
        vanillaSwap.setPricingEngine(engine)
        # It is easier to work with legs instead of Swap intruments
        legs = [vanillaSwap.fixedLeg(), vanillaSwap.floatingLeg()]
        pors = [1.0, -1.0] if swapType==ql.VanillaSwap.Receiver else [-1.0, 1.0]
        swap = Swap(legs,pors,discYtsH)
        #swap.cashFlows(8.1)
        timeLine = swap.timeLine([1.0, 2.0, 3.0])
        #
        for t in timeLine:
            print('ObsTime: %.2f' % t)
            for p in timeLine[t]:
                print(p)


if __name__ == '__main__':
    unittest.main()
    
