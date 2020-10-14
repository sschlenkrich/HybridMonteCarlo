#!/usr/bin/python

import sys
sys.path.append('./')

import QuantLib as ql

from src.simulations.Payoffs import Payoff, Fixed, ZeroBond, LiborRate
from src.products.Product import Product

def PayoffFromCashFlow(cf, obsTime, payTime, payOrReceive, discYtsH=None):
    # this is a bit dangerous if someone changes evaluation date
    today = ql.Settings.instance().getEvaluationDate()
    # model time is measured via Act/365 (Fixed)
    dc = ql.Actual365Fixed()
    # first we try fixed rate cash flow
    cp = ql.as_fixed_rate_coupon(cf)
    if cp is not None:
        return (Fixed(payOrReceive*cp.amount()) * ZeroBond(obsTime,payTime)) @ obsTime
    # second try a libor coupon
    cp = ql.as_floating_rate_coupon(cf)
    if cp is not None:
        # first we need to puzzle out the dates for the index
        projIndex  = ql.as_iborindex(cp.index())
        fixingDate = cp.fixingDate()
        startDate  = projIndex.valueDate(fixingDate)
        endDate    = projIndex.maturityDate(startDate)
        #print(index, fixingDate, startDate, endDate)
        tau        = projIndex.dayCounter().yearFraction(startDate,endDate)
        tenorBasis = 1.0  #  default
        if discYtsH is not None:
            # we apply deterministic basis calculation
            dfProj = 1.0 + tau*projIndex.fixing(fixingDate)
            discIndex = projIndex.clone(discYtsH)
            dfDisc = 1.0 + tau*discIndex.fixing(fixingDate)
            tenorBasis = dfProj / dfDisc
            #print(tenorBasis)
        fixingTime = dc.yearFraction(today,fixingDate)
        startTime = dc.yearFraction(today,startDate)
        endTime = dc.yearFraction(today,endDate)
        #  fixed Libor or Libor forward rate
        L = LiborRate(min(fixingTime,obsTime),startTime,endTime,tau,tenorBasis)
        factor = payOrReceive * cp.nominal() * cp.accrualPeriod()
        return ( factor * (L + cp.spread()) * ZeroBond(obsTime,payTime) ) @ obsTime
    return None


class Swap(Product):
    # Python constructor
    def __init__(self, qlLegs, payOrRecs, discYtsH=None):
        self.qlLegs = qlLegs           #  a list of Legs
        self.payOrRecs = payOrRecs     #  a list of Payer (-1) or Receiver (+1) flags
        self.discYtsH = discYtsH

    def cashFlows(self, obsTime):
        # we calculate times relative to global evaluation date
        # this is a bit dangerous if someone changes evaluation date
        today = ql.Settings.instance().getEvaluationDate()
        # model time is measured via Act/365 (Fixed)
        dc = ql.Actual365Fixed()
        # we assume our swap has exactly two legs
        cfs = []
        for leg, por in zip(self.qlLegs, self.payOrRecs):
            for cf in leg:
                payTime = dc.yearFraction(today,cf.date())
                if payTime>obsTime:  # only consider future cash flows
                    #print('%s:  %f,  %f' % (cf.date(),cf.amount(),payTime))
                    p = PayoffFromCashFlow(cf,obsTime,payTime,por,self.discYtsH)
                    cfs.append(p)
                    # print(p)
        return cfs
                    
