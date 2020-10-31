#!/usr/bin/python

import sys
sys.path.append('./')

import QuantLib as ql

from hybmc.simulations.Payoffs import Payoff, Fixed, ZeroBond, LiborRate, Cache, Asset
from hybmc.simulations.AmcPayoffs import AmcSum
from hybmc.products.Product import Product

def DiscountedPayoffFromCashFlow(cf, obsTime, payOrReceive, discYtsH=None, currencyAlias=None):
    # this is a bit dangerous if someone changes evaluation date
    today = ql.Settings.instance().getEvaluationDate()
    # model time is measured via Act/365 (Fixed)
    dc = ql.Actual365Fixed()
    # first try a libor coupon
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
        L = LiborRate(min(fixingTime,obsTime),startTime,endTime,tau,tenorBasis,currencyAlias)
        if cp.spread()!=0.0:
            L = L + cp.spread()
        #  we treat deterministic factors separately to avoid unneccessary multiplications
        cpFactor = cp.nominal() * cp.accrualPeriod()
    else:
        L = 1.0  #  used as pseudo-rate here
        cpFactor = cf.amount()  # treat it as a fixed cash flow
    #
    payTime = dc.yearFraction(today,cf.date())
    cashFlow = payOrReceive * cpFactor * L * ZeroBond(obsTime,payTime,currencyAlias)
    if currencyAlias is not None:
        cashFlow = Asset(obsTime,currencyAlias) * cashFlow
    #
    return cashFlow @ obsTime

class Swap(Product):
    # Python constructor
    def __init__(self, qlLegs, payOrRecs, discYtsHs=None, currencyAliases=None):
        self.qlLegs = qlLegs           #  a list of Legs
        self.payOrRecs = payOrRecs     #  a list of Payer (-1) or Receiver (+1) flags
        # we need to normalise optional inputs
        if type(discYtsHs)==list:
            self.discYtsHs = discYtsHs
        else:
            self.discYtsHs = [ discYtsHs for l in self.qlLegs ]
        if type(currencyAliases)==list:
            self.currencyAliases = currencyAliases
        else:
            self.currencyAliases = [ currencyAliases for l in self.qlLegs ]

    def cashFlows(self, obsTime):
        # we calculate times relative to global evaluation date
        # this is a bit dangerous if someone changes evaluation date
        today = ql.Settings.instance().getEvaluationDate()
        # model time is measured via Act/365 (Fixed)
        dc = ql.Actual365Fixed()
        # we assume our swap has exactly two legs
        cfs = []
        for leg, por, discYtsH, alias in zip(self.qlLegs, self.payOrRecs,self.discYtsHs,self.currencyAliases):
            for cf in leg:
                payTime = dc.yearFraction(today,cf.date())
                if payTime>obsTime:  # only consider future cash flows
                    #print('%s:  %f,  %f' % (cf.date(),cf.amount(),payTime))
                    p = DiscountedPayoffFromCashFlow(cf,obsTime,por,discYtsH,alias)
                    cfs.append(p)
                    # print(p)
        return cfs
                    

def PayoffFromCashFlow(cf, payOrReceive, discYtsH=None):
    # this is a bit dangerous if someone changes evaluation date
    today = ql.Settings.instance().getEvaluationDate()
    # model time is measured via Act/365 (Fixed)
    dc = ql.Actual365Fixed()
    #
    payTime = dc.yearFraction(today,cf.date())
    # first we try fixed rate cash flow
    cp = ql.as_fixed_rate_coupon(cf)
    if cp is not None:
        return Fixed(payOrReceive*cp.amount()) @ payTime
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
        L = LiborRate(fixingTime,startTime,endTime,tau,tenorBasis)
        factor = payOrReceive * cp.nominal() * cp.accrualPeriod()
        return ( factor * (L + cp.spread()) ) @ payTime
    return None

class AmcSwap(Product):
    # Python constructor, only single-currency
    def __init__(self, qlLegs, payOrRecs, mcSim, maxDegree=2, discYtsH=None):
        self.qlLegs = qlLegs           #  a list of Legs
        self.payOrRecs = payOrRecs     #  a list of Payer (-1) or Receiver (+1) flags
        self.mcSim = mcSim
        self.maxDegree = maxDegree
        self.discYtsH = discYtsH
        # we want to re-use payoffs
        today = ql.Settings.instance().getEvaluationDate()
        # model time is measured via Act/365 (Fixed)
        dc = ql.Actual365Fixed()
        self.cfs = []
        for leg, por in zip(self.qlLegs, self.payOrRecs):
            for cf in leg:
                payTime = dc.yearFraction(today,cf.date())
                if payTime>0.0:  # only consider future cash flows
                    p = PayoffFromCashFlow(cf,por,self.discYtsH)
                    self.cfs.append(Cache(p))   # this is the magic


    def cashFlows(self, obsTime):
        cfs = []
        for cf in self.cfs:
            if cf.obsTime>obsTime:
                cfs.append(cf)
        # we use a 'co-terminal' Libor rate as observable
        tMax = max([ cf.obsTime for cf in self.cfs ])
        L = LiborRate(obsTime,obsTime,tMax)
        p = AmcSum(obsTime,cfs,[L],self.mcSim,self.maxDegree)
        return [ p ]


