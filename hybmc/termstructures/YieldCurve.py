#!/usr/bin/python

import math

class YieldCurve:

    # Python constructor
    def __init__(self, rate):
        self.rate = rate


    # zero coupon bond
    def discount(self, T):
        return math.exp(-self.rate*T)


