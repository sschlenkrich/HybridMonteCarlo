#!/usr/bin/python

# create QuantLib models from Python models

try:
    import QuantLib as ql
except ModuleNotFoundError as e:
    print('Error: Module QuantLibPayoffs requires a (custom) QuantLib installation.')
    raise e

try:
    from QuantLib import QuasiGaussianModel as qlQuasiGaussianModel
except ImportError as e:
    print('Error: Module QuantLibPayoffs requires a custom QuantLib installation.')
    print('It seems you do have a QuantLib installed but maybe not the custom version')
    print('with payoff scripting. Checkout https://github.com/sschlenkrich/QuantLib')
    raise e

import numpy as np

import sys
sys.path.append('./')

from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.HullWhiteModel import HullWhiteModel
from hybmc.models.QuasiGaussianModel import QuasiGaussianModel


def QuantLibYieldCurve(curve):
    if isinstance(curve,YieldCurve):
        today = ql.Settings.instance().evaluationDate
        return ql.YieldTermStructureHandle(
            ql.FlatForward(today,curve.rate,ql.Actual365Fixed()))
    else:
        raise NotImplementedError('Implementation of JuliaYieldCurve for %s required.' % str(type(curve)))
    return


def QuantLibModel(model):
    #
    if isinstance(model,QuasiGaussianModel):
        ytsh = QuantLibYieldCurve(model.yieldCurve)
        # we need to add trivial stoch vol parameters
        eta = np.array([ 0.0 for t in model._times ])
        theta = 0.1 # does not matter for zero vol of vol
        Gamma = np.identity(model._d + 1)
        for i in range(model._d):
            for j in range(model._d):
                Gamma[i,j] = model._Gamma[i,j]
        return qlQuasiGaussianModel(ytsh,model._d,
            model._times, model._sigma, model._slope, model._curve, eta,
            model._delta, model._chi, model._Gamma, theta)
    else:
        raise NotImplementedError('Implementation of JuliaModel for %s required.' % str(type(model)))
    return

