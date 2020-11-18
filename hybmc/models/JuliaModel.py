#!/usr/bin/python

# create Julia models from Python models

from julia import Main
Main.include('./hybmc/termstructures/YieldCurve.jl')
Main.include('./hybmc/models/HullWhitemodel.jl')
Main.include('./hybmc/models/QuasiGaussianModel.jl')

import sys
sys.path.append('./')

from hybmc.termstructures.YieldCurve import YieldCurve
from hybmc.models.HullWhiteModel import HullWhiteModel
from hybmc.models.QuasiGaussianModel import QuasiGaussianModel


def JuliaYieldCurve(curve):
    if isinstance(curve,YieldCurve):
        return Main.YieldCurve(curve.rate)
    else:
        raise NotImplementedError('Implementation of JuliaYieldCurve for %s required.' % str(type(curve)))
    return


def JuliaModel(model):
    #
    if isinstance(model,HullWhiteModel):
        yc = JuliaYieldCurve(model.yieldCurve)
        return Main.HullWhiteModel(yc,model.meanReversion,model.volatilityTimes,model.volatilityValues)
    #
    if isinstance(model,QuasiGaussianModel):
        yc = JuliaYieldCurve(model.yieldCurve)
        return Main.QuasiGaussianModel(yc,model._d,
            model._times, model._sigma, model._slope, model._curve,
            model._delta, model._chi, model._Gamma)
    else:
        raise NotImplementedError('Implementation of JuliaModel for %s required.' % str(type(model)))
    return
