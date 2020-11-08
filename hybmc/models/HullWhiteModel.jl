include("../mathutils/Helpers.jl")

using Roots

# the actual data structure for the model
struct HullWhiteModel
    yieldCurve
    meanReversion
    volatilityTimes::Array
    volatilityValues::Array
    y_
end

# constructor

function HullWhiteModel(yieldCurve, meanReversion, volatilityTimes::Array, volatilityValues::Array)
    #
    t0 = 0.0
    y0 = 0.0
    y_ = zeros(size(volatilityTimes))
    for i = 1:size(y_)[1]
        y_[i] = (GPrime(meanReversion,t0,volatilityTimes[i])^2) * y0 +
            (volatilityValues[i]^2) * 
            (1.0 - exp(-2*meanReversion*(volatilityTimes[i]-t0))) /
            (2.0 * meanReversion)
        t0 = volatilityTimes[i]
        y0 = y_[i]
    end
    return HullWhiteModel(yieldCurve,meanReversion,volatilityTimes,volatilityValues,y_)
end

# auxilliary functions/methods

G(meanReversion,t,T) =
    (1.0 - exp(-meanReversion*(T-t))) / meanReversion

GPrime(meanReversion,t,T) = exp(-meanReversion*(T-t))

G(self::HullWhiteModel,t,T) = G(self.meanReversion,t,T)

GPrime(self::HullWhiteModel,t,T) = GPrime(self.meanReversion,t,T)

function y(self::HullWhiteModel,t)
    idx = searchsortedfirst(self.volatilityTimes,t)  # first element of tupel
    t0 = (idx==1) ? 0.0 : self.volatilityTimes[idx-1]
    y0 = (idx==1) ? 0.0 : self.y_[idx-1]
    s1 = self.volatilityValues[min(idx,size(self.volatilityValues)[1])]  # flat extrapolation
    y1 = (GPrime(self,t0,t)^2) * y0 +
            s1^2 * (1.0 - exp(-2*self.meanReversion*(t-t0))) /
            (2.0 * self.meanReversion)
    return y1
end

function riskNeutralExpectationX(self::HullWhiteModel, t, xt, T)
    # E[x] = G'(t,T)x + \int_t^T G'(u,T)y(u)du
    f(u) = GPrime(self,u,T)*y(self,u)
    # use Simpson's rule to approximate integral, this should better be solved analytically
    integral = (T-t) / 6 * (f(t) + 4*f((t+T)/2) + f(T)) 
    return GPrime(self,t,T)*xt + integral
end

function sigma(self::HullWhiteModel,t)   # Todo test this method
    # find idx s.t. t[idx-1] < t <= t[idx]
    idx = searchsortedfirst(self.volatilityTimes,t)
    return self.volatilityValues[min(idx,size(self.volatilityValues)[1])]
end

# bond volatility is used in hybrid model vol adjuster

function zeroBondVolatility(self::HullWhiteModel, t, T)
    # we wrap scalar bond volatility into array to allow
    # for generalisation to multi-factor models
    return [ sigma(self,t)*G(self,t,T) ]
end

function zeroBondVolatilityPrime(self::HullWhiteModel, t, T)
    # we wrap scalar bond volatility into array to allow
    # for generalisation to multi-factor models
    return [ sigma(self,t)*GPrime(self,t,T) ]
end

# model methods

# conditional expectation in T-forward measure
function expectationX(self::HullWhiteModel, t, xt, T)
    return GPrime(self,t,T)*(xt + G(self,t,T)*y(self,t))
end

function varianceX(self::HullWhiteModel, t, T)
    return y(self,T) - GPrime(self,t,T)^2 * y(self,t)
end

function zeroBondPrice(self::HullWhiteModel, t, T, xt)
    G_ = G(self,t,T)
    return discount(self.yieldCurve,T) / discount(self.yieldCurve,t) *
        exp(-G_*xt - 0.5 * G_^2 * y(self,t) )
end

function zeroBondOption(self::HullWhiteModel, expiryTime, maturityTime, strikePrice, callOrPut)
    nu2 = G(self,expiryTime,maturityTime)^2 * y(self,expiryTime)
    P0  = discount(self.yieldCurve, expiryTime)
    P1  = discount(self.yieldCurve, maturityTime)
    return P0 * Black(strikePrice,P1/P0,sqrt(nu2),1.0,callOrPut)
end

function couponBondOption(self::HullWhiteModel, expiryTime, payTimes, cashFlows, strikePrice, callOrPut)
    function objective(x)
        bond = 0
        for i = 1:size(payTimes)[1]
            bond += cashFlows[i] * zeroBondPrice(self,expiryTime,payTimes[i],x)
        end
        return bond - strikePrice
    end
    xStar = find_zero(objective,(-1.0,1.0),Roots.Brent(),xatol=1.0e-8)
    bondOption = 0.0
    for i = 1:size(payTimes)[1]
        strike = zeroBondPrice(self,expiryTime,payTimes[i],xStar)
        bondOption += cashFlows[i] * zeroBondOption(self,expiryTime,payTimes[i],strike,callOrPut)
    end
    return bondOption
end

# stochastic process interface
    
function stateSize(self::HullWhiteModel)   # dimension of X(t)
    return 2      # [x, s], we also need for numeraire s = \int_0^t r dt
end

function factors(self::HullWhiteModel)   # dimension of W(t)
    return 1
end

function initialValues(self::HullWhiteModel)
    return [ 0.0, 0.0 ]
end

function zeroBond(self::HullWhiteModel, t, T, X, alias)
    return zeroBondPrice(self, t, T, X[1])
end

function numeraire(self::HullWhiteModel, t, X)
    return exp(X[2]) / discount(self.yieldCurve,t)
end

# evolve X(t0) -> X(t0+dt) using independent Brownian increments dW
# t0, dt are assumed float, X0, X1, dW are np.array
function evolve(self::HullWhiteModel, t0, X0, dt, dW, X1)
    x1 = riskNeutralExpectationX(self,t0,X0[1],t0+dt)
    # x1 = X0[0] + (self.y(t0) - self.meanReversion*X0[0])*dt
    nu = sqrt(varianceX(self,t0,t0+dt))
    x1 = x1 + nu*dW[1]
    # s1 = s0 + \int_t0^t0+dt x dt via Trapezoidal rule
    s1 = X0[2] + (X0[1] + x1) * dt / 2
    # gather results
    X1[1] = x1
    X1[2] = s1
    return nothing
end

# the short rate over an integration time period
# this is required for drift calculation in multi-asset and hybrid models
function shortRateOverPeriod(self::HullWhiteModel, t0, dt, X0, X1)
    B_d = discount(self.yieldCurve,t0) / discount(self.yieldCurve,t0 + dt)  # deterministic drift part for r_d
    x_av = 0.5 * (X0[1] + X1[1])
    return log(B_d) / dt + x_av
end
