
using Distributions, Roots

function BlackOverK(moneyness, stdDev, callOrPut)
    d1 = log(moneyness) / stdDev + stdDev / 2.0
    d2 = d1 - stdDev
    norm = Normal()
    return callOrPut * (moneyness*cdf(norm,callOrPut*d1)-cdf(norm,callOrPut*d2))
end

function Black(strike, forward, sigma, T, callOrPut)
    nu = sigma*sqrt(T)
    if nu<1.0e-12   # assume zero
        return max(callOrPut*(forward-strike),0.0)  # intrinsic value
    end
    return strike * BlackOverK(forward/strike,nu,callOrPut)
end

function BlackVega(strike, forward, sigma, T)
    stdDev = sigma*sqrt(T)
    d1 = log(forward/strike) / stdDev + stdDev / 2.0    
    norm = Normal()
    return forward * cdf(norm,d1) * sqrt(T)
end

function BlackImpliedVol(price, strike, forward, T, callOrPut)
    objective(sigma) = Black(strike, forward, sigma, T, callOrPut) - price
    return find_zero(objective,(0.01,1.0),Roots.Brent(),xatol=1.0e-8)
end
