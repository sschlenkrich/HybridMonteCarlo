
using Distributions

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