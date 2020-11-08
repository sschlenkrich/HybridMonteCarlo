
struct AssetModel
    X0
    sigma
end

# stochastic process interface
    
function stateSize(self::AssetModel)   # dimension of X(t)
    return 1
end

function factors(self::AssetModel)   # dimension of W(t)
    return 1
end

function initialValues(self::AssetModel)
    return [ 0.0 ]
end

# calculate volatility taking into account exogenous hybrid adjuster
function volatility(self::AssetModel, t, Y)
    return self.sigma + Y[3]
end

function evolve(self::AssetModel, t0, Y0, dt, dW, Y1)
    sigma = volatility(self,t0,Y0)
    Y1[1] = Y0[1] + (Y0[2] - 0.5*sigma*sigma)*dt + sigma*dW[1]*sqrt(dt)
end

function asset(self::AssetModel, t, Y, alias)
    return self.X0 * exp(Y[1])
end

