
abstract type StochasticProcess end

# process

# length of state X
function stateSize(self::StochasticProcess)
    throw(ArgumentError("Implementation of method stateSize() required."))
end

# lengh of stochastic factors, dW
function factors(self::StochasticProcess)
    throw(ArgumentError("Implementation of method factors() required."))
end

# initial values for simulation
function initialValues(self::StochasticProcess)
    throw(ArgumentError("Implementation of method initialValues() required."))
end

# evolve X(t0) -> X(t0+dt) using independent Brownian increments dW
# t0, dt are assumed float, X0, X1, dW are array
function evolve(self::StochasticProcess, t0, X0, dt, dW, X1)
    throw(ArgumentError("Implementation of method evolve() required."))
end

# hybrid evolution

# the short rate over an integration time period
# this is required for drift calculation in multi-asset and hybrid models
function shortRateOverPeriod(self::StochasticProcess, t0, dt, X0, X1)
    throw(ArgumentError("Implementation of method shortRateOverPeriod() required."))
end

function zeroBondVolatility(self::StochasticProcess, t, T)
    throw(ArgumentError("Implementation of method zeroBondVolatility() required."))
end

function zeroBondVolatilityPrime(self::StochasticProcess, t, T)
    throw(ArgumentError("Implementation of method zeroBondVolatilityPrime() required."))
end

# model

# the numeraire in the domestic currency used for discounting future payoffs
function numeraire(self::StochasticProcess, t, X)
    throw(ArgumentError("Implementation of method numeraire() required."))
end

# an asset price for a given currency alias
function asset(self::StochasticProcess, t, X, alias)
    throw(ArgumentError("Implementation of method asset() required."))
end

# a domestic/foreign currency zero coupon bond
function zeroBond(self::StochasticProcess, t, T, X, alias)
    throw(ArgumentError("Implementation of method zeroBond() required."))
end

# additional simulated credit quantities, we use a Cox process
# for references, see
#   - Brigo/Mercurio, 2007, Sec. 22.2.3
#   - Brigo/Vrins, Disentangling wrong-way risk, 2016, Sec. 3.3

# Cumulated intensity; probability of tau > t, conditional on F_t
function hazardProcess(self::StochasticProcess, t, X, alias)
    throw(ArgumentError("Implementation of method hazardProcess() required."))
end

# instantanous probability of default
function hazardRate(self::StochasticProcess, t, X, alias)
    throw(ArgumentError("Implementation of method hazardRate() required."))
end

# probavility of survival consitional on information at t
function survivalProb(self::StochasticProcess, t, T, X, alias)
    throw(ArgumentError("Implementation of method survivalProb() required."))
end
