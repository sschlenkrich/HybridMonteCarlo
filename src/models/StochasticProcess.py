#!/usr/bin/python


class StochasticProcess:

    # We specify the interface for Monte Carlo simulation

    # length of state X
    def size(self):
        raise NotImplementedError('Implementation of method size required.')

    # lengh of stochastic factors, dW
    def factors(self):
        raise NotImplementedError('Implementation of method factors required.')
    
    # initial values for simulation
    def initialValues(self):
        raise NotImplementedError('Implementation of method initialValues required.')

    # the short rate over an integration time period
    # this is required for drift calculation in multi-asset and hybrid models
    def shortRateOverPeriod(self, t0, dt, X0, X1):
        raise NotImplementedError('Implementation of method shortRateOverPeriod required.')

    # evolve X(t0) -> X(t0+dt) using independent Brownian increments dW
    # t0, dt are assumed float, X0, X1, dW are np.array
    def evolve(self, t0, X0, dt, dW, X1):
        raise NotImplementedError('Implementation of method evolve required.')
    
    # We specify the interface for payoff calculation
    
    # the numeraire in the domestic currency used for discounting future payoffs
    def numeraire(self, t, X):
        raise NotImplementedError('Implementation of method numeraire required.')
    
    # a domestic/foreign currency zero coupon bond
    def zeroBond(self, t, T, X, alias):
        raise NotImplementedError('Implementation of method zeroBond required.')
    
    # an asset price for a given currency alias
    def asset(self, t, X, alias):
        raise NotImplementedError('Implementation of method asset required.')
    
    
    
    