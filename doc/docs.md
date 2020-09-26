# HybridMonteCarlo Documentation

This module implements a hybrid modelling framework with Monte Carlo simulation and payoff scripting.

Use cases are pricing complex financial instruments and exposure simulation.

## Monte Carlo Framework

The Monte Carlo framework consists of four key elements:

  - StochastiProcess (and derived concrete models)
  - MCSimulation
  - a general Path representation
  - MCPayoffs that are used to compose cash flows of financial instruments
  
### Stochastic Process Interface

We specify a general StochasticProcess class that declares the interface used by the Monte Carlo simulation as well as the payoffs.

Monte Carlo simulation represents evolution of a *State*. The State captures all the information that is required to recontruct our *World* and calculate any required payoff. 

A StochasticProcess specifies the *State* and how it is to be interpreted.

A StochasticProcess further describes how the states change or *evolve* from a given point in time to another point in time. In mathematical terms this is integrating an SDE. For financial modelling we may need additional information, e.g. auxilliary variables to calculate future numeraires. These additional information are also evolved.

The *World* is described by a set of functions that calculate the basic quantities used for financial modelling. These quantities are:

  - numeraire associated with the chosen pricing measure
  - asset for a given alias; this is typically an FX rate or stock price identified by the alias
  - zero coupon bond price for a (currency) alias

These quantities can be calculated based on a given input *State*.
  
The StochasticProcess interface is implemented by concrete models. Obviously, not all financial models describe all quantities. For example, an interest rate model does not know about any FX rates or stock prices. If a model is asked by a payoff for a particular quantity which it does not implement then it raises an exception.

### Monte Carlo Simulation

Monte Carlo simulation implements the simulation of *States*. By doing so it is agnostic to the concrete interpretation of the *States* or the *World* they describe.

The key purpose of the Monte Carlo simulation is generating random numbers. Based on that random numbers sequences of *States* are generated and stored. Such a sequence of *States* represents a path (or its key building blocks) of the Monte Carlo simulation.

Each sequence of *States* is also a scenario of the evolution of our *World*. The Monte Carlo simulation generates a set of scenarios which are used to calculate outcomes of payoffs and cash flows of financial instruments.

### Monte Carlo Paths

A Monte Carlo path is essentially specified by the sequence of states in a Monte Carlo simulation. However, to construct the information of our *World* each state needs to be interpreted by the concrete model.

However, a payoff should be agnostic to the way a *World* is constructed. That way a payoff can be evaluated with any kind of scenario generation (random or deterministic) and any kind of modelling.

The desired level of abstraction is implemented via a *Path* object. The *Path* provides all the functions that describe the *World* to the payoff. However, it ommits the input *State*. 

The linkage of *Path* to input *State* and model is established by the Monte Carlo simulation. The Monte Carlo simulation knows the model and the simulated *States*. Therefore, we use Monte Carlo simulation to *create* given *Paths* which can subsequently be fed to payoffs.

### Payoffs

*Payoffs* represent functions from a given *Path* to a scalar number. We say, we evaluate a *Payoff at a given Path*.

*Payoffs* typically have an observation date. This is the point in time when the *Payoff* is observable, fixed or paid. In the financial modelling context we are typically interested in *discounted Payoffs*. Such *discounted Payoff* is evaluated as a *Payoff at a given Path* divided by the numeraire at observation time.

We implement elementary payoffs and define operations on these payoffs. Elementary payoffs are, e.g. fixed amounts, Libor rates, asset prices and payments. Oprations are basic arithmetic operations or cloning a payoff at a different observation time.

The elementary payoffs and operations allow constructing complex cash flows. The cash flows are used to compose instruments. Note, cash flow and instrument specification is independent from models and Monte Carlo simulation.

## Model Implementations

In this section we specify the concrete model implementations.

### One-factor Gaussian Short Rate Model

Hull-White model based on HJM representation.

### Lognormal Asset Model

Black-Scholes.

### Hybrid Model

A hybrid model composed of short rate models and asset models












