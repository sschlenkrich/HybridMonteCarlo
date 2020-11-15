
using LinearAlgebra

include("../models/StochasticProcess.jl")

# the actual data structure for the model
struct QuasiGaussianModel <: StochasticProcess
    yieldCurve  # initial yield term structure 
    d           # specify d-dimensional Brownian motion for x(t)
                # we do not use stoch vol z(t) for now
    times       # time-grid of left-constant model parameter values
    sigma       # volatility
    slope       # skew
    curve       # smile
    delta       # maturity of benchmark rates f(t,t+delta_i)
    chi         # mean reversion
    Gamma       # (benchmark rate) correlation matrix
    # additional model parameters
    DfT
    HHfInv
end

# constructor

function QuasiGaussianModel(yieldCurve,d,times,sigma,slope,curve,delta,chi,Gamma)
    #
    # run some thorough checks on the input parameters...
    #
    DfT = cholesky(Gamma).L
    Hf_H_inv = [ exp(-chi_ * delta_) for delta_ in delta, chi_ in chi ]  # beware the order of loops!
    HHfInv = inv(Hf_H_inv)
    return QuasiGaussianModel(yieldCurve,d,times,sigma,slope,curve,delta,chi,Gamma,DfT,HHfInv)
end

# time-dependent model parameters are assumed (left-) piece-wise constant
_idx(self::QuasiGaussianModel,t) = 
    min(searchsortedfirst(self.times,t), size(self.times)[1])

sigma(self::QuasiGaussianModel, i, t) = self.sigma[i,_idx(self,t)]
slope(self::QuasiGaussianModel, i, t) = self.slope[i,_idx(self,t)]
curve(self::QuasiGaussianModel, i, t) = self.curve[i,_idx(self,t)]

# additional model functions

G(self::QuasiGaussianModel, i, t, T) = (1.0-exp(-self.chi[i]*(T-t)))/self.chi[i]
GPrime(self::QuasiGaussianModel, i, t, T) = exp(-self.chi[i]*(T-t))

# path/payoff interfaces

@views function zeroBond(self::QuasiGaussianModel, t, T, X, alias; G_=nothing)
    if (t==T)
        return 1.0  # short cut
    end
    if (t==0)
        return discount(self.yieldCurve,T)  # short cut, assume x=0, y=0 initial values
    end
    if isnothing(G_)
        G_ = [ G(self, i, t, T) for i=1:self.d ]
    end
    DF1 = discount(self.yieldCurve, t)
    DF2 = discount(self.yieldCurve, T)
    Gx = transpose(G_) * X[begin:self.d]
    GyG = 0.0
    for i = 1:self.d
        y_i = X[self.d + (i-1)*self.d + 1 : self.d + i*self.d]
        tmp = transpose(y_i) * G_  # s.y(i,j) * G[j], j = 1..d
        GyG += G_[i]*tmp
    end
    ZCB = DF2 / DF1 * exp(-Gx - 0.5*GyG)
    return ZCB
end

@views function numeraire(self::QuasiGaussianModel, t, X)
    return exp(X[end]) / discount(self.yieldCurve, t)  # depends on stoch process spec
end

# process simulation interface
# X = [ x, y, s ] (y row-wise)

stateSize(self::QuasiGaussianModel) = self.d + self.d^2 + 1  # dimension of X(t) = [ x, y, s ]

factors(self::QuasiGaussianModel) = self.d  # dimension of W(t)

initialValues(self::QuasiGaussianModel) = zeros(stateSize(self))

# volatility specification is key to model properties

# f - f0 for local volatility calculation
@views function deltaForwardRate(self::QuasiGaussianModel, t, T, X; GPrime_=nothing, G_=nothing)
    # we allow GPrime, G to be provided by user to avoid costly exponentials
    if isnothing(GPrime_)
        GPrime_ = [ GPrime(self, i, t, T) for i = 1:self.d ]
    end
    if isnothing(G_)
        G_ = (1.0 .- GPrime_) ./ self.chi
    end
    df = 0.0
    for i = 1:self.d
        tmp = X[i]  # s.x(i)
        y_i = X[self.d + (i-1)*self.d + 1 : self.d + i*self.d]
        tmp = X[i] + transpose(y_i) * G_  # s.x(i) + sum[ s.y(i,j) * G[j], j = 1..d ]
        df  += GPrime_[i] * tmp
    end
    return df
end

# local volatiity spec, interpreted as diagonal matrix
function sigma_f(self::QuasiGaussianModel, t, X)
    res = zeros(self.d)
    eps = 1.0e-4   # 1bp lower cutoff for volatility
    maxFactor = 10.0  # 10 sigma upper cut-off
    for k = 1:self.d
        df = deltaForwardRate(self, t, t + self.delta[k], X)
        sigma_ = sigma(self,k,t)
        slope_ = slope(self,k,t)
        curve_ = curve(self,k,t)
        vol = sigma_ + slope_*df + curve_*abs(df)
        vol = min(max(vol,eps),maxFactor*sigma_)
        res[k] = vol
    end
    return res
end

function sigma_xT(self, t, X)
    return self.HHfInv * Diagonal(sigma_f(self,t,X)) * self.DfT
end

# simulate Quasi-Gaussian model as Gaussian model with frozen vol matrix
function evolve(self::QuasiGaussianModel, t0, X0, dt, dW, X1)
    # first we may simulate stochastic vol via lognormal approximation (or otherwise)
    # next we need V = z * sigmaxT sigmax
    sigmaxT = sigma_xT(self, t0, X0)
    V = sigmaxT * transpose(sigmaxT)   # we need V later for x simulation
    # we also calculate intermediate variables; these do strictly depend on X and could be cached as well
    GPrime_ = [ GPrime(self,i,0,dt) for i = 1:self.d ]
    G_      = (1.0 .- GPrime_) ./ self.chi
    # now we can calculate y
    chi_i_p_chi_j = [ (chi_i + chi_j) for chi_i in self.chi, chi_j in self.chi ]
    b = V ./ chi_i_p_chi_j
    y0 = reshape(X0[self.d+1 : self.d + self.d^2], (self.d,self.d)) # create a view and re-shape as matrix
    a = y0 - b
    # y[i,j] = a[i,j] exp{-(chi_i + chi_j)(T-t)} + b[i,j]
    for i = 1:self.d
        for j = 1:self.d
            X1[self.d + (i-1)*self.d + j] = a[i,j] * GPrime_[i] * GPrime_[j] + b[i,j]
        end
    end
    # finally we calculate x
    for i = 1:self.d  # E[ x1 | x0 ]
        X1[i] = X0[i]
        for j = 1:self.d
            X1[i] += a[i,j] * G_[j]
        end
        X1[i] *= GPrime_[i]
        sumB = sum(b[i,:])
        X1[i] += sumB * G_[i]
    end
    # we overwrite V by the variance; better re-use b and matrix op's
    for i = 1:self.d
        for j = 1:self.d
            V[i,j] *= (1.0 - GPrime_[i] * GPrime_[j]) / (self.chi[i] + self.chi[j])
        end
    end
    L = cholesky(V).L  # maybe better do this in-place
   
    for i = 1:self.d
        for j = 1:self.d
            X1[i] += L[i,j] * dW[j]  # maybe also exploit that L is lower triangular
        end
    end
    # finally, we need to update s as well
    x0 = sum(X0[begin:self.d])
    x1 = sum(X1[begin:self.d])
    X1[end] = X0[end] + 0.5 * (x0 + x1) * dt
    # all done
    return nothing
end

# the short rate over an integration time period
# this is required for drift calculation in multi-asset and hybrid models
@views function shortRateOverPeriod(self::QuasiGaussianModel, t0, dt, X0, X1)
    B_d = discount(self.yieldCurve,t0) / discount(self.yieldCurve,t0 + dt)  # deterministic drift part for r_d
    x_av = 0.5 * (sum(X0[begin:self.d]) + sum(X1[begin:self.d]))
    return log(B_d) / dt + x_av
end

# bond volatility is used in hybrid model vol adjuster

function zeroBondVolatility(self::QuasiGaussianModel, t, T)
    # sigma_r^T = sigma_x^T sqrt(z) 
    # sigma_P^T = G(t,T)^T sigma_x^T sqrt(z)
    G_ = [ G(self, i, t, T) for i = 1:self.d ]
    sigmaxT = sigma_xT(self, t, initialValues(self))  # we approximate at x=0
    return transpose(sigmaxT) * G_
end

function zeroBondVolatilityPrime(self::QuasiGaussianModel, t, T)
    # we wrap scalar bond volatility into array to allow
    # for generalisation to multi-factor models
    GPrime_ = [ GPrime(self, i, t, T) for i = 1:self.d ]
    sigmaxT = sigma_xT(self, t, initialValues(self))  # we approximate at x=0
    return transpose(sigmaxT) * GPrime_
end
