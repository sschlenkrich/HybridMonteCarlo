
using Printf, Test, Plots, LinearAlgebra, Distributions

include("../../hybmc/termstructures/YieldCurve.jl")
include("../../hybmc/models/QuasiGaussianModel.jl")

function setUp()
    curve = YieldCurve(0.03)
    d     = 3

    times = [ 1.0,    2.0,    5.0,    10.0     ]

    sigma = [  0.0060 0.0070 0.0080  0.0100 ; 
               0.0030 0.0035 0.0040  0.0050 ; 
               0.0025 0.0025 0.0025  0.0025 ]

    slope = [  0.10   0.15   0.20    0.25   ; 
               0.20   0.35   0.40    0.55   ; 
               0.10   0.10   0.10    0.10   ]

    curve = [  0.05   0.05   0.05    0.05   ; 
               0.10   0.10   0.10    0.10   ; 
               0.20   0.15   0.10    0.00   ]

    delta = [ 1.0,  5.0,  20.0 ]
    chi   = [ 0.01, 0.05, 0.15 ]
  
    Gamma = [ 1.0  0.8  0.6 ; 
              0.8  1.0  0.8 ;
              0.6  0.8  1.0 ]

    # test matricies
    return QuasiGaussianModel(curve,d,times,sigma,slope,curve,delta,chi,Gamma)
end


@testset "test_ModelSetup" begin
    model = setUp()
    @test norm(model.DfT * transpose(model.DfT) - model.Gamma) == 0.0
    # Hf_H_inv from manual calculation
    Hf_H_inv = [ 0.99004983 0.95122942 0.86070798 ;
                 0.95122942 0.77880078 0.47236655 ;
                 0.81873075 0.36787944 0.04978707 ]
    @test norm(model.HHfInv * Hf_H_inv - Matrix{Float64}(I, 3, 3)) < 1.0e-7
    @test stateSize(model) == 13
    @test factors(model) == 3
    # test parameter functions
    times = Array(range(0.0, stop=11.0, length=23))

    sigma_ = [ sigma(model,i,t) for t in times, i = 1:model.d ]
    sigma_ = hcat(times, sigma_)
    sigmaRef = [
        0.00e+00  6.00e-03  3.00e-03  2.50e-03 ;
        5.00e-01  6.00e-03  3.00e-03  2.50e-03 ;
        1.00e+00  6.00e-03  3.00e-03  2.50e-03 ;
        1.50e+00  7.00e-03  3.50e-03  2.50e-03 ;
        2.00e+00  7.00e-03  3.50e-03  2.50e-03 ;
        2.50e+00  8.00e-03  4.00e-03  2.50e-03 ;
        3.00e+00  8.00e-03  4.00e-03  2.50e-03 ;
        3.50e+00  8.00e-03  4.00e-03  2.50e-03 ;
        4.00e+00  8.00e-03  4.00e-03  2.50e-03 ;
        4.50e+00  8.00e-03  4.00e-03  2.50e-03 ;
        5.00e+00  8.00e-03  4.00e-03  2.50e-03 ;
        5.50e+00  1.00e-02  5.00e-03  2.50e-03 ;
        6.00e+00  1.00e-02  5.00e-03  2.50e-03 ;
        6.50e+00  1.00e-02  5.00e-03  2.50e-03 ;
        7.00e+00  1.00e-02  5.00e-03  2.50e-03 ;
        7.50e+00  1.00e-02  5.00e-03  2.50e-03 ;
        8.00e+00  1.00e-02  5.00e-03  2.50e-03 ;
        8.50e+00  1.00e-02  5.00e-03  2.50e-03 ;
        9.00e+00  1.00e-02  5.00e-03  2.50e-03 ;
        9.50e+00  1.00e-02  5.00e-03  2.50e-03 ;
        1.00e+01  1.00e-02  5.00e-03  2.50e-03 ;
        1.05e+01  1.00e-02  5.00e-03  2.50e-03 ;
        1.10e+01  1.00e-02  5.00e-03  2.50e-03 ]
    @test norm(sigma_-sigmaRef) == 0.0

    slope_ = [ slope(model,i,t) for t in times, i = 1:model.d ]
    slope_ = hcat(times, slope_)
    slopeRef = [
         0.     0.1    0.2    0.1 ;
         0.5    0.1    0.2    0.1 ;
         1.     0.1    0.2    0.1 ;
         1.5    0.15   0.35   0.1 ;
         2.     0.15   0.35   0.1 ;
         2.5    0.2    0.4    0.1 ;
         3.     0.2    0.4    0.1 ;
         3.5    0.2    0.4    0.1 ;
         4.     0.2    0.4    0.1 ;
         4.5    0.2    0.4    0.1 ;
         5.     0.2    0.4    0.1 ;
         5.5    0.25   0.55   0.1 ;
         6.     0.25   0.55   0.1 ;
         6.5    0.25   0.55   0.1 ;
         7.     0.25   0.55   0.1 ;
         7.5    0.25   0.55   0.1 ;
         8.     0.25   0.55   0.1 ;
         8.5    0.25   0.55   0.1 ;
         9.     0.25   0.55   0.1 ;
         9.5    0.25   0.55   0.1 ;
        10.     0.25   0.55   0.1 ;
        10.5    0.25   0.55   0.1 ;
        11.     0.25   0.55   0.1 ]
    @test norm(slope_-slopeRef) == 0.0

    curve_ = [ curve(model,i,t) for t in times, i = 1:model.d ]
    curve_ = hcat(times, curve_)
    curveRef = [
         0.     0.05   0.1    0.2  ;
         0.5    0.05   0.1    0.2  ;
         1.     0.05   0.1    0.2  ;
         1.5    0.05   0.1    0.15 ;
         2.     0.05   0.1    0.15 ;
         2.5    0.05   0.1    0.1  ;
         3.     0.05   0.1    0.1  ;
         3.5    0.05   0.1    0.1  ;
         4.     0.05   0.1    0.1  ;
         4.5    0.05   0.1    0.1  ;
         5.     0.05   0.1    0.1  ;
         5.5    0.05   0.1    0.   ;
         6.     0.05   0.1    0.   ;
         6.5    0.05   0.1    0.   ;
         7.     0.05   0.1    0.   ;
         7.5    0.05   0.1    0.   ;
         8.     0.05   0.1    0.   ;
         8.5    0.05   0.1    0.   ;
         9.     0.05   0.1    0.   ;
         9.5    0.05   0.1    0.   ;
        10.     0.05   0.1    0.   ;
        10.5    0.05   0.1    0.   ;
        11.     0.05   0.1    0.   ]
        @test norm(curve_-curveRef) == 0.0
end

@testset "test_ZeroBond" begin
    yts = YieldCurve(0.03)
    d     = 3
    times = [  10.0   ]
    sigma = [  0.0060 ;
               0.0050 ;
               0.0040 ]
    slope = [  0.10   ;
               0.20   ;
               0.30   ]
    curve = [  0.05   ;
               0.10   ;
               0.20   ]
    delta = [ 1.0, 5.0, 20.0 ]
    chi   = [ 0.01, 0.05, 0.15 ]
    Gamma = [ 1.0  0.8  0.6 ; 
              0.8  1.0  0.8 ;
              0.6  0.8  1.0 ]
    model = QuasiGaussianModel(yts,d,times,sigma,slope,curve,delta,chi,Gamma)
    uni = Uniform(0.0,0.05)
    x = rand(uni, d)
    y = rand(uni, (d,d))
    y = y * transpose(y)
    t = 5.0
    T = 10.0
    X = cat(x,vec(y),dims=1)
    X = cat(X,[0],dims=1)
    zcb = zeroBond(model,t,T,X,nothing)
    #
    G_ = (1.0 .- exp.(-chi*(T-t))) ./ chi
    zcbRef = discount(yts,T) / discount(yts,t) * exp(-transpose(G_) * x - 0.5 * transpose(G_) * y * G_)
    @test zcb == zcbRef    
end

@testset "test_Sigma_xT" begin
    yts = YieldCurve(0.03)
    d     = 3
    times = [  10.0   ]
    sigma = [  0.0060 ;
               0.0050 ;
               0.0040 ]
    slope = [  0.10   ;
               0.20   ;
               0.30   ]
    curve = [  0.05   ;
               0.10   ;
               0.20   ]
    delta = [ 1.0, 5.0, 20.0 ]
    chi   = [ 0.01, 0.05, 0.15 ]
    Gamma = [ 1.0  0.8  0.6 ; 
              0.8  1.0  0.8 ;
              0.6  0.8  1.0 ]
    model = QuasiGaussianModel(yts,d,times,sigma,slope,curve,delta,chi,Gamma)
    x = [ 0.00206853, 0.01093424, 0.00696041 ]  # np.random.uniform(0.0,0.05,d)
    y = [ 0.02980810  0.00656372 0.0041344  ;
          0.04282753  0.02406725 0.0001211  ;
          0.02906687  0.01078535 0.02328022 ] # np.random.uniform(0.0,0.05,[d,d])
    y = y * transpose(y)
    t = 5.0
    X = cat(x,vec(y),dims=1)
    X = cat(X,[0],dims=1)
    sigma_xT_ = sigma_xT(model,t,X)
    # reference result from QuantLib
    sigma_xT_Ref = [
         0.031658475910   0.017375011335   0.084573118461 ;
        -0.007058576846   0.021654204395  -0.134352049767 ;
        -0.016417240779  -0.043917617245   0.051199735933 ]
    @test norm(sigma_xT_ - sigma_xT_Ref) < 1.0e-12
end

@testset "test_Evolve" begin
    yts = YieldCurve(0.03)
    d     = 3
    times = [  10.0   ]
    sigma = [  0.0060 ;
               0.0050 ;
               0.0040 ]
    slope = [  0.10   ;
               0.20   ;
               0.30   ]
    curve = [  0.05   ;
               0.10   ;
               0.20   ]
    delta = [ 1.0, 5.0, 20.0 ]
    chi   = [ 0.01, 0.05, 0.15 ]
    Gamma = [ 1.0  0.8  0.6 ; 
              0.8  1.0  0.8 ;
              0.6  0.8  1.0 ]
    model = QuasiGaussianModel(yts,d,times,sigma,slope,curve,delta,chi,Gamma)
    times = [0.0, 5.0, 10.0]
    dW = [ -0.8723107  -0.00585635  0.31102388 ;
           -0.15673275  0.28482762  0.79041947 ]
    Xref = [ 0.          0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.           0.         ;
            -0.02182255  0.0397456   -0.01717905   0.0006412   -0.00116167   0.00059747  -0.00116167   0.00259077  -0.00141925   0.00059747  -0.00141925   0.00088815   0.00185998 ;
            -0.02476015  0.04705765  -0.0083194    0.00127306  -0.00232704   0.00102049  -0.00232704   0.00518216  -0.00243878   0.00102049  -0.00243878   0.00133699   0.03866521 ]
   #
   X = zeros((3,13))
   evolve(model, 0.0, X[1,:], 5.0, dW[1,:], @view X[2,:])
   evolve(model, 5.0, X[2,:], 5.0, dW[2,:], @view X[3,:])
   @test norm(X-Xref) < 1.0e-7
   # test zero coupon bond (again)
   zcbRef = 0.851676232405887
   zcb = zeroBond(model,5.0,10.0,X[2,:],nothing)
   @test abs(zcb - zcbRef) < 2.0e-10
end
