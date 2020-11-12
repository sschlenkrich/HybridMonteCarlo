
using Printf, Test

include("../../hybmc/termstructures/YieldCurve.jl")
include("../../hybmc/models/DeterministicModel.jl")
include("../../hybmc/simulations/McSimulation.jl")
include("../../hybmc/simulations/Payoffs.jl")

function deterministicPath()
    yc = YieldCurve(0.0)
    model = DeterministicModel("EUR",yc,["USD"],[1.0],[yc])
    sim = McSimulation(model,[0.0],1)
    return Path(sim,1)
end

@testset "test_payoffSetup" begin
    p = deterministicPath()
    a  = 0.5
    px = 1.0
    py = 2.0
    x = Fixed(px)
    y = Fixed(py)
    # 
    @test isequal(at(x+y,p), px+py)
    @test isequal(at(x-y,p), px-py)
    @test isequal(at(x*y,p), px*py)
    @test isequal(at(x/y,p), px/py)
    # 
    @test isequal(at(x+a,p), px+a)
    @test isequal(at(x-a,p), px-a)
    @test isequal(at(x*a,p), px*a)
    @test isequal(at(x/a,p), px/a)
    # 
    @test isequal(at(a+y,p), a+py)
    @test isequal(at(a-y,p), a-py)
    @test isequal(at(a*y,p), a*py)
    @test isequal(at(a/y,p), a/py)
    #
    @test isequal(at(x<y,p),float(px<py))
    @test isequal(at(x<=y,p),float(px<=py))
    @test isequal(at(x==y,p),float(px==py))
    @test isequal(at(x>=y,p),float(px>=py))
    @test isequal(at(x>y,p),float(px>py))
    #
    @test isequal(at(x<a,p),float(px<a))
    @test isequal(at(x<=a,p),float(px<=a))
    @test isequal(at(x==a,p),float(px==a))
    @test isequal(at(x>=a,p),float(px>=a))
    @test isequal(at(x>a,p),float(px>a))
    #
    @test isequal(at(a<y,p),float(a<py))
    @test isequal(at(a<=y,p),float(a<=py))
    @test isequal(at(a==y,p),float(a==py))
    @test isequal(at(a>=y,p),float(a>=py))
    @test isequal(at(a>y,p),float(a>py))
    #
    @test isequal(obsTime(x + y), 0.0)
    @test isequal(obsTime((x+y)%1.0), 1.0)
end

