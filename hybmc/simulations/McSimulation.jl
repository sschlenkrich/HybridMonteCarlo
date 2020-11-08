
using Random, Distributions

struct McSimulation
    model
    times
    nPaths
    seed
    X
end

function McSimulation(model,times,nPaths,seed)
    Random.seed!(seed) # Setting the seed
    norm = Normal()
    dW = rand( norm, (nPaths,size(times)[1]-1,factors(model)) )
    X = zeros((nPaths,size(times)[1],stateSize(model)))
    @views for i = 1:nPaths
        X[i,1,:] = initialValues(model)
        for j = 1:size(times)[1]-1
            evolve(model,times[j],X[i,j,:],times[j+1]-times[j],dW[i,j,:],X[i,j+1,:])
        end
    end
    return McSimulation(model,times,nPaths,seed,X)
end
