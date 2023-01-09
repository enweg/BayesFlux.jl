# Simulate data coming from an AR(1). This will be used for the examples. 

using Distributions
using Random
using BFlux
using Serialization

function ar1(gamma; c = 0, dist = Normal(0, 1), N = 500, burnin = 1000, seed = 6150533)
    rng = Random.MersenneTwister(seed)
    y = zeros(N+burnin+1)
    for t=2:length(y)
        y[t] = c + gamma*y[t-1] + rand(rng, dist)
    end
    return y[end-N+1:end]
end

gamma = 0.5
y = ar1(gamma; N = 1000)
# BFlux by default works with Float32 data so lets convert it to that
y = Float32.(y)

data = (
    y = y
)

serialize("./examples/data_ar1.jld", data)