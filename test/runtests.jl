using BFlux
using Flux
using Distributions, Random, Bijectors
using Test

@testset "BFlux" begin

    # destructing Networks and layers
    # include("./deconstruct.jl")
    # likelihoods
    # include("./likelihoods.jl")
    # network priors
    include("./networkpriors.jl")

    # include("laplace.jl")
    # include("bbb.jl")
    # include("modes.jl")
    # include("ggmc.jl")
end