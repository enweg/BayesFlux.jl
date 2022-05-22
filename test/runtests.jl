using BFlux
using Flux
using Distributions, Random, Bijectors
using Test

# @testset "BFlux" begin

    # destructing Networks and layers
    include("./deconstruct.jl")
    # likelihoods
    include("./likelihoods.jl")
    # network priors
    include("./networkpriors.jl")
    # initialisers 
    include("./initialisers.jl")
    # BNN basic operations
    include("./bnn.jl")
    # Mode Finding
    include("./modes.jl")
    # MCMC
    include("./sgld.jl")


    # include("laplace.jl")
    # include("bbb.jl")
    # include("modes.jl")
    # include("ggmc.jl")
# end