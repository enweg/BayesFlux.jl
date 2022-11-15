using BFlux
using Flux
using Distributions, Random, Bijectors
using Test
using LinearAlgebra

@testset "BFlux" begin
    Random.seed!(6150533)
    # destructing Networks and layers
    include("./deconstruct.jl")
    # # likelihoods
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
    # include("./sgld.jl")
    # include("./sgnht.jl")
    # include("./sgnht-s.jl")
    # include("./ggmc.jl")
    # include("./amh.jl")
    # include("./hmc.jl")
    # # vi
    # include("./bbb.jl")
end