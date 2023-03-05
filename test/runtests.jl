using BayesFlux
using Flux
using Distributions, Random, Bijectors
using Test
using LinearAlgebra

println("Hostname: $(gethostname())")

@testset "BayesFlux" begin
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
    # Posterior Predictive Draws
    include("./posterior_predict.jl")
    # Checking whether all derivatives work. See issue #6
    include("./derivatives.jl")

    # Tests after this line are reduced in the number of samples when run 
    # on GitHub actions. 
    if gethostname()[1:2] == "fv" 
        @info "Tests run on GitHub actions are reduced. For the full tests suit, please run tests on another machine."
    end
    # MCMC
    include("./sgld.jl")
    include("./sgnht.jl")
    include("./sgnht-s.jl")
    include("./ggmc.jl")
    include("./amh.jl")
    include("./hmc.jl")
    # vi
    include("./bbb.jl")
end