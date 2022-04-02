module BFlux

include("./model/BNN.jl")
include("./layers/dense.jl")
include("./layers/recurrent.jl")
include("./likelihoods/feedforward.jl")
include("./likelihoods/seq_to_one.jl")
include("./optimise/modes.jl")
include("./sampling/laplace.jl")

###### These are for testing
include("./sampling/nuts_test.jl")
export sample_nuts

###### Exports
export BNN, BLayer
export lp, reconstruct_sample
export FeedforwardNormal, FeedforwardTDist
export SeqToOneNormal, SeqToOneTDist
export find_mode
export laplace, SIR_laplace

end # module
