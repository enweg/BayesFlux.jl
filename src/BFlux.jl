module BFlux

include("./model/BNN.jl")
include("./layers/dense.jl")
include("./layers/recurrent.jl")
include("./likelihoods/feedforward.jl")
include("./likelihoods/seq_to_one.jl")

export BNN, BLayer
export lp, reconstruct_sample
export FeedforwardNormal, FeedforwardTDist
export SeqToOneNormal


end # module
