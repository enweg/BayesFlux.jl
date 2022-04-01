module BFlux

include("./model/BNN.jl")
include("./layers/dense.jl")
include("./likelihoods/feedforward.jl")

export BNN, BLayer
export lp, reconstruct_sample
export FeedforwardNormal


end # module
