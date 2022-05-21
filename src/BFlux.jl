module BFlux

include("./model/BNN.jl")
include("./model/deconstruct.jl")

include("./likelihoods/abstract.jl")
include("./likelihoods/feedforward.jl")
include("./likelihoods/seq_to_one.jl")

include("./netpriors/abstract.jl")
include("./netpriors/gaussian.jl")

include("./initialisers/abstract.jl")
include("./initialisers/basics.jl")

include("./model/posterior.jl")
include("./layers/dense.jl")
include("./layers/recurrent.jl")
include("./optimise/modes.jl")
include("./sampling/laplace.jl")
include("./sampling/advi.jl")
include("./sampling/bbb.jl")
include("./sampling/sgld.jl")
include("./sampling/ggmc.jl")
include("./simulations/AR.jl")
include("./utils/rnn_utils.jl")

###### Exports
export destruct

export BNNLikelihood, predict
export FeedforwardNormal, FeedforwardTDist
export SeqToOneNormal, SeqToOneTDist

export NetworkPrior, sample_prior 
export GaussianPrior

export BNNInitialiser
export InitialiseAllSame



export BNN, BLayer
export posterior_predict
export lp, reconstruct_sample
export find_mode, find_mode_sgd
export laplace, SIR_laplace
export advi, bbb
export sgld, ggmc

end # module
