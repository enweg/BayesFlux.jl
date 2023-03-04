module BayesFlux

include("./utils/gradient_utils.jl")

include("./layers/dense.jl")
include("./layers/recurrent.jl")

include("./model/deconstruct.jl")
export destruct
export NetConstructor

include("./likelihoods/abstract.jl")
include("./likelihoods/feedforward.jl")
include("./likelihoods/seq_to_one.jl")
export BNNLikelihood, posterior_predict
export FeedforwardNormal, FeedforwardTDist
export SeqToOneNormal, SeqToOneTDist

include("./netpriors/abstract.jl")
include("./netpriors/gaussian.jl")
include("./netpriors/mixturescale.jl")
export NetworkPrior, sample_prior
export GaussianPrior
export MixtureScalePrior

include("./initialisers/abstract.jl")
include("./initialisers/basics.jl")
export BNNInitialiser
export InitialiseAllSame

include("./model/BNN.jl")
export BNN
export split_params
export loglikeprior, ∇loglikeprior
export sample_prior_predictive, get_posterior_networks, sample_posterior_predict

include("./inference/mode/abstract.jl")
include("./inference/mode/flux.jl")
export BNNModeFinder, find_mode, step!
export FluxModeFinder

# Abstract MCMC
include("./inference/mcmc/abstract.jl")
# Mass Adapters
include("./inference/mcmc/adapters/mass/abstract_mass.jl")
include("./inference/mcmc/adapters/mass/diagcovariancemassadapter.jl")
include("./inference/mcmc/adapters/mass/fixedmassmatrix.jl")
include("./inference/mcmc/adapters/mass/fullcovariancemassadapter.jl")
include("./inference/mcmc/adapters/mass/rmspropmassadapter.jl")
export MassAdapter
export DiagCovMassAdapter, FixedMassAdapter, FullCovMassAdapter, RMSPropMassAdapter
# Stepsize Adapters
include("./inference/mcmc/adapters/stepsize/abstract_stepsize.jl")
include("./inference/mcmc/adapters/stepsize/constantstepsize.jl")
include("./inference/mcmc/adapters/stepsize/dualaveragestepsize.jl")
export StepsizeAdapter
export ConstantStepsize, DualAveragingStepSize
# MCMC Methods
include("./inference/mcmc/sgld.jl")
include("./inference/mcmc/sgnht.jl")
include("./inference/mcmc/sgnht-s.jl")
include("./inference/mcmc/ggmc.jl")
include("./inference/mcmc/amh.jl")
include("./inference/mcmc/hmc.jl")
export MCMCState, mcmc
export SGLD
export SGNHT
export SGNHTS
export GGMC
export AdaptiveMH
export HMC


# Variational Inference Methods
# include("./inference/vi/advi.jl")
include("./inference/vi/bbb.jl")
# export advi
export bbb

# Utilities
include("./utils/rnn_utils.jl")
export make_rnn_tensor

end # module
