module BFlux

include("./model/BNN.jl")
export BNN
export split_params
export loglikeprior, ∇loglikeprior
export sample_prior_predictive, get_posterior_networks, sample_posterior_predict

include("./layers/dense.jl")
include("./layers/recurrent.jl")

include("./model/deconstruct.jl")
export destruct

include("./likelihoods/abstract.jl")
include("./likelihoods/feedforward.jl")
include("./likelihoods/seq_to_one.jl")
export BNNLikelihood, predict
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
include("./inference/mcmc/ggmc.jl")
include("./inference/mcmc/amh.jl")
include("./inference/mcmc/hmc.jl")
export MCMCState, mcmc
export SGLD
export GGMC
export AdaptiveMH
export HMC


# Variational Inference Methods
# include("./inference/vi/advi.jl")
include("./inference/vi/bbb.jl")
# export advi
export bbb


# include("./model/posterior.jl")
# include("./optimise/modes.jl")
# include("./sampling/laplace.jl")
# include("./sampling/advi.jl")
# include("./sampling/bbb.jl")
# include("./sampling/sgld.jl")
# include("./sampling/ggmc.jl")
# include("./simulations/AR.jl")
# include("./utils/rnn_utils.jl")

# ###### Exports
# export posterior_predict
# export lp, reconstruct_sample
# export find_mode, find_mode_sgd
# export laplace, SIR_laplace
# export advi, bbb
# export sgld, ggmc

end # module
