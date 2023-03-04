# BayesFlux.jl a Bayesian extension to [Flux.jl](https://fluxml.ai)

````julia
using BayesFlux, Flux
using Random, Distributions
using StatsPlots
using LinearAlgebra

Random.seed!(6150533)
````

```@meta
CurrentModule = BayesFlux
DocTestSetup = quote
    using BayesFlux
end
```

BayesFlux is meant to be an extension to Flux.jl, a machine learning library written
entirely in Julia. BayesFlux will and is not meant to be the fastest production
ready library, but rather is meant to make research and experimentation easy.

BayesFlux is part of my Master Thesis in Economic and Financial Research -
specialisation Econometrics at Maastricht University and will therefore likely
still go through some revisions in the coming months.

## Structure

Every Bayesian model can in general be broken down into the probabilistic
model, which gives the likelihood function and the prior on all parameters of
the probabilistic model. BayesFlux somewhat follows this and splits every Bayesian
Network into the following parts:

1. **Network**: Every BNN must have some general network structure. This is
   defined using Flux and currently supports Dense, RNN, and LSTM layers. More
   on this later
2. **Network Constructor**: Since BayesFlux works with vectors of parameters, we
   need to be able to go from a vector to the network and back. This works by
   using the NetworkConstructor.
3. **Likelihood**: The likelihood function. In traditional estimation of NNs,
   this would correspond to the negative loss function. BayesFlux has a twist on
   this though and nomenclature might change because of this twist: The
   likelihood also contains all additional parameters and priors. For example,
   for a Gaussian likelihood, the likelihood object also defines the standard
   deviation and the prior for the standard deviation. This design choice was
   made to keep the likelihood and everything belonging to it separate from
   the network; Again, due to the potential confusion, the nomenclature might
   change in later revisions.
4. **Prior on network parameters**: A prior on all network parameters.
   Currently the RNN layers do not define priors on the initial state and thus
   the initial state is also not sampled. Priors can have hyper-priors.
5. **Initialiser**: Unless some special initialisation values are given, BayesFlux
   will draw initial values as defined by the initialiser. An initialiser
   initialises all network and likelihood parameters to reasonable values.

All the above are then used to create a BNN which can then be estimated
using the MAP, can be sampled from using any of the MCMC methods implemented,
or can be estimated using Variational Inference.

The examples and the sections below hopefully clarify everything. If any
questions remain, please open an issue.
