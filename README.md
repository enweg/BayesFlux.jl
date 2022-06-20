<!-- Create the .md file by running
Literate.markdown("./README.jl", flavor = Literate.CommonMarkFlavor()) -->

````julia
using BFlux, Flux
using Random, Distributions
using StatsPlots

Random.seed!(6150533)
````

## BFlux (Bayesian extension for Flux)
BFlux is meant to be an extension to Flux.jl, a machine learning
library written entirely in Julia. BFlux will and is not meant to be the
fastest production ready library, but rather is meant to make research and
experimentation easy.

BFlux is part of my Master Thesis in Economic and Financial Research -
specialisation Econometrics and will therefore likelily still go through some
revisions in the coming months.

## Structure

Every Bayesian model can in general be broken down into the probablistic
model, which gives the likelihood function and the prior on all parameters of
the probabilistic model. BFlux somewhat follows this and splits every Bayesian
Network into the following parts:

1. **Network**: Every BNN must have some general network structure. This is
   defined using Flux and currently supports Dense, RNN, and LSTM layers. More
   on this later
2. **Network Constructor**: Since BFlux works with vectors of parameters, we
   need to be able to go from a vector to the network and back. This works by
   using the NetworkConstructor.
3. **Likelihood**: The likelihood function. In traditional estimation of NNs,
   this would correspond to the negative loss function. BFlux has a twist on
   this though and nomenclature might change because of this twist: The
   likelihood also contains all additional parameters and priors. For example,
   for a Gaussian likelihood, the likelihood object also defines the standard
   deviation and the prior for the standard deviation. This desing choice was
   made to keep the likelihood and everything belonging to it separate from
   the network; Again, due to the potential confusion, the nomenclature might
   change in later revisions.
4. **Prior on network parameters**: A prior on all network parameters.
   Currently the RNN layers do not define priors on the initial state and thus
   the initial state is also not sampled. Priors can have hyper-priors.
5. **Initialiser**: Unless some special initialisation values are given, BFlux
   will draw initial values as defined by the initialiser. An initialiser
   initialises all network and likelihood parameters to reasonable values.

All of the above are then used to create a BNN which can then be estimated
using the MAP, can be sampled from using any of the MCMC methods implemented,
or can be estimated using Variational Inference.

The examples and the sections below hopefully clarify everything. If any
questions remain, please open an issue.

## Linear Regression using BFlux

Although not meant for Simple Linear Regression, BFlux can be used for it, and
we will do so in this section. This will hopefully demonstrate the basics.
Later sections will show better examples.

Let's say we have the idea that the data can be modelled via a linear model of
the form
$$y_i = x_i'\beta + e_i$$
with $e_i \sim N(0, 1)$

````julia
k = 5
n = 500
x = randn(Float32, k, n);
β = randn(Float32, k);
y = x'*β + randn(Float32, n);
````

This is a standard linear model and we would likely be better off using STAN
or Turing for this, but due to the availability of a Dense layer with linear
activation function, we can also implent it in BFlux.

The first step is to define the network. As mentioned above, the network
consists of a single Dense layer with a linear activation function (the
default activation in Flux and hence not explicitly shown).

````julia
net = Chain(Dense(k, 1))  # k inputs and one output
````

Since BFlux works with vectors, we need to be able to transform a vector to
the above network and back. We thus need a NetworkConstructor, which we obtain
as a the return value of a `destruct`

````julia
nc = destruct(net)
````

We can check whether everything work by just creating a random vector of the
right dimension and calling the NetworkConstructor using this vector.

````julia
θ = randn(Float32, nc.num_params_network)
nc(θ)
````

We indeed obtain a network of the right size and structure.
Next, we will define a prior for all parameters of the network. Since weight
decay is a popular regularisation method in standard ML estimation, we will be
using a Gaussian prior, which is the Bayesian weight decay:

````julia
prior = GaussianPrior(nc, 0.5f0)  # the last value is the standard deviation
````

We also need a likelihood and a prior on all parameters the likelihood
introduces to the model. We will go for a Gaussian likelihood, which
introduces the standard deviation of the model. BFlux currently implements
Gaussian and Student-t likelihoods for Feedforward and Seq-to-one cases but
more can easily be implemented. See **TODO HAR link** for an example.

````julia
like = FeedforwardNormal(nc, Gamma(2.0, 0.5))  # Second argument is prior for standard deviation.
````

Lastly, when no explicit initial value is given, BFlux will draw it from an
initialiser. Currently only one type of initialiser is implemented in BFlux,
but this can easily be extended by the user itself.

````julia
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)  # First argument is dist we draw parameters from.
````

Given all the above, we can now define the BNN:

````julia
bnn = BNN(x, y, like, prior, init)
````

### MAP estimate.

It is always a good idea to first find the MAP estimate. This can serve two
purposes:

1. It is faster than fully estimating the model using MCMC or VI and can thus
   serve as a quick check; If the MAP estimate results in bad point
   predictions, so will likely the full estimation results.
2. It can serve as a starting value for the MCMC samplers.

To find a MAP estimate, we must first specify how we want to find it: We need
to define an optimiser. BFlux currently only implements optimisers derived
from Flux itself, but this can be extended by the user.

````julia
opt = FluxModeFinder(bnn, Flux.ADAM())  # We will use ADAM
θmap = find_mode(bnn, 10, 500, opt)  # batchsize 10 with 500 epochs
````

We can already use the MAP estimate to make some predictions and calculate the
RMSE.

````julia
nethat = nc(θmap)
yhat = vec(nethat(x))
sqrt(mean(abs2, y .- yhat))
````

### MCMC - SGLD

If the MAP estimate does not show any problems, it can be used as the starting
point for SGLD or any of the other MCMC methods (see later section).

Simulations have shown that using a relatively large initial stepsize with a
slow decaying stepsize schedule often results in the best mixing. *Note: We
would usually use samplers such as NUTS for linear regressions, which are much
more efficient than SGLD*

````julia
sampler = SGLD(Float32; stepsize_a = 10f-0, stepsize_b = 0.0f0, stepsize_γ = 0.55f0)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]
````

We can obtain summary statistics and trace and density plots of network
parameters and likelihood parameters by transforming the BFlux chain into a
MCMCChain.

````julia
using MCMCChains
chain = Chains(ch')
plot(chain)
````

In more complicated networks, it is usually a hopeless goal to obtain good
mixing in parameter space and thus we rather focus on the output space of the
network. *Mixing in parameter space is hopeless due to the very complicated
topology of the posterior; see ...*
We will use a little helper function to get the output values of the network:

````julia
function naive_prediction(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    yhats = Array{T, 2}(undef, length(y), size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        yh = vec(net(x))
        yhats[:,i] = yh
    end
    return yhats
end

yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
maximum(summarystats(chain_yhat)[:, :rhat])
````

Similarly, we can obtain posterior predictive values and evaluate quantiles
obtained using these to how many percent of the actual data fall below the
quantiles. What we would like is that 5% of the data fall below the 5%
quantile of the posterior predictive draws.

````julia
function get_observed_quantiles(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    qs = [quantile(yr, target_q) for yr in eachrow(posterior_yhat)]
    qs = reduce(hcat, qs)
    observed_q = mean(reshape(y, 1, :) .< qs; dims = 2)
    return observed_q
end

posterior_yhat = sample_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y, posterior_yhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")
````

### MCMC - SGNHTS

Just like SGLD, SGNHTS also does not apply a Metropolis-Hastings correction
step. Contrary to SGLD though, SGNHTS implementes a Thermostat, whose task it
is to keep the temperature in the dynamic system close to one, and thus the
sampling more accurate. Although the thermostats goal is often not achieved,
samples obtained using SGNHTS often outperform those obtained using SGLD.

````julia
sampler = SGNHTS(1f-2, 2f0; xi = 2f0^2, μ = 50f0)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')
````

----

````julia
yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
maximum(summarystats(chain_yhat)[:, :rhat])
````

----

````julia
posterior_yhat = sample_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y, posterior_yhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")
````

### MCMC - GGMC

As pointed out above, neither SGLD nor SGNHTS apply a Metropolis-Hastings
acceptance step and are thus difficult to monitor. Indeed, draws from SGLD or
SGNHTS should perhaps rather be considered as giving and ensemble of models
rather than draws from the posterior, since without any MH step, it is unclear
whether the chain actually will converge to the posterior.

BFlux also implements three methods that do apply a MH step and are thus
easier to monitor. These are GGMC, AdaptiveMH, and HMC. Both GGMC and HMC do
allow for taking stochastic gradients. GGMC also allows to use delayed
acceptance in which the MH step is only applied after a couple of steps,
rather than after each step (see ... for details).

Because both GGMC and HMC use a MH step, they provide a measure of the mean
acceptance rate, which can be used to tune the stepsize using Dual Averaging
(see .../STAN for details). Similarly, both also make use of mass matrices,
which can also be tuned.

BFlux implements both stepsize adapters and mass adapters but to this point
does not implement a smart way of combining them (this will come in the
future). In my experience, naively combining them often only helps in more
complex models and thus we will only use a stepsize adapter here.

````julia
sadapter = DualAveragingStepSize(1f-9; target_accept = 0.55f0, adapt_steps = 10000)
sampler = GGMC(Float32; β = 0.1f0, l = 1f-9, sadapter = sadapter)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')
````

----

````julia
yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
maximum(summarystats(chain_yhat)[:, :rhat])
````

----

````julia
posterior_yhat = sample_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y, posterior_yhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")
````

The above uses a MH correction after each step. This can be costly in big-data
environments or when the evaluation of the likelihood is costly. If either of
the above applies, delayed acceptance can speed up the process.

````julia
sadapter = DualAveragingStepSize(1f-9; target_accept = 0.25f0, adapt_steps = 10000)
sampler = GGMC(Float32; β = 0.1f0, l = 1f-9, sadapter = sadapter, steps = 3)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')
````

----

````julia
yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
maximum(summarystats(chain_yhat)[:, :rhat])
````

----

````julia
posterior_yhat = sample_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y, posterior_yhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")
````

### MCMC - HMC

Since HMC showed some mixing problems for some variables during the testing of
this README, we decided to use a mass matrix adaptation. This turned out to
work better even in this simple case.

````julia
sadapter = DualAveragingStepSize(1f-9; target_accept = 0.55f0, adapt_steps = 10000)
madapter = DiagCovMassAdapter(5000, 1000)
sampler = HMC(1f-9, 5; sadapter = sadapter)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')
````

----

````julia
yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
maximum(summarystats(chain_yhat)[:, :rhat])
````

----

````julia
posterior_yhat = sample_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y, posterior_yhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")
````

### MCMC - Adaptive Metropolis-Hastings

As a derivative free alternative, BFlux also implements Adaptive MH as
introduced in (...). This is currently quite a costly method for complex
models since it needs to evaluate the MH ratio at each step. Plans exist to
parallelise the calculation of the likelihood which should speed up Adaptive
MH.

````julia
sampler = AdaptiveMH(diagm(ones(Float32, bnn.num_total_params)), 1000, 0.5f0, 1f-4)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')
````

----

````julia
yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
maximum(summarystats(chain_yhat)[:, :rhat])
````

----

````julia
posterior_yhat = sample_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y, posterior_yhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")
````

## Variation Inference

In some cases MCMC method either do not work well or even the methods above
take too long. For these cases BFlux currently implements Bayes-By-Backprop
(...); One shortcoming of the current implementation is that the variational
family is constrained to a diagonal multivariate gaussian and thus any
correlations between network parameters are set to zero. This can cause
problems in some situations and plans exist to allow for more felxible
covariance specifications.

````julia
q, params, losses = bbb(bnn, 10, 2_000; mc_samples = 1, opt = Flux.ADAM(), n_samples_convergence = 10)
ch = rand(q, 20_000)
posterior_yhat = sample_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y, posterior_yhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

