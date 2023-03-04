# Example: Recurrent Neural Networks

Next to Dense layers, BayesFlux also implements RNN and LSTM layers. These two do
require some additional care though, since the layout of the data must be
adjusted. In general, the last dimension of `x` and `y` is always the dimension
along which BayesFlux batches, which is also what Flux does. Thus, if we are in a
seq-to-one setting then the sequences must be along the last dimension (here the
third). To demonstrate this, let us simulate some AR1 data

!!! note "Note" 
    BayesFlux currently only implements univariate regression problems (a single
    dependent variable) and for recurrent structures only seq-to-one type of
    settings. This can be extended by the user. For this see
    [`BNNLikelihood`](@ref)

````julia
Random.seed!(6150533)
gamma = 0.8
N = 500
burnin = 1000
y = zeros(N + burnin + 1)
for t=2:(N+burnin+1)
    y[t] = gamma*y[t-1] + randn()
end
y = Float32.(y[end-N+1:end])
````

Just like in the FNN case, we need a network structure and its constructor, a
prior on the network parameters, a likelihood with a prior on the additional
parameters introduced by the likelihood, and an initialiser. Note how most
things are the same as for the FNN case, with the differences being the actual
network defined and the likelihood.

````julia
net = Chain(RNN(1, 1), Dense(1, 1))  # last layer is linear output layer
nc = destruct(net)
like = SeqToOneNormal(nc, Gamma(2.0, 0.5))
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
````

We are given a single sequence (time series). To exploit batching and to not
always have to feed through the whole sequence, we will split the single
sequence into overlapping subsequences of length 5 and store these in a
tensor. Note that we add 1 to the subsequence length, because the last
observation of each subsequence will be our training observation to predict
using the fist five items in the subsequence.

````julia
x = make_rnn_tensor(reshape(y, :, 1), 5 + 1)
y = vec(x[end, :, :])
x = x[1:end-1, :, :]
````

We are now ready to create the BNN and find the MAP estimate. The MAP will be
used to check whether the overall network structure makes sense (does provide
at least good point estimates).

````julia
bnn = BNN(x, y, like, prior, init)
opt = FluxModeFinder(bnn, Flux.RMSProp())
θmap = find_mode(bnn, 10, 1000, opt)
````

When checking the performance we need to make sure to feed the sequences
through the network observation by observation:

````julia
nethat = nc(θmap)
yhat = vec([nethat(xx) for xx in eachslice(x; dims =1 )][end])
sqrt(mean(abs2, y .- yhat))
````

The rest works just like before with some minor adjustments to the helper
functions.

````julia
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^2, μ = 10f0)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')

function naive_prediction_recurrent(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    yhats = Array{T, 2}(undef, length(y), size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        yh = vec([net(xx) for xx in eachslice(x; dims = 1)][end])
        yhats[:,i] = yh
    end
    return yhats
end
````

----

````julia
yhats = naive_prediction_recurrent(bnn, ch)
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
