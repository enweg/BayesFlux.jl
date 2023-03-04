# Example: Feedforward NN Regression 

Let's say we have the same setting as in [Example: Linear Regression](@ref) but
are not aware that it is a linear model and thus decide to use a Feedforward
Neural Network. 

````julia
k = 5
n = 500
x = randn(Float32, k, n);
β = randn(Float32, k);
y = x'*β + randn(Float32, n);
````

While some might think this will change a lot, given that the
model we are estimating is a lot more complicated than a linear regression
model, BayesFlux abstracts away all of this and all that changes is the network
definition. 

````julia
net = Chain(Dense(k, k, relu), Dense(k, k, relu), Dense(k, 1))
````

We can then still use the same prior, likelihood, and initialiser. But we do
need to change the NetworkConstructor, which we still obtain in the same way by
calling `destruct`

````julia
nc = destruct(net)
like = FeedforwardNormal(nc, Gamma(2.0, 0.5))
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
bnn = BNN(x, y, like, prior, init)
````

The rest is the same as for the linear regression case. We can, for example,
first find the MAP:

````julia
opt = FluxModeFinder(bnn, Flux.ADAM())  # We will use ADAM
θmap = find_mode(bnn, 10, 500, opt)  # batchsize 10 with 500 epochs
````

----

````julia
nethat = nc(θmap)
yhat = vec(nethat(x))
sqrt(mean(abs2, y .- yhat))
````

Or we can use any of the MCMC or VI method - SGNHTS is just one option:

````julia
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^2, μ = 10f0)
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