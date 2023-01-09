# Bayesian Feedforward 

using BFlux 
using Flux
using Serialization
using StatsPlots

data = deserialize("./examples/data_ar1.jld")
y = data
# althugh the data come from an AR(1) model, we will use five lags
x = reduce(hcat, [y[i-5:i-1] for i=6:size(y, 1)])
y = y[6:end]
# We will use the first 500 observations of these vectors as training data and 
# the ramining 495 as test data
x_train, x_test = x[:, 1:500], x[:, 501:end]
y_train, y_test = y[1:500], y[501:end]

net = Chain(Dense(5, 5, relu), Dense(5, 1))
nc = destruct(net)
like = FeedforwardNormal(nc, Gamma(2.0, 0.5))
prior = GaussianPrior(nc, 0.8f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
bnn = BNN(x_train, y_train, like, prior, init)

Random.seed!(6150533)
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^2, μ = 10f0)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]

ypp = sample_posterior_predict(bnn, ch; x = x_test)
ypp_mean = mean(ypp; dims = 2)
plot(ypp_mean)
plot!(y_test)

qs = [quantile(x, 0.05) for x in eachrow(ypp)]
plot(qs; label = "5% predicted quantile", color = :red)
plot!(y_test; label = "Test data", color = :black)
mean(y_test .< qs)

predict(net) = vec(net(x_test))
y_prior = sample_prior_predictive(bnn, predict, 20_000)
y_prior = reduce(hcat, y_prior)

i = 110  # Observation for which we plot densities
density(y_prior[i,:]; label = "prior")
density!(ypp[i,:]; label = "posterior")

