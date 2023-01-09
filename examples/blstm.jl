# Bayesian LSTM network
using BFlux 
using Flux
using Serialization
using StatsPlots

data = deserialize("./examples/data_ar1.jld")
y = data

x = make_rnn_tensor(reshape(y, :, 1), 5 + 1)
y = vec(x[end,:,:])
y_train, y_test = y[1:500], y[501:end]
x = x[1:end-1,:,:]
x_train, x_test = x[:,:,1:500], x[:,:,501:end]

net = Chain(LSTM(1, 1), Dense(1, 1))  # last layer is linear output layer
nc = destruct(net)
like = SeqToOneNormal(nc, Gamma(2.0, 0.5))
prior = GaussianPrior(nc, 1.8f0)
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

predict(net) = vec([net(xx) for xx in eachslice(x_test; dims=1)][end])
y_prior = sample_prior_predictive(bnn, predict, 20_000)
y_prior = reduce(hcat, y_prior)

i = 110  # Observation for which we plot densities
density(y_prior[i,:]; label = "prior")
density!(ypp[i,:]; label = "posterior")



