# ## BFlux (Bayesian extension for Flux)
# BFlux is meant to be an extension to Flux.jl, a machine learning 
# library written entirely in Julia. BFlux will and is not meant to be the 
# fastest production ready library, but rather is meant to make research and 
# expeimentation easy. Currently it is in its infancy and a lot of work still 
# needs to be done.

# ## Basics

# To demonstrate the basics, we will first simulate some data. Here we will 
# focus on regression problems: 

using Distributions, Random
using StatsPlots
rng = Random.MersenneTwister(123)
X = randn(rng, 3, 100) # Note the dimensions (3 variables, 100 observations)
β = randn(rng, 3)
y = X'*β + randn(rng, 100);


# The basics of BFlux are very simple. Given a Flux model (a `Chain` object)
# and a desired likelihood and data `y` and `x`, a Bayesian version of the 
# network can be created by calling `BNN` in the following way: 

using Flux, BFlux
net = Chain(Dense(3, 3, sigmoid), Dense(3, 1))
loglike = BFlux.FeedforwardNormal(Gamma(2.0, 2.0), Float64)
bnn = BNN(net, loglike, y, X);

# The first thing we can then do is to find the mode and use the mode to 
# check how well the model fits. Note that we are just using all data as 
# training data for now.  

mode = find_mode(bnn, 10_000, 1e-6)
nethat = bnn.re(mode[1]) # reconstruct the original network using the mode parameters
yhat = nethat(X)
mean((y .- yhat).^2)

# The problem with this mode estimate, and the reason for extending Flux is, 
# that this does not give us any uncertainty bounds. While this is sufficient
# for many applications, financial and economic applications can often not do
# without uncertainty estimates. The most basic way in which we could obtain 
# rough uncertainty estimates is by using a Laplace approximation around the mode. 

# Note: Currently only the diagonal covariance estimation is truly working. 

lapprox = laplace(bnn, 10_000, 1e-6; init_θ = mode[1], diag = true)
θhat = rand(rng, lapprox[1], 1000) # obtaining 1000 draws 
nethats = [bnn.re(θ) for θ in eachcol(θhat)] # Giving us 1000 networks
yhats = [vec(nn(X)) for nn in nethats] # Giving us 1000 draws
yhats = hcat(yhats...)
yhat = mean(yhats; dims = 2)
mean((y.-yhat).^2)

# The above analysis does not actually draw from the posterior predictive. If we 
# truly want to draw from the posterior predictive, we can use our draws in the 
# following way: 

ppred = posterior_predict(bnn, θhat)
yhat = mean(ppred; dims = 2)
mean((y.-yhat).^2)

# The whole reason for doing Bayesian NN is though to obtain uncertainty estimates. 
# Se we can use our posterior predictive draws to obtain credible intervals 
# We can see below that the posterior predictive credible intervals are rather good. 
# So in this case, already a simple Laplace approximation around the mode is providing 
# us with reasonable intervals (again only on training data). 

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y, upper)]
sum(outside)/length(y)

# We can obviously also test everything on data not seen before. 

X_test = randn(rng, 3, 100)
y_test = X_test'*β + randn(rng, 100)
ppred = posterior_predict(bnn, θhat; newx = X_test)
yhat_test = mean(ppred; dims = 2)
mean((y_test .- yhat_test).^2)

# And similarly to above we can test how good the posterior predictive credible 
# intervals are. We find that we are still doing quite well dispite the simplicity.   

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y_test, upper)]
sum(outside)/length(y_test)

# Now what happens when the model is more complicated? 

X = randn(rng, 3, 200)
β1 = randn(rng, 3)
β2 = randn(rng, 3)
β3 = randn(rng, 3)
y = X'*β1 .+ (X.^2)'*β2 .+ (X.^3)'*β3 + randn(rng, 200)
y_test = y[101:end]
y = y[1:100]
X_test = X[:, 101:end]
X = X[:, 1:100];

# Using the same method as above would now result in: 

bnn = BNN(net, loglike, y, X)
mode = find_mode(bnn, 20_000, 1e-6)
lapprox = laplace(bnn, 10_000, 1e-6; init_θ = mode[1], diag = true)
θhat = rand(rng, lapprox[1], 1000)
ppred = posterior_predict(bnn, θhat, newx = X_test)
yhat = mean(ppred; dims = 2)
mean((y_test .- yhat).^2)


# Checking again the credible intervals: 

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y_test, upper)]
sum(outside)/length(y_test)

# Since Neural Networks are often overparameterised, they often are multimodel. 
# It is thus likely not a good idea to rely on a Laplace approximation within 
# a single mode. We can run multiple Laplace approximations in parallel by 
# defining the number of mode searches we would like to run. Here we go for ten.

lapprox_M = laplace(bnn, 50_000, 10, 1e-6; diag = true)
θhat = rand(rng, lapprox_M, 1000)
ppred = posterior_predict(bnn, θhat, newx = X_test)
yhat = mean(ppred; dims = 2)
mean((y_test .- yhat).^2)

# Seems like in this case, taking a mixture of Laplace approximations does not 
# actually improve the estimates. This might simply be due to having found some 
# modes that drastically overfit the data and generalise bad. Or in genral having
# formed some very bad approximations. We can try to correct this by using 
# Sampling-Importance-Resampling. We use SIR without replacement, and thus 
# the initial samples (here the first number) should be much larger than the final 
# sample size. The printed info shows us that SIR actually performes really bad 
# (importance weights are very close to zero while we would like them to be close to one)
# We can see that dispite this, it still seems to be able to correct for some of 
# the bad Laplace approximations. 

sir = SIR_laplace(bnn, lapprox_M, 100_000, 1000)
ppred = posterior_predict(bnn, sir, newx = X_test)
yhat = mean(ppred; dims = 2)
mean((y_test .- yhat).^2)

# Also checking the credible intervals again:  

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y_test, upper)]
sum(outside)/length(y_test)

# Can we do better by using better posterior approximations? 
# We can try to investigate this using BFlux's other methods. Currently BFlux
# also implements Variational Inference and MCMC. We will now focus on VI.

# ### Variational Inference

# BFlux currently implementes two very basic Variational Inference methods. 
# (1) it implements ADVI which uses AdvancedVI.jl 
# (2) it implements Bayes By Backprop allowing for stochastic gradients and 
# thus better scaling in large datasets. 

# We can see that standard ADVI seems to perform worse on a MSE basis but 
# seems to perform better with respect to the coverage of credible intervals. 

getq(θ) = MvNormal(θ[1:bnn.totparams], exp.(θ[bnn.totparams+1:end]))
vi = advi(bnn, getq, 10, 30_000)
θhat = rand(rng, vi, 1000)
ppred = posterior_predict(bnn, θhat, newx = X_test)
yhat = mean(ppred; dims = 2)
mean((y_test .- yhat).^2)

# 

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y_test, upper)]
sum(outside)/length(y_test)

# The second implemented methods is BBB which allows for stochastic gradients and
# thus scales better with the data set size. Currently BBB is only implemented for 
# diagonal covariance matrices but future work will extend this. We see that BBB 
# performs better than ADVI and but worse than the SIR Laplace method in both MSE
# and credible interval coverage sense.  

vi = bbb(bnn, 10, 50_000, 50)
θhat = rand(rng, vi[1], 1000)
ppred = posterior_predict(bnn, θhat, newx = X_test)
yhat = mean(ppred; dims = 2)
mean((y_test .- yhat).^2)

# 

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y_test, upper)]
sum(outside)/length(y_test)

# ### MCMC

# BFlux currently also implementes two MCMC samples: 
# (1) Stochastic Gradient Langevin Dynamics introduced by ... but shown by ...
# to have a zero MH acceptance probability, and
# (2) Gradient Guided Monte Carlo as proposed by ... as an alternative to SGLD. 
# Note: Both MCMC methods need a lot of tuning. Current experience tells me that 
# small step sizes and no metric tuning work best. The small step sizes are 
# likely needed due to the complex topology of the posterior distribution. Also note, 
# that the chains generally mix very badly and have a very high autocorrelation in 
# most experiments. Future work will look into how this could be improved. Noteworthy 
# is though, that although mixing is often bad in parameter space, posterior predictive 
# values often mix rather well. 

# We see that SGLD and GGMC perform well, with GGMC being better than SGLD,
# but also that a very low stepsize was needed. 

samples = sgld(bnn, 50, mode[1], 500_000, stepsize_γ = 0.55, stepsize_b = 0, stepsize_a = 1e-10)
ppred = posterior_predict(bnn, samples[:, 500_000:end], newx = X_test)
yhat = mean(ppred; dims = 2)
mean((y_test .- yhat).^2)

# 

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y_test, upper)]
sum(outside)/length(y_test)


# **GGMC**

samples = BFlux.ggmc(bnn, 50, randn(bnn.totparams), 100_000, adaptM = false, adapth = true, l = 1e-10, β = 0.3, goal_accept_rate = 0.2, keep_every = 1)
ppred = posterior_predict(bnn, samples[1][:, 100_000:end], newx = X_test)
yhat = mean(ppred; dims = 2)
mean((y_test .- yhat).^2)

# 

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y_test, upper)]
sum(outside)/length(y_test)

# ## Recurrent Neural Networks
# 
# BFlux also allows for Recurrent Layers. We will demonstrate this by using 
# a simple AR(1) model 

ar1 = BFlux.AR([0.5])
y = ar1(;N=500)
X = [hcat(y[i:i+9]...) for i=1:length(y)-10] # we consider subsequences of length 10
X = BFlux.to_RNN_format(X)
y = y[11:end];

# BFlux supports both LSTM and RNN currently, as well as any composition of those with 
# Dense layers. I will demonstrate the use of LSTM layers here. 

net = Chain(LSTM(1, 1), Dense(1, 1))
loglike = BFlux.SeqToOneNormal(Gamma(2.0, 2.0), Float64)
bnn = BNN(net, loglike, y, X);

# We can again estimate the network using any of the methods demonstrated above. 
# So, for example we could use BBB

vi_bbb = bbb(bnn, 10, 10_000, 98)
θhat = rand(vi_bbb[1], 10_000)
ppred = posterior_predict(bnn, θhat)
yhat = mean(ppred; dims = 2)
mean((y.-yhat).^2)

# 

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y, upper)]
sum(outside)/length(y)

# What about larger AR models? 

ar10 = BFlux.AR(0.2*randn(10))
y = ar10(;N=500)
X = [hcat(y[i:i+9]...) for i=1:length(y)-10] # we consider subsequences of length 10
X = BFlux.to_RNN_format(X)
y = y[11:end]
net = Chain(LSTM(1, 1), Dense(1, 1))
loglike = BFlux.SeqToOneNormal(Gamma(2.0, 2.0), Float64)
bnn = BNN(net, loglike, y, X);

# 

vi_bbb = bbb(bnn, 10, 10_000, 98)
θhat = rand(vi_bbb[1], 10_000)
ppred = posterior_predict(bnn, θhat)
yhat = mean(ppred; dims = 2)
mean((y.-yhat).^2)

# 

upper = [quantile(yh, 0.975) for yh in eachrow(ppred)]
lower = [quantile(yh, 0.025) for yh in eachrow(ppred)]
outside = [yy < l || yy > u for (l, yy, u) in zip(lower, y, upper)]
sum(outside)/length(y)

# We could also estimate the model using GGMC. This will naturally take longer, 
# but will give us an average acceptance rate, which could be taken as a measure
# of how well of an approximation we obtain. Better measures would be to look at 
# whether chains are actually mixing and what the effective sample size is. Both 
# are still open projects. 

# TODO

# ## Implemented Priors

# TODO

# ## Implemented Likelihoods

# TODO

# ## Current Shortcomings
# 
# - Currently only allows for one output variable
# - Problems with mixing chains