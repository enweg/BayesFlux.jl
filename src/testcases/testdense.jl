using QuickBNN, Flux, Zygote, Distributions, StatsPlots
using BFlux

ar1 = AR([0.5])
y = ar1(;N=100);
x = hcat(y[1:end-1]...);
y = y[2:end];

################################################################################
# NORMAL 
################################################################################

################################################################################
# Small network testcase
net = Chain(Dense(1, 1))
loglike = FeedforwardNormal(Gamma(1.0, 1.0), Float64)
bnn = BFlux.BNN(net, loglike, y, x)

using ProgressBars
opt = Flux.ADAGrad()
θ = randn(bnn.totparams)
lp(bnn, θ)
for i in ProgressBar(1:10_000)
    g = Zygote.gradient(θ -> lp(bnn, θ), θ)
    Flux.update!(opt, θ, -g[1])
end
lp(bnn, θ)

th = BFlux.get_network_params(bnn, θ)
nethat = bnn.re(th)
yhat = vec(nethat(x))
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# two layer testcase
net = Chain(Dense(1, 1, sigmoid), Dense(1, 1))
loglike = FeedforwardNormal(Gamma(1.0, 1.0), Float64)
bnn = BFlux.BNN(net, loglike, y, x)

opt = Flux.ADAGrad()
θ = randn(bnn.totparams)
lp(bnn, θ)
for i in ProgressBar(1:10_000)
    g = Zygote.gradient(θ -> lp(bnn, θ), θ)
    Flux.update!(opt, θ, -g[1])
end
lp(bnn, θ)

th = BFlux.get_network_params(bnn, θ)
nethat = bnn.re(th)
yhat = vec(nethat(x))
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# larger network 
net = Chain(Dense(1, 10, sigmoid), Dense(10, 1))
loglike = FeedforwardNormal(Gamma(1.0, 1.0), Float64)
bnn = BFlux.BNN(net, loglike, y, x)

opt = Flux.ADAGrad()
θ = randn(bnn.totparams)
lp(bnn, θ)
for i in ProgressBar(1:10_000)
    g = Zygote.gradient(θ -> lp(bnn, θ), θ)
    Flux.update!(opt, θ, -g[1])
end
lp(bnn, θ)

th = BFlux.get_network_params(bnn, θ)
nethat = bnn.re(th)
yhat = vec(nethat(x))
plot(y, label = "y")
plot!(yhat, lebel = "yhat")


################################################################################
# TDIST 
################################################################################

################################################################################
# Small network testcase
net = Chain(Dense(1, 1))
loglike = FeedforwardTDist(Gamma(1.0, 1.0), 5.0)
bnn = BFlux.BNN(net, loglike, y, x)

using ProgressBars
opt = Flux.ADAGrad()
θ = randn(bnn.totparams)
lp(bnn, θ)
for i in ProgressBar(1:10_000)
    g = Zygote.gradient(θ -> lp(bnn, θ), θ)
    Flux.update!(opt, θ, -g[1])
end
lp(bnn, θ)

th = BFlux.get_network_params(bnn, θ)
nethat = bnn.re(th)
yhat = vec(nethat(x))
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# two layer testcase
net = Chain(Dense(1, 1, sigmoid), Dense(1, 1))
loglike = FeedforwardTDist(Gamma(1.0, 1.0), 5.0)
bnn = BFlux.BNN(net, loglike, y, x)

opt = Flux.ADAGrad()
θ = randn(bnn.totparams)
lp(bnn, θ)
for i in ProgressBar(1:10_000)
    g = Zygote.gradient(θ -> lp(bnn, θ), θ)
    Flux.update!(opt, θ, -g[1])
end
lp(bnn, θ)

th = BFlux.get_network_params(bnn, θ)
nethat = bnn.re(th)
yhat = vec(nethat(x))
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# larger network 
net = Chain(Dense(1, 10, sigmoid), Dense(10, 1))
loglike = FeedforwardTDist(Gamma(1.0, 1.0), 5.0)
bnn = BFlux.BNN(net, loglike, y, x)

opt = Flux.ADAGrad()
θ = randn(bnn.totparams)
lp(bnn, θ)
for i in ProgressBar(1:10_000)
    g = Zygote.gradient(θ -> lp(bnn, θ), θ)
    Flux.update!(opt, θ, -g[1])
end
lp(bnn, θ)

th = BFlux.get_network_params(bnn, θ)
nethat = bnn.re(th)
yhat = vec(nethat(x))
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

