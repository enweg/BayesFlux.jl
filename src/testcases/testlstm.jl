using QuickBNN, Flux, Zygote, Distributions, StatsPlots
using BFlux
using LinearAlgebra

ar1 = AR([0.5])
y = ar1(;N=1000);
x = [hcat(y[i:i+9]...) for i=1:(length(y) - 10)];
x = to_RNN_format(x);
y = y[11:end];

################################################################################
# Normal Distribution
################################################################################

################################################################################
#  Single layer
net = Chain(LSTM(1, 1))
loglike = SeqToOneNormal(Gamma(1.0, 1.0), Float64)
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
yhat = vec([nethat(xx) for xx in x][end])
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# Two layers
net = Chain(LSTM(1, 1), Dense(1, 1))
loglike = SeqToOneNormal(Gamma(1.0, 1.0), Float64)
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
yhat = vec([nethat(xx) for xx in x][end])
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# larger network
net = Chain(LSTM(1, 10), Dense(10, 1))
loglike = SeqToOneNormal(Gamma(1.0, 1.0), Float64)
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
yhat = vec([nethat(xx) for xx in x][end])
plot(y, label = "y")
plot!(yhat, lebel = "yhat")


################################################################################
# TDIST 
################################################################################

################################################################################
# Small network testcase
net = Chain(RNN(1, 1))
loglike = SeqToOneTDist(Gamma(1.0, 1.0), 5.0)
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
yhat = vec([nethat(xx) for xx in x][end])
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# two layer testcase
net = Chain(RNN(1, 1), Dense(1, 1))
loglike = SeqToOneTDist(Gamma(1.0, 1.0), 5.0)
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
yhat = vec([nethat(xx) for xx in x][end])
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# larger network 
net = Chain(RNN(1, 10), Dense(10, 1))
loglike = SeqToOneTDist(Gamma(1.0, 1.0), 5.0)
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
yhat = vec([nethat(xx) for xx in x][end])
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

