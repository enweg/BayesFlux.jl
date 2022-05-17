
using QuickBNN, Flux, Zygote, Distributions, StatsPlots
using BFlux
using LinearAlgebra

ar1 = AR([0.5])
y = ar1(;N=100);
x = [hcat(y[i:i+9]...) for i=1:(length(y) - 10)];
x = to_RNN_format(x);
y = y[11:end];

################################################################################
# Normal Distribution
################################################################################

################################################################################
#  Single layer
net = Chain(RNN(1, 1))
loglike = SeqToOneNormal(Gamma(1.0, 1.0), Float64)
bnn = BFlux.BNN(net, loglike, y, x)

initθ = randn(bnn.totparams)
θsgd = BFlux.find_mode_sgd(bnn, 33, initθ, 10_000, 0.000001)

th = BFlux.get_network_params(bnn, θ)
nethat = bnn.re(th)
yhat = vec([nethat(xx) for xx in x][end])
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# Two layers
net = Chain(RNN(1, 1), Dense(1, 1))
loglike = SeqToOneNormal(Gamma(1.0, 1.0), Float64)
bnn = BFlux.BNN(net, loglike, y, x)

initθ = randn(bnn.totparams)
θsgd = BFlux.find_mode_sgd(bnn, 33, initθ, 10_000, 0.000001)

th = BFlux.get_network_params(bnn, θ)
nethat = bnn.re(th)
yhat = vec([nethat(xx) for xx in x][end])
plot(y, label = "y")
plot!(yhat, lebel = "yhat")

################################################################################
# larger network
net = Chain(RNN(1, 10), Dense(10, 1))
loglike = SeqToOneNormal(Gamma(1.0, 1.0), Float64)
bnn = BFlux.BNN(net, loglike, y, x)

initθ = randn(bnn.totparams)
θsgd = BFlux.find_mode_sgd(bnn, 33, initθ, 10_000, 0.000001)

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

θ = BFlux.find_mode(bnn, 10_000; opt = Flux.ADAGrad())
la = laplace(bnn, 10_000, 20; diag = true, opt = Flux.ADADelta()) # using 20 modes
all(la.c) # did all converge? 
# Sampling Importance Sampling correction 
draws = SIR_laplace(bnn, la, 100_000, 10_000)
draws = hcat(draws...)
histogram(draws[end-1, :])

th = BFlux.get_network_params.([bnn], eachcol(draws))
nethats = bnn.re.(th)
predict(net, x) = vec([net(xx) for xx in x][end])
yhats = hcat([predict(nn, bnn.x) for nn in nethats]...)
p = plot(y, legend = false, color = :black, linewidth = 3.0)
for i=1:1000
    plot!(p, yhats[:,i], color = :red, alpha = 0.1)
end
p

################################################################################
# two layer testcase
net = Chain(RNN(1, 1), Dense(1, 1))
loglike = SeqToOneTDist(Gamma(1.0, 1.0), 5.0)
bnn = BFlux.BNN(net, loglike, y, x)

θ = BFlux.find_mode(bnn, 10_000; opt = Flux.ADAGrad())
la = laplace(bnn, 10_000, 20; diag = true, opt = Flux.ADADelta()) # using 20 modes
all(la.c) # did all converge? 
# Sampling Importance Sampling correction 
draws = SIR_laplace(bnn, la, 100_000, 10_000)
draws = hcat(draws...)
histogram(draws[end-1, :])

th = BFlux.get_network_params.([bnn], eachcol(draws))
nethats = bnn.re.(th)
predict(net, x) = vec([net(xx) for xx in x][end])
yhats = hcat([predict(nn, bnn.x) for nn in nethats]...)
p = plot(y, legend = false, color = :black, linewidth = 3.0)
for i=1:1000
    plot!(p, yhats[:,i], color = :red, alpha = 0.1)
end
p

################################################################################
# larger network 
net = Chain(RNN(1, 10), Dense(10, 1))
loglike = SeqToOneTDist(Gamma(1.0, 1.0), 5.0)
bnn = BFlux.BNN(net, loglike, y, x)

θ = BFlux.find_mode(bnn, 10_000; opt = Flux.ADAGrad())
la = laplace(bnn, 10_000, 20; diag = true, opt = Flux.ADADelta()) # using 20 modes
all(la.c) # did all converge? 
# Sampling Importance Sampling correction 
draws = SIR_laplace(bnn, la, 100_000, 10_000)
draws = hcat(draws...)
histogram(draws[end-1, :])

th = BFlux.get_network_params.([bnn], eachcol(draws))
nethats = bnn.re.(th)
predict(net, x) = vec([net(xx) for xx in x][end])
yhats = hcat([predict(nn, bnn.x) for nn in nethats]...)
p = plot(y, legend = false, color = :black, linewidth = 3.0)
for i=1:1000
    plot!(p, yhats[:,i], color = :red, alpha = 0.1)
end
p
