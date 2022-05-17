using QuickBNN, Flux, Zygote, Distributions, StatsPlots
using BFlux
using Random

ar1 = AR([0.5]; rng = Random.MersenneTwister(6150533))
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

θ = BFlux.find_mode(bnn, 10_000; opt = Flux.ADAGrad())
initθ = randn(bnn.totparams)
θsgd = BFlux.find_mode_sgd(bnn, 33, initθ, 10_000, 0.000001)
la = laplace(bnn, 10_000, 20; diag = true) # using 20 modes
all(la.c) # did all converge? 
draws = rand(la, 10_000)
histogram(draws[end-1, :])


th = BFlux.get_network_params.([bnn], eachcol(draws))
nethats = bnn.re.(th)
predict(net, x) = vec(net(x))
yhats = hcat([predict(nn, bnn.x) for nn in nethats]...)
p = plot(y, legend = false, color = :black, linewidth = 3.0)
for i=1:1000
    plot!(p, yhats[:,i], color = :red, alpha = 0.01)
end
p

################################################################################
# two layer testcase
net = Chain(Dense(1, 1, sigmoid), Dense(1, 1))
loglike = FeedforwardNormal(Gamma(1.0, 1.0), Float64)
bnn = BFlux.BNN(net, loglike, y, x)

θ = BFlux.find_mode(bnn, 10_000; opt = Flux.ADAGrad())
la = laplace(bnn, 10_000, 20; diag = true) # using 20 modes
all(la.c) # did all converge? 
draws = rand(la, 10_000)
histogram(draws[end-1, :])

th = BFlux.get_network_params.([bnn], eachcol(draws))
nethats = bnn.re.(th)
predict(net, x) = vec(net(x))
yhats = hcat([predict(nn, bnn.x) for nn in nethats]...)
p = plot(y, legend = false, color = :black, linewidth = 3.0)
for i=1:1000
    plot!(p, yhats[:,i], color = :red, alpha = 0.01)
end
p

################################################################################
# larger network 
net = Chain(Dense(1, 10, sigmoid), Dense(10, 1))
loglike = FeedforwardNormal(Gamma(1.0, 1.0), Float64)
bnn = BFlux.BNN(net, loglike, y, x)

θ = BFlux.find_mode(bnn, 10_000; opt = Flux.ADAGrad())
la = laplace(bnn, 10_000, 20; diag = true, opt = Flux.ADADelta()) # using 20 modes
all(la.c) # did all converge? 
draws = rand(la, 10_000)
histogram(draws[end-1, :])

th = BFlux.get_network_params.([bnn], eachcol(draws))
nethats = bnn.re.(th)
predict(net, x) = vec(net(x))
yhats = hcat([predict(nn, bnn.x) for nn in nethats]...)
p = plot(y, legend = false, color = :black, linewidth = 3.0)
for i=1:1000
    plot!(p, yhats[:,i], color = :red, alpha = 0.01)
end
p

################################################################################
# TDIST 
################################################################################

################################################################################
# Small network testcase
net = Chain(Dense(1, 1))
loglike = FeedforwardTDist(Gamma(1.0, 1.0), 5.0)
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
predict(net, x) = vec(net(x))
yhats = hcat([predict(nn, bnn.x) for nn in nethats]...)
p = plot(y, legend = false, color = :black, linewidth = 3.0)
for i=1:1000
    plot!(p, yhats[:,i], color = :red, alpha = 0.1)
end
p

################################################################################
# two layer testcase
net = Chain(Dense(1, 1, sigmoid), Dense(1, 1))
loglike = FeedforwardTDist(Gamma(1.0, 1.0), 5.0)
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
predict(net, x) = vec(net(x))
yhats = hcat([predict(nn, bnn.x) for nn in nethats]...)
p = plot(y, legend = false, color = :black, linewidth = 3.0)
for i=1:1000
    plot!(p, yhats[:,i], color = :red, alpha = 0.01)
end
p

################################################################################
# larger network 
net = Chain(Dense(1, 10, sigmoid), Dense(10, 1))
loglike = FeedforwardTDist(Gamma(1.0, 1.0), 5.0)
bnn = BFlux.BNN(net, loglike, y, x)

θ = BFlux.find_mode(bnn, 10_000; opt = Flux.ADAGrad())
la = laplace(bnn, 10_000, 20; diag = true, opt = Flux.ADADelta()) # using 20 modes
all(la.c) # did all converge? 
# Sampling Importance Sampling correction 
draws = SIR_laplace(bnn, la, 1_000_000, 1_000)
draws = hcat(draws...)
histogram(draws[end-1, :])

th = BFlux.get_network_params.([bnn], eachcol(draws))
nethats = bnn.re.(th)
predict(net, x) = vec(net(x))
yhats = hcat([predict(nn, bnn.x) for nn in nethats]...)
p = plot(y, legend = false, color = :black, linewidth = 3.0)
for i=1:1000
    plot!(p, yhats[:,i], color = :red, alpha = 0.01)
end
p
