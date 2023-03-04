using BayesFlux
using Flux, Distributions, Random
using Test

@testset "Posterior Predictive Draws" begin
    n = 1_000
    k = 5
    x = randn(Float32, k, n)
    β = randn(Float32, k)
    y = x' * β + 1.0f0 * randn(Float32, n)

    net = Chain(Dense(k, 1))
    nc = destruct(net)
    sigma_prior = Gamma(2.0f0, 0.5f0)
    like = FeedforwardNormal(nc, sigma_prior)
    prior = GaussianPrior(nc, 10.0f0)
    init = InitialiseAllSame(Normal(0.0f0, 1.0f0), like, prior)
    bnn = BNN(x, y, like, prior, init)

    sampler = SGLD(; stepsize_a=1.0f0)
    ch = mcmc(bnn, 1000, 20_000, sampler; showprogress=true)
    ch_short = ch[:, end-9999:end]

    pp = sample_posterior_predict(bnn, ch_short)
    @test(size(pp, 1) == n)
    @test(size(pp, 2) == size(ch_short, 2))
end