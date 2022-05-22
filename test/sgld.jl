# Testing SGLD
using BFlux
using Flux, Distributions, Random

@testset "SGLD" begin
    @testset "Linear Regression" begin
        k = 5
        n = 100_000
        x = randn(Float32, k, n)
        β = randn(Float32, k)
        y = x'*β + 0.1f0*randn(Float32, n)

        net = Chain(Dense(5, 1))
        nc = destruct(net)
        sigma_prior = Gamma(2.0f0, 0.5f0)
        like = FeedforwardNormal(nc, sigma_prior)
        prior = GaussianPrior(nc, 10.0f0)
        init = InitialiseAllSame(Normal(0.0f0, 1.0f0), like, prior)
        bnn = BNN(x, y, like, prior, init)

        sampler = SGLD(; stepsize_a = 1.0f0)
        ch = mcmc(bnn, 1000, 15_000, sampler)

        θmean = mean(ch[:, end-10000:end]; dims=2)
        βhat = θmean[1:length(β)]
        # coefficient estimate
        @test maximum(abs, β - βhat) < 0.05

        # Continue sampling
        @test BFlux.calculate_epochs(sampler, 100, 20_000; continue_sampling = true) == 50
        ch_longer = mcmc(bnn, 1000, 20_000, sampler; continue_sampling = true)
        @test all(ch_longer[:, 1:15_000] .== ch)
    end
end