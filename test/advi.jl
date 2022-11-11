using Flux, BFlux
using Random, Distributions
using ReTest

@testset "ADVI" begin
    @testset "Linear Regression" begin
        k = 5
        n = 100_000
        x = randn(Float32, k, n)
        β = randn(Float32, k)
        y = x' * β + 0.1f0 * randn(Float32, n)

        net = Chain(Dense(5, 1))
        nc = destruct(net)
        sigma_prior = Gamma(2.0f0, 0.5f0)
        like = FeedforwardNormal(nc, sigma_prior)
        prior = GaussianPrior(nc, 10.0f0)
        init = InitialiseAllSame(Normal(0.0f0, 1.0f0), like, prior)
        bnn = BNN(x, y, like, prior, init)

        # Using default variational distribution: MvNormal 
        q = advi(bnn, 10, 10_000)

    end
end