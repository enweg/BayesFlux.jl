using BFlux
using Flux, Distributions, Random

function test_BBB_regression(; k=5, n=10_000)
    x = randn(Float32, k, n)
    β = randn(Float32, k)
    y = x' * β + 1.0f0 * randn(Float32, n)

    net = Chain(Dense(5, 1))
    nc = destruct(net)
    sigma_prior = InverseGamma(2.0f0, 0.5f0)
    like = FeedforwardNormal(nc, sigma_prior)
    prior = GaussianPrior(nc, 10.0f0)
    init = InitialiseAllSame(Normal(0.0f0, 1.0f0), like, prior)
    bnn = BNN(x, y, like, prior, init)

    q, params, losses = bbb(bnn, 1000, 1000; mc_samples=1, opt=Flux.RMSProp(), n_samples_convergence = 1)

    μ = mean(q)
    test1 = maximum(abs, β - μ[1:length(β)]) < 0.05
    test2 = abs(μ[end-1]) < 0.05
    test3 = 0.9f0 <= exp(μ[end]) <= 1.1f0

    return [test1, test2, test3]
end


@testset "BBB" begin
    @testset "Linear Regression" begin
        # Because GitHub Actions seem very slow and occasionally run out of 
        # memory, we will decrease the number of tests if the tests are run on 
        # GitHub actions. Hostnames on GH actions seem to always start with fv
        ntests = gethostname()[1:2] == "fv" ? 3 : 10
        results = fill(false, ntests, 3)
        for i = 1:ntests
            results[i, :] = test_BBB_regression()
        end
        pct_pass = mean(results; dims=1)

        @test pct_pass[1] > 0.9
        @test pct_pass[2] > 0.9
        @test pct_pass[3] > 0.8  # variances are difficult to estimate
    end
end