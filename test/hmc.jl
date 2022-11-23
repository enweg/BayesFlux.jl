# Testing HMC
using BFlux
using Flux, Distributions, Random
using LinearAlgebra

function test_HMC_regression(madapter; k=5, n=10_000)
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

    l = 1.0f-4
    sadapter = DualAveragingStepSize(l; adapt_steps=1000, target_accept=0.5f0)
    # Below was tested together with DiagCovMassAdapter and worked but results
    # are very sensitive to the stepsize set and thus proper testing must be 
    # deferred for now. 
    # TODO: properly test ConstantStepsize
    # l = 1f-3
    # sadapter = ConstantStepsize(l)

    leapfrog_steps = 5
    sampler = HMC(l, leapfrog_steps; sadapter=sadapter, madapter=madapter)

    opt = FluxModeFinder(bnn, Flux.ADAM())
    θmap = find_mode(bnn, 1000, 100, opt; showprogress=false)

    # Commented below was also tested and always worked. But good practice 
    # for BNNs is to have a warm start.
    # ch = mcmc(bnn, 1000, 20_000, sampler; showprogress = true)
    ch = mcmc(bnn, 1000, 20_000, sampler; showprogress=true, θstart=θmap)
    ch_short = ch[:, end-9999:end]

    θmean = mean(ch_short; dims=2)
    βhat = θmean[1:length(β)]
    # coefficient estimate
    test1 = maximum(abs, β - βhat) < 0.05
    # Intercept
    test2 = abs(θmean[end-1]) < 0.05
    # Variance
    test3 = 0.9f0 <= mean(exp.(ch_short[end, :])) <= 1.1f0

    # Continue sampling
    test4 = BFlux.calculate_epochs(sampler, 100, 25_000; continue_sampling=true) == 50 * leapfrog_steps
    ch_longer = mcmc(bnn, 1000, 25_000, sampler; continue_sampling=true)
    test5 = all(ch_longer[:, 1:20_000] .== ch)

    return [test1, test2, test3, test4, test5]
end


Random.seed!(6150533)
@testset "HMC" begin
    madapter1 = DiagCovMassAdapter(1000, 100; kappa=0.1f0)
    madapter2 = FullCovMassAdapter(1000, 100; kappa=0.1f0)
    madapter3 = RMSPropMassAdapter(1000)
    mas = [madapter1, madapter2, madapter3]
    @testset "Linear Regression" for madapter in mas
        @testset "Linear Regression madapter = $madapter" begin
            # Because GitHub Actions seem very slow and occasionally run out of 
            # memory, we will decrease the number of tests if the tests are run on 
            # GitHub actions. Hostnames on GH actions seem to always start with fv
            ntests = gethostname()[1:2] == "fv" ? 3 : 10
            results = fill(false, ntests, 5)
            for i = 1:ntests
                results[i, :] = test_HMC_regression(madapter)
            end
            pct_pass = mean(results; dims=2)

            @test pct_pass[1] > 0.9
            @test pct_pass[2] > 0.9
            @test pct_pass[3] > 0.8  # variances are difficult to estimate
            @test pct_pass[4] == 1
            @test pct_pass[5] == 1
        end
    end
end