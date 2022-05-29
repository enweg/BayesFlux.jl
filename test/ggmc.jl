using BFlux
using Flux, Distributions, Random
using Test

function test_GGMC_regression(steps, k = 5, n = 100_000)
        x = randn(Float32, k, n)
        β = randn(Float32, k)
        y = x'*β + 1f0*randn(Float32, n)

        net = Chain(Dense(k, 1))
        nc = destruct(net)
        sigma_prior = Gamma(2.0f0, 0.5f0)
        like = FeedforwardNormal(nc, sigma_prior)
        prior = GaussianPrior(nc, 10.0f0)
        init = InitialiseAllSame(Normal(0.0f0, 1.0f0), like, prior)
        bnn = BNN(x, y, like, prior, init)

        # firts run optimisation 
        opt = FluxModeFinder(bnn, Flux.RMSProp(); windowlength = 50)
        θmode = find_mode(bnn, 1000, 100, opt; showprogress = false)

        # stepsize_adapter = StepsizeConstantAdapter(1000)
        stepsize_adapter = StepsizeStochasticOptAdapter(0.55f0, 0.35f0, 2000)
        # madapter = MassIdentityAdapter(1000)
        # madapter = MassRMSPropAdapter(bnn.num_total_params, 250)
        madapter = MassVarianceAdapter(1000, 1)
        sampler = GGMC(;l = 1f-15, β = 0.1f0,steps = steps, 
            stepsize_adapter = stepsize_adapter, M_adapter = madapter)

        ch = mcmc(bnn, 1_000, 20_000, sampler; showprogress = true, θstart = θmode)
        ch_short = ch[:, end-9999:end]

        θmean = mean(ch_short; dims=2)
        βhat = θmean[1:length(β)]
        # coefficients
        test1 = maximum(abs, β - βhat) < 0.05
        # intercept
        test2 = abs(θmean[end-1]) < 0.05
        # variance
        test3 = 0.9f0 <= mean(exp.(ch_short[end, :])) <= 1.1f0

        ch_longer = mcmc(bnn, 1000, 25_000, sampler; continue_sampling = true, showprogress = false)
        test4 = all(ch_longer[:, 1:20_000] .== ch)

       
        return [test1, test2, test3, test4]
end



@testset "GGMC" begin
    @testset "Linear Regression" for steps in [1, 2, 5]
        @testset "Steps = $steps" begin
            ntests = 10
            results = fill(false, ntests, 4)
            for i=1:ntests
                results[i, :] = test_GGMC_regression(steps)
            end
            pct_pass = mean(results; dims = 2)
            @test pct_pass[1] > 0.9
            @test pct_pass[2] > 0.9
            @test pct_pass[3] > 0.8  # variances are difficult to estimate
            @test pct_pass[4] == 1
        end
    end
end


