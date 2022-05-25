using BFlux
using Flux, Distributions, Random
using Test

function ggmc_regression_test(steps, k = 5, n = 100_000)
        x = randn(Float32, k, n)
        β = randn(Float32, k)
        y = x'*β + 0.1f0*randn(Float32, n)

        net = Chain(Dense(k, 1))
        nc = destruct(net)
        sigma_prior = Gamma(2.0f0, 0.5f0)
        like = FeedforwardNormal(nc, sigma_prior)
        prior = GaussianPrior(nc, 10.0f0)
        init = InitialiseAllSame(Normal(0.0f0, 1.0f0), like, prior)
        bnn = BNN(x, y, like, prior, init)

        # stepsize_adapter = StepsizeConstantAdapter(1000)
        stepsize_adapter = StepsizeStochasticOptAdapter(0.55f0, 0.75f0, 500)
        # madapter = MassIdentityAdapter(1000)
        madapter = MassRMSPropAdapter(bnn.num_total_params, 250)
        sampler = GGMC(;l = 0.001f0, β = 0.005f0 ,steps = steps, 
            stepsize_adapter = stepsize_adapter, M_adapter = madapter)

        ch = mcmc(bnn, 1_000, 15_000, sampler; showprogress = false)

        θmean = mean(ch[:, end-10000:end]; dims=2)
        βhat = θmean[1:length(β)]

        ch_longer = mcmc(bnn, 1000, 20_000, sampler; continue_sampling = true, showprogress = false)
       
        return maximum(abs, β - βhat),  all(ch_longer[:, 1:15_000] .== ch)
end



@testset "GGMC" begin
    @testset "Linear Regression" for steps in [1, 2, 5]
        @testset "Steps = $steps" begin
            diffs , conts  = Array{Float64}(undef, 5), Array{Bool}(undef, 5)
            for retry=1:5
                diff, cont = ggmc_regression_test(steps) 
                diffs[retry] = diff 
                conts[retry] = cont
            end
            @test sum(diffs .<= 0.5) >= 4
            @test sum(conts) == 5
        end
    end
end



# @testset "GGMC" begin
#     @testset "Linear Regression" for _ in 1:10 
#         n = 1000
#         k = 5
#         β = randn(k)
#         X = randn(k, n)
#         y = X'*β + randn(n)

#         β_priordist = MvNormal(zeros(k), I)
#         σ_priordist = Gamma(2.0, 2.0)
#         tσ_priordist = transformed(σ_priordist)

#         llike(θ, y, X) = logpdf(MvNormal(X'*θ[2:end], exp(θ[1])*I), y)
#         lprior(θ) = logpdf(β_priordist, θ[2:end]) + logpdf(tσ_priordist, θ[1])
#         samples = ggmc(llike, lprior, 1000, y, X, randn(k+1), 30000; 
#                     adapruns = 1000, keep_every = 1, 
#                     adapth = true, 
#                     h_adapter = BFlux.hStochasticAdapter(0.1; goal_accept_rate = 0.65), 
#                     adaptM = true)

#         θ = mean(samples[1][:, 1001:end]; dims = 2)
#         σhat = invlink(σ_priordist, θ[1])
#         βhat = θ[2:end]

#         βdiff = maximum(abs, β - βhat)
#         σdiff = maximum(abs, 1.0 - σhat)

#         @test βdiff < 0.1
#         @test σdiff < 0.1
        
#     end

#     @testset "NN Linear Regresssion" for _ in 1:10 
#         n = 1000
#         k = 5
#         β = randn(k)
#         X = randn(k, n)
#         y = X'*β + randn(n)

#         net = Chain(Dense(k, 1))
#         loglike = BFlux.FeedforwardNormal(Gamma(2.0, 2.0), Float64)
#         bnn = BNN(net, loglike, y, X)

#         samples = ggmc(bnn, 100, rand(bnn.totparams).-0.5, 4_000; 
#                        keep_every = 1, adapruns = 1000,
#                        h_adapter = BFlux.hStochasticAdapter(0.1; goal_accept_rate = 0.65))
#         s = samples[1][:, 1001:end]
#         netparams = reduce(hcat, [BFlux.get_network_params(bnn, θ) for θ in eachcol(s)])
#         βhat = mean(netparams; dims = 2)
#         # βhat constains constant as last value 
#         βhat = βhat[1:end-1]

#         @test maximum(abs, β - βhat) < 0.1
#     end
# end
