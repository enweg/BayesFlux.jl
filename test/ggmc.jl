# TODO: implement test for ggmc
using LinearAlgebra
using Bijectors

@testset "GGMC" begin
    @testset "Linear Regression" for _ in 1:10 
        n = 1000
        k = 5
        β = randn(k)
        X = randn(k, n)
        y = X'*β + randn(n)

        β_priordist = MvNormal(zeros(k), I)
        σ_priordist = Gamma(2.0, 2.0)
        tσ_priordist = transformed(σ_priordist)

        llike(θ, y, X) = logpdf(MvNormal(X'*θ[2:end], exp(θ[1])*I), y)
        lprior(θ) = logpdf(β_priordist, θ[2:end]) + logpdf(tσ_priordist, θ[1])
        samples = ggmc(llike, lprior, 1000, y, X, randn(k+1), 30000; 
                    adapruns = 1000, keep_every = 1, 
                    adapth = true, 
                    h_adapter = BFlux.hStochasticAdapter(0.1; goal_accept_rate = 0.65), 
                    adaptM = true)

        θ = mean(samples[1][:, 1001:end]; dims = 2)
        σhat = invlink(σ_priordist, θ[1])
        βhat = θ[2:end]

        βdiff = maximum(abs, β - βhat)
        σdiff = maximum(abs, 1.0 - σhat)

        @test βdiff < 0.1
        @test σdiff < 0.1
        
    end
end
