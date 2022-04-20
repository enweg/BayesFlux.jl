using Distributions, Random
using LinearAlgebra

@testset "Bayes By Backprop" begin
    @testset "Mean Only" begin
        n = 1000
        X = randn(3, n)
        β = randn(3)
        y = X'*β + randn(n) 

        llike(β, y, X) = logpdf(MvNormal(X'*β, I), y)
        lpriorβ(β) = logpdf(MvNormal(zeros(3), ones(3)), β)
        function get_mu_sig(ψ)
            return ψ, diagm(ones(3)) 
        end

        β̂ = inv(X*X')*X*y
        post_mean = inv(X*X' + I)*(X*X'*β̂)

        bbb_post = bbb(llike, lpriorβ, randn(3), get_mu_sig, 10, 5_000, 500, y, X)
        @test isapprox(post_mean, bbb_post[2], atol = 0.05)
    end
    
    # TODO: also test with variance
end