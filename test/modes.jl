using Distributions, Random, LinearAlgebra
using Bijectors

@testset "Mode Finding" begin
    @testset "Full Gradient $i" for i=2:100
        μ = randn(i)
        lpdf(θ) = logpdf(MvNormal(μ, I), θ)
        mode = find_mode(lpdf, randn(i), 10_000, 1e-10; 
                         showprogress = false, verbose = false)
        @test isapprox(mode[1], μ, atol = 0.01)
    end

    # TODO: implement SGD test; How to design a good test? 
end
