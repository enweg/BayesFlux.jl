using Flux
using Distributions, Random, LinearAlgebra
using Bijectors


@testset "Mode Finding" begin
    @testset "Mode Linear Regression" begin
        k = 5
        n = 100_000
        x = randn(Float32, k, n)
        β = randn(Float32, k)
        y = x'*β + 0.1f0*randn(Float32, n)

        net = Chain(Dense(k, 1))
        nc = destruct(net)
        like = FeedforwardNormal(nc, Gamma(2.0, 2.0))
        prior = GaussianPrior(nc, 10.0f0)
        init = InitialiseAllSame(Normal(0.0f0, 1.0f0), like, prior)
        bnn = BNN(x, y, like, prior, init)

        opt = FluxModeFinder(bnn, Flux.ADAM(); windowlength = 50)
        θmode = find_mode(bnn, 10000, 1000, opt; showprogress = false)

        # We do not have a constant in the original model so discard bias
        βhat = θmode[1:bnn.like.nc.num_params_network-1]

        @test maximum(abs, β .- βhat) < 0.01

    end
end


# @testset "Mode Finding" begin
#     @testset "Full Gradient $i" for i=2:100
#         μ = randn(i)
#         lpdf(θ) = logpdf(MvNormal(μ, I), θ)
#         mode = find_mode(lpdf, randn(i), 10_000, 1e-10; 
#                          showprogress = false, verbose = false)
#         @test isapprox(mode[1], μ, atol = 0.01)
#     end

#     # TODO: implement SGD test; How to design a good test? 
# end
