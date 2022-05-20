# Each likelihood must be a subtype of BNNLikelihood and must have implemented 
# at least the fields `num_params_likelihood::Int` and `nc<:NetConstructor`.
using Distributions
using Flux
using BFlux
using Bijectors

@testset "Likelihood Feedforward" begin

    @testset "Gaussian" begin
        net = Chain(Dense(10, 10, sigmoid), Dense(10, 1))
        nc = destruct(net)
        gl = FeedforwardNormal(nc, Gamma(2.0, 2.0))

        # A Gaussian likelihood has one additional parameter: σ
        @test gl.num_params_like == 1

        T = eltype(nc.θ)

        # If we set all network parameters to zero then the prediction will
        # always be zero. If additionally we set σ = 1, then we can compare the
        # network likelihood to a standard normal 
        # We only need to subtract the contribution of the prior for σ
        y = T.(quantile.(Normal(), 0.1:0.1:0.9))
        x = randn(T, 10, length(y))
        θ = zeros(eltype(nc.θ), nc.num_params_network)

        tdist = transformed(gl.prior_σ)
        tσ = link(gl.prior_σ, 1.0f0)

        @test gl(x, y, θ, [tσ]) ≈ sum(logpdf.(Normal(), y)) + logpdf(tdist, tσ)

        # Similarly, using any x should result in predictions that are 
        # distributed according to a standard normal
        x = randn(T, 10, 10_000)
        ypp = predict(gl, x, θ, [tσ])
        q = T.(quantile.([ypp], 0.1:0.1:0.9))
        @test maximum(abs, q - y) < 0.05
    end

    @testset "TDist" begin
        net = Chain(Dense(10, 10, sigmoid), Dense(10, 1))
        nc = destruct(net)
        tl = FeedforwardTDist(nc, Gamma(2.0, 2.0), 10.0f0)

        @test tl.num_params_like == 1

        T = eltype(nc.θ)

        # We can do the same as for the Gaussian case. 
        y = T.(quantile.(TDist(tl.ν), 0.1:0.1:0.9))
        x = randn(T, 10, length(y))
        θ = zeros(T, nc.num_params_network)

        tdist = transformed(tl.prior_σ)
        tσ = link(tl.prior_σ, 1.0f0)

        @test tl(x, y, θ, [tσ]) ≈ sum(logpdf.(TDist(tl.ν), y)) + logpdf(tdist, tσ)

        # And doing the same for prediction
        x = randn(T, 10, 10_000)
        ypp = predict(tl, x, θ, [tσ])
        q = T.(quantile.([ypp], 0.1:0.1:0.9))
        @test maximum(abs, q - y) < 0.05
    end
end