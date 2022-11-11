using BFlux
using Flux, Distributions, Random
using Bijectors
using Zygote

@testset "BNN" begin

    @testset "Split Parameter Vector" begin

        n = 1000
        k = 5
        x = randn(Float32, k, n)
        β = randn(Float32, k)
        y = x' * β + randn(Float32, n)

        net = Chain(Dense(5, 10, relu), Dense(10, 1))
        nc = destruct(net)
        like = FeedforwardNormal(nc, Gamma(2.0, 2.0))
        prior = GaussianPrior(nc, 10.0f0)
        init = InitialiseAllSame(Uniform(-0.5f0, 0.5f0), like, prior)

        bnn = BNN(x, y, like, prior, init)
        θ = vcat(init()...)
        θnet, θhyper, θlike = split_params(bnn, θ)
        @test length(θnet) == nc.num_params_network
        @test all(θnet .== θ[1:nc.num_params_network])
        @test θnet == @view θ[1:nc.num_params_network]
        @test length(θhyper) == 0
        @test length(θlike) == like.num_params_like
        @test θlike == @view θ[end:end]
    end

    @testset "Gradient" begin
        # This test is done using Float64 to avoid numerical differences 
        # But we usually work with Float32

        x = randn(2, 1)
        y = randn()
        net = Chain(Dense(randn(2, 2), randn(2), sigmoid), Dense(randn(1, 2), randn(1)))
        nc = destruct(net)
        like = FeedforwardNormal(nc, Gamma(2.0, 2.0))
        prior = GaussianPrior(nc, 1.0)
        init = InitialiseAllSame(Normal(), like, prior)
        bnn = BNN(x, y, like, prior, init)

        θnet, θhyper, θlike = init()
        net = nc(θnet)

        # ∂yhat/∂θ
        yhat = net(x)[1]
        z = vec(net[1].weight * x + net[1].bias)
        a = vec(sigmoid.(z))

        ∇W2 = reshape(vec(a), 1, 2)
        ∇b2 = 1.0
        ∇a = vec(net[2].weight)
        ∇z = ∇a .* sigmoid.(z) .* (1.0 .- sigmoid.(z))
        ∇W1r1 = reshape(∇z[1] .* vec(x), 1, 2)
        ∇W1r2 = reshape(∇z[2] .* vec(x), 1, 2)
        ∇W1 = vcat(∇W1r1, ∇W1r2)
        ∇b1 = ∇z

        # Comparing the above derivatives with those obtained from Zygote

        g = Zygote.gradient(() -> net(x)[1], Flux.params(net))

        @test all(∇b1 .≈ g[net[1].bias])
        @test all(∇W1 .≈ g[net[1].weight])
        @test all(∇b2 .≈ g[net[2].bias])
        @test all(∇W2 .≈ g[net[2].weight])

        ∇yhat_θnet = vcat(vec(∇W1), vec(∇b1), vec(∇W2), ∇b2)

        # Guassian Likelihood contribution
        sigma = exp(θlike[1])
        ∇like_sigma = -sigma^(-1) + sigma^(-3) * (y - yhat)^2
        ∇like_θlike = ∇like_sigma * exp(θlike[1])
        ∇like_yhat = sigma^(-2) * (y - yhat)
        ∇like_θnet = ∇like_yhat .* ∇yhat_θnet

        g = Zygote.gradient((yhat, sigma) -> logpdf(Normal(yhat, sigma), y), yhat, sigma)
        @test ∇like_yhat ≈ g[1]
        @test ∇like_sigma ≈ g[2]

        # σ prior contribution
        # We will trust Zygote here
        # TODO: change this
        tdist = transformed(Gamma(2.0, 2.0))
        ∇prior_θlike = Zygote.gradient(θlike -> logpdf(tdist, θlike[1]), θlike)[1][1]

        # Priors for network parameters are standard normal 
        ∇prior_θnet = -θnet


        # Final 
        ∇θnet = ∇like_θnet .+ ∇prior_θnet
        ∇θlike = ∇like_θlike .+ ∇prior_θlike

        ∇θ = vcat(∇θnet, ∇θlike)

        # Comparing 
        θ = vcat(θnet, θhyper, θlike)
        v, g = ∇loglikeprior(bnn, θ, x, [y])

        @test all(∇θ .≈ g)


    end
end