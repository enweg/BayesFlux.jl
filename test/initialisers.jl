using Flux
using Distributions, Random
using BFlux

@testset "Initialise" begin

    @testset "Basic Initialiser" for dist in [Normal(), Normal(0.0f0, 10.0f0), Uniform(-0.5f0, 0.5f0)]
        net = Chain(LSTM(1, 10), Dense(10, 1))
        nc = destruct(net)
        like = FeedforwardNormal(nc, Gamma(2.0, 2.0))
        prior = GaussianPrior(nc, 10.0f0)
        init = InitialiseAllSame(dist, like, prior)

        θnet, θhyper, θlike = init()
        @test length(θnet) == nc.num_params_network
        @test length(θhyper) == prior.num_params_hyper
        @test length(θlike) == like.num_params_like

        draws = [vcat(init()...) for _ in 1:1_000_000]
        draws = reduce(hcat, draws)
        mindraw = minimum(draws; dims=2)
        maxdraw = maximum(draws; dims=2)
        meandraw = mean(draws; dims=2)
        vardraw = var(draws; dims=2)

        supdist = support(dist)
        mindist = supdist.lb
        maxdist = supdist.ub
        meandist = mean(dist)
        vardist = var(dist)

        @test all(mindist .<= mindraw .<= maxdist)
        @test all(mindist .<= maxdraw .<= maxdist)
        @test maximum(abs, meandraw .- meandist) < 0.1
        @test maximum(abs, vardraw ./ vardist) < 1.1

    end
end