using BayesFlux
import BayesFlux
using Flux, Zygote


@testset "Feedforward derivatives" begin
    net = Chain(Dense(1, 1))
    nc = destruct(net)
    x = randn(Float32, 1, 100)
    y = randn(Float32, 100)
    like = FeedforwardNormal(nc, Gamma(2.0, 0.5))
    prior = GaussianPrior(nc, 0.5f0)
    init = InitialiseAllSame(Normal(0f0, 0.5f0), like, prior)
    bnn = BNN(x, y, like, prior, init)

    θ = randn(Float32, bnn.num_total_params)
    ∇loglikeprior(bnn, θ, x, y)

    # If derivative above does not fail, the test has passed
    @test true
end

@testset "Recurrent derivatives" for rnn in [RNN, LSTM] 
    net = Chain(rnn(1, 1))
    nc = destruct(net)
    x = randn(Float32, 10, 1, 100)
    y = randn(Float32, 100)
    like = SeqToOneNormal(nc, Gamma(2.0, 0.5))
    prior = GaussianPrior(nc, 0.5f0)
    init = InitialiseAllSame(Normal(0f0, 0.5f0), like, prior)
    bnn = BNN(x, y, like, prior, init)

    θ = randn(Float32, bnn.num_total_params)
    ∇loglikeprior(bnn, θ, x, y)

    # If derivative above does not fail, the test has passed
    @test true
end