using Flux

@testset "Dense Layer" begin
    @testset "Destructuring" begin
        net = Dense(10, 10)
        x = randn(10, 10)
        θ, re = BFlux.destruct(net)
        net_re = re(θ)

        yoriginal = net(x)
        yre = net_re(x)

        @test all(yoriginal .== yre)

        θnew = randn(length(θ))
        netnew = re(θnew)
        ynew = netnew(x)

        # The chance that they are equal if everything is correct is 0
        @test all(yoriginal .!= ynew)
    end
end