# Given a Flux.Chain object, we need to destruct the network. Returned should
# be a NetConstructor object which must have fields `num_params_network`, `θ`
# holding the current parameterisation and must be callable. Calling the
# NetConstructor with a vector of length `num_params_network` then should return
# a network of the same structure as the original network with the parameters
# being taken from the input vector. 
using Flux
using BFlux

@testset "Destruct" begin
    @testset "destruct Dense" for in=1:3 for out=1:3 for act in [sigmoid, tanh, identity]
        @testset "Chain(Dense($in, $out))" begin
            net = Chain(Dense(in, out, act))
            net_constructor = destruct(net)
            # W is in×out 
            # b is out×1
            @test net_constructor.num_params_network == in*out + out

            T = eltype(net_constructor.θ)

            netre = net_constructor(net_constructor.θ)
            x = randn(T, in, 10)
            y = net(x)
            yre = netre(x)
            @test all(y .== yre)

            # The next test is only violated with very small probability
            θrand = randn(T, net_constructor.num_params_network)
            netfalse = net_constructor(θrand)
            yfalse = netfalse(x)
            @test all(y .!= yfalse)
        end
    end end end

    @testset "destruct RNN" for in=1:3 for out=1:3
        @testset "Chain(RNN($in, $out))" begin
            net = Chain(RNN(in, out))
            net_constructor = destruct(net)
            # Wx is in×out
            # Wh is out×out 
            # b is out×1
            # h0 is out×1 but we do not infer h0 so we drop it
            @test net_constructor.num_params_network == in*out + out*out + out

            T = eltype(net_constructor.θ)

            netre = net_constructor(net_constructor.θ)
            x = [randn(T, in, 1) for _ in 1:10]
            y = [net(xx) for xx in x][end]
            yre = [netre(xx) for xx in x][end]
            @test all(y .== yre)

            # Next one should only fail with very small probability (basically zero)
            θrand = randn(T, net_constructor.num_params_network)
            netfalse = net_constructor(θrand)
            yfalse = [netfalse(xx) for xx in x][end]
            @test all(y .!= yfalse)
        end
    end end


    @testset "destruct LSTM" for in=1:3 for out=1:3
        @testset "Chain(LSTM($in, $out))" begin
            net = Chain(LSTM(in, out))
            net_constructor = destruct(net)
            # Wx is (out*4)×in because we need four Wx for input, forget, output and
            # cell input activation 
            # Wh is (out*4)×(out) because we need four Wh for input, foget, output and
            # cell input activation
            # b is  (out*4)×1 
            # We have two hidden states, c and h, both are out×1. We do not count
            # these since we do not infer them
            @test net_constructor.num_params_network == in*out*4 + out*out*4 + out*4

            T = eltype(net_constructor.θ)

            netre = net_constructor(net_constructor.θ)
            x = [randn(T, in, 1) for _ in 1:10]
            y = [net(xx) for xx in x][end]
            yre = [netre(xx) for xx in x][end]
            @test all(y .== yre)

            # The following should only fail with probability practically zero
            θfalse = randn(T, net_constructor.num_params_network)
            netfalse = net_constructor(θfalse)
            yfalse = [netfalse(xx) for xx in x][end]
            @test all(y .!= yfalse)
        end
    end end

    @testset "destruct Dense-Dense" for in=1:3 for out=1:3 for act in [sigmoid, tanh, identity]
        @testset "Chain(Dense($in, $in), Dense($in, $out))" begin
            net = Chain(Dense(in, in, act), Dense(in, out))
            net_constructor = destruct(net)
            @test net_constructor.num_params_network == (in*in + in) + (in*out + out)

            T = eltype(net_constructor.θ)
            x = randn(T, in, 10)
            netre = net_constructor(net_constructor.θ)
            y = net(x)
            yre = netre(x)
            @test all(y .== yre)

            θfalse = randn(T, net_constructor.num_params_network)
            netfalse = net_constructor(θfalse)
            yfalse = netfalse(x)
            @test all(y .!= yfalse)
        end
    end end end

    @testset "destruct RNN-Dense" for in=1:3 for out=1:3 for act in [sigmoid, tanh, identity]
        @testset "Chain(RNN($in, $in), Dense($in, $out))" begin
            net = Chain(RNN(in, in), Dense(in, out, act))
            net_constructor = destruct(net)
            @test net_constructor.num_params_network == (in*in + in*in + in) + (in*out + out)

            T = eltype(net_constructor.θ)
            x = [randn(T, in, 1) for _ in 1:10]
            netre = net_constructor(net_constructor.θ)
            y = [net(xx) for xx in x][end]
            yre = [netre(xx) for xx in x][end]
            @test all(y .== yre)

            θfalse = randn(T, net_constructor.num_params_network)
            netfalse = net_constructor(θfalse)
            yfalse = [netfalse(xx) for xx in x][end]
            @test all(y .!= yfalse)
        end
    end end end

    @testset "destruct LSTM-Dense" for in=1:3 for out=1:3 for act in [sigmoid, tanh, identity]
        @testset "Chain(LSTM($in, $in), Dense($in, $out))" begin
            net = Chain(LSTM(in, in), Dense(in, out, act))
            net_constructor = destruct(net)
            @test net_constructor.num_params_network == (in*in*4 + in*in*4 + in*4) + (in*out + out)

            T = eltype(net_constructor.θ)
            x = [randn(T, in, 1) for _ in 1:10]
            netre = net_constructor(net_constructor.θ)
            y = [net(xx) for xx in x][end]
            yre = [netre(xx) for xx in x][end]
            @test all(y .== yre)

            θfalse = randn(T, net_constructor.num_params_network)
            netfalse = net_constructor(θfalse)
            yfalse = [netfalse(xx) for xx in x][end]
            @test all(y .!= yfalse)
        end
    end end end
end