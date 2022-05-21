# Every prior of network parameters must be a subtype of NetworkPrior. It must
# be callable and return the logprior density and it must implement a sample
# method, sampling a vector of network parameters from the prior

@testset "Network Prior" begin
   @testset "Gaussian" for σ0 in [0.5, 1.0, 3.0, 10.0]
        @testset "Gaussian σ0 = $σ0" begin
            net = Chain(Dense(10, 10, sigmoid), Dense(10, 1))
            nc = destruct(net)
            T = eltype(nc.θ)
            gp = GaussianPrior(nc, T(σ0))

            @test gp.num_params_hyper == 0


            n = nc.num_params_network
            θ = T.(collect(0.1:0.1:0.9))
            # out prior is standard normal
            @test gp(θ, Float32[]) ≈ T(sum(logpdf.(Normal(T(0), T(σ0)), 0.1:0.1:0.9)))

            θdraws = reduce(hcat, [sample_prior(gp) for _ in 1:1_000_000])
            𝔼θdraws = vec(mean(θdraws; dims = 2))
            @test maximum(abs, 𝔼θdraws) < 0.1

            𝕍θdraws = vec(var(θdraws; dims = 2))
            @test maximum(𝕍θdraws ./ (σ0^2)) < 1.01
        end
   end
end