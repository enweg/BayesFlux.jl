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
            𝔼θdraws = vec(mean(θdraws; dims=2))
            @test maximum(abs, 𝔼θdraws) < 0.1

            𝕍θdraws = vec(var(θdraws; dims=2))
            @test maximum(𝕍θdraws ./ (σ0^2)) < 1.01
        end
    end

    @testset "Mixture Gaussian" for (μ1, σ1, σ2) in zip([0.01f0, 0.1f0, 0.5f0, 0.9f0], [0.001f0, 0.1f0, 1.0f0], [1.0f0, 5.0f0, 10.0f0])
        @testset "Mixture Gaussian μ1=$μ1, σ1=$σ1, σ2=$σ2" begin
            net = Chain(Dense(10, 10, sigmoid), Dense(10, 1))
            nc = destruct(net)
            T = eltype(nc.θ)
            prior = MixtureScalePrior(nc, σ1, σ2, μ1)

            @test prior.num_params_hyper == 0

            # Both have zero mean so mixture has zero mean
            θdraws = reduce(hcat, [sample_prior(prior) for _ in 1:1_000_000])
            𝔼θdraws = vec(mean(θdraws; dims=2))
            @test maximum(abs, 𝔼θdraws) < 0.1

            # Gaussian are independent so Var(Mixture) = π1^2Var(G1) + π2^2Var(G2)
            # θ = z₁θ₁ + (1-z₁)θ₂ where z₁ ~ Bernoulli(π1) and thus 1-z₁ ~ Bernoulli(π2)
            # This gives a theoretical variance of 
            # V(θ) = π1*σ1^2 + π2*σ2^2
            𝕍θdraws = vec(var(θdraws; dims=2))
            var_theoretic = prior.π1 * prior.σ1^2 + prior.π2 * prior.σ2^2
            @test maximum(𝕍θdraws ./ var_theoretic) < 1.01

        end
    end
end