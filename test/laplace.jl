using Distributions, Random


@testset "Laplace" begin

    # note, this actually means a variance of 4
    lpdf(θ) = logpdf(MvNormal(5*ones(2), 2*ones(2)), θ)

    @testset "Single Laplace" begin
        lp = laplace(lpdf, randn(2), 50_000, 1e-10; diag = true)
        draws = rand(lp[1], 10_000)
        @test all(isapprox.(mean(draws; dims = 2), [5.0, 5.0], atol = 0.5))
        @test all(isapprox.(vec(cov(draws')), vec([4.0 0.0; 0.0 4.0]), atol = 0.5))
    end

    @testset "Multiple Laplace" begin
        lps = laplace(lpdf, MvNormal(zeros(2), ones(2)), 10_000, 5, 1e-10)
        draws = rand(lps, 10_000)
        @test all(isapprox.(mean(draws; dims = 2), [5.0, 5.0], atol = 0.5))
        @test all(isapprox.(vec(cov(draws')), vec([4.0 0.0; 0.0 4.0]), atol = 0.5))
    end

    # @testset "SIR Laplace" begin
    #     lps = laplace(lpdf, MvNormal(zeros(2), ones(2)), 10_000, 5, 1e-10)
    #     draws = SIR_laplace(lpdf, lps, 100_000, 10_000; verbose = false)
    #     @test all(isapprox.(mean(draws; dims = 2), [5.0, 5.0], atol = 0.5))
    #     @test all(isapprox.(vec(cov(draws')), vec([4.0 0.0; 0.0 4.0]), atol = 0.5))
    # end
end