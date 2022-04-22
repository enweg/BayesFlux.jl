# Find modes by optimising the network from multiple random starting points
# Forms a laplace approximation of all modes. 
using ProgressMeter
using Distributions
import Distributions: _rand!
import Base: length
using StatsBase
using Random

struct LaplaceApproximation{D<:Distributions.MultivariateNormal, W<:StatsBase.AbstractWeights} <: Distributions.Sampleable{Multivariate, Continuous}
    dists::Vector{D}
    w::W
    c::Array{Bool}
end
length(s::LaplaceApproximation) = length(s.dists[1].μ)
function _rand!(rng::AbstractRNG, s::LaplaceApproximation, x::AbstractVector{<:Real})
    dists, w = s.dists, s.w
    if !all(s.c)
        @warn "not all approximations have converged. Only sampling from converged ones"
        sum(s.c)==0 && error("No apprixomation has converged. Run apprixmation for longer")
        dists = s.dists[s.c]
        w = StatsBase.ProbabilityWeights(s.w.values[s.c] ./ sum(s.w.values[s.c]))
    end
    x .= rand(rng, StatsBase.sample(dists, w))
end

function laplace(lpdf, initθ::AbstractVector, maxiter::Int, ϵ::Float64=0.01; diag = false, kwargs...) 
    θ, conv = find_mode(lpdf, initθ, maxiter, ϵ; verbose = false, kwargs...)
    hessian = diag ? Zygote.diaghessian : Zygote.hessian
    H = hessian(θ -> lpdf(θ), θ)
    Hinv = diag ? sqrt.(1 ./ -H[1]) : (-H)\I 
    if !diag && !LinearAlgebra.isposdef(Hinv)
        error("Hessian was not positive definite. Try using a diagonal approximation")
    end

    if any(isnan.(θ)) error("Produced NaN during mode finding.") end

    weight = 1.0

    return MvNormal(θ, Hinv), weight, conv 
end

function laplace(lpdf, initθ_distribution::D, maxiter::Int, M::Int, args...; kwargs...) where {D<:Distributions.Distribution}
    if Threads.nthreads() == 1
        @warn "Only one thread Available. Will not work in parallel."
    end
    dists = Array{Distributions.MultivariateNormal}(undef, M)
    weights = ones(M)
    conv = Array{Bool}(undef, M)
    starting_θ = rand(initθ_distribution, M) 

    p = Progress(M)
    update!(p, 0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()

    Threads.@threads for m=1:M 
        dists[m], weights[m], conv[m] = laplace(lpdf, starting_θ[:, m], maxiter, args...; showprogress = false, kwargs...)
        Threads.atomic_add!(jj, 1)
        Threads.lock(l)
        update!(p, jj[])
        Threads.unlock(l)
    end
    return LaplaceApproximation(dists, StatsBase.ProbabilityWeights(weights ./ sum(weights)), conv) 
end

function laplace(bnn::BNN, maxiter::Int, ϵ::Float64=0.01; init_θ = randn(bnn.totparams), 
                diag = false, kwargs...)
    return laplace(θ -> lp(bnn, θ), init_θ, maxiter, ϵ; diag = diag, kwargs...)
end

function laplace(bnn::BNN, maxiter::Int, M::Int, args...; kwargs...)
    return laplace(θ -> lp(bnn, θ), MvNormal(zeros(bnn.totparams), 10*ones(bnn.totparams)), maxiter, M, args...; kwargs...)
end

function mixture_dens(dists::Vector{D}, w::AbstractVector, θ::AbstractVector) where {D<:Distributions.MultivariateDistribution}
    dens = zero(eltype(θ))
    for (di,wi) in zip(dists, w)
        dens += wi*pdf(di, θ)
    end
    return dens
end

function SIR_laplace(lpdf, la::LaplaceApproximation, n::Int, k::Int; verbose = true)
    # Sampling Importance Resampling of Laplace approximation
    sum(la.c) == 0 && error("No laplace approximation converged. Consider running them for longer.")
    !all(la.c) && @warn "Not all laplace approximation converged. Will only used converged ones." 

    proposal = Distributions.MixtureModel(MvNormal[la.dists...], la.w)
    initial_samples = rand(proposal, n)
    initial_samples = collect(eachcol(initial_samples))
    weights = exp.(lpdf.(initial_samples)) ./ pdf(proposal, initial_samples)
    choose = StatsBase.sample(1:n, StatsBase.ProbabilityWeights(weights), k; replace = false)
    chosen_weights = weights[choose]
    chosen_samples = initial_samples[choose]
    verbose && @info "Mean of importance weights (should be close to one): $(mean(weights))"
    verbose && @info "Mean of importance weights of chosen samples: $(mean(chosen_weights))"

    return hcat(chosen_samples...)
end

function SIR_laplace(bnn::BNN, la::LaplaceApproximation, n::Int, k::Int)
    return SIR_laplace(θ -> lp(bnn, θ), la, n, k)
end
