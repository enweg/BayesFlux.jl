# Find modes by optimising the network from multiple random starting points
# Forms a laplace approximation of all modes. 
include("../optimise/modes.jl")
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
    x .= rand(rng, StatsBase.sample(s.dists, s.w))
end


function laplace(bnn::BNN, maxiter::Int, ϵ::Float64=0.01; init_θ = randn(bnn.totparams), 
                 showprogress=true, opt = Flux.RMSProp(), diag = false)
   
    θ, conv = find_mode(bnn, maxiter, ϵ; init_θ = init_θ, showprogress = showprogress, opt = opt, verbose = false)
    hessian = diag ? Zygote.diaghessian : Zygote.hessian
    H = hessian(θ -> lp(bnn, θ), θ)
    Hinv = diag ? 1 ./ H[1] : H\I 
    if !diag && !LinearAlgebra.isposdef(Hinv)
        error("Hessian was not positive definite. Try using a diagonal approximation")
    end

    weight = diag ? det(diagm(Hinv)) : det(Hinv)
    weight *= lp(bnn, θ)

    return MvNormal(θ, Hinv), weight, conv
end

function laplace(bnn::BNN, maxiter::Int, M::Int, args...; kwargs...)
    if Threads.nthreads() == 1
        @warn "Only one thread Available. Will not work in parallel."
    end
    dists = Array{Distributions.MultivariateNormal}(undef, M)
    weights = ones(M)
    conv = Array{Bool}(undef, M)
    starting_θ = 10 .* rand(bnn.totparams, M) .- 5

    p = Progress(M)
    update!(p, 0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()

    Threads.@threads for m=1:M 
        dists[m], weights[m], conv[m] = laplace(bnn, maxiter, args...; init_θ = starting_θ[:, m], showprogress = false, kwargs...)
        Threads.atomic_add!(jj, 1)
        Threads.lock(l)
        update!(p, jj[])
        Threads.unlock(l)
    end
    return LaplaceApproximation(dists, StatsBase.ProbabilityWeights(weights ./ sum(weights)), conv)
end