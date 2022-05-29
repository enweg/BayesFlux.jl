
using StatsFuns
using Random, Distributions
using ProgressMeter

function loss_bbb(bnn::B, ψ::AbstractVector{T}, 
    x::Union{Vector{Matrix{T}}, Matrix{T}, Array{T, 3}}, 
    y::Union{Vector{T}, Matrix{T}}; num_batches = T(1)) where {B<:BNN, T}

    n = Int(length(ψ)/2)
    μ = ψ[1:n]
    σ = log1pexp.(ψ[n+1:end])
    ϵ = randn(T, n)
    θ = μ .+ σ.*ϵ
    return logpdf(MvNormal(μ, σ), θ) - loglikeprior(bnn, θ, x, y; num_batches = num_batches)
end

function ∇bbb(bnn::B, ψ::AbstractVector{T}, 
    x::Union{Vector{Matrix{T}}, Matrix{T}, Array{T, 3}}, 
    y::Union{Vector{T}, Matrix{T}}; num_batches = T(1)) where {B<:BNN, T}

    return Zygote.gradient(ψ -> loss_bbb(bnn, ψ, x, y; num_batches = num_batches), ψ)[1]
end

function bbb(bnn::BNN, batchsize::Int, epochs::Int; 
    mc_samples = 1, shuffle = true, partial = true, 
    showprogress = true, opt = Flux.ADAM())

    T = eltype(bnn.like.nc.θ)

    if !partial && !shuffle 
        @warn """shuffle and partial should not be both false unless the data is
        perfectly divided by the batchsize. If this is not the case, some data
        would never be considered"""
    end

    θnet, θhyper, θlike = bnn.init()
    initμ = vcat(θnet, θhyper, θlike)
    initσ = ones(T, length(initμ))
    ψ = vcat(initμ, initσ)

    batcher = Flux.Data.DataLoader((x = bnn.x, y = bnn.y), 
        batchsize = batchsize, shuffle = shuffle, partial = partial)
    
    num_batches = length(batcher)
    prog = Progress(num_batches * epochs; enabled = showprogress, 
        desc = "BBB...", showspeed = true)

    ∇ψ(ψ, x, y) = ∇bbb(bnn, ψ, x, y; num_batches = num_batches)

    ψs = Array{T}(undef, length(ψ), epochs)
    
    gs = Array{T}(undef, length(ψ), mc_samples)
    for e=1:epochs
        for (x, y) in batcher
            # Threads.@threads for s=1:mc_samples
            @simd for s=1:mc_samples
                gs[:, s] = ∇ψ(ψ, x, y)
            end
            g = vec(mean(gs; dims = 2))
            Flux.update!(opt, ψ, g)
            next!(prog)
        end
        ψs[:, e] = copy(ψ)
    end

    n = bnn.num_total_params
    μ = ψ[1:n]
    σ = log1pexp.(ψ[(n+1):end])
    return MvNormal(μ, σ), ψs
end
