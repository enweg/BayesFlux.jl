# Find the map estimate (mode of the posteior)
using ProgressBars

function find_mode(bnn::BNN, maxiter::Int, ϵ::Float64=0.0001; init_θ = randn(bnn.totparams), kwargs...)
    return find_mode(θ -> lp(bnn, θ), init_θ, maxiter, ϵ; kwargs...)
end

function find_mode(lpdf, initθ::AbstractVector, maxiter::Int, ϵ::Float64=0.0001; 
                    showprogress = true, opt = Flux.RMSProp(), verbose = true)
    iterator = showprogress ? ProgressBar(1:maxiter) : 1:maxiter
    θ = copy(initθ)
    lj = lpdf(θ)
    for t in iterator
        lj_, g = Zygote.withgradient(θ -> lpdf(θ), θ)
        Flux.update!(opt, θ, -g[1])
        if  t > 1 && abs((lj_ / lj) - 1) < ϵ
            if (verbose) @info "Found mode in $t iterations." end
            return θ, true
        end
        lj = lj_
    end
    if (verbose) @info "Did not converge within $maxiter iterations." end
    return θ, false 
end

function sgd_shuffle(y::AbstractVector{T}, x::Vector{Matrix{T}}) where {T<:Real}
    yshuffle = copy(y)
    xshuffle = deepcopy(x)
    order = shuffle(1:length(y))
    return yshuffle[order], [xx[:,order] for xx in xshuffle] 
end

function sgd_shuffle(y::AbstractVector{T}, x::Matrix{T}) where {T<:Real}
    yshuffle = copy(y)
    xshuffle = copy(x)
    order = shuffle(1:length(y))
    return yshuffle[order], xshuffle[:, order]
end

sgd_y_batch(yshuffle::AbstractVector{T}, b::Int, bsize::Int) where {T<:Real} = yshuffle[((b-1)*bsize+1):(b*bsize)]

sgd_x_batch(xshuffle::Matrix{T}, b::Int, bsize::Int) where {T<:Real} = xshuffle[:, ((b-1)*bsize+1):(b*bsize)]

sgd_x_batch(xshuffle::Vector{Matrix{T}}, b::Int, bsize::Int) where {T<:Real} = [sgd_x_batch(xx, b, bsize) for xx in xshuffle]

function find_mode_sgd(llike, lpriorθ, batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}, 
                       initθ::AbstractVector, maxiter::Int, ϵ::Float64=0.0001;
                       showprogress = true, opt = Flux.ADAM(), verbose = true) where {T<:Real}
    θ = copy(initθ)

    lj = llike(θ, y, x) + lpriorθ(θ)
    losses = Float64[]
    push!(losses, lj)
    ralength = 5
    ljrunningaverage = zeros(ralength)
    ljrunningaverage[1] = lj

    # yshuffel, xshuffel = sgd_shuffle(y, x) 
    if mod(length(y), batchsize) != 0
        @warn "Batchsize does not properly partition data. Some data will be left out in each cycle."
    end
    num_batches = floor(Int64, length(y)/batchsize)

    p = Progress(maxiter * num_batches, 1, "Finding mode ...")

    for t=1:maxiter
        yshuffel, xshuffel = sgd_shuffle(y, x) 
        gprior = Zygote.gradient(lpriorθ, θ)[1]
        for b=1:num_batches 
            xbatch = sgd_x_batch(xshuffel, b, batchsize)
            ybatch = sgd_y_batch(yshuffel, b, batchsize)
            glike = (length(y)/batchsize) * Zygote.gradient(θ -> llike(θ, ybatch, xbatch), θ)[1]
            g = glike .+ gprior
            Flux.update!(opt, θ, -g)
            update!(p, (t-1)*num_batches + b)
        end

        ljindex = mod(t, ralength) + 1
        ljprior = lpriorθ(θ)
        ljlike = llike(θ, y, x)
        ljrunningaverage[ljindex] = ljlike + ljprior

        if t > ralength
            if abs(mean(ljrunningaverage)/lj - 1) < ϵ
                if (verbose) @info "Converged in iteration $t" end
                return θ, true, losses
            end
            lj = mean(ljrunningaverage)
            push!(losses, lj)
        end
    end

    if (verbose) @info "Did not converge within $maxiter iterations" end
    return θ, false, losses

end 

function find_mode_sgd(bnn::BNN, batchsize::Int, initθ::AbstractVector, maxiter::Int, ϵ::Float64=0.0001; kwargs...) where {T<:Real}
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    return find_mode_sgd(llike, lpriorθ, batchsize, bnn.y, bnn.x, initθ, maxiter, ϵ; kwargs...)
end
