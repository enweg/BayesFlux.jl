# Bayes by Backprop implementation
# Focusing on diagonal covariances and normal reparameterisation

using ProgressMeter

"""
    getθ(ψ, ϵ, get_μ_Σ)

Get θ, the network parameters.

Given a vector ψ corresponding to the reparameterised variational distribution
and a vector ϵ, obtain the mean, μ, and the variance, Σ, using `get_μ_Σ` and 
calculate the network parameters, θ, as θ = μ + Σϵ ~ Normal(μ, Σ'Σ)

# Arguments
- `ψ`: Vector of parameters for the variational family. Usually contains both μ and Σ in vector form 
- `ϵ`: A random draw form the reparameterised variational family. This is usually a standard (multivariate)normal
- `get_μ_Σ`: A function that takes ψ and returns μ and Σ. 

"""
function getθ(ψ, ϵ, get_μ_Σ)
    μ, Σ = get_μ_Σ(ψ)
    return μ .+ Σ * ϵ
end


function bbb_step!(ψ, ybatch, xbatch, nsamples, nparams, bbb_objective, opt)
    g = zeros(eltype(ψ), size(ψ, 1), nsamples)
    for s=1:nsamples
        ϵ = randn(nparams)
        g[:, s] = Zygote.gradient(ψ -> bbb_objective(ψ, ϵ, ybatch, xbatch), ψ)[1]
    end
    g = mean(g; dims = 2)
    Flux.update!(opt, ψ, g)
end

function bbb(llike, lpriorθ, initψ::Vector{T}, get_μ_Σ, nsamples::Int, maxiter::Int, 
             batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}; 
             opt = Flux.RMSProp(), windowlength = 100, showprogress = true, 
             p = Progress(maxiter*floor(Int, length(y)/batchsize); dt = 1, desc = "Bayes by Backprop ...", enabled = showprogress)) where {T<:Real}

    ψ = copy(initψ)
    nparams = length(get_μ_Σ(ψ)[1])
    nbatches = floor(Int, length(y)/batchsize)
    if mod(length(y), batchsize) != 0
        @warn "length(y) is not a multiple of batchsize. Discarding some observations each iteration."
    end

    @info """
    Running BBB with $nsamples samples to estimate expectation and batchsize of $batchsize, resulting in $nbatches batchruns each iteration.
    The maximal total number of iterations is thus $maxiter × $nbatches = $(maxiter * nbatches).
    Losses will be calculated using a window length of $windowlength. 
    """

    bbb_objective(ψ, ϵ, ybatch, xbatch) = logpdf(MvNormal(get_μ_Σ(ψ)[1], Symmetric(get_μ_Σ(ψ)[2]*get_μ_Σ(ψ)[2]')), getθ(ψ, ϵ, get_μ_Σ)) - nbatches * llike(getθ(ψ, ϵ, get_μ_Σ), ybatch, xbatch) - lpriorθ(getθ(ψ, ϵ, get_μ_Σ))

    ravalues = zeros(windowlength)
    losses = zeros(maxiter-windowlength)

    for i=1:maxiter
        yshuffle, xshuffle = sgd_shuffle(y, x)
        for b=1:nbatches
            ybatch = sgd_y_batch(yshuffle, b, batchsize)
            xbatch = sgd_x_batch(xshuffle, b, batchsize)
            bbb_step!(ψ, ybatch, xbatch, nsamples, nparams, bbb_objective, opt)
            next!(p)
        end

        # Calculating the moving average of the loss
        ϵ = randn(nparams)
        loss = bbb_objective(ψ, ϵ, y, x)
        raindex = mod(i, windowlength) + 1
        ravalues[raindex] = loss
        if i > windowlength
            losses[i-windowlength] = mean(ravalues)
        end
    end

    return MvNormal(get_μ_Σ(ψ)[1], Symmetric(get_μ_Σ(ψ)[2]*get_μ_Σ(ψ)[2]')), initψ, ψ, losses
end

function bbb(bnn::BNN, nsamples::Int, maxiter::Int, batchsize::Int; kwargs...)
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    initψ = 0.1*randn(bnn.totparams*2)
    get_mu_sig(ψ) = (ψ[1:bnn.totparams], diagm(exp.(ψ[bnn.totparams+1:end])))
    return bbb(llike, lpriorθ, initψ, get_mu_sig, nsamples, maxiter, batchsize, bnn.y, bnn.x; kwargs...)
end

function bbb(bnn::BNN, nsamples::Int, maxiter::Int, batchsize::Int, nchains::Int; kwargs...)
    chains = Array{Any}(undef, nchains)
    if Threads.nthreads() == 1
        @warn "Only one thread available."
    end
    p = Progress(maxiter*floor(Int, length(bnn.y)/batchsize)*nchains; dt = 1, desc = "Bayes by Backprop using $nchains chains ...")
    Threads.@threads for ch=1:nchains
        chains[ch] = bbb(bnn, nsamples, maxiter, batchsize; p = p, kwargs...)
    end

    dist = MixtureModel([ch[1] for ch in chains])
    initψs = hcat([ch[2] for ch in chains]...)
    ψs = hcat([ch[3] for ch in chains]...)
    losses = hcat([ch[4] for ch in chains]...)

    return dist, initψs, ψs, losses
end
