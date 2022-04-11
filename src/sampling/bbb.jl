# Bayes by Backprop implementation
# Focusing on diagonal covariances and normal reparameterisation

using ProgressMeter

function getθ(ψ, ϵ, get_μ_Σ)
    μ, Σ = get_μ_Σ(ψ)
    return μ .+ Σ * ϵ
end

function bbb(llike, lpriorθ, initψ::Vector{T}, get_μ_Σ, nsamples::Int, maxiter::Int, batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}; opt = Flux.RMSProp(), windowlength = 100) where {T<:Real}

    nbatches = floor(Int, length(y)/batchsize)
    if mod(length(y), batchsize) != 0
        @warn "length(y) is not a multiple of batchsize. Discarding some observations each iteration."
    end

    @info """
    Running BBB with $nsamples samples to estimate expectation and batchsize of $batchsize, resulting in $nbatches batchruns each iteration.
    The maximal total number of iterations is thus $maxiter × $nbatches = $(maxiter * nbatches).
    Losses will be calculated using a window length of $windowlength. 
    """

    ψ = copy(initψ)
    bbb_kl(ψ, ϵ, ybatch, xbatch) = logpdf(MvNormal(get_μ_Σ(ψ)[1], Symmetric(get_μ_Σ(ψ)[2]*get_μ_Σ(ψ)[2]')), getθ(ψ, ϵ, get_μ_Σ)) - length(y)/nbatches * llike(getθ(ψ, ϵ, get_μ_Σ), ybatch, xbatch) - lpriorθ(getθ(ψ, ϵ, get_μ_Σ))


    p = Progress(maxiter*nbatches, 1, "Bayes by Backprop ...")
    nparams = length(get_μ_Σ(ψ)[1])

    ravalues = zeros(windowlength)
    losses = zeros(maxiter-windowlength)

    for i=1:maxiter
        yshuffle, xshuffle = sgd_shuffle(y, x)
        for b=1:nbatches
            ybatch = sgd_y_batch(yshuffle, b, batchsize)
            xbatch = sgd_x_batch(xshuffle, b, batchsize)

            g = zeros(eltype(ψ), size(ψ)[1], nsamples)
            Threads.@threads for s=1:nsamples
                ϵ = randn(nparams)
                g[:, s] = Zygote.gradient(ψ -> bbb_kl(ψ, ϵ, ybatch, xbatch), ψ)[1]
            end
            g = mean(g; dims = 2)
            # ϵ = rand(MvNormal(zeros(nparams), ones(nparams)), nsamples)
            # g = Zygote.gradient(ψ -> mean(bbb_kl.([ψ], eachcol(ϵ), [ybatch], [xbatch])), ψ)[1]
            Flux.update!(opt, ψ, g)
            update!(p, (i-1)*nbatches + b)
        end

        ϵ = randn(nparams)
        loss = bbb_kl(ψ, ϵ, y, x)
        raindex = mod(i, windowlength) + 1
        ravalues[raindex] = loss
        if i > windowlength
            losses[i-windowlength] = mean(ravalues)
        end

    end

    return MvNormal(get_μ_Σ(ψ)[1], Symmetric(get_μ_Σ(ψ)[2]*get_μ_Σ(ψ)[2]')), ψ, losses
end

function bbb(bnn::BNN, nsamples::Int, maxiter::Int, batchsize::Int; kwargs...)
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    initψ = randn(bnn.totparams*2)
    get_mu_sig(ψ) = (ψ[1:bnn.totparams], diagm(exp.(ψ[bnn.totparams+1:end])))
    return bbb(llike, lpriorθ, initψ, get_mu_sig, nsamples, maxiter, batchsize, bnn.y, bnn.x; kwargs...)
end

