# Testing Bayes By Backprop 
using ProgressBars

function bbb_loss(bnn::BNN, ψdist::D) where {D<:Distribution}
    θ = rand(ψdist)
    return logpdf(ψdist, θ) - lp(bnn, θ)
end

function bbb_loss(bnn::BNN, ψdist::D, samples::Int) where {D<:Distribution}
    θ = rand(ψdist, samples)
    return sum(logpdf.([ψdist], eachcol(θ))) - sum(lp.([bnn], eachcol(θ)))
end

function bbb_test(bnn::BNN, maxiters::Int, samples::Int = 10;showprogress = true)
    @info "Using $samples samples per interator"
    μ = zeros(bnn.totparams)
    Σdiag = log.(ones(bnn.totparams))
    ψ = vcat(μ, Σdiag)
    iterator = showprogress ? ProgressBar(1:maxiters) : 1:maxiters
    getdist(ψ) = MvNormal(ψ[1:Int(length(ψ)/2)], exp.(ψ[Int(length(ψ)/2)+1:end]))
    opt = Flux.ADAM()

    losses = ones(maxiters+1)
    losses[1] = bbb_loss(bnn, getdist(ψ))

    for t in iterator
        losses[t+1], g = Zygote.withgradient(ψ -> bbb_loss(bnn, getdist(ψ), samples), ψ)
        Flux.update!(opt, ψ, g[1])
        if abs(losses[t+1]/losses[t] - 1) < 0.00001
            @info "Converged in iteration $t"
            return getdist(ψ), losses[1:(t+1)]
        end
    end
        
    return getdist(ψ), losses
end

function vec_to_ltri(v::Vector{T}) where {T<:Real}
    cols = Int(-1/2 + sqrt(1+8*length(v))/2)
    ltri = Zygote.Buffer(ones(T), cols, cols)
    # ltri = Matrix{T}(undef, cols, cols)
    k = 1
    for i=1:cols for j=1:cols
        ltri[i, j] = i >= j ? v[k] : 0
        k += i >= j ? 1 : 0
    end end
    return LinearAlgebra.LowerTriangular(copy(ltri))
end


function bbb_grad1(ψ, ϵ, n)
    μ = ψ[1:n]
    ltri = vec_to_ltri(ψ[n+1:end])
    θs = [μ .+ ltri * ϵi for ϵi in ϵ]
    d = MvNormal(μ, Symmetric(ltri*ltri'))
    return mean(logpdf.([d], θs))
end

function bbb_grad2(bnn, ψ, ϵ, n)
    μ = ψ[1:n]
    ltri = vec_to_ltri(ψ[n+1:end])
    θs = [μ .+ ltri * ϵi for ϵi in ϵ]
    return mean(loglike.([bnn], [bnn.loglikelihood], θs, [bnn.y], [bnn.x])) 
end

function bbb_grad3(bnn, ψ, ϵ, n)
    μ = ψ[1:n]
    ltri = vec_to_ltri(ψ[n+1:end])
    θs = [μ .+ ltri * ϵi for ϵi in ϵ]
    return mean(lprior.([bnn], θs)) 
end


# TODO: use stochastic gradients
# TODO: reduce variance using ... ?
# TODO: use lower triangular Matrix
# TODO: properly implement.
function bbb_test_repara(bnn::BNN, maxiters::Int, samples::Int = 10;showprogress = true)
    @info "Using $samples samples per interator"
    μ = zeros(bnn.totparams)
    Σdiag = log.(ones(bnn.totparams))
    n = bnn.totparams
    Σ = randn(Int(n*(n+1)/2))
    
    ψ = vcat(μ, Σ)
    iterator = showprogress ? ProgressBar(1:maxiters) : 1:maxiters
    opt = Flux.ADAM()

    windowlength = 20
    losses = ones(maxiters - windowlength)
    runaverage = zeros(20)

    for t in iterator
        ϵ = rand(MvNormal(zeros(n), ones(n)), samples)
        ϵ = collect(eachcol(ϵ))
        # try
        #     sig = vec_to_ltri(exp.(ψ[n+1:end]))*vec_to_ltri(exp.(ψ[n+1:end]))'
        #     cholesky(sig)
        # catch
        #     sig = vec_to_ltri(exp.(ψ[n+1:end]))*vec_to_ltri(exp.(ψ[n+1:end]))'
        #     println("Something went wrong during cholesky")
        #     println("$sig")
        # end
        
        # v1, g1 = Zygote.withgradient(ψ -> mean(logpdf.([MvNormal(ψ[1:n], Symmetric(vec_to_ltri(exp.(ψ[n+1:end]))*vec_to_ltri(exp.(ψ[n+1:end]))'))], [ψ[1:n] .+ vec_to_ltri(exp.(ψ[n+1:end])) * ϵi for ϵi in ϵ])), ψ)
        # v2, g2 = Zygote.withgradient(ψ -> mean(loglike.([bnn], [bnn.loglikelihood], [ψ[1:n] .+ vec_to_ltri(exp.(ψ[n+1:end])) * ϵi for ϵi in ϵ], [bnn.y], [bnn.x])), ψ)
        # v3, g3 = Zygote.withgradient(ψ -> mean(lprior.([bnn], [ψ[1:n] .+ vec_to_ltri(exp.(ψ[n+1:end])) * ϵi for ϵi in ϵ])), ψ)
        v1, g1 = Zygote.withgradient(ψ -> bbb_grad1(ψ, ϵ, n), ψ)
        v2, g2 = Zygote.withgradient(ψ -> bbb_grad2(bnn, ψ, ϵ, n), ψ)
        v3, g3 = Zygote.withgradient(ψ -> bbb_grad3(bnn, ψ, ϵ, n), ψ)

        loss = v1 - v2 - v3
        raindex = mod(t, windowlength) + 1
        runaverage[raindex] = loss
        g = g1[1] .- g2[1] .- g3[1]
        if (t>windowlength)
            losses[t-windowlength] = mean(runaverage)
        end

        Flux.update!(opt, ψ, g)
        if t>windowlength+1 && abs(losses[t-windowlength]/losses[t-windowlength-1] - 1) < 0.0000000001
            @info "Converged in iteration $t"
            return MvNormal(ψ[1:n], Symmetric(vec_to_ltri(exp.(ψ[n+1:end]))*vec_to_ltri(exp.(ψ[n+1:end]))')), losses[1:t-windowlength]
        end
    end
        
    return MvNormal(ψ[1:n], Symmetric(vec_to_ltri(exp.(ψ[n+1:end]))*vec_to_ltri(exp.(ψ[n+1:end]))')), losses
end