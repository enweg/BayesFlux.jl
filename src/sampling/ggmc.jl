# Gradient Guided Monte Carlo
# Following papers critices sgld and variants relying on Euler-Maruyama scheme
# and instead proposed to use GGMC with stochastic gradients. 
# Garriga-Alonso, A., & Fortuin, V. (2021). Exact langevin dynamics with stochastic gradients. arXiv preprint arXiv:2102.01691.
using LinearAlgebra
using Parameters

U(θ, llike, lpriorθ, y, x) = -llike(θ, y, x) - lpriorθ(θ)
K(m, M) = 1/2 * m' * inv(M) * m

################################################################################
#### Mass Matrix Adaptation 
################################################################################

abstract type MAdapter end

# Using the same adapter used in 
# Scalable Bayesian Learning of Recurrent Neural Networks for Language Modeling
# just for ggmc instead of sgld
struct MRMSPropAdapter <: MAdapter
    v::AbstractArray{Float64}
    β::Float64
    λ::Float64
end
MRMSPropAdapter(size; β = 0.99, λ = 1e-8) = MRMSPropAdapter(zeros(size), β, λ)
function (m_adapter::MRMSPropAdapter)(g::AbstractArray)
    @unpack v, β, λ = m_adapter
    m_adapter.v .= β*v .+ (1-β)*g.*g
    G = diagm(1 ./ (λ .+ sqrt.(v)))
    Minv = G
    M = inv(G)
    Mhalf = sqrt.(M)
    return Minv, M, Mhalf
end

# No adapting of M
struct MIdentityAdapter <: MAdapter
end
function (m_adapter::MIdentityAdapter)(g::AbstractArray)
    M = diagm(ones(length(g)))
    Minv = copy(M)
    Mhalf = copy(M)
    return Minv, M, Mhalf
end

################################################################################
#### Step Length Adaptation 
################################################################################

abstract type hAdapter end

# Very simplistic adaptation of h using stochastic optimisation
# 0.5 < κ < 1
mutable struct hStochasticAdapter <: hAdapter
    h::Float64
    κ::Float64
    goal_accept_rate::Float64
end
hStochasticAdapter(h ;κ = 0.55, goal_accept_rate = 0.55) = hStochasticAdapter(h, κ, goal_accept_rate)
function (h_adapter::hStochasticAdapter)(t::Int64, hastings::Float64)
    @unpack h, κ, goal_accept_rate = h_adapter
    arate = min(1, hastings)
    H = goal_accept_rate - arate 
    eta = t^(-κ)
    h_adapter.h = exp(log(h) - eta*H) 
    return h_adapter.h
end

struct hFixedAdapter <: hAdapter
    h::Float64
end
function (h_adapter::hFixedAdapter)(t::Int64, hastings::Float64)
    return h_adapter.h
end

# Gradient can sometimes be exploding, for that reason we often need to clip 
# gradients so that we are not running into numerical problems. 
∇U(θ, llike, lpriorθ, y, x, nbatches) = Zygote.gradient(θ -> nbatches * -llike(θ, y, x) - lpriorθ(θ), θ)[1]


function ggmc(llike::Function, lpriorθ::Function, batchsize::Int, y::Vector{T}, x::Union{Vector{Matrix{T}}, Matrix{T}}, 
              initθ::Vector{T}, maxiter::Int;
              l = 0.01, β = 0.5, beta = 0.5, temp = 1, keep_every = 10, 
              adapruns = 5000, 
              adaptM = true, m_adapter = MRMSPropAdapter(size(initθ)),
              adapth = true, h_adapter = hStochasticAdapter(sqrt(l / length(y)); goal_accept_rate = 0.55),
              showprogress = true, debug = false, 
              p = Progress(maxiter * floor(Int, length(y)/batchsize); dt=1, desc="GGMC ...", enabled = showprogress)) where {T <: Real}
    
    # Allowing for unicode/no-unicode
    β = beta
    # We are exposing a parameterisation in terms of learning rate and momentum parameter
    # but the updates work using h and γ. See paper for details
    γ = -sqrt(length(y)/l)*log(β)
    h = adapth ? h_adapter.h : sqrt(l / length(y))
    
    θ = copy(initθ)
    M = diagm(ones(length(θ)))
    if mod(length(y), batchsize) != 0
        @warn "Batchsize does not properly partition data. Some data will be left out in each cycle."
    end
    num_batches = floor(Int64, length(y)/batchsize)
    @info "Using $num_batches batches."
    θprop = copy(θ)
    tot_iters = maxiter*num_batches

    samples = zeros(eltype(initθ), length(initθ), tot_iters + 1) 
    accept = zeros(tot_iters + 1)
    samples[:, 1] = θ
    accept[1] = 1
    lastθi = 1 # column of last added samples

    momentum14 = zeros(size(θ))
    momentum12 = zeros(size(θ))
    momentum34 = zeros(size(θ))
    momentum = zeros(size(θ))
    Mhalf = cholesky(M, check = false).L
    Minv = inv(M)
    naccepts = 0

    lMH = -U(θ, llike, lpriorθ, y, x) 
    hastings = zeros(tot_iters+1)
    momenta = zeros(length(θ), tot_iters+1)
    for t=1:maxiter
        yshuffel, xshuffel = sgd_shuffle(y, x) 
        for b=1:num_batches 
            a = exp(-γ*h)
            xbatch = sgd_x_batch(xshuffel, b, batchsize)
            ybatch = sgd_y_batch(yshuffel, b, batchsize)

            g = ∇U(θ, llike, lpriorθ, ybatch, xbatch, num_batches)
            g = clip_gradient_value!(g)
            if any(isnan.(g))
                @warn "NaN in gradient. Returning results obtained until now."
                return samples[:, 1:lastθi], hastings[1:lastθi], momenta[:, 1:((t-1)*num_batches + b)]
            end

            if lastθi <= adapruns && adaptM
                Minv, M, Mhalf = m_adapter(g)
            end

            momentum14 .= sqrt(a)*momentum .+ sqrt((1-a)*temp)*Mhalf*randn(length(θ))
            momentum12 .= momentum14 .- h/2 * g
            θ .= θ .+ h*Minv*momentum12
            θprop = θ

            g = ∇U(θ, llike, lpriorθ, ybatch, xbatch, num_batches)
            g = clip_gradient_value!(g)
            if any(isnan.(g))
                @warn "NaN in gradient. Returning results until obtained until now."
                return samples[:, 1:lastθi], hastings[1:lastθi], momenta[:, 1:((t-1)*num_batches + b)]
            end

            momentum34 .= momentum12 .- h/2 * g 
            momentum .= sqrt(a)*momentum34 .+ sqrt((1-a)*temp)*Mhalf*randn(length(θ))
            momenta[:, (t-1)*num_batches + b] = momentum

            lMH += K(momentum34, M) - K(momentum14, M)

            if mod((t-1)*num_batches + b, keep_every) == 0
                # time to evaluate MH ratio and keep last draw or start new 
                lMH += U(θ, llike, lpriorθ, y, x)
                lMH /= -temp
                lr = log(rand())
                hastings[lastθi+1] = exp(lMH)
                if lr < lMH
                    samples[:, lastθi+1] .= copy(θprop)
                    accept[lastθi+1] = 1
                    if (lastθi >= adapruns) naccepts += 1 end
                else 
                    samples[:, lastθi+1] .= copy(samples[:, lastθi])
                    θ = copy(samples[:, lastθi])
                end
                lastθi += 1

                if  adapth && lastθi <= adapruns
                    # adapting step size/learning rate
                    h = h_adapter(lastθi, hastings[lastθi])
                    lastθi == adapruns && @info "l=$(length(y)*h^2)"
                end

                lMH = -U(θ, llike, lpriorθ, y, x) 

            end

            if debug
                update!(p, (t-1)*num_batches + b, showvalues = [(:arate, naccepts/lastθi), 
                                                            (:iter, (t-1)*num_batches + b), 
                                                            (:samples, lastθi), 
                                                            (:Madapt, adaptM && in_mass_window), 
                                                            (:hadapt, adapth && in_h_window)])
            else 
                next!(p)
            end
        end

    end

    return samples[:, 1:lastθi], hastings[1:lastθi], momenta, accept[1:lastθi]

end

function ggmc(bnn::BNN, batchsize::Int, initθ::AbstractVector, maxiter::Int; kwargs...)
    llike(θ, y, x) = loglike(bnn, bnn.loglikelihood, θ, y, x)
    lpriorθ(θ) = lprior(bnn, θ)
    return ggmc(llike, lpriorθ, batchsize, bnn.y, bnn.x, initθ, maxiter; kwargs...)
end

function ggmc(bnn::BNN, batchsize::Int, initθ::Vector{Vector{T}}, maxiter::Int, nchains::Int; kwargs...) where {T<:Real}
    chains = Array{Any}(undef, nchains)
    Threads.nthreads() == 1 && @warn "Only one thread available."
    p = Progress(maxiter * floor(Int, length(bnn.y)/batchsize) * nchains; dt=1, desc="GGMC with $nchains chains ...") 
    Threads.@threads for ch=1:nchains
        chains[ch] = ggmc(bnn, batchsize, initθ[ch], maxiter; p = p, kwargs...)
    end

    samples = cat([ch[1] for ch in chains]...; dims = 3)
    hastings = cat([ch[2] for ch in chains]...; dims = 3)
    momenta = cat([ch[3] for ch in chains]...; dims = 3)
    accept = cat([ch[4] for ch in chains]...; dims = 3)

    return samples, hastings, momenta, accept
end