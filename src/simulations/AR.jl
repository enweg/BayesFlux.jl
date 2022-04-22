
################################################################################
# Simulation and construction of AR(p) model
################################################################################
struct AR{T<:Real, D<:Distribution, R<:Random.AbstractRNG}
    Φ::Vector{T} # lag Vector
    burnin::Int
    e_dist::D   
    rng::R
    seed::Int
end

# Note that the normal is parameterised by the SD not VAR
AR(Φ::Vector; burnin = 1000, e_dist = Normal(0, 1), rng = Random.GLOBAL_RNG) = AR(Φ, burnin, e_dist, rng, rng == Random.GLOBAL_RNG ? 0 : Int(rng.seed[1]))

function (mod::AR)(;N=100, starting_values = zeros(size(mod.Φ)[1]), reseed = false)
    # Starting the random number genrator from the beginning again? 
    if reseed
        if (mod.rng == Random.GLOBAL_RNG) error("Cannot reset seed for global RNG") end
         Random.seed!(mod.rng, mod.seed) 
    end
    p = size(mod.Φ)[1]
    iters = N + p + mod.burnin
    y = zeros(iters)
    y[1:p] = starting_values 
    for n=(p+1):iters
        y[n] = y[(n-p):(n-1)]'*mod.Φ + rand(mod.rng, mod.e_dist)
    end
    return y[(end-N+1):end] 
end

function theoretical_dist(mod::AR, data::Vector{Float64})
    # This return the theoretical distribution at each point t given 
    # information up until time t-1 
    # note that the theoretical distribution can 
    # only be provided for observations from p+1 onwards
    p = length(mod.Φ)
    mu = zeros(length(data)-p)
    v = zeros(length(data)-p)
    for t=(p+1):length(data)
        mu[t-p] = data[t-p:t-1]'*mod.Φ
        v[t-p] = var(mod.e_dist)
    end
    return mu, v
end