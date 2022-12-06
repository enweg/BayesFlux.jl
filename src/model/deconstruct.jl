


"""
    NetConstructor{T, F}

Used to construct a network from a vector.

The `NetConstructor` constains all important information to construct a network
like the original network from a given vector. 

# Fields 
- `num_params_net`: Number of network parameters 
- `θ`: Vector of network parameters of the original network
- `starts`: Vector containing the starting points of each layer in θ
- `ends`: Vector containing the end points of each layer in θ
- `reconstructors`: Vector containing the reconstruction functions for each
  layer

"""
struct NetConstructor{T,F}
    num_params_network::Int
    θ::Vector{T}
    starts::Vector{Int}
    ends::Vector{Int}
    reconstructors::Vector{F}
end

function (nc::NetConstructor{T})(θ::Vector{T}) where {T}
    layers = [re(θ[s:e]) for (re, s, e) in zip(nc.reconstructors, nc.starts, nc.ends)]
    return Flux.Chain(layers...)
end

"""
    destruct(net::Flux.Chain{T}) where {T}

Destruct a network

Given a `Flux.Chain` network, destruct it and create a NetConstructor. 
Each layer type must implement a destruct method taking the layer and returning
a vector containing the original layer parameters, and a function that given a
vector of the right length constructs a layer like the original using the
parameters given in the vector
"""
function destruct(net::Flux.Chain{T}) where {T}
    θre = [destruct(layer) for layer in net]
    θ = vcat([item[1] for item in θre]...)
    res = [item[2] for item in θre]
    s = ones(Int, length(θre))
    e = similar(s)
    for i in eachindex(θre)
        e[i] = s[i] + length(θre[i][1]) - 1
        if (i < length(θre))
            s[i+1] = e[i] + 1
        end
    end
    return NetConstructor(length(θ), θ, s, e, res)
end