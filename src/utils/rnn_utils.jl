"""
    make_rnn_tensor(m::Matrix{T}, seq_to_one_length = 10) where {T}

Create a Tensor used for RNNs 

Given an input matrix of dimensions timesteps×features transform it into a
Tensor of dimension timesteps×features×sequences where the sequences are
overlapping subsequences of length `seq_to_one_length` of the orignal
`timesteps` long sequence
"""
function make_rnn_tensor(m::Matrix{T}, seq_to_one_length=10) where {T}
    nfeatures = size(m, 2)
    nsequences = size(m, 1) - seq_to_one_length + 1

    tensor = Array{T}(undef, seq_to_one_length, nfeatures, nsequences)
    for i = 1:nsequences
        tensor[:, :, i] = m[i:i+seq_to_one_length-1, :]
    end

    return tensor
end