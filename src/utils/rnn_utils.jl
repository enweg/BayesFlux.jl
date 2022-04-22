function to_RNN_format(sequences::Vector{Matrix{T}}) where {T}
    # Assumes that each matrix is of the format n_features x len_seq 
    # We need a vector of matrices of format n_featurs x n_seq for RNN layers 
    n_seq = length(sequences)
    n_features, len_seq = size(sequences[1])
    out = [ones(n_features, n_seq) for _ in 1:len_seq]
    for t=1:len_seq 
        out[t] = hcat([seq[:,t] for seq in sequences]...)
    end
    return out
end

function to_RNN_format(tensor::Array{T, 3}) where {T}
    # R cannot handle vectors of matrices, so R will communicate using tensors
    # these must be retransformed into a vector of sequences. 
    return [tensor[:,:,i] for i=1:size(tensor, 3)]
end