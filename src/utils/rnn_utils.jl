"""
    to_RNN_format(sequences::Vector{Matrix{T}}) 

Transform a data into a format used for Recurrent Networks 

Recurrent Networks need a Vecotor of Matrices where each matrix has the format of 
`features` x `num_sequences` and the vector has length `len_seq`. 

# Arguments
- `sequences`: A Vecotor of Matrices where each matrix is of the form `n_features` x `len_seq`

"""
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

"""
    to_RNN_format(tensor::Array{T, 3})

Transform a tensor of data into a format used by Recurrent Networks

This function is usually only needed for BFluxR
"""
function to_RNN_format(tensor::Array{T, 3}) where {T}
    # R cannot handle vectors of matrices, so R will communicate using tensors
    # these must be retransformed into a vector of sequences. 
    return [tensor[:,:,i] for i=1:size(tensor, 3)]
end