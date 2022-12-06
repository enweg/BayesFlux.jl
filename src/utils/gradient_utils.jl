using LinearAlgebra

"""
Clip the norm of the gradient.
"""
function clip_gradient!(g; maxnorm=5.0f0)
    ng = norm(g)
    g .= ng > maxnorm ? maxnorm .* g ./ ng : g
    return g
end