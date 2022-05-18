using BFlux
using Test

@testset "BFlux" begin
    include("laplace.jl")
    include("bbb.jl")
    include("modes.jl")
    include("ggmc.jl")
end