# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

struct CouplingRQSpline <: Function
    nn1::Chain
    nn2::Chain
end

function CouplingRQSpline(n_dims::Integer, K::Integer = 20)
    nn1, nn2 = _get_nns(n_dims, K)
    return CouplingRQSpline(nn1, nn2)
end

# function CouplingRQSpline(w1::AbstractMatrix, w2::AbstractMatrix, K::Integer = 20)

#     nn1 = Chain(Dense(w1, true, relu),
#                 Dense(20 => 20, relu),
#                 Dense(20 => (3K-1))
#                 )

#     nn2 = Chain(Dense(w2, true, relu),
#                 Dense(20 => 20, relu),
#                 Dense(20 => (3K-1))
#                 )
    
#     return CouplingRQSpline(nn1, nn2)
# end

export CouplingRQSpline
@functor CouplingRQSpline

Base.:(==)(a::CouplingRQSpline, b::CouplingRQSpline) = a.nn1 == b.nn1 &&  a.nn2 == b.nn1

Base.isequal(a::CouplingRQSpline, b::CouplingRQSpline) = isequal(a.nn1, b.nn1)  && isequal(a.nn2, b.nn2)

Base.hash(x::CouplingRQSpline, h::UInt) =  hash(x.nn1, hash(x.nn2, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h))))

(f::CouplingRQSpline)(x::AbstractMatrix{<:Real}) = coupling_trafo(f, x)[1]


struct CouplingRQSplineInv <: Function
    nn1::Chain
    nn2::Chain
end

export CouplingRQSplineInv
@functor CouplingRQSplineInv

Base.:(==)(a::CouplingRQSplineInv, b::CouplingRQSplineInv) = a.nn1 == b.nn1 &&  a.nn2 == b.nn1

Base.isequal(a::CouplingRQSplineInv, b::CouplingRQSplineInv) = isequal(a.nn1, b.nn1)  && isequal(a.nn2, b.nn2)

Base.hash(x::CouplingRQSplineInv, h::UInt) = hash(x.nn1, hash(x.nn2, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h))))

(f::CouplingRQSplineInv)(x::AbstractMatrix{<:Real}) = coupling_trafo(f, x)[1]


function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQSpline,
    x::AbstractMatrix{<:Real}
)
    return coupling_trafo(f, x)
end

function InverseFunctions.inverse(f::CouplingRQSpline)
    return CouplingRQSplineInv(f.nn1, f.nn2)
end


function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQSplineInv,
    x::AbstractMatrix{<:Real}
)
    return coupling_trafo(f, x)
end

function InverseFunctions.inverse(f::CouplingRQSplineInv)
    return CouplingRQSpline(f.nn1, f.nn2)
end


function coupling_trafo(trafo::Union{CouplingRQSpline, CouplingRQSplineInv}, x::AbstractMatrix{<:Real})
    b = round(Int, size(x,1)/2)
    inv = trafo isa CouplingRQSplineInv 

    x₁ = x[1:b, :]
    x₂ = x[b+1:end, :]

    y₁, LogJac₁ = partial_coupling_trafo(trafo.nn1, x₁, x₂, inv)
    y₂, LogJac₂ = partial_coupling_trafo(trafo.nn2, x₂, y₁, inv)

    return vcat(y₁, y₂), LogJac₁ + LogJac₂
end

export coupling_trafo

function partial_coupling_trafo(nn::Chain, x₁::AbstractMatrix{<:Real}, x₂::AbstractMatrix{<:Real}, inv::Bool)
    N = size(x₁,2)

    θ = nn(x₂)
    K = Int((size(θ,1) + 1) / 3)
    w, h, d = get_params(θ, N, K)
    spline = inv ? RQSplineInv(w, h, d) : RQSpline(w, h, d)

    return with_logabsdet_jacobian(spline, x₁)
end

export partial_coupling_trafo
