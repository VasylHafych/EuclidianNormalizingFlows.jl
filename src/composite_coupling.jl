# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

struct CouplingRQSpline <: Function
    nn1::Chain
    nn2::Chain
end

function CouplingRQSpline(n_dims::Integer, K::Integer = 20)
    nn1, nn2 = _get_nns(n_dims, K)
    return CouplingRQSpline(nn1, nn2)
end

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

Base.:(==)(a::RQSplineCouplingInv, b::RQSplineCouplingInv) = a.nn1 == b.nn1 &&  a.nn2 == b.nn1

Base.isequal(a::RQSplineCouplingInv, b::RQSplineCouplingInv) = isequal(a.nn1, b.nn1)  && isequal(a.nn2, b.nn2)

Base.hash(x::RQSplineCouplingInv, h::UInt) = hash(x.nn1, hash(x.nn2, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h))))

(f::RQSplineCouplingInv)(x::AbstractMatrix{<:Real}) = coupling_trafo(f, x)[1]


function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQSpline,
    x::AbstractMatrix{<:Real}
)
    return coupling_trafo(f, x)
end

function InverseFunctions.inverse(f::RQSplineCoupling)
    return CouplingRQSplineInv(f.nn1, f.nn2)
end


function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineCouplingInv,
    x::AbstractMatrix{<:Real}
)
    return coupling_trafo(f, x)
end

function InverseFunctions.inverse(f::RQSplineCouplingInv)
    return CouplingRQSpline(f.nn1, f.nn2)
end


function coupling_trafo(trafo::Union{CouplingRQSpline, CouplingRQSplineInv}, x::AbstractMatrix{<:Real})
    b = round(Int, size(x,1)/2)
    N = size(x,2)

    x₁ = x[1:b, :]
    x₂ = x[b+1:end, :]

    θ1 = trafo.nn1(x₂)
    K = Int((size(θ1,1) / b + 1) / 3)
    w1, h1, d1 = get_params(θ1, N, K)
    Spline1 = trafo isa CouplingRQSpline ? RQSpline(w1, h1, d1) : RQSplineInv(w1, h1, d1)
    y₁, LogJac₁ = spline_forward(Spline1, x₁)

    θ2 = trafo.nn2(y₁)
    w2, h2, d2 = get_params(θ2, N, K)
    Spline2 = trafo isa CouplingRQSpline ? RQSpline(w2, h2, d2) : RQSplineInv(w2, h2, d2)
    y₂, LogJac₂ = spline_forward(Spline2, x₂)

    return vcat(y₁, y₂), LogJac₁ + LogJac₂
end
