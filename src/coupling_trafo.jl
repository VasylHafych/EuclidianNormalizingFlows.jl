
struct RQSplineCoupling <: Function
    weights::AbstractMatrix{<:Real}
    bias::AbstractVector{<:Real}
end

export RQSplineCoupling
@functor RQSplineCoupling

struct RQSplineCouplingInv <: Function
    weights::AbstractMatrix{<:Real}
    bias::AbstractVector{<:Real}
end

export RQSplineCouplingInv
@functor RQSplineCouplingInv

Base.:(==)(a::RQSplineCoupling, b::RQSplineCoupling) = a.weights == b.weights &&  a.bias == b.bias

Base.isequal(a::RQSplineCoupling, b::RQSplineCoupling) = isequal(a.weights, b.weights)  && isequal(a.bias, b.bias)

Base.hash(x::RQSplineCoupling, h::UInt) =  hash(x.weights, hash(x.bias, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h))))

(f::RQSplineCoupling)(x::AbstractMatrix{<:Real}) = forward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineCoupling,
    x::AbstractMatrix{<:Real}
)
    return forward(f, x)
end

function InverseFunctions.inverse(f::RQSplineCoupling)
    return RQSplineCouplingInv(f.weights)
end

Base.:(==)(a::RQSplineCouplingInv, b::RQSplineCouplingInv) = a.weights == b.weights  &&  a.bias == b.bias

Base.isequal(a::RQSplineCouplingInv, b::RQSplineCouplingInv) = isequal(a.weights, b.weights)  && isequal(a.bias, b.bias)

Base.hash(x::RQSplineCouplingInv, h::UInt) = hash(x.weights, hash(x.bias, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h))))

(f::RQSplineCouplingInv)(x::AbstractMatrix{<:Real}) = backward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineCouplingInv,
    x::AbstractMatrix{<:Real}
)
    return backward(f, x)
end

function InverseFunctions.inverse(f::RQSplineCouplingInv)
    return TrainableRQSpline(f.weights)
end

# To do: Implement forward pass with Dense Layer: 
# function forward(trafo::RQSplineCoupling, x::AbstractMatrix{<:Real})
#     ind_range = 1:1
#     k = 8
#     x_flat = [x[ind_range, :]...]
#     forward_pass = _softplus(trafo.weights * x_flat + trafo.bias)
# end