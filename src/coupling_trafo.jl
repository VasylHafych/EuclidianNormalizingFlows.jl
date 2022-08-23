
struct RQSplineCoupling <: Function
    nn::Chain
end

function RQSplineCoupling(;K::Integer, d::Integer)
    l1 = Dense(d => 20, relu)
    l2 = Dense(20 => 20, relu)
    l3 = Dense(20 => 3K-1)
    return RQSplineCoupling(Chain(l1, l2, l3))
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

(f::RQSplineCoupling)(x::AbstractMatrix{<:Real}) = forward_coupling(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineCoupling,
    x::AbstractMatrix{<:Real}
)
    return forward_coupling(f, x)
end

function InverseFunctions.inverse(f::RQSplineCoupling)


    
    return RQSplineCouplingInv(f.weights, f.bias)
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
    return RQSplineCoupling(f.weights, f.bias)
end


function forward_coupling(coup_trafo::RQSplineCoupling, x::AbstractMatrix{<:Real})
    b = round(Int, size(x,1)/2)
    N = size(x,2)

    x₂ = x[1:b, :]'
    x₁ = x[b+1:end, :]

    # x₁ = x[1:b, :]
    # x₂ = x[b+1:end, :]'

    θ = coup_trafo.nn(x₁)

    K = Int((size(θ,1) + 1) / 3)

    w = _cumsum(_softmax(θ[1:K,:]'))
    h = _cumsum(_softmax(θ[K+1:2K,:]'))
    d = _softplus(θ[2K+1:end,:]')

    w = hcat(repeat([-5,], N,1), w)
    h = hcat(repeat([-5,], N,1), h)
    d = hcat(repeat([1,], N,1), d)
    d = hcat(d, repeat([1,], N,1))

    Spline = RQSpline(w,h,d)
    y₁ = x₁'
    y₂, LogJac₂ = spline_forward(Spline, x₂)

    return hcat(y₁, y₂)', LogJac₂
end
export forward_coupling

function get_spline(θ::AbstractMatrix, K::Integer, N::Integer)
    w = _cumsum(_softmax(θ[1:K,:]'))
    h = _cumsum(_softmax(θ[K+1:2K,:]'))
    d = _softplus(θ[2K+1:end,:]')

    w = hcat(repeat([-5,], N,1), w)
    h = hcat(repeat([-5,], N,1), h)
    d = hcat(repeat([1,], N,1), d)
    d = hcat(d, repeat([1,], N,1))

    return RQSpline(w,h,d)
end
