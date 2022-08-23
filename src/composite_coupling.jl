
struct RQSplineCoupling <: Function
    nn::Chain
end

function RQSplineCoupling(;K::Integer, d::Integer)
    l1 = Dense(d => 20, relu)
    l2 = Dense(20 => 20, relu)
    l3 = Dense(20 => 3K-1)
    return RQSplineCoupling(Chain(l1, l2, l3))
end

struct RQSplineDoubleCoupling <: Function
    nn1::Chain
    nn2::Chain
end

function RQSplineDoubleCoupling(;K::Integer, d::Integer, n_dims::Integer)
    nn1 = Chain(Dense(d => 20, relu),
                Dense(20 => 20, relu),
                Dense(20 => 3K-1))
    
    nn2 = Chain(Dense(n_dims - d => 20, relu),
                Dense(20 => 20, relu),
                Dense(20 => 3K-1))
    return RQSplineDoubleCoupling(nn1, nn2)
end
struct RQSplineCouplingOnepass <: Function
    nn::Chain
    RQS::TrainableRQSpline
end

function RQSplineCouplingOnepass(;K::Integer, d::Integer, n_smpls::Integer)
    l1 = Dense(d => 20, relu)
    l2 = Dense(20 => 20, relu)
    l3 = Dense(20 => 3K-1)
    nn = Chain(l1, l2, l3)

    w = ones(n_smpls, K)
    h = ones(n_smpls, K)
    d = ones(n_smpls, K -1)
    return RQSplineCouplingOnepass(nn, TrainableRQSpline(w,h,d))
end

function RQSplineCoupling(weights::AbstractMatrix, bias::AbstractVector)
    nn = Dense(weights, bias)
    return RQSplineCoupling(nn)
end

# struct RQSplineCoupling <: Function
#     weights::AbstractMatrix{<:Real}
#     bias::AbstractVector{<:Real}
#     train_trafo::TrainableRQSpline
# end

# function RQSplineCoupling(K::Integer, d::Integer, n_smpls::Integer)
#     weights = zeros(d * (3K-1), d * n_smpls)
#     bias = zeros(d * (3K-1))
#     train_trafo = TrainableRQSpline(ones(d,K), ones(d,K), ones(d,K-1))
#     return RQSplineCoupling(weights, bias, train_trafo)
# end

export RQSplineCoupling
@functor RQSplineCoupling

export RQSplineDoubleCoupling
@functor RQSplineDoubleCoupling

export RQSplineCouplingOnepass
@functor RQSplineCouplingOnepass

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

(f::RQSplineDoubleCoupling)(x::AbstractMatrix{<:Real}) = forward_coupling(f, x)[1]

(f::RQSplineCouplingOnepass)(x::AbstractMatrix{<:Real}) = forward_coupling(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineCoupling,
    x::AbstractMatrix{<:Real}
)
    return forward_coupling(f, x)
end

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineDoubleCoupling,
    x::AbstractMatrix{<:Real}
)
    return forward_coupling(f, x)
end

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineCouplingOnepass,
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

function forward_coupling(coup_trafo::RQSplineCouplingOnepass, x::AbstractMatrix{<:Real})
    b = round(Int, size(x,1)/2)
    N = size(x,2)

    # x₂ = x[1:b, :]'
    # x₁ = x[b+1:end, :]

    x₁ = x[1:b, :]
    x₂ = x[b+1:end, :]'

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

    y₁, LogJac₁ = spline_forward(coup_trafo.RQS, x₁')
    y₂, LogJac₂ = spline_forward(Spline, x₂)

    return hcat(y₁, y₂)', LogJac₁ + LogJac₂
end

function forward_coupling(coup_trafo::RQSplineDoubleCoupling, x::AbstractMatrix{<:Real})
    b = round(Int, size(x,1)/2)
    N = size(x,2)

    x₁ = x[1:b, :]
    x₂ = x[b+1:end, :]

    θ1 = coup_trafo.nn1(x₂)
    θ2 = coup_trafo.nn2(x₁)
    K = Int((size(θ1,1) + 1) / 3)

    Spline1 = get_spline(θ1, K, N)
    Spline2 = get_spline(θ2, K, N)

    y₁, LogJac₁ = spline_forward(Spline1, x₁')
    y₂, LogJac₂ = spline_forward(Spline2, x₂')

    return hcat(y₁, y₂)', LogJac₁ + LogJac₂
end

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

# function forward_coupling(coup_trafo::RQSplineCoupling, x::AbstractMatrix{<:Real})
#     d = round(Int, size(x,1)/2)
#     x₁ = x[1:d, :]
#     x₂ = x[d+1:end, :]
#     x₁_flat = vec(x₁)
#     K = Int((length(coup_trafo.bias) + 1) / 3)

#     norm = mean(x₁_flat) * length(x₁_flat)  # doesn't work without this, because otherwise the 
#                                             # elements of the result of the muladd become gigantic, and the learned 
#                                             # weigths and bias are NaNs for large numbers of samples

#     θ = _sigmoid(muladd(coup_trafo.weights, x₁_flat ./ norm , coup_trafo.bias))

#     par_shapes = NamedTupleShape(w = ArrayShape{Real}(d, K),
#                                  h = ArrayShape{Real}(d, K),
#                                  d = ArrayShape{Real}(d, K-1)
#                                  )
#     params = par_shapes(θ)

#     RQS_trafo = TrainableRQSpline(params.w, params.h, params.d)
#     y₁, LogJac₁ = spline_forward(coup_trafo.train_trafo, x₁)
#     y₂, LogJac₂ = spline_forward(RQS_trafo, x₂)
    
#     return vcat(y₁, y₂), LogJac₁ + LogJac₂
# end
# export forward_coupling
