
struct RQSplineCoupling <: Function
    weights::AbstractMatrix{<:Real}
    bias::AbstractVector{<:Real}
    train_trafo::TrainableRQSpline
end

function RQSplineCoupling(K::Integer, d::Integer, n_smpls::Integer)
    weights = zeros(d * (3K-1), d * n_smpls)
    bias = zeros(d * (3K-1))
    train_trafo = TrainableRQSpline(ones(d,K), ones(d,K), ones(d,K-1))
    return RQSplineCoupling(weights, bias, train_trafo)
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
    d = round(Int, size(x,1)/2)
    x₁ = x[1:d, :]
    x₂ = x[d+1:end, :]
    x₁_flat = vec(x₁)
    K = Int((length(coup_trafo.bias) + 1) / 3)

    norm = mean(x₁_flat) * length(x₁_flat)  # doesn't work without this, because otherwise the 
                                            # elements of the result of the muladd become gigantic, and the learned 
                                            # weigths and bias are NaNs for large numbers of samples

    θ = _sigmoid(muladd(coup_trafo.weights, x₁_flat ./ norm , coup_trafo.bias))

    par_shapes = NamedTupleShape(w = ArrayShape{Real}(d, K),
                                 h = ArrayShape{Real}(d, K),
                                 d = ArrayShape{Real}(d, K-1)
                                 )
    params = par_shapes(θ)

    RQS_trafo = TrainableRQSpline(params.w, params.h, params.d)
    y₁, LogJac₁ = spline_forward(coup_trafo.train_trafo, x₁)
    y₂, LogJac₂ = spline_forward(RQS_trafo, x₂)
    
    return vcat(y₁, y₂), LogJac₁ + LogJac₂
end
export forward_coupling
