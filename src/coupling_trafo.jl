
# struct RQSplineCoupling <: Function
#     nn::Chain
# end

# function RQSplineCoupling(dims::Integer, K::Integer)
#     d = round(Integer, dims / 2)

#     hidden = dims * 10

#     l1 = Dense(d => hidden, relu)
#     l2 = Dense(hidden => hidden, relu)
#     l3 = Dense(hidden => 3K-1)
#     return RQSplineCoupling(Chain(l1, l2, l3))
# end

# export RQSplineCoupling
# @functor RQSplineCoupling

# struct RQSplineCouplingInv <: Function
#     nn::Chain
# end

# export RQSplineCouplingInv
# @functor RQSplineCouplingInv

# Base.:(==)(a::RQSplineCoupling, b::RQSplineCoupling) = a.nn == b.nn

# Base.isequal(a::RQSplineCoupling, b::RQSplineCoupling) = isequal(a.nn, b.nn)

# Base.hash(x::RQSplineCoupling, h::UInt) =  hash(x.nn, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h)))

# (f::RQSplineCoupling)(x::AbstractMatrix{<:Real}) = forward_coupling(f, x)[1]

# function ChangesOfVariables.with_logabsdet_jacobian(
#     f::RQSplineCoupling,
#     x::AbstractMatrix{<:Real}
# )
#     return forward_coupling(f, x)
# end

# function InverseFunctions.inverse(f::RQSplineCoupling)
#     return RQSplineCouplingInv(f.nn)
# end

# Base.:(==)(a::RQSplineCouplingInv, b::RQSplineCouplingInv) = a.nn == b.nn

# Base.isequal(a::RQSplineCouplingInv, b::RQSplineCouplingInv) = isequal(a.nn, b.nn)

# Base.hash(x::RQSplineCouplingInv, h::UInt) = hash(x.nn, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h)))

# (f::RQSplineCouplingInv)(x::AbstractMatrix{<:Real}) = backward_coupling(f, x)[1]

# function ChangesOfVariables.with_logabsdet_jacobian(
#     f::RQSplineCouplingInv,
#     x::AbstractMatrix{<:Real}
# )
#     return backward_coupling(f, x)
# end

# function InverseFunctions.inverse(f::RQSplineCouplingInv)
#     return RQSplineCoupling(f.weights, f.bias)
# end


# function get_RQSCoupling_trafo(dims::Integer, N::Integer = 1, K::Integer = 20)
#     layers = Function[]
#     for i in 1:2N
#         push!(layers, RQSplineCoupling(dims, K))
#     end
#     return fchain(layers)
# end

# export get_RQSCoupling_trafo

# function forward_coupling(coup_trafo::RQSplineCoupling, x::AbstractMatrix{<:Real})
#     b = round(Int, size(x,1)/2)
#     N = size(x,2)

#     x₂ = x[1:b, :]'
#     x₁ = x[b+1:end, :]

#     # x₁ = x[1:b, :]
#     # x₂ = x[b+1:end, :]'

#     θ = coup_trafo.nn(x₁)
#     K = Int((size(θ,1) + 1) / 3)
    
#     w, h, d = _get_params(θ, K, N)

#     Spline = RQSpline(w,h,d)
#     y₁ = x₁'
#     y₂, LogJac₂ = spline_forward(Spline, x₂)

#     return hcat(y₁, y₂)', LogJac₂
# end

# export forward_coupling

# function backward_coupling(coup_trafo::RQSplineCouplingInv, y::AbstractMatrix{<:Real})
#     b = round(Int, size(y,1)/2)
#     N = size(y,2)

#     # y₂ = y[1:b, :]'
#     # y₁ = y[b+1:end, :]

#     y₁ = y[1:b, :]
#     y₂ = y[b+1:end, :]'

#     θ = coup_trafo.nn(y₁)
#     K = Int((size(θ,1) + 1) / 3)

#     w, h, d = _get_params(θ, K, N)

#     Spline = RQSplineInv(w,h,d)
#     x₁, LogJac₁ = spline_backward(Spline, y₂)
#     x₂ = y₁'

#     return hcat(x₁, x₂)', LogJac₁
# end

# export backward_coupling



# function _get_params(θ::AbstractMatrix, K::Integer, N::Integer)
#     w = _cumsum(_softmax(θ[1:K,:]'))
#     h = _cumsum(_softmax(θ[K+1:2K,:]'))
#     d = _softplus(θ[2K+1:end,:]')

#     w = hcat(repeat([-5,], N,1), w)
#     h = hcat(repeat([-5,], N,1), h)
#     d = hcat(repeat([1,], N,1), d)
#     d = hcat(d, repeat([1,], N,1))

#     return w, h, d
# end
