abstract type SplineTrafo end

struct RationalQuadratic{P<:Union{Real,AbstractVector{<:Real}}} <: SplineTrafo
    widths::P
    heights::P
    derivatives::P
end

export RationalQuadratic
@functor RationalQuadratic

Base.:(==)(a::RationalQuadratic, b::RationalQuadratic) = a.widths == b.widths && a.heights == b.heights && a.derivatives == b.derivatives

Base.isequal(a::RationalQuadratic, b::RationalQuadratic) = isequal(a.widths, b.widths) && isequal(a.heights, b.heights) && isequal(a.derivatives, b.derivatives)

Base.hash(x::RationalQuadratic, h::UInt) = hash(x.widths, hash(x.heights, hash(x.derivatives, hash(:RationalQuadratic, hash(:EuclidianNormalizingFlows, h)))))

(f::RationalQuadratic)(x::Real) = _spline_transform(f, x)[1]
(f::RationalQuadratic)(x::AbstractMatrix{<:Real}) = map(x->f(x...), x)

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RationalQuadratic,
    x::AbstractMatrix{<:Real}
)
    res = map(x->_spline_transform(f, x), x)
    xtr = map(x->x[1],res)
    ladj =  map(x->sum(map(x->x[2],x)) ,eachcol(res))
    
    return xtr, ladj'
end

function InverseFunctions.inverse(f::RationalQuadratic)
    return RationalQuadraticInv(f.widths, f.heights, f.derivatives)
end


# **** Inverse (to do: Add log Jacobian)


struct RationalQuadraticInv{P<:Union{Real,AbstractVector{<:Real}}} <: SplineTrafo
    widths::P
    heights::P
    derivatives::P
end

@functor RationalQuadraticInv

Base.:(==)(a::RationalQuadraticInv, b::RationalQuadraticInv) = a.widths == b.widths && a.heights == b.heights && a.derivatives == b.derivatives

Base.isequal(a::RationalQuadraticInv, b::RationalQuadraticInv) = isequal(a.widths, b.widths) && isequal(a.heights, b.heights) && isequal(a.derivatives, b.derivatives)

Base.hash(x::RationalQuadraticInv, h::UInt) = hash(x.widths, hash(x.heights, hash(x.derivatives, hash(:RationalQuadraticInv, hash(:EuclidianNormalizingFlows, h)))))

(f::RationalQuadraticInv)(x::Real) = _spline_transform(f, x)[1]
(f::RationalQuadraticInv)(x::AbstractMatrix{<:Real}) = map(x->f(x...), x)

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RationalQuadraticInv,
    x::AbstractMatrix{<:Real}
)
    res = map(x->_spline_transform(f, x), x)
    xtr = map(x->x[1],res)
    ladj =  map(x->sum(map(x->x[2],x)) ,eachcol(res))
    
    return xtr, nothing
end

function InverseFunctions.inverse(f::RationalQuadraticInv)
    return RationalQuadratic(f.widths, f.heights, f.derivatives)
end

# Utills 

function _searchsortedfirst(trafo::RationalQuadratic, x::Real) 
    return searchsortedfirst(trafo.widths, x) - 1 
end

function _searchsortedfirst(trafo::RationalQuadraticInv, x::Real) 
    return searchsortedfirst(trafo.heights, x) - 1 
end

function _compute_params(trafo, T, x, k, K)
    # Width
    # If k == 0 put it in the bin `[-B, widths[1]]`
    w_k = trafo.widths[k] #(k == 0) ? -widths[end] :
    w = trafo.widths[k + 1] - w_k

    # Height
    h_k = trafo.heights[k] # (k == 0) ? -heights[end] : 
    h_kplus1 = (k == K) ? trafo.heights[end] : trafo.heights[k + 1]
    
    # Slope 
    Δy = trafo.heights[k + 1] - h_k
    s = Δy / w

    # Derivatives
    d_k = trafo.derivatives[k] #(k == 0) ? one(T) :
    d_kplus1 = (k == K - 1) ? one(T) : trafo.derivatives[k + 1]
    
    return [x, w_k, w, h_k, h_kplus1, Δy, s, d_k, d_kplus1]
end

function _compute_vals(trafo::RationalQuadratic, x::Real, w_k::Real, w::Real, h_k::Real, h_kplus1::Real, Δy::Real, s::Real, d_k::Real, d_kplus1::Real)

    ξ = (x - w_k) / w
    # Eq. (14) from [1]
    numerator = Δy * (s * ξ^2 + d_k * ξ * (1 - ξ))
    numerator_jl = s^2 * (d_kplus1 * ξ^2 + 2 * s * ξ * (1 - ξ) + d_k * (1 - ξ)^2)

    denominator = s + (d_kplus1 + d_k - 2s) * ξ * (1 - ξ)

    g = h_k + numerator / denominator
    logjac = log(numerator_jl) - 2 * log(denominator)

    return (g, logjac)
end

function _compute_vals(trafo::RationalQuadraticInv, x::Real, w_k::Real, w::Real, h_k::Real, h_kplus1::Real, Δy::Real, s::Real, d_k::Real, d_kplus1::Real)

    ds = d_kplus1 + d_k - 2 * s

    # Eq.s (25) through (27) from [1]
    a1 = Δy * (s - d_k) + (x - h_k) * ds
    a2 = Δy * d_k - (x - h_k) * ds
    a3 = - s * (x - h_k)

    # Eq. (24) from [1]. There's a mistake in the paper; says `x` but should be `ξ`
    numerator = - 2 * a3
    denominator = (a2 + sqrt(a2^2 - 4 * a1 * a3))
    ξ = numerator / denominator

    g = ξ * w + w_k
    return (g, nothing)
end

function _spline_transform(trafo::ST, x::Real) where {ST<:SplineTrafo}
    
    widths = trafo.widths
    heights = trafo.heights
    derivatives = trafo.derivatives
    
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(x))

    # Number of bins K
    K = length(widths)
    
    k = _searchsortedfirst(trafo, x)

    # If x outside interval mask apply identity transform
    if (k >= K) || (k == 0)
        return (one(T) * x, zero(T) * x)
    end
            
    params = _compute_params(trafo, T, x, k, K)
    
    return _compute_vals(trafo, params...)
end   

