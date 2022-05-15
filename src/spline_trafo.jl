const KERNEL_DEVICE = CPU()
const KERNEL_SIZE = 16

struct RationalQuadratic{R<:AbstractMatrix{<:Real}} <: Function
    widths::R
    heights::R
    derivatives::R
end

export RationalQuadratic
@functor RationalQuadratic

Base.:(==)(a::RationalQuadratic, b::RationalQuadratic) = a.widths == b.widths && a.heights == b.heights && a.derivatives == b.derivatives

Base.isequal(a::RationalQuadratic, b::RationalQuadratic) = isequal(a.widths, b.widths) && isequal(a.heights, b.heights) && isequal(a.derivatives, b.derivatives)

Base.hash(x::RationalQuadratic, h::UInt) = hash(x.widths, hash(x.heights, hash(x.derivatives, hash(:RationalQuadratic, hash(:EuclidianNormalizingFlows, h)))))

(f::RationalQuadratic)(x::AbstractMatrix{<:Real}) = spline_forward(f.widths, f.heights, f.derivatives, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RationalQuadratic,
    x::AbstractMatrix{<:Real}
)
    return spline_forward(f.widths, f.heights, f.derivatives, x)
end

function InverseFunctions.inverse(f::RationalQuadratic)
    return RationalQuadraticInv(f.widths, f.heights, f.derivatives)
end


struct RationalQuadraticInv{R<:AbstractMatrix{<:Real}} <: Function
    widths::R
    heights::R
    derivatives::R
end

@functor RationalQuadraticInv
export RationalQuadraticInv

Base.:(==)(a::RationalQuadraticInv, b::RationalQuadraticInv) = a.widths == b.widths && a.heights == b.heights && a.derivatives == b.derivatives

Base.isequal(a::RationalQuadraticInv, b::RationalQuadraticInv) = isequal(a.widths, b.widths) && isequal(a.heights, b.heights) && isequal(a.derivatives, b.derivatives)

Base.hash(x::RationalQuadraticInv, h::UInt) = hash(x.widths, hash(x.heights, hash(x.derivatives, hash(:RationalQuadraticInv, hash(:EuclidianNormalizingFlows, h)))))

(f::RationalQuadraticInv)(x::AbstractMatrix{<:Real}) = spline_backward(f.widths, f.heights, f.derivatives, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RationalQuadraticInv,
    x::AbstractMatrix{<:Real}
)
    return f(x)
end

function InverseFunctions.inverse(f::RationalQuadraticInv)
    return RationalQuadratic(f.widths, f.heights, f.derivatives)
end


function _compute_vals_forw(x::Real, w_k::Real, w::Real, h_k::Real, h_kplus1::Real, Δy::Real, s::Real, d_k::Real, d_kplus1::Real)

    ξ = (x - w_k) / w
    # Eq. (14) from [1]
    numerator = Δy * (s * ξ^2 + d_k * ξ * (1 - ξ))
    numerator_jl = s^2 * (d_kplus1 * ξ^2 + 2 * s * ξ * (1 - ξ) + d_k * (1 - ξ)^2)

    denominator = s + (d_kplus1 + d_k - 2s) * ξ * (1 - ξ)

    g = h_k + numerator / denominator
    logjac = log(numerator_jl) - 2 * log(denominator)

    return (g, logjac)
end

function _compute_params(
        widths::AbstractVector{<:Real}, 
        heights::AbstractVector{<:Real}, 
        derivatives::AbstractVector{<:Real}, 
        T::DataType, 
        x::Real, 
        k::Real, 
        K::Real
    ) 
    
    # Width
    # If k == 0 put it in the bin `[-B, widths[1]]`
    w_k = widths[k] #(k == 0) ? -widths[end] :
    w = widths[k + 1] - w_k

    # Height
    h_k = heights[k] # (k == 0) ? -heights[end] : 
    h_kplus1 = (k == K) ? heights[end] : heights[k + 1]

    # Slope 
    Δy = heights[k + 1] - h_k
    s = Δy / w

    # Derivatives
    d_k = derivatives[k] #(k == 0) ? one(T) :
    d_kplus1 = (k == K - 1) ? one(T) : derivatives[k + 1]

    return [x, w_k, w, h_k, h_kplus1, Δy, s, d_k, d_kplus1]
end

function spline_forward(
        width::AbstractMatrix{<:Real}, 
        height::AbstractMatrix{<:Real}, 
        deriv::AbstractMatrix{<:Real}, 
        X::AbstractMatrix{<:Real}
    )
    
    Y = zeros(size(X))
    LADJ = zeros(size(X))
    
    kernel! = spline_forward_kernel!(KERNEL_DEVICE, KERNEL_SIZE)
    
    ev = kernel!(Y, LADJ, width, height, deriv, X, ndrange=size(X)) 
    
    wait(ev)
    
    return Y, LADJ
end

@kernel function spline_forward_kernel!(
        Y::AbstractMatrix{<:Real},
        LADJ::AbstractMatrix{<:Real}, 
        widths::AbstractMatrix{<:Real}, 
        heights::AbstractMatrix{<:Real}, 
        derivs::AbstractMatrix{<:Real}, 
        X::AbstractMatrix{<:Real}
    )
    
    i, j = @index(Global, NTuple)
    
    T = promote_type(eltype(widths[i,1]), eltype(heights[i,1]), eltype(derivs[i,1]), eltype(X[i,j]))
    
    K = length(widths[i,:]) 
    k = searchsortedfirst(widths[i,:], X[i,j]) - 1 
    
    
    if (k >= K) || (k == 0)
        Y[i,j] = X[i,j]
        LADJ[i,j] = zero(typeof(X[i,j]))
    else
        params = _compute_params(widths[i,:], heights[i,:], derivs[i,:], T, X[i,j], k, K)
        Y[i,j], LADJ[i,j] = _compute_vals_forw(params...)
        
    end
end

function spline_backward(
        width::AbstractMatrix{<:Real}, 
        height::AbstractMatrix{<:Real}, 
        deriv::AbstractMatrix{<:Real}, 
        X::AbstractMatrix{<:Real}
    )
    
    Y = zeros(size(X))
    LADJ = zeros(size(X))
    
    kernel! = spline_backward_kernel!(KERNEL_DEVICE, KERNEL_SIZE)
    
    ev = kernel!(Y, LADJ, width, height, deriv, X, ndrange=size(X)) 
    
    wait(ev)
    
    return Y, LADJ
end

@kernel function spline_backward_kernel!(
        Y::AbstractMatrix{<:Real},
        LADJ::AbstractMatrix{<:Real}, 
        widths::AbstractMatrix{<:Real}, 
        heights::AbstractMatrix{<:Real}, 
        derivs::AbstractMatrix{<:Real}, 
        X::AbstractMatrix{<:Real}
    )
    
    i, j = @index(Global, NTuple)
    
    T = promote_type(eltype(widths[i,1]), eltype(heights[i,1]), eltype(derivs[i,1]), eltype(X[i,j]))
    
    K = length(widths[i,:]) 
    k = searchsortedfirst(heights[i,:], X[i,j]) - 1 
    
    
    if (k >= K) || (k == 0)
        Y[i,j] = X[i,j]
        LADJ[i,j] = zero(typeof(X[i,j]))
    else
        params = _compute_params(widths[i,:], heights[i,:], derivs[i,:], T, X[i,j], k, K)
        ytransf, _ = _compute_vals_bw(params...)
    
        params = _compute_params(widths[i,:], heights[i,:], derivs[i,:], T, ytransf, k, K)
        _, logjac = _compute_vals_forw(params...)
        
        Y[i,j] = ytransf
        LADJ[i,j] = -logjac
    end
    
end

function _compute_vals_bw(x::Real, w_k::Real, w::Real, h_k::Real, h_kplus1::Real, Δy::Real, s::Real, d_k::Real, d_kplus1::Real)

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
