struct DimFlip <: Function
end

export DimFlip
@functor DimFlip

function ChangesOfVariables.with_logabsdet_jacobian(
    f::DimFlip,
    x::AbstractMatrix{<:Real}
)
    return dim_flip(x)
end

(f::DimFlip)(x::AbstractMatrix{<:Real}) = dim_flip(x)[1]

function dim_flip(x::AbstractMatrix)
    d = round(Int, size(x,1)/2)
    x₁ = x[1:d,:]
    x₂ = x[d+1:end,:]
    return vcat(x₂,x₁), ones(size(x,2))
end
