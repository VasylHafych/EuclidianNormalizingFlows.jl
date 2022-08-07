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

function dim_flip(x::Abstractmatrix)
    e = size(x,1)
    negs = collect(1:e)
    negs .-= 1
return vcat([x[e-i,:] for i in negs]), 1

function bu()
    println("ju")
end
export bu
