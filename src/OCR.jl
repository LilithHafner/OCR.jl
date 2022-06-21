module OCR

using Images, BasicTextRender

export training_data

function training_data(n=512, m=500, textsize=10)
    img = rand(Gray{Float64},n,n) ./ 2 .+ .5

    chars = vcat('a':'z', 'A':'Z', '0':'9')

    locs = []
    n0 = n - ceil(Int, ℯ * textsize)
    for _ in 1:m
        xy = rand(1:n0), rand(1:n0)
        _textsize = round(Int, textsize*exp(rand()))
        _textsize == 23 && continue # https://github.com/IanButterworth/BasicTextRender.jl/issues/2
        overlay = overlaytext!(img, string(rand(chars)), _textsize, xy)
        push!(locs, xy .+ size(overlay) ./ 2)
    end
    img    
end

end
