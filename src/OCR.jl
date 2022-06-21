module OCR

using Images, BasicTextRender, StatsBase

export training_data, overlay

function training_data(n=512, font_size=10)
    n1 = n + ceil(Int, (1 + ℯ) * font_size)
    img = rand(Gray{Float64},n1,n1) ./ 2 .+ .5

    chars = vcat('a':'z', 'A':'Z', '0':'9')

    labels = similar(img, Char)
    distances = similar(img, Int)
    distances .= typemax(Int)
    for x in 1:1.5font_size:n
        for y in 1:1.5font_size:n
            xy = round(Int, x+rand(0:font_size)), round(Int, y+rand(0:font_size))
            char = rand(chars)
            _font_size = round(Int, font_size*exp(rand()))
            _font_size == 23 && continue # https://github.com/IanButterworth/BasicTextRender.jl/issues/2
            overlay = overlaytext!(img, string(char), _font_size, xy)
            center = xy .+ size(overlay) .÷ 2
            labels[center...] = char
            distances[center...] = 0
        end
    end

    img = img[1:n, 1:n]
    labels = labels[1:n, 1:n]
    distances = distances[1:n, 1:n]

    labels, distances = propagate!(labels, distances)

    img, labels, distances
end

# Way (asymptotically) slower than it needs to be,
# esp. if there are margins or a large font size
function propagate!(l, d0)
    d1 = copy(d0)
    changed = true
    while changed
        changed = false
        for i in eachindex(IndexCartesian(), l, d0)
            v = Tuple(i)
            for k in eachindex(v)
                for d in (-1, 1)
                    v2 = ntuple(j -> j == k ? v[j]+d : v[j], length(v))
                    if checkbounds(Bool, d0, v2...) && d0[v2...] < d1[i] - 1
                        l[i] = l[v2...]
                        d1[i] = d0[v2...] + 1
                        changed = true
                    end
                end
            end
        end
        d1, d0 = d0, d1
    end
    l, d0
end

function overlay(image, marks)
    out = similar(image, RGB{Float64})
    copyto!(out, image)
    out[marks] .= RGB(1.0,0.0,0.0)
    out
end

function test_data()
    img, lab, dist = training_data(512); OCR.overlay(img, dist .== 0)
end

weights(shape) = map(randn, shape[1:end-1], shape[2:end])

function forward(weights, x)
    history = similar(weights, typeof(x))
    for i in eachindex(weights)
        history[i] = x
        x = max.(0, x * weights[i])
    end
    x, history
end

function backward!(weights, history, grad)
    for i in reverse(eachindex(weights, history))
        weights[i] += transpose(history[i]) * grad
        grad = grad * transpose(weights[i])
        grad[history[i] .== 0] .= 0
    end
    grad # gradient with respect to input
end

function test_network()
    input = rand(17, 5)
    target = rand(17, 5)
    network = weights([5, 15, 15, 5])
    for i in 1:10000
        prediction, history = forward(network, input)
        grad = .0001(target .- prediction)
        isinteger(log2(i)) && println(mean((target .- prediction).^2))
        backward!(network, history, grad)
    end
end

end # module
