module OCR

using Images, StatsBase

using BasicTextRender: overlaytext!
using Random: shuffle

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
    target = rand(17, 4)
    network = weights([5, 15, 15, 4])
    for i in 1:10000
        prediction, history = forward(network, input)
        grad = .0001(target .- prediction)
        isinteger(log2(i)) && println(mean((target .- prediction).^2))
        backward!(network, history, grad)
    end
end

margin(source, x, y) = source[begin+x:end-x, begin+y:end-y]

function epoch(network; coords, img, dist, window, train, window_length, batch_size)
    loss = []
    live = []
    for i in firstindex(coords):batch_size:lastindex(coords)
        bs = min(batch_size, lastindex(coords)-i+1)
        input = Matrix{Float64}(undef, bs, window_length)
        target = Matrix{Float64}(undef, bs, 1)
        for (j, loc) in enumerate(coords[i:i+bs-1])
            input[j, :] = vec(img[ntuple(i->loc[i]-window[i]:loc[i]+window[i], 2)...])
            target[j, 1] = 1/(2+dist[loc])
        end

        prediction, history = forward(network, input)
        push!(loss, mean((target .- prediction).^2))
        push!(live, mean(hcat(history...) .> 0))

        if train
            grad = 1e-3(target .- prediction)
            backward!(network, history, grad)
        end
    end
    mean(loss), mean(live)
end

function train(epochs)
    window = (7,4)
    window_length = prod(window .* 2 .+ 1)

    network = weights([window_length, 200, 70, 1]) * .05
    batch_size = 200

    for _ in 1:epochs

        img, _, dist = training_data(256)
        coords = shuffle(vec(margin(eachindex(IndexCartesian(), img), window...)))
        println(epoch(network; coords, img, dist, window, batch_size, window_length, train=true))

    end

    network, (;window, window_length, batch_size)
end

function apply(network; img, window, window_length, batch_size)
    out = zeros(size(img)...)
    coords = margin(eachindex(IndexCartesian(), img), window...)
    for i in firstindex(coords):batch_size:lastindex(coords)
        bs = min(batch_size, lastindex(coords)-i+1)
        input = Matrix{Float64}(undef, bs, window_length)
        for (j, loc) in enumerate(coords[i:i+bs-1])
            input[j, :] = vec(img[ntuple(i->loc[i]-window[i]:loc[i]+window[i], 2)...])
        end
        prediction, _ = forward(network, input)
        out[coords[i:i+bs-1]] .= vec(prediction)
    end

    out
end

function blur(img)
    a = 2 * img
    a[begin+1:end, :] += img[begin:end-1, :]
    a[begin:end-1, :] += img[begin+1:end, :]
    out = 2 * a
    out[:, begin+1:end] += a[:, begin:end-1]
    out[:, begin:end-1] += a[:, begin+1:end]
    out ./= 16
end

function peaks(img)
    out = img .> quantile(vec(img), .8)
    out[begin+1:end, :] .&= img[begin+1:end, :] .>= img[begin:end-1, :]
    out[begin:end-1, :] .&= img[begin:end-1, :] .> img[begin+1:end, :]
    out[:, begin+1:end] .&= img[:, begin+1:end] .>= img[:, begin:end-1]
    out[:, begin:end-1] .&= img[:, begin:end-1] .> img[:, begin+1:end]
    out
end

function make_system(network, params)
    function f(img)
        apply(network; img, params...) |>
        blur |>
        blur |>
        peaks |>
        findall
    end
end

standard_training_data = training_data(256)
function test_system(system, source=standard_training_data; show_img=true)
    img, _, dists = source
    guess = system(img)
    show_img && display(overlay(img, guess))

    truth = findall(dists .== 0)
    evaluate(guess, truth)
end

function evaluate(guess, truth, proximity=(a,b)->sum((Tuple(a).-Tuple(b)).^2), threshold=5^2)
    hits = fill(0, axes(truth)...)
    false_positive = 0
    for g in guess
        i, t = argmin(((i, t),) -> proximity(g, t), enumerate(truth))
        dist = proximity(g, t)
        if dist <= threshold
            hits[i] += 1
        else
            false_positive += 1
        end
    end
    undetected = sum(hits .== 0)
    doublecount = sum(max.(0, hits .- 1))
    map(x -> x/length(truth), (;undetected, false_positive, doublecount))
end

function test_all(epochs; show_img=true)
    time = @elapsed system = make_system(train(epochs)...)
    out = (;time, test_system(system; show_img)...)
    map(x->round(1000x)/1000, out)
end

end # module
