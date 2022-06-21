using OCR
using Test

@testset "OCR.jl" begin
    OCR.test_data()
    OCR.test_network()
    OCR.test_train(2)
end
