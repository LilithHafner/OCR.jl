using OCR
using Test

@testset "OCR.jl" begin
    OCR.test_data()
    OCR.test_network()
    println(OCR.test_all(4; show_img=false))
end
