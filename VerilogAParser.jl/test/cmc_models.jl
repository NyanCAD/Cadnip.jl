import VerilogAParser

# Use local cmc_models directory
const local_cmc_dir = joinpath(dirname(@__DIR__), "cmc_models")

# Only test models available locally
# Full CMC test suite requires the CMC package
models = [
    "bsimcmg107/bsimcmg.va",
]

@testset "CMC Models (local)" begin
    for model in models
        model_path = joinpath(local_cmc_dir, model)
        if isfile(model_path)
            @testset "$model" begin
                local va = VerilogAParser.parsefile(model_path)
                # CMC models shouldn't have errors since they presumably work in Cadence, and others.
                buf = IOBuffer()
                out = IOContext(buf, :color=>true, :displaysize => (80, 240))
                VerilogAParser.VerilogACSTParser.visit_errors(va; io=out)

                out = String(take!(buf))
                @test isempty(out)
            end
        else
            @warn "Skipping $model (not found at $model_path)"
        end
    end
end
