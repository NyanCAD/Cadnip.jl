import VerilogAParser
using AbstractTrees
using VerilogAParser.VerilogACSTParser: virtrange

# Use local cmc_models directory
const local_cmc_dir = joinpath(dirname(@__DIR__), "cmc_models")

macro_va = VerilogAParser.parse("""
    `define f(arg) arg
    `define g(arg) `f(arg)

    module test(x);
            analog begin
                    `g(VMAX_s) = 1;
            end
    endmodule
    """)

# Test with local bsimcmg107 model
test_vas = [macro_va]
bsimcmg107_path = joinpath(local_cmc_dir, "bsimcmg107/bsimcmg.va")
if isfile(bsimcmg107_path)
    push!(test_vas, VerilogAParser.parsefile(bsimcmg107_path))
end

for va in test_vas
    ls = collect(Leaves(VerilogAParser.VerilogACSTParser.ChunkTree(va.ps)))
    @test all(1:(length(ls)-1)) do i
        first(virtrange(ls[i+1])) == last(virtrange(ls[i]))+1
    end
end
