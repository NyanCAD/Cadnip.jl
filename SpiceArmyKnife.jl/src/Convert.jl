"""
SPICE/Spectre/Verilog-A Netlist Converter

Convert netlists between different SPICE/Spectre dialects and to Verilog-A format.

Usage:
    spak-convert input.sp output.sp --input-simulator pspice --output-simulator ngspice
    spak-convert circuit.sp circuit.va --input-simulator ngspice --output-simulator openvaf

Examples:
    # Convert Cordell models to ngspice-compatible format (filters doc parameters)
    spak-convert Cordell-Models.txt ngspice-models.sp --input-simulator pspice --output-simulator ngspice

    # Convert SPICE to Verilog-A for OpenVAF
    spak-convert diode_divider.lib diode_divider.va --input-simulator ngspice --output-simulator openvaf

    # Convert SPICE to Spectre format
    spak-convert circuit.sp circuit.scs --input-simulator ngspice --output-simulator spectre
"""
module Convert

using ..SpiceArmyKnife
using NyanSpectreNetlistParser
using ArgParse

function parse_commandline(args)
    s = ArgParseSettings(
        description = "Convert SPICE/Spectre netlists between different dialects or to Verilog-A",
        epilog = """
            Examples:
              # Convert to ngspice (filters documentation parameters)
              \$ spak-convert models.txt ngspice-models.sp --input-simulator pspice --output-simulator ngspice

              # Convert SPICE to Verilog-A for OpenVAF
              \$ spak-convert diode_divider.lib diode_divider.va --input-simulator ngspice --output-simulator openvaf

              # Convert SPICE to Spectre
              \$ spak-convert circuit.sp circuit.scs --input-simulator ngspice --output-simulator spectre

              # Specify input simulator explicitly
              \$ spak-convert input.cir output.sp --input-simulator hspice --output-simulator ngspice
            """
    )

    @add_arg_table! s begin
        "input"
            help = "Input netlist file"
            required = true
        "output"
            help = "Output netlist file"
            required = true
        "--input-simulator"
            help = "Input simulator: ngspice, hspice, pspice, xyce, spectre (determines language via language() trait)"
            arg_type = Symbol
            required = true
        "--output-simulator", "-s"
            help = "Output simulator: ngspice, hspice, pspice, xyce, spectre, openvaf, gnucap (required)"
            arg_type = Symbol
            required = true
        "--va-include"
            help = "Verilog-A file to extract model definitions from (can be specified multiple times)"
            action = :append_arg
            default = String[]
    end

    return parse_args(args, s)
end

function (@main)(ARGS)
    args = parse_commandline(ARGS)

    input_file = args["input"]
    output_file = args["output"]
    input_simulator_sym = args["input-simulator"]
    output_simulator_sym = args["output-simulator"]

    # Validate input file
    if !isfile(input_file)
        println("Error: Input file not found: $input_file")
        exit(1)
    end

    # Determine input language
    input_sim = simulator_from_symbol(input_simulator_sym)
    input_lang = language(input_sim)
    println("Input simulator: $input_simulator_sym → language: $input_lang")

    # Map output simulator symbol to simulator instance
    output_sim = simulator_from_symbol(output_simulator_sym)
    output_lang = language(output_sim)

    println("=" ^ 80)
    println("SPICE/Spectre/Verilog-A Netlist Converter")
    println("=" ^ 80)
    println("Input:  $input_file ($input_lang)")
    println("Output: $output_file ($output_simulator_sym → $output_lang)")
    println("-" ^ 80)

    try
        # Parse input file
        println("Parsing input file...")
        ast = if input_lang == :spice
            NyanSpectreNetlistParser.parsefile(input_file; start_lang=:spice, spice_dialect=input_simulator_sym, implicit_title=true)
        else
            NyanSpectreNetlistParser.parsefile(input_file; start_lang=:spectre)
        end

        if ast.ps.errored
            println("⚠ Warning: Parse errors encountered in input file")
            println("  Attempting to generate output anyway...")
        else
            println("✓ Successfully parsed input file")
        end

        # Extract Verilog-A model definitions if provided
        model_database = if !isempty(args["va-include"])
            println("Extracting Verilog-A model definitions...")
            dbs = map(args["va-include"]) do va_file
                if !isfile(va_file)
                    println("  ⚠ Warning: VA file not found: $va_file")
                    return ModelDatabase([], Dict{Symbol, Int}())
                end
                println("  Reading: $va_file")
                try
                    extract_model_definitions(va_file)
                catch e
                    println("  ⚠ Warning: Failed to extract from $va_file: $e")
                    ModelDatabase([], Dict{Symbol, Int}())
                end
            end
            merged_db = merge_model_databases(dbs)
            println("  ✓ Extracted $(length(merged_db.models)) model(s): $(join(keys(merged_db.model_lookup), ", "))")
            merged_db
        else
            ModelDatabase([], Dict{Symbol, Int}())
        end

        # Generate output
        println("Generating output code...")
        # Set up includepaths with input file's directory for resolving relative includes
        input_file_abs = abspath(input_file)
        source_root = dirname(input_file_abs)
        includepaths = [source_root]

        # Write output file - use file IO directly to enable separate include file generation
        println("Writing output file...")
        output_file_abs = abspath(output_file)
        output_dir = dirname(output_file_abs)

        # Use output filename (without extension) as library name for Spectre
        lib_name = splitext(basename(output_file))[1]

        options = Dict{Symbol, Any}(
            :output_dir => output_dir,
            :source_root => source_root,
            :main_output_file => output_file_abs,
            :spice_dialect => input_simulator_sym,
            :va_models => model_database,
            :library_name => lib_name,
            :model_prefix => "m_",
            :ckt_prefix => ""
        )
        open(output_file, "w") do io
            generate_code(ast, io, output_sim; options=options, includepaths=includepaths)
        end

        file_size_kb = round(stat(output_file).size / 1024, digits=1)
        println("✓ Successfully wrote $output_file ($(file_size_kb) KB)")

        println("=" ^ 80)
        println("✓ CONVERSION COMPLETE")
        println("=" ^ 80)

    catch e
        println("=" ^ 80)
        println("✗ CONVERSION FAILED")
        println("=" ^ 80)
        showerror(stdout, e, catch_backtrace())
        println()
        exit(1)
    end
end

end # module Convert
