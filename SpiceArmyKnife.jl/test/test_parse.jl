# Tests for the Mosaic model-database conversion (to_mosaic_format) — the new
# schema: flat `tags`, flat `models` entry list, flat `ports` list, explicit
# `port-order`.

using Test
using SpiceArmyKnife

@testset "Mosaic format conversion" begin

    @testset "spice_type_letter" begin
        @test SpiceArmyKnife.spice_type_letter("resistor") == "R"
        @test SpiceArmyKnife.spice_type_letter("capacitor") == "C"
        @test SpiceArmyKnife.spice_type_letter("inductor") == "L"
        @test SpiceArmyKnife.spice_type_letter("diode") == "D"
        @test SpiceArmyKnife.spice_type_letter("nmos") == "M"
        @test SpiceArmyKnife.spice_type_letter("pmos") == "M"
        @test SpiceArmyKnife.spice_type_letter("npn") == "Q"
        @test SpiceArmyKnife.spice_type_letter("pnp") == "Q"
        @test SpiceArmyKnife.spice_type_letter("ckt") == ""
    end

    @testset "ports_to_entries" begin
        layout = Dict("top" => ["vdd"], "bottom" => ["vss"],
                      "left" => ["in"], "right" => ["out"])
        entries = SpiceArmyKnife.ports_to_entries(layout)
        @test length(entries) == 4
        @test all(e -> e["type"] == "electric", entries)
        @test Dict("name" => "in", "side" => "left", "type" => "electric") in entries
        @test Dict("name" => "vdd", "side" => "top", "type" => "electric") in entries
    end

    @testset "inline .model primitive" begin
        models = [(:res_model, :r, :nmos, ".model res_model R (...)")]
        result = to_mosaic_format(models, []; base_category=["Basic"],
                                  source_file="basic.lib", mode=:inline)
        @test length(result) == 1
        doc = result[1]
        @test startswith(doc["_id"], "models:")
        @test doc["name"] == "res_model"
        @test doc["type"] == "resistor"
        @test doc["tags"] == ["Basic", "basic.lib"]
        @test doc["props"] == []
        # New schema: no legacy keys, no `ports` for primitives
        @test !haskey(doc, "category")
        @test !haskey(doc, "templates")
        @test !haskey(doc, "ports")
        # Single inline entry with the element letter and the raw code
        @test length(doc["models"]) == 1
        entry = doc["models"][1]
        @test entry["language"] == "spice"
        @test entry["spice-type"] == "R"
        @test entry["code"] == ".model res_model R (...)"
        @test !haskey(entry, "port-order")
        @test !haskey(entry, "library")
    end

    @testset "inline subcircuit" begin
        subs = [(:myamp, [:in, :out, :vdd, :vss], [:gain], ".subckt myamp in out vdd vss\n...")]
        result = to_mosaic_format([], subs; base_category=["Lib"],
                                  source_file="amps.lib", mode=:inline)
        @test length(result) == 1
        doc = result[1]
        @test doc["name"] == "myamp"
        @test doc["tags"] == ["Lib", "amps.lib"]
        # Flat port list and props from params
        @test length(doc["ports"]) == 4
        @test all(p -> p["type"] == "electric", doc["ports"])
        @test doc["props"] == [Dict("name" => "gain", "tooltip" => "")]
        # SUBCKT entry with inline code and the original ordered pin list
        entry = doc["models"][1]
        @test entry["spice-type"] == "SUBCKT"
        @test entry["code"] == ".subckt myamp in out vdd vss\n..."
        @test entry["port-order"] == ["in", "out", "vdd", "vss"]
        @test !haskey(doc, "category")
        @test !haskey(doc, "templates")
    end

    @testset ":lib subcircuit -> structured library + sections" begin
        subs = [(:nfet, [:d, :g, :s, :b], Symbol[], ".subckt nfet d g s b\n...")]
        result = to_mosaic_format([], subs; base_category=["Sky130"],
                                  source_file="sky130.lib.spice",
                                  mode=:lib, archive_url="https://example.com/pdk.zip",
                                  lib_sections=["tt"], file_device_type="nmos")
        doc = result[1]
        @test doc["type"] == "nmos"
        entry = doc["models"][1]
        @test entry["spice-type"] == "SUBCKT"
        @test entry["library"] == "https://example.com/pdk.zip#sky130.lib.spice"
        @test entry["sections"] == ["tt"]
        @test entry["port-order"] == ["d", "g", "s", "b"]
        # No {corner} token / inline code in :lib mode
        @test !haskey(entry, "code")
    end

    @testset ":lib .model primitive -> structured library, element letter" begin
        models = [(:sg13_nmos, :nmos, :nmos, ".model sg13_nmos nmos (...)")]
        result = to_mosaic_format(models, []; base_category=["IHP"],
                                  source_file="cornerMOSlv.lib",
                                  mode=:lib, archive_url="https://example.com/ihp.zip",
                                  lib_sections=["mos_tt"])
        entry = result[1]["models"][1]
        @test entry["spice-type"] == "M"
        @test entry["library"] == "https://example.com/ihp.zip#cornerMOSlv.lib"
        @test entry["sections"] == ["mos_tt"]
        @test !haskey(entry, "port-order")  # primitives have no pin list
        @test !haskey(entry, "code")
    end

    @testset "bare file (no archive_url) uses file path as library" begin
        subs = [(:foo, [:a, :b], Symbol[], ".subckt foo a b\n...")]
        result = to_mosaic_format([], subs; base_category=["X"],
                                  source_file="https://example.com/foo.lib",
                                  mode=:include, archive_url=nothing)
        entry = result[1]["models"][1]
        @test entry["library"] == "https://example.com/foo.lib"
        @test !haskey(entry, "sections")  # no corner whitelist for :include
    end
end
