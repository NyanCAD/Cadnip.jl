#!/usr/bin/env julia
#==============================================================================#
# PSP103VA Bridge Tests - Incremental Complexity from Integration to Benchmark
#
# This file bridges the gap between simple PSP103 integration tests and the
# full 9-stage ring oscillator benchmark by testing increasingly complex circuits:
#
# 1. Single NMOS with full model card (DC)
# 2. Single PMOS with full model card (DC)
# 3. Single inverter with full model card (DC)
# 4. 3-stage ring oscillator (minimal ring, transient)
# 5. 5-stage ring oscillator (transient)
# 6. 7-stage ring oscillator (transient)
# 7. 9-stage ring oscillator (matches benchmark, transient)
#
# Note: We skip simple transistor transient tests as they are numerically unstable.
# Ring oscillators provide much better test coverage for transient behavior.
#==============================================================================#

using CedarSim
using CedarSim.SpectreNetlistParser
using CedarSim.MNA: MNAContext, MNASpec, solve_dc, voltage, current, MNACircuit, tran!, CedarTranOp
using PSPModels
using OrdinaryDiffEq: FBDF
using Test

# Full PSP103 model card (280+ parameters) - extracted from benchmark
const FULL_MODEL_CARD = """
.model psp103n psp103va
+    type=1
+    tr=27.0
+    dta=0
+    swgeo=1
+    qmc=1.0
+    lvaro=-10.0e-9
+    lvarl=0
+    lvarw=0
+    lap=10.0e-9
+    wvaro=10.0e-9
+    wvarl=0
+    wvarw=0
+    wot=0
+    dlq=0
+    dwq=0
+    vfbo=-1.1
+    vfbl=0
+    vfbw=0
+    vfblw=0
+    stvfbo=5.0e-4
+    stvfbl=0
+    stvfbw=0
+    stvfblw=0
+    toxo=1.5e-9
+    epsroxo=3.9
+    nsubo=3.0e+23
+    nsubw=0
+    wseg=1.5e-10
+    npck=1.0e+24
+    npckw=0
+    wsegp=0.9e-8
+    lpck=5.5e-8
+    lpckw=0
+    fol1=2.0e-2
+    fol2=5.0e-6
+    facneffaco=0.8
+    facneffacl=0
+    facneffacw=0
+    facneffaclw=0
+    gfacnudo=0.1
+    gfacnudl=0
+    gfacnudlexp=1
+    gfacnudw=0
+    gfacnudlw=0
+    vsbnudo=0
+    dvsbnudo=1
+    vnsubo=0
+    nslpo=0.05
+    dnsubo=0
+    dphibo=0
+    dphibl=0
+    dphiblexp=1.0
+    dphibw=0
+    dphiblw=0
+    delvtaco=0
+    delvtacl=0
+    delvtaclexp=1
+    delvtacw=0
+    delvtaclw=0
+    npo=1.5e+26
+    npl=10.0e-18
+    cto=5.0e-15
+    ctl=4.0e-2
+    ctlexp=0.6
+    ctw=0
+    ctlw=0
+    toxovo=1.5e-9
+    toxovdo=2.0e-9
+    lov=10.0e-9
+    lovd=0
+    novo=7.5e+25
+    novdo=5.0e+25
+    cfl=3.0e-4
+    cflexp=2.0
+    cfw=5.0e-3
+    cfbo=0.3
+    uo=3.5e-2
+    fbet1=-0.3
+    fbet1w=0.15
+    lp1=1.5e-7
+    lp1w=-2.5e-2
+    fbet2=50.0
+    lp2=8.5e-10
+    betw1=5.0e-2
+    betw2=-2.0e-2
+    wbet=5.0e-10
+    stbeto=1.75
+    stbetl=-2.0e-2
+    stbetw=-2.0e-3
+    stbetlw=-3.0e-3
+    mueo=0.6
+    muew=-1.2e-2
+    stmueo=0.5
+    themuo=2.75
+    stthemuo=-0.1
+    cso=1.0e-2
+    csl=0
+    cslexp=1
+    csw=0
+    cslw=0
+    stcso=-5.0
+    xcoro=0.15
+    xcorl=2.0e-3
+    xcorw=-3.0e-2
+    xcorlw=-3.5e-3
+    stxcoro=1.25
+    fetao=1
+    rsw1=50
+    rsw2=5.0e-2
+    strso=-2.0
+    rsbo=0
+    rsgo=0
+    thesato=1.0e-6
+    thesatl=0.6
+    thesatlexp=0.75
+    thesatw=-1.0e-2
+    thesatlw=0
+    stthesato=1.5
+    stthesatl=-2.5e-2
+    stthesatw=-2.0e-2
+    stthesatlw=-5.0e-3
+    thesatbo=0.15
+    thesatgo=0.75
+    axo=20
+    axl=0.2
+    alpl=7.0e-3
+    alplexp=0.6
+    alpw=5.0e-2
+    alp1l1=2.5e-2
+    alp1lexp=0.4
+    alp1l2=0.1
+    alp1w=8.5e-3
+    alp2l1=0.5
+    alp2lexp=0
+    alp2l2=0.5
+    alp2w=-0.2
+    vpo=0.25
+    a1o=1.0
+    a1l=0
+    a1w=0
+    a2o=10.0
+    sta2o=-0.5
+    a3o=1.0
+    a3l=0
+    a3w=0
+    a4o=0
+    a4l=0
+    a4w=0
+    gcoo=5.0
+    iginvlw=50.0
+    igovw=10.0
+    igovdw=0
+    stigo=1.5
+    gc2o=1.0
+    gc3o=-1.0
+    chibo=3.1
+    agidlw=50.0
+    agidldw=0
+    bgidlo=35.0
+    bgidldo=41
+    stbgidlo=-5.0e-4
+    stbgidldo=0
+    cgidlo=0.15
+    cgidldo=0
+    cgbovl=0
+    cfrw=5.0e-17
+    cfrdw=0
+    fnto=1
+    nfalw=8.0e+22
+    nfblw=3.0e7
+    nfclw=0
+    rgo=0
+    rint=0
+    rvpoly=0
+    rshg=0
+    dlsil=0
+    rbulko=0
+    rwello=0
+    rjundo=0
+    rjunso=0
+    swjunexp=0
+    trj=27.0
+    imax=1.0e3
+    vjunref=2.5
+    fjunq=0.03
+    cjorbot=1.0e-3
+    cjorsti=1.0e-9
+    cjorgat=0.5e-9
+    vbirbot=0.75
+    vbirsti=1.0
+    vbirgat=0.75
+    pbot=0.35
+    psti=0.35
+    pgat=0.6
+    phigbot=1.16
+    phigsti=1.16
+    phiggat=1.16
+    idsatrbot=5.0e-9
+    idsatrsti=1.0e-18
+    idsatrgat=1.0e-18
+    csrhbot=5.0e2
+    csrhsti=0
+    csrhgat=1.0e3
+    xjunsti=1.0e-8
+    xjungat=1.0e-9
+    ctatbot=5.0e2
+    ctatsti=0
+    ctatgat=1.0e3
+    mefftatbot=0.25
+    mefftatsti=0.25
+    mefftatgat=0.25
+    cbbtbot=1.0e-12
+    cbbtsti=1.0e-18
+    cbbtgat=1.0e-18
+    fbbtrbot=1.0e9
+    fbbtrsti=1.0e9
+    fbbtrgat=1.0e9
+    stfbbtbot=-1.0e-3
+    stfbbtsti=-1.0e-3
+    stfbbtgat=-1.0e-2
+    vbrbot=10.0
+    vbrsti=10.0
+    vbrgat=10.0
+    pbrbot=3
+    pbrsti=4
+    pbrgat=3
+    vjunrefd=2.5
+    fjunqd=0.03
+    cjorbotd=1.0e-3
+    cjorstid=1.0e-9
+    cjorgatd=1.0e-9
+    vbirbotd=1.0
+    vbirstid=1.0
+    vbirgatd=1.0
+    pbotd=0.5
+    pstid=0.5
+    pgatd=0.5
+    phigbotd=1.16
+    phigstid=1.16
+    phiggatd=1.16
+    idsatrbotd=1.0e-12
+    idsatrstid=1.0e-18
+    idsatrgatd=1.0e-18
+    csrhbotd=1.0e+2
+    csrhstid=1.0e-4
+    csrhgatd=1.0e-4
+    xjunstid=1.0e-7
+    xjungatd=1.0e-7
+    ctatbotd=1.0e+2
+    ctatstid=1.0e-4
+    ctatgatd=1.0e-4
+    mefftatbotd=0.25
+    mefftatstid=0.25
+    mefftatgatd=0.25
+    cbbtbotd=1.0e-12
+    cbbtstid=1.0e-18
+    cbbtgatd=1.0e-18
+    fbbtrbotd=1.0e9
+    fbbtrstid=1.0e9
+    fbbtrgatd=1.0e9
+    stfbbtbotd=-1.0e-3
+    stfbbtstid=-1.0e-3
+    stfbbtgatd=-1.0e-3
+    vbrbotd=10.0
+    vbrstid=10.0
+    vbrgatd=10.0
+    pbrbotd=4
+    pbrstid=4
+    pbrgatd=4

.model psp103p psp103va
+    type=-1
+    tr=27.0
+    dta=0
+    swgeo=1
+    qmc=1.0
+    lvaro=-10.0e-9
+    lvarl=0
+    lvarw=0
+    lap=10.0e-9
+    wvaro=10.0e-9
+    wvarl=0
+    wvarw=0
+    wot=0
+    dlq=0
+    dwq=0
+    vfbo=-1.1
+    vfbl=0
+    vfbw=0
+    vfblw=0
+    stvfbo=5.0e-4
+    stvfbl=0
+    stvfbw=0
+    stvfblw=0
+    toxo=1.5e-9
+    epsroxo=3.9
+    nsubo=3.0e+23
+    nsubw=0
+    wseg=1.5e-10
+    npck=1.0e+24
+    npckw=0
+    wsegp=0.9e-8
+    lpck=5.5e-8
+    lpckw=0
+    fol1=2.0e-2
+    fol2=5.0e-6
+    facneffaco=0.8
+    facneffacl=0
+    facneffacw=0
+    facneffaclw=0
+    gfacnudo=0.1
+    gfacnudl=0
+    gfacnudlexp=1
+    gfacnudw=0
+    gfacnudlw=0
+    vsbnudo=0
+    dvsbnudo=1
+    vnsubo=0
+    nslpo=0.05
+    dnsubo=0
+    dphibo=0
+    dphibl=0
+    dphiblexp=1.0
+    dphibw=0
+    dphiblw=0
+    delvtaco=0
+    delvtacl=0
+    delvtaclexp=1
+    delvtacw=0
+    delvtaclw=0
+    npo=1.5e+26
+    npl=10.0e-18
+    cto=5.0e-15
+    ctl=4.0e-2
+    ctlexp=0.6
+    ctw=0
+    ctlw=0
+    toxovo=1.5e-9
+    toxovdo=2.0e-9
+    lov=10.0e-9
+    lovd=0
+    novo=7.5e+25
+    novdo=5.0e+25
+    cfl=3.0e-4
+    cflexp=2.0
+    cfw=5.0e-3
+    cfbo=0.3
+    uo=3.5e-2
+    fbet1=-0.3
+    fbet1w=0.15
+    lp1=1.5e-7
+    lp1w=-2.5e-2
+    fbet2=50.0
+    lp2=8.5e-10
+    betw1=5.0e-2
+    betw2=-2.0e-2
+    wbet=5.0e-10
+    stbeto=1.75
+    stbetl=-2.0e-2
+    stbetw=-2.0e-3
+    stbetlw=-3.0e-3
+    mueo=0.6
+    muew=-1.2e-2
+    stmueo=0.5
+    themuo=2.75
+    stthemuo=-0.1
+    cso=1.0e-2
+    csl=0
+    cslexp=1
+    csw=0
+    cslw=0
+    stcso=-5.0
+    xcoro=0.15
+    xcorl=2.0e-3
+    xcorw=-3.0e-2
+    xcorlw=-3.5e-3
+    stxcoro=1.25
+    fetao=1
+    rsw1=50
+    rsw2=5.0e-2
+    strso=-2.0
+    rsbo=0
+    rsgo=0
+    thesato=1.0e-6
+    thesatl=0.6
+    thesatlexp=0.75
+    thesatw=-1.0e-2
+    thesatlw=0
+    stthesato=1.5
+    stthesatl=-2.5e-2
+    stthesatw=-2.0e-2
+    stthesatlw=-5.0e-3
+    thesatbo=0.15
+    thesatgo=0.75
+    axo=20
+    axl=0.2
+    alpl=7.0e-3
+    alplexp=0.6
+    alpw=5.0e-2
+    alp1l1=2.5e-2
+    alp1lexp=0.4
+    alp1l2=0.1
+    alp1w=8.5e-3
+    alp2l1=0.5
+    alp2lexp=0
+    alp2l2=0.5
+    alp2w=-0.2
+    vpo=0.25
+    a1o=1.0
+    a1l=0
+    a1w=0
+    a2o=10.0
+    sta2o=-0.5
+    a3o=1.0
+    a3l=0
+    a3w=0
+    a4o=0
+    a4l=0
+    a4w=0
+    gcoo=5.0
+    iginvlw=50.0
+    igovw=10.0
+    igovdw=0
+    stigo=1.5
+    gc2o=1.0
+    gc3o=-1.0
+    chibo=3.1
+    agidlw=50.0
+    agidldw=0
+    bgidlo=35.0
+    bgidldo=41
+    stbgidlo=-5.0e-4
+    stbgidldo=0
+    cgidlo=0.15
+    cgidldo=0
+    cgbovl=0
+    cfrw=5.0e-17
+    cfrdw=0
+    fnto=1
+    nfalw=8.0e+22
+    nfblw=3.0e7
+    nfclw=0
+    rgo=0
+    rint=0
+    rvpoly=0
+    rshg=0
+    dlsil=0
+    rbulko=0
+    rwello=0
+    rjundo=0
+    rjunso=0
+    swjunexp=0
+    trj=27.0
+    imax=1.0e3
+    vjunref=2.5
+    fjunq=0.03
+    cjorbot=1.0e-3
+    cjorsti=1.0e-9
+    cjorgat=0.5e-9
+    vbirbot=0.75
+    vbirsti=1.0
+    vbirgat=0.75
+    pbot=0.35
+    psti=0.35
+    pgat=0.6
+    phigbot=1.16
+    phigsti=1.16
+    phiggat=1.16
+    idsatrbot=5.0e-9
+    idsatrsti=1.0e-18
+    idsatrgat=1.0e-18
+    csrhbot=5.0e2
+    csrhsti=0
+    csrhgat=1.0e3
+    xjunsti=1.0e-8
+    xjungat=1.0e-9
+    ctatbot=5.0e2
+    ctatsti=0
+    ctatgat=1.0e3
+    mefftatbot=0.25
+    mefftatsti=0.25
+    mefftatgat=0.25
+    cbbtbot=1.0e-12
+    cbbtsti=1.0e-18
+    cbbtgat=1.0e-18
+    fbbtrbot=1.0e9
+    fbbtrsti=1.0e9
+    fbbtrgat=1.0e9
+    stfbbtbot=-1.0e-3
+    stfbbtsti=-1.0e-3
+    stfbbtgat=-1.0e-2
+    vbrbot=10.0
+    vbrsti=10.0
+    vbrgat=10.0
+    pbrbot=3
+    pbrsti=4
+    pbrgat=3
+    vjunrefd=2.5
+    fjunqd=0.03
+    cjorbotd=1.0e-3
+    cjorstid=1.0e-9
+    cjorgatd=1.0e-9
+    vbirbotd=1.0
+    vbirstid=1.0
+    vbirgatd=1.0
+    pbotd=0.5
+    pstid=0.5
+    pgatd=0.5
+    phigbotd=1.16
+    phigstid=1.16
+    phiggatd=1.16
+    idsatrbotd=1.0e-12
+    idsatrstid=1.0e-18
+    idsatrgatd=1.0e-18
+    csrhbotd=1.0e+2
+    csrhstid=1.0e-4
+    csrhgatd=1.0e-4
+    xjunstid=1.0e-7
+    xjungatd=1.0e-7
+    ctatbotd=1.0e+2
+    ctatstid=1.0e-4
+    ctatgatd=1.0e-4
+    mefftatbotd=0.25
+    mefftatstid=0.25
+    mefftatgatd=0.25
+    cbbtbotd=1.0e-12
+    cbbtstid=1.0e-18
+    cbbtgatd=1.0e-18
+    fbbtrbotd=1.0e9
+    fbbtrstid=1.0e9
+    fbbtrgatd=1.0e9
+    stfbbtbotd=-1.0e-3
+    stfbbtstid=-1.0e-3
+    stfbbtgatd=-1.0e-3
+    vbrbotd=10.0
+    vbrstid=10.0
+    vbrgatd=10.0
+    pbrbotd=4
+    pbrstid=4
+    pbrgatd=4
"""

const NMOS_PMOS_SUBCIRCUITS = """
.subckt nmos d g s b w=1u l=0.2u ld=0.5u ls=0.5u
  nm d g s b psp103n w={w} l={l} ad={w*ld} as={w*ls} pd={2*(w+ld)} ps={2*(w+ls)}
.ends

.subckt pmos d g s b w=1u l=0.2u ld=0.5u ls=0.5u
  nm d g s b psp103p w={w} l={l} ad={w*ld} as={w*ls} pd={2*(w+ld)} ps={2*(w+ls)}
.ends
"""

@testset "PSP103VA Bridge Tests" begin

    @testset "1. Single NMOS with full model card (DC)" begin
        netlist = """
* Single NMOS with full 280+ param model card - DC test
$FULL_MODEL_CARD

M1 d g 0 0 psp103n W=10u L=1u
Vds d 0 DC 1.2
Vgs g 0 DC 0.6
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)

        spec = MNASpec(temp=27.0, mode=:dcop)
        sol = solve_dc(circuit_fn, (;), spec)

        @test isapprox(voltage(sol, :d), 1.2, atol=1e-6)
        @test isapprox(voltage(sol, :g), 0.6, atol=1e-6)

        # Current should be reasonable
        Id = current(sol, :I_vds)
        @test abs(Id) > 10e-6 && abs(Id) < 10e-3
        println("  ✓ Single NMOS DC: Id = $(Id*1e6) µA")
    end

    @testset "2. Single PMOS with full model card (DC)" begin
        netlist = """
* Single PMOS with full 280+ param model card - DC test
$FULL_MODEL_CARD

M1 d g vdd vdd psp103p W=20u L=1u
Vdd vdd 0 DC 1.2
Vgs g 0 DC 0.6
Vds d 0 DC 0.0
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)

        spec = MNASpec(temp=27.0, mode=:dcop)
        sol = solve_dc(circuit_fn, (;), spec)

        @test isapprox(voltage(sol, :vdd), 1.2, atol=1e-6)
        @test isapprox(voltage(sol, :g), 0.6, atol=1e-6)

        # Current should be reasonable for PMOS
        Id = current(sol, :I_vdd)
        @test abs(Id) > 10e-6 && abs(Id) < 10e-3
        println("  ✓ Single PMOS DC: Id = $(Id*1e6) µA")
    end

    @testset "3. Single inverter with full model card (DC)" begin
        netlist = """
* Single inverter DC operating point
$FULL_MODEL_CARD
$NMOS_PMOS_SUBCIRCUITS

.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends

xinv in out vdd 0 inverter w=10u l=1u

Vin in 0 DC 0.6
Vdd vdd 0 DC 1.2
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)

        spec = MNASpec(temp=27.0, mode=:dcop)
        sol = solve_dc(circuit_fn, (;), spec)

        @test isapprox(voltage(sol, :vdd), 1.2, atol=1e-6)
        @test isapprox(voltage(sol, :in), 0.6, atol=1e-6)

        # Output should be in valid range
        vout = voltage(sol, :out)
        @test vout >= 0.0 && vout <= 1.2
        println("  ✓ Single inverter DC: Vin = 0.6V, Vout = $(vout) V")
    end

    @testset "4. 3-stage ring oscillator" begin
        netlist = """
* 3-stage ring oscillator (minimal odd-stage ring)
$FULL_MODEL_CARD
$NMOS_PMOS_SUBCIRCUITS

.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends

* Current pulse to kick-start oscillation
i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n

* 3-stage ring
xu1 1 2 vdd 0 inverter w=10u l=1u
xu2 2 3 vdd 0 inverter w=10u l=1u
xu3 3 1 vdd 0 inverter w=10u l=1u

Vdd vdd 0 DC 1.2
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)

        circuit = MNACircuit(circuit_fn)

        # Use benchmark configuration
        sol = tran!(circuit, (0.0, 1e-6);
                   solver=FBDF(autodiff=false),
                   initializealg=CedarTranOp(),
                   dtmax=0.01e-9,
                   maxiters=10_000_000,
                   dense=false)

        @test sol.retcode == :Success
        println("  ✓ 3-stage ring: $(length(sol.t)) timepoints, $(sol.stats.nnonliniter) NR iters")
    end

    @testset "5. 5-stage ring oscillator" begin
        netlist = """
* 5-stage ring oscillator
$FULL_MODEL_CARD
$NMOS_PMOS_SUBCIRCUITS

.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends

i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n

xu1 1 2 vdd 0 inverter w=10u l=1u
xu2 2 3 vdd 0 inverter w=10u l=1u
xu3 3 4 vdd 0 inverter w=10u l=1u
xu4 4 5 vdd 0 inverter w=10u l=1u
xu5 5 1 vdd 0 inverter w=10u l=1u

Vdd vdd 0 DC 1.2
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)

        circuit = MNACircuit(circuit_fn)

        sol = tran!(circuit, (0.0, 1e-6);
                   solver=FBDF(autodiff=false),
                   initializealg=CedarTranOp(),
                   dtmax=0.01e-9,
                   maxiters=10_000_000,
                   dense=false)

        @test sol.retcode == :Success
        println("  ✓ 5-stage ring: $(length(sol.t)) timepoints, $(sol.stats.nnonliniter) NR iters")
    end

    @testset "6. 7-stage ring oscillator" begin
        netlist = """
* 7-stage ring oscillator
$FULL_MODEL_CARD
$NMOS_PMOS_SUBCIRCUITS

.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends

i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n

xu1 1 2 vdd 0 inverter w=10u l=1u
xu2 2 3 vdd 0 inverter w=10u l=1u
xu3 3 4 vdd 0 inverter w=10u l=1u
xu4 4 5 vdd 0 inverter w=10u l=1u
xu5 5 6 vdd 0 inverter w=10u l=1u
xu6 6 7 vdd 0 inverter w=10u l=1u
xu7 7 1 vdd 0 inverter w=10u l=1u

Vdd vdd 0 DC 1.2
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)

        circuit = MNACircuit(circuit_fn)

        sol = tran!(circuit, (0.0, 1e-6);
                   solver=FBDF(autodiff=false),
                   initializealg=CedarTranOp(),
                   dtmax=0.01e-9,
                   maxiters=10_000_000,
                   dense=false)

        @test sol.retcode == :Success
        println("  ✓ 7-stage ring: $(length(sol.t)) timepoints, $(sol.stats.nnonliniter) NR iters")
    end

    @testset "7. 9-stage ring oscillator (matches benchmark)" begin
        netlist = """
* 9-stage ring oscillator - matches full benchmark
$FULL_MODEL_CARD
$NMOS_PMOS_SUBCIRCUITS

.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends

i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n

xu1 1 2 vdd 0 inverter w=10u l=1u
xu2 2 3 vdd 0 inverter w=10u l=1u
xu3 3 4 vdd 0 inverter w=10u l=1u
xu4 4 5 vdd 0 inverter w=10u l=1u
xu5 5 6 vdd 0 inverter w=10u l=1u
xu6 6 7 vdd 0 inverter w=10u l=1u
xu7 7 8 vdd 0 inverter w=10u l=1u
xu8 8 9 vdd 0 inverter w=10u l=1u
xu9 9 1 vdd 0 inverter w=10u l=1u

Vdd vdd 0 DC 1.2
"""
        ast = SpectreNetlistParser.parse(IOBuffer(netlist); start_lang=:spice, implicit_title=true)
        code = CedarSim.make_mna_circuit(ast; imported_hdl_modules=[PSPModels])
        circuit_fn = eval(code)

        circuit = MNACircuit(circuit_fn)

        sol = tran!(circuit, (0.0, 1e-6);
                   solver=FBDF(autodiff=false),
                   initializealg=CedarTranOp(),
                   dtmax=0.01e-9,
                   maxiters=10_000_000,
                   dense=false)

        @test sol.retcode == :Success
        println("  ✓ 9-stage ring: $(length(sol.t)) timepoints, $(sol.stats.nnonliniter) NR iters")
        println("  ✓ Matches benchmark configuration!")
    end

end
