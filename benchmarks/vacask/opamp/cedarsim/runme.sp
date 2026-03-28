5-Transistor OTA: CMOS Differential Pair with Current Mirror Load
* PSP103 MOSFET models from VACASKModels
*
* Topology:
*   - NMOS differential pair (M1, M2) with tail current source (M5)
*   - PMOS current mirror load (M3 diode-connected, M4 mirror)
*   - Bias generator: diode-connected NMOS (M6) + current source
*   - Output loaded with capacitor (models next-stage gate cap)
*
* This is the canonical "medium complexity" analog circuit for
* accuracy benchmarking: rich dynamics (slew + settling) exercise
* the adaptive error estimator rather than just dtmax.

* === OTA subcircuit ===
* Wrapped in .subckt so imported nmos/pmos builders resolve correctly
.subckt ota inp inn out vdd vstep=0.01

* PMOS Current Mirror Load
* M3: diode-connected (gate=drain), sets mirror voltage
xmp3 mirr mirr vdd vdd pmos w=20u l=1u
* M4: mirrors M3 current to output
xmp4 out mirr vdd vdd pmos w=20u l=1u

* NMOS Differential Pair
* M1: positive input, drain to mirror node
xmn1 mirr inp tail 0 nmos w=10u l=1u
* M2: negative input, drain to output
xmn2 out inn tail 0 nmos w=10u l=1u

* Tail Current Source
* M5: sets differential pair bias current
xmn5 tail bias 0 0 nmos w=20u l=1u

* Bias Generator
* M6: diode-connected NMOS sets Vgs for tail current source
xmn6 bias bias 0 0 nmos w=20u l=1u
* Current source defines bias current (20uA)
Ibias vdd bias DC 20u

.ends

* === Top-level testbench ===

* Instantiate OTA
xota inp inn out vdd ota

* Supply voltage
Vdd vdd 0 DC 1.8

* Inputs: common-mode at mid-rail, small step on inp
* PULSE: initial=0.9, pulsed=0.91, delay=100n, rise=1n, fall=1n, width=400n, period=1u
.param vstep=0.01
Vcm inn 0 DC 0.9
Vinp inp 0 DC 0.9 PULSE 0.9 {0.9+vstep} 100n 1n 1n 400n 1u

* Output load capacitor (1pF models gate cap of next stage)
Cload out 0 1p

.end
