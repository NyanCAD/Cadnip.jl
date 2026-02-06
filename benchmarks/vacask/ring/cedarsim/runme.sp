9 stage ring oscillator with PSP103 MOSFETs

* nmos/pmos subcircuit builders provided by VACASKModels (precompiled PSP103)

* Inverter subcircuit
.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends

* Current pulse to kick-start oscillation (100uA to overcome load caps)
i0 0 1 dc 0 pulse 0 100u 1n 0.5n 0.5n 2n

* 9-stage ring oscillator
xu1 1 2 vdd 0 inverter w={10u} l={1u}
xu2 2 3 vdd 0 inverter w={10u} l={1u}
xu3 3 4 vdd 0 inverter w={10u} l={1u}
xu4 4 5 vdd 0 inverter w={10u} l={1u}
xu5 5 6 vdd 0 inverter w={10u} l={1u}
xu6 6 7 vdd 0 inverter w={10u} l={1u}
xu7 7 8 vdd 0 inverter w={10u} l={1u}
xu8 8 9 vdd 0 inverter w={10u} l={1u}
xu9 9 1 vdd 0 inverter w={10u} l={1u}

* Load capacitors (wiring + gate capacitance of next stage)
c1 1 0 5p
c2 2 0 5p
c3 3 0 5p
c4 4 0 5p
c5 5 0 5p
c6 6 0 5p
c7 7 0 5p
c8 8 0 5p
c9 9 0 5p

* Supply voltage
vdd vdd 0 1.2

.end
