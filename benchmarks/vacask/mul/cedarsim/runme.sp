Diode cascade (Voltage multiplier)
* Note: Using phase=90 to start at peak (dV/dt=0) for better Newton convergence
* at t=0. The cascaded diode topology has convergence issues with phase=0.

.model diode d is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45

vs a 0 dc=0 sin 0 50 100k 0 0 90
r1 a 1 0.01
c1 1 2 100n
d1 0 1 diode
c2 0 10 100n
d2 1 10 diode
c3 1 2 100n
d3 10 2 diode
c4 10 20 100n
d4 2 20 diode

.end
