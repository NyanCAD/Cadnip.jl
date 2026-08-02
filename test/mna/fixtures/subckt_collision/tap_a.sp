* Same subckt name as tap_b, different internals: 1k/1k taps 2V.
.subckt divider p out n
R1 p out 1k
R2 out n 1k
.ends
V1 in 0 DC 4
X1 in vout 0 divider
