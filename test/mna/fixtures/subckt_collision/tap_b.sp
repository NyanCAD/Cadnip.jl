* Same subckt name as tap_a, different internals: 1k/3k taps 3V.
.subckt divider p out n rbot=3k
R1 p out 1k
R2 out n {rbot}
.ends
V1 in 0 DC 4
X1 in vout 0 divider
