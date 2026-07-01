Third order low pass Butterworth filter, omega_c = 1 rad/s
* Linear LC ladder with a known closed-form transient response (see
* test/transients.jl). Driven by a unit sine at f = 1/(2*pi) Hz so omega = 1.
* Output node `vout`. Used as the analytic anchor in the work-precision sweep.

V1 vin 0 SIN(0, 1, 0.15915494309189535)
L1 vin n1 1.5
C2 n1 0 1.3333333333333333
L3 n1 vout 0.5
R4 vout 0 1.0

.end
