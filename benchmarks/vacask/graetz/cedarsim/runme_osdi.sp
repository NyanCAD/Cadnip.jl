Full-wave rectifier with smoothing and load (OSDI diode)

.model d1n4007 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45

vs inp inn 0 sin 0.0 20 50.0

N1 inp outp d1n4007
N2 outn inp d1n4007
N3 inn outp d1n4007
N4 outn inn d1n4007
cl outp outn 100u
rl outp outn 1k
rgnd1 inn 0 1meg
rgnd2 outn 0 1meg

.end
