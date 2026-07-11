Darlington pair switch driven by a pulse train

* Q1 emitter feeds Q2 base; both collectors share the 1k load. The 3V pulse
* through 10k drives the pair between cutoff and saturation every period -
* each edge slams all six limited junctions (vbe, vbc, vsub per BJT), the
* PCNR in-step limiting stress case. rbleed gives Q2's base a DC path and
* speeds turn-off; cje/cjc/tf add real charge dynamics to the switching.

.model qmod npn bf=100 is=1e-15 cje=10p cjc=5p tf=0.3n

vcc vcc 0 dc 5
vs vin 0 dc 0 pulse 0 3 0 10n 10n 0.99u 2u
rb vin b1 10k
q1 coll b1 b2 qmod
q2 coll b2 0 qmod
rbleed b2 0 10k
rl vcc coll 1k
cl coll 0 100p

.end
