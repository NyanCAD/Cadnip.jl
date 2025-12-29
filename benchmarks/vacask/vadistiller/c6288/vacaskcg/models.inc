// BSIM3 wrappers and models

subckt nmos (d g s b)
parameters w=2u l=0.35u ld=0.5u ls=0.5u
  m (d g s b) n1 w=w l=l ad=w*ld as=w*ls pd=2*(w+ld) ps=2*(w+ls)
ends

subckt pmos (d g s b)
parameters w=2u l=0.35u ld=0.5u ls=0.5u
  m (d g s b) p1 w=w l=l ad=w*ld as=w*ls pd=2*(w+ld) ps=2*(w+ls)
ends

// model = bsim3v3
// Berkeley Spice Compatibility 
// Lmin= .35 Lmax= 20 Wmin= .6 Wmax= 20

model n1 bsim3 ( type=1
  tnom=27.0
  nqsmod=0 elm=3
  capmod=3 
  nch= 2.498e+17  tox=9e-09 xj=1.00000e-07
  lint=9.36e-8 wint=1.47e-7
  vth0= 0.6322    k1= 0.756  k2= -3.83e-2  k3= -2.612 
  dvt0= 2.812  dvt1= 0.462  dvt2=-9.17e-2 
  nlx= 3.52291e-08  w0= 1.163e-6 
  k3b= 2.233 
  vsat= 86301.58  ua= 6.47e-9  ub= 4.23e-18  uc=-4.706281e-11 
  rdsw= 650  u0= 388.3203 wr=1
  a0= 0.3496967 ags=0.1    b0=0.546    b1= 1   
  dwg = -6.0e-09 dwb = -3.56e-09 prwb = -0.213
  keta=-3.605872e-02  a1= 2.778747e-02  a2= 0.9 
  voff=-6.735529e-02  nfactor= 1.139926  cit= 1.622527e-04 
  cdsc=-2.147181e-05   
  cdscb= 0  dvt0w =  0 dvt1w =  0 dvt2w =  0 
  cdscd =  0 prwg =  0 
  eta0= 1.0281729e-02  etab=-5.042203e-03 
  dsub= 0.31871233 
  pclm= 1.114846  pdiblc1= 2.45357e-03  pdiblc2= 6.406289e-03 
  drout= 0.31871233  pscbe1= 5000000  pscbe2= 5e-09 pdiblcb = -0.234
  pvag= 0 delta=0.01
   wl =  0 ww = -1.420242e-09 wwl =  0 
   wln =  0 wwn =  0.2613948 ll =  1.300902e-10 
   lw =  0 lwl =  0 lln =  0.316394 
   lwn =  0
  kt1=-0.3  kt2=-0.051 
  at= 22400 
  ute=-1.48 
  ua1= 3.31e-10  ub1= 2.61e-19 uc1= -3.42e-10 
  kt1l=0  prt=764.3
  kf=0.032f af=1 rsh=7
)

model p1 bsim3 ( type=-1
  tnom=27.0
  nqsmod=0 elm=3
  nch= 3.533024e+17  tox=9e-09 xj=1.00000e-07
  lint=6.23e-8 wint=1.22e-7
  vth0=-.6732829 k1= .8362093  k2=-8.606622e-02  k3= 1.82 
  dvt0= 1.903801  dvt1= .5333922  dvt2=-.1862677 
  nlx= 1.28e-8  w0= 2.1e-6 
  k3b= -0.24 prwg=-0.001 prwb=-0.323 
  vsat= 103503.2  ua= 1.39995e-09  ub= 1.e-19  uc=-2.73e-11 
   rdsw= 460  u0= 138.7609 
  a0= .4716551 ags=0.12 
  keta=-1.871516e-03  a1= .3417965  a2= 0.83 
  voff=-.074182  nfactor= 1.54389  cit=-1.015667e-03 
  cdsc= 8.937517e-04 
  cdscb= 1.45e-4  cdscd=1.04e-4
   dvt0w=0.232 dvt1w=4.5e6 dvt2w=-0.0023
  eta0= 6.024776e-02  etab=-4.64593e-03 
  dsub= .23222404 
  pclm= .989  pdiblc1= 2.07418e-02  pdiblc2= 1.33813e-3 
  drout= .3222404  pscbe1= 118000  pscbe2= 1e-09 
  pvag= 0 
  kt1= -0.25  kt2= -0.032 prt=64.5 
  at= 33000 
  ute= -1.5 
  ua1= 4.312e-9 ub1= 6.65e-19  uc1= 0 
  kt1l=0
  kf=0.032f af=1 rsh=7
)
