# VADistiller Models

This folder contains Verilog-A models from the [VADistiller](https://codeberg.org/arpadbuermen/VADistiller) project.

## License

These models are licensed under **AGPL-3.0-or-later**. See [LICENSE-AGPL-3.0.txt](LICENSE-AGPL-3.0.txt) for details.

Copyright (c) 2025 Arpad Buermen. All rights reserved.

## Source

- Repository: https://codeberg.org/arpadbuermen/VADistiller
- Description: SPICE3 models converted to Verilog-A

## Models Included

| Model | Description | Status |
|-------|-------------|--------|
| resistor.va | SPICE resistor with temp coefficients | Working |
| capacitor.va | SPICE capacitor | Working |
| inductor.va | SPICE inductor | Working |
| diode.va | SPICE diode | Parser limitation (analysis() function) |
| bjt.va | Bipolar Junction Transistor | Needs analysis() support |
| mos1.va | Level 1 MOSFET | Needs analysis() support |
| mos2.va | Level 2 MOSFET | Needs analysis() support |
| mos3.va | Level 3 MOSFET | Needs analysis() support |
| mos6.va | Level 6 MOSFET | Needs analysis() support |
| mos9.va | Level 9 MOSFET (BSIM1) | Needs analysis() support |
| jfet1.va | Level 1 JFET | Needs analysis() support |
| jfet2.va | Level 2 JFET (Parker-Skellern) | Needs analysis() support |
| mes1.va | Level 1 MESFET (Statz) | Needs analysis() support |
| vdmos.va | Vertical DMOS | Needs analysis() support |
| bsim3v3.va | BSIM3v3 MOSFET | Complex - needs many features |
| bsim4v8.va | BSIM4v8 MOSFET | Complex - needs many features |

## Notes

- Parameter `m` (device multiplicity) is replaced with `$mfactor`
- Model parameters that conflict with instance parameters are prefixed with `model_`
- These are the "default" variant models (not the "sn" stripped-node variants)
