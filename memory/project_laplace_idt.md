---
name: Laplace/idt implementation planning
description: Design decisions for implementing laplace_* and idt() VA operators - pole-residue vs state-space, numerical stability concerns
type: project
---

Implementing Verilog-A `laplace_nd`, `laplace_zp`, and `idt()` operators.

**Decided approach:**
- Use DescriptorSystems.jl (already a dep) for tf→dss conversion via `dss(rtf(Polynomial(num), Polynomial(den)))`
- Use `gprescale` for numerical conditioning (TunableFilter has 70 OOM coefficient span)
- Parse VA `{...}` array literals as Julia tuples for natural constant folding
- Only supporting `laplace_nd` and `laplace_zp`, not `_np`/`_zd`
- `idt()` as internal state node: `dx/dt = expr`, `x(0) = ic`
- ControlSystems.jl and DescriptorSystems.jl are separate ecosystems with incompatible types

**Why:** Photonic models (PhotoDetector, TunableFilter) need `laplace_nd`, PSP103 NQS needs `idt()`.

**How to apply:** See plan file for full implementation details.
