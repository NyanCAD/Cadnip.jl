# Claude Development Notes

## Environment

- **Julia is installed via juliaup** and is NOT on PATH
  - Use `~/.juliaup/bin/julia` to run Julia
  - Example: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'`

## Development Guidelines

### MNA Backend Migration

- **DO NOT maintain backward compatibility with DAECompiler**
- Update existing APIs to use the new MNA backend directly
- When modifying sweep/simulation code, replace DAECompiler patterns with MNA equivalents
- Do not create duplicate types (e.g., `MNACircuitSweep`) - modify existing types instead

### Key MNA Components

- `MNASim`: Parameterized circuit simulation wrapper
- `MNAContext`: Circuit builder context for stamping components
- `alter()`: Create new simulation with modified parameters
- `dc!()` / `tran!()`: DC and transient analysis
- `CircuitSweep`: Parameter sweep over MNA circuits

## MNA Documentation

Read these files in `doc/` for detailed design information:

| File | Description |
|------|-------------|
| `doc/mna_design.md` | **Start here.** Core design decisions, key principles, sign conventions, and why we're migrating from DAECompiler |
| `doc/mna_architecture.md` | Detailed architecture: out-of-place evaluation, explicit parameter passing, GPU compatibility design |
| `doc/mna_ad_stamping.md` | Advanced: how to use ForwardDiff to extract Jacobians and handle Verilog-A contributions without AST analysis |
| `doc/mna_changelog.md` | Migration progress tracker - shows which phases are complete and what's remaining |

## Testing

Run MNA tests directly to avoid Enzyme precompilation issues:
```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["mna"])'
```

Or run specific test files:
```bash
~/.juliaup/bin/julia --project=. test/mna/core.jl
```
