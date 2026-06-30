#!/usr/bin/env julia
# Profile c6288 setup + per-iter cost. Print phase times so we can see where time goes.

using Cadnip
using Cadnip.MNA
using Cadnip.MNA: CedarUICOp, CedarTranOp, fast_rebuild!, fast_residual!, fast_jacobian!,
                  build_with_detection, compile_structure, create_workspace, system_size,
                  reset_for_restamping!, assemble!
using PSPModels
using Printf
using SparseArrays
using LinearAlgebra

# ---- 1. parse + sema + codegen + eval (top-level) -------------------------
# PSP103VA_module is exported by PSPModels
const spice_file = joinpath(@__DIR__, "runme.sp")

@printf("=== c6288 profiling ===\n")

t0 = time()
ast = Cadnip.NyanSpectreNetlistParser.parsefile(spice_file; start_lang=:spice, implicit_title=true)
t_parse = time() - t0
@printf("parse:                %8.2f s\n", t_parse)

t0 = time()
sema_result = Cadnip.sema(ast; imported_hdl_modules=[PSP103VA_module])
t_sema = time() - t0
@printf("sema:                 %8.2f s\n", t_sema)

t0 = time()
quoted = Cadnip._make_mna_circuit_with_sema(sema_result; circuit_name=:c6288_circuit)
t_codegen = time() - t0
@printf("codegen (build expr): %8.2f s\n", t_codegen)

t0 = time()
eval(quoted)
t_eval = time() - t0
@printf("eval (compile):       %8.2f s   (Julia JIT)\n", t_eval)

# ---- 2. first builder run (allocates MNAContext) --------------------------
t0 = time()
circuit = MNACircuit(c6288_circuit)
t_circuit_obj = time() - t0
@printf("MNACircuit ctor:      %8.2f s\n", t_circuit_obj)

# ---- 3. build_with_detection (5 builder passes) ---------------------------
t0 = time()
ctx = build_with_detection(circuit)
t_detect = time() - t0
@printf("build_with_detection: %8.2f s   (5 passes)\n", t_detect)
n = system_size(ctx)
@printf("  system size:        %8d vars\n", n)
@printf("  n_nodes:            %8d\n", ctx.n_nodes)
@printf("  n_currents:         %8d\n", ctx.n_currents)
@printf("  n_charges:          %8d\n", ctx.n_charges)
@printf("  G_I COO entries:    %8d\n", length(ctx.G_I))
@printf("  C_I COO entries:    %8d\n", length(ctx.C_I))
@printf("  internal nodes:     %8d\n", count(ctx.internal_node_flags))

# ---- 4. compile_structure (sparse pattern + COO->nzval mapping) -----------
t0 = time()
cs = compile_structure(circuit.builder, circuit.params, circuit.spec; ctx=ctx)
t_compile_struct = time() - t0
@printf("compile_structure:    %8.2f s\n", t_compile_struct)
@printf("  G nnz:              %8d\n", nnz(cs.G))
@printf("  C nnz:              %8d\n", nnz(cs.C))

# ---- 5. create_workspace + first fast_rebuild! ----------------------------
t0 = time()
ws = create_workspace(cs; ctx=ctx)
t_ws = time() - t0
@printf("create_workspace:     %8.2f s\n", t_ws)

# warm up + measure fast_rebuild!, residual, jacobian
u = zeros(n)
du = zeros(n)
resid = zeros(n)

t0 = time()
fast_rebuild!(ws, u, 0.0)
t_rebuild_first = time() - t0
@printf("fast_rebuild! (1st):  %8.2f s   (JIT warmup)\n", t_rebuild_first)

t0 = time()
fast_rebuild!(ws, u, 0.0)
t_rebuild = time() - t0
@printf("fast_rebuild! (warm): %8.4f s\n", t_rebuild)

t0 = time()
fast_residual!(resid, du, u, ws, 0.0)
t_resid = time() - t0
@printf("fast_residual! warm:  %8.4f s\n", t_resid)

J = copy(cs.G)  # sparse pattern
t0 = time()
fast_jacobian!(J, du, u, ws, 1.0, 0.0)
t_jac = time() - t0
@printf("fast_jacobian! warm:  %8.4f s\n", t_jac)

# ---- 6. measure single KLU factorize+solve --------------------------------
using LinearSolve
b = randn(n)
prob = LinearProblem(J, b)
t0 = time()
sol = solve(prob, KLUFactorization())
t_klu = time() - t0
@printf("KLU full solve (1st): %8.2f s\n", t_klu)
t0 = time()
sol2 = solve(prob, KLUFactorization())
t_klu_warm = time() - t0
@printf("KLU full solve warm:  %8.4f s\n", t_klu_warm)

@printf("\n=== summary ===\n")
@printf("setup total: %.1f s\n", t_parse + t_sema + t_codegen + t_eval + t_circuit_obj + t_detect + t_compile_struct + t_ws + t_rebuild_first)
@printf("per-step:    rebuild=%.3fs  resid=%.3fs  jac=%.3fs  klu=%.3fs\n", t_rebuild, t_resid, t_jac, t_klu_warm)
