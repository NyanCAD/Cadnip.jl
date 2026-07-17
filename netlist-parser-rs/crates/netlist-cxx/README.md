# netlist-cxx

A clean **C++ binding** for the Spectre netlist parser, built with
[cxx](https://cxx.rs).

It exposes an *eager, owned projection* of the Spectre typed AST
(`netlist_syntax::spectre_ast`) as plain-old-data value structs (`netlist::Netlist`
with `rust::Vec<netlist::Instance>`, `rust::String` fields). A C++ consumer sees
ordinary value types and range-for — no opaque handles, no FFI lifetime
management:

```cpp
#include "netlist_cxx/netlist.h"

netlist::Netlist nl = netlist::parse_spectre_netlist(rust::Str(source));
for (const auto &inst : nl.instances)
    std::cout << std::string(inst.name) << " -> " << std::string(inst.master) << "\n";
```

## Why eager POD (not lazy handles)?

rowan already materialises the whole tree at parse time, so a lazy view saves no
parsing. What it *adds* is a heap allocation + an FFI crossing per node touched —
and any real consumer (building simulator parser tables, lowering to a flat
device list) walks essentially the whole netlist. The eager projection walks the
tree once in Rust and hands the data across in one transfer. It is also the
natural shape of the neutral structural IR downstream tools want.

Parameter/expression values are carried as **verbatim source text**; the consumer
re-parses them with its own expression engine.

## FFI boundary

Rust has no stable ABI and cannot emit the C++ ABI, so the linkage boundary is
`extern "C"` under the hood — cxx generates ergonomic C++ types on top of it. This
crate is the *exporting* side (`extern "Rust"` only), so it is pure Rust and needs
no `build.rs`: the C++ shim + header are produced by the `cxxbridge` CLI at the
consumer's build time.

## Building a C++ consumer

Prerequisites on `PATH`: `cargo`, and `cxxbridge` (`cargo install cxxbridge-cmd`).

The recipe (see `demo/CMakeLists.txt` for a working CMake version):

```sh
# 1. static library
cargo build -p netlist-cxx                       # -> target/debug/libnetlist_cxx.a

# 2. codegen: C++ header + shim from the bridge
cxxbridge crates/netlist-cxx/src/lib.rs --header -o gen/netlist_cxx/netlist.h
cxxbridge crates/netlist-cxx/src/lib.rs          -o gen/netlist_cxx/shim.cc

# 3. compile + link
g++ -std=c++20 -Igen consumer.cpp gen/netlist_cxx/shim.cc \
    target/debug/libnetlist_cxx.a -lpthread -ldl -lm -o consumer
```

## Demo

```sh
cmake -S crates/netlist-cxx/demo -B build
cmake --build build
./build/smoke crates/netlist-cxx/demo/example.scs
```
