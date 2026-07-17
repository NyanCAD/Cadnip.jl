# Demonstrates consuming the Spectre parser's clean C++ API from CMake.
#
# The Rust crate `netlist-cxx` (../) exposes a `#[cxx::bridge]`. This build:
#   1. compiles the crate to a static library via cargo,
#   2. runs the `cxxbridge` codegen tool to emit the C++ header + shim source,
#   3. compiles the shim + the demo, and links the static library.
#
# No registry publish and no FetchContent required — everything is path-local.
#
# Prerequisites on PATH: cargo, and cxxbridge (`cargo install cxxbridge-cmd`).
#
#   cmake -S . -B build && cmake --build build
#   ./build/smoke ../example.scs

cmake_minimum_required(VERSION 3.18)
project(netlist_cxx_demo CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- locate the tools -------------------------------------------------------
find_program(CARGO cargo REQUIRED)
find_program(CXXBRIDGE cxxbridge
  HINTS "$ENV{HOME}/.cargo/bin"
  REQUIRED)

# Crate layout: <workspace>/crates/netlist-cxx, workspace root two levels up.
set(CRATE_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(WORKSPACE   "${CRATE_DIR}/../..")
set(BRIDGE_RS   "${CRATE_DIR}/src/lib.rs")

# Build profile → cargo target subdir + flag.
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
  set(CARGO_PROFILE_DIR "debug")
  set(CARGO_PROFILE_FLAG "")
else()
  set(CARGO_PROFILE_DIR "release")
  set(CARGO_PROFILE_FLAG "--release")
endif()

set(RUST_LIB "${WORKSPACE}/target/${CARGO_PROFILE_DIR}/libnetlist_cxx.a")

# --- 1. build the Rust static library --------------------------------------
add_custom_command(
  OUTPUT "${RUST_LIB}"
  COMMAND ${CARGO} build ${CARGO_PROFILE_FLAG} -p netlist-cxx
  WORKING_DIRECTORY "${WORKSPACE}"
  DEPENDS "${BRIDGE_RS}"
  COMMENT "cargo build netlist-cxx (${CARGO_PROFILE_DIR})"
  VERBATIM)
add_custom_target(rust_netlist_cxx DEPENDS "${RUST_LIB}")

# --- 2. generate the cxx header + shim from the bridge ----------------------
set(GEN_DIR    "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GEN_HEADER "${GEN_DIR}/netlist_cxx/netlist.h")
set(GEN_SHIM   "${GEN_DIR}/netlist_cxx/shim.cc")

file(MAKE_DIRECTORY "${GEN_DIR}/netlist_cxx")
add_custom_command(
  OUTPUT "${GEN_HEADER}" "${GEN_SHIM}"
  COMMAND ${CXXBRIDGE} "${BRIDGE_RS}" --header -o "${GEN_HEADER}"
  COMMAND ${CXXBRIDGE} "${BRIDGE_RS}" -o "${GEN_SHIM}"
  DEPENDS "${BRIDGE_RS}"
  COMMENT "cxxbridge codegen"
  VERBATIM)

# --- 3. compile + link ------------------------------------------------------
add_executable(smoke "${CMAKE_CURRENT_SOURCE_DIR}/smoke.cpp" "${GEN_SHIM}")
add_dependencies(smoke rust_netlist_cxx)
target_include_directories(smoke PRIVATE "${GEN_DIR}")
target_link_libraries(smoke PRIVATE "${RUST_LIB}" pthread dl m)
