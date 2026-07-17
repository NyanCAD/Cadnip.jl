// Smoke test / usage example for the Spectre parser's clean C++ API.
//
// Demonstrates that a C++ consumer sees ordinary value types — `netlist::Netlist`
// with `rust::Vec<netlist::Instance>`, `rust::String` fields, range-for — with no
// handle lifetimes to manage. Reads a .scs file, parses it, prints a structured
// summary, and returns non-zero if the parse reported errors.

#include "netlist_cxx/netlist.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string str(const rust::String &s) { return std::string(s); }

void print_params(const rust::Vec<netlist::Param> &params, const char *indent) {
  for (const auto &p : params) {
    std::cout << indent << str(p.name) << " = " << str(p.value) << "\n";
  }
}

void print_subckt(const netlist::Subckt &s, const std::string &indent) {
  std::cout << indent << (s.is_inline ? "inline subckt " : "subckt ")
            << str(s.name) << " (";
  bool first = true;
  for (const auto &port : s.ports) {
    std::cout << (first ? "" : " ") << str(port);
    first = false;
  }
  std::cout << ")\n";
  print_params(s.params, (indent + "  params: ").c_str());
  for (const auto &m : s.models) {
    std::cout << indent << "  model " << str(m.name) << " " << str(m.master)
              << "\n";
  }
  for (const auto &inst : s.instances) {
    std::cout << indent << "  " << str(inst.name) << " -> " << str(inst.master)
              << " [" << inst.nodes.size() << " nodes]\n";
  }
  for (const auto &cond : s.conditionals) {
    std::cout << indent << "  conditional (" << cond.clauses.size()
              << " clauses):\n";
    for (const auto &cl : cond.clauses) {
      std::cout << indent << "    "
                << (cl.condition.size() ? str(cl.condition) : std::string("else"))
                << " => " << str(cl.instance.name) << "\n";
    }
  }
  for (const auto &nested : s.subckts) {
    print_subckt(nested, indent + "  ");
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <file.scs>\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  std::string source = ss.str();

  netlist::Netlist nl = netlist::parse_spectre_netlist(rust::Str(source));

  std::cout << "=== global parameters ===\n";
  print_params(nl.params, "  ");

  std::cout << "=== top-level instances ===\n";
  for (const auto &inst : nl.instances) {
    std::cout << "  " << str(inst.name) << " -> " << str(inst.master) << " (";
    bool first = true;
    for (const auto &n : inst.nodes) {
      std::cout << (first ? "" : " ") << str(n);
      first = false;
    }
    std::cout << ")\n";
    print_params(inst.params, "      ");
  }

  std::cout << "=== models ===\n";
  for (const auto &m : nl.models) {
    std::cout << "  " << str(m.name) << " " << str(m.master) << " ["
              << m.params.size() << " params]\n";
  }

  std::cout << "=== subckts ===\n";
  for (const auto &s : nl.subckts) {
    print_subckt(s, "  ");
  }

  std::cout << "=== analyses ===\n";
  for (const auto &a : nl.analyses) {
    std::cout << "  " << str(a.name) << " : " << str(a.analysis_type) << "\n";
    print_params(a.params, "      ");
  }

  std::cout << "=== saves / ic / globals / includes ===\n";
  for (const auto &s : nl.saves)
    std::cout << "  save " << str(s.signal) << "\n";
  for (const auto &ic : nl.ics)
    std::cout << "  ic " << str(ic.node) << " = " << str(ic.value) << "\n";
  for (const auto &g : nl.globals)
    std::cout << "  global " << str(g) << "\n";
  for (const auto &inc : nl.includes)
    std::cout << "  include " << str(inc.path)
              << (inc.section.size() ? " section=" + str(inc.section) : "")
              << "\n";

  if (!nl.errors.empty()) {
    std::cout << "=== parse errors ===\n";
    for (const auto &e : nl.errors)
      std::cout << "  error at bytes [" << e.start << ", " << e.end << ")\n";
    return 1;
  }
  std::cout << "parse OK, no errors\n";
  return 0;
}
