"""Smoke tests for the netlist_parser Python bindings."""
import netlist_parser


def find_child(node, kind):
    """Return first direct child with the given kind, or None."""
    for child in node.children:
        if child.kind == kind:
            return child
    return None


def walk(node):
    """Yield node and all descendants (pre-order)."""
    yield node
    for child in node.children:
        yield from walk(child)


def test_parse_basic():
    src = "* t\nR1 a b 1k\n"
    root = netlist_parser.parse_spice(src)

    # Root kind
    assert root.kind == "SPICENetlistSource", f"expected SPICENetlistSource, got {root.kind!r}"

    # Direct children include Title and Resistor
    kinds = [c.kind for c in root.children]
    assert "Title" in kinds, f"no Title child; children: {kinds}"
    assert "Resistor" in kinds, f"no Resistor child; children: {kinds}"

    # Resistor span is (4, 14) — covers "R1 a b 1k\n"
    resistor = find_child(root, "Resistor")
    assert resistor is not None
    assert resistor.span == (4, 14), f"Resistor span: {resistor.span}"

    # Deep walk: find Identifier token with text "R1"
    idents = [n for n in walk(root) if n.kind == "Identifier" and n.text == "R1"]
    assert len(idents) >= 1, "no Identifier with text 'R1' found in tree"


def test_errors():
    # ".model" without a name is malformed — parser emits exactly one Error token
    src = "* t\n.model\nR2 a b 2k\n"
    root = netlist_parser.parse_spice(src)
    errs = netlist_parser.errors(root)
    assert len(errs) == 1, f"expected 1 error, got {len(errs)}: {errs}"
    start, end = errs[0]
    assert end - start == 1, f"expected error span width 1, got {end - start}: ({start}, {end})"
