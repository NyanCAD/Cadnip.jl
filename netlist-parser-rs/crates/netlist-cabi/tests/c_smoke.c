/* C ABI smoke test: parse a netlist, walk the root's children, exercise the
 * span/text/error API. Compiled and run by run_c_smoke.sh (and, in turn, the
 * Rust integration test). Exits non-zero on any failed assertion. */
#include "netlist_parser.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "FAIL: %s\n", msg);                                 \
            failures++;                                                         \
        }                                                                       \
    } while (0)

int main(void) {
    /* Title, a resistor, a malformed `.model`, then another resistor. */
    const char *src = "* t\nR1 a b 1k\n.model\nR2 a b 2k\n";
    size_t len = strlen(src);

    NlTree *tree = nl_parse_spice(src, len, 0 /* ngspice */);
    CHECK(tree != NULL, "parse returned null");
    if (!tree)
        return 1;

    NlNode *root = nl_tree_root(tree);
    CHECK(root != NULL, "root null");

    uint32_t rs = 0, re = 0;
    nl_node_span(root, &rs, &re);
    CHECK(rs == 0 && re == (uint32_t)len, "root span covers whole source");

    size_t n = nl_node_child_count(root);
    /* Title, Resistor, Incomplete(model), Resistor. */
    CHECK(n == 4, "root has 4 statement children");
    printf("root kind=%u span=%u-%u children=%zu\n", nl_node_kind(root), rs, re, n);

    for (size_t i = 0; i < n; i++) {
        NlNode *c = nl_node_child(root, i);
        CHECK(c != NULL, "child not null");
        uint32_t s = 0, e = 0;
        nl_node_span(c, &s, &e);
        char buf[64];
        size_t tl = nl_node_text(c, buf, sizeof(buf) - 1);
        if (tl > sizeof(buf) - 1)
            tl = sizeof(buf) - 1;
        buf[tl] = '\0';
        /* strip newlines for tidy printing */
        for (char *p = buf; *p; p++)
            if (*p == '\n')
                *p = ' ';
        printf("  child[%zu] kind=%u span=%u-%u text=\"%s\"\n", i, nl_node_kind(c), s, e, buf);
        nl_node_free(c);
    }

    /* The malformed `.model` line yields exactly one Error leaf (the newline). */
    size_t errc = nl_tree_error_count(tree);
    CHECK(errc == 1, "one error leaf");
    if (errc >= 1) {
        uint32_t es = 0, ee = 0;
        int ok = nl_tree_error(tree, 0, &es, &ee);
        CHECK(ok, "error 0 fetched");
        printf("error[0] span=%u-%u\n", es, ee);
        CHECK(ee - es == 1, "error span is one byte (the newline)");
    }

    nl_node_free(root);
    nl_tree_free(tree);

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("c_smoke: OK\n");
    return 0;
}
