//! Spectre keyword recognition via a trie.
//!
//! Ports `Tries.jl` + the tail of `lex_identifier` (tokenize/lexer.jl). The trie
//! is built from Spectre `reserved_words`, which is derived from the keyword
//! `Kind`s by this Julia rule (token_kinds.jl):
//!
//! ```julia
//! for k in instances(Kind)
//!     is_kw(k) || continue
//!     all(x->isuppercase(x)||x=='_', string(k)) || continue   # excludes names w/ digits
//!     str = is_builtin_const(k) ? string(k) : lowercase(string(k))
//!     reserved_words[str] = k
//! end
//! ```
//!
//! Two consequences we reproduce exactly:
//! 1. Enum names containing a DIGIT are excluded (`M_PI` in, `M_1_PI`/`M_SQRT2`
//!    out; `ATAN` in, `ATAN2`/`LOG10` out). The lowercase `begin_*`/`end_*`
//!    markers are likewise excluded (they contain lowercase letters).
//! 2. Builtin *constants* keep their upper-case source spelling (`M_PI`, `P_Q`);
//!    every other reserved word is the lower-cased enum name (`model`, `dc`,
//!    `sqrt`).
//!
//! CRITICAL difference from `keywords.rs` (SPICE): the Spectre `lex_identifier`
//! (lexer.jl lines 428-432) does NOT do unique-prefix completion — it matches
//! the *entire* identifier text against the trie exactly (`t.is_key` at the
//! terminal node). So `.param` does NOT complete to `parameters` here; only the
//! full word `parameters` resolves. `lookup` therefore performs an exact match.

use crate::spectre_syntax_kind::TokenKind;
use crate::spectre_syntax_kind::TokenKind::*;
use std::collections::BTreeMap;

/// Every reserved word and its source spelling, per the Julia rule above.
/// Non-builtin-const keywords are lower-cased; builtin constants keep uppercase.
const KEYWORDS: &[(&str, TokenKind)] = &[
    // --- general keywords (lower-cased) ---
    ("correlate", CORRELATE),
    ("else", ELSE),
    ("end", END),
    ("ends", ENDS),
    ("export", EXPORT),
    ("for", FOR),
    ("function", FUNCTION),
    ("global", GLOBAL),
    ("if", IF),
    ("inline", INLINE),
    ("library", LIBRARY),
    ("local", LOCAL),
    ("march", MARCH),
    ("model", MODEL),
    ("parameters", PARAMETERS),
    ("paramtest", PARAMTEST),
    ("plot", PLOT),
    ("print", PRINT),
    ("real", REAL),
    ("return", RETURN),
    ("subckt", SUBCKT),
    ("to", TO),
    ("vary", VARY),
    ("simulator", SIMULATOR),
    ("lang", LANG),
    ("spectre", SPECTRE),
    ("spice", SPICE),
    ("include", INCLUDE),
    ("ahdl_include", AHDL_INCLUDE),
    // --- control (lower-cased) ---
    ("alter", ALTER),
    ("altergroup", ALTERGROUP),
    ("assert", ASSERT),
    ("check", CHECK),
    ("checklimit", CHECKLIMIT),
    ("info", INFO),
    ("options", OPTIONS),
    ("paramset", PARAMSET),
    ("set", SET),
    ("shell", SHELL),
    ("ic", IC),
    ("nodeset", NODESET),
    ("save", SAVE),
    ("statistics", STATISTICS),
    // --- analyses (lower-cased) ---
    ("dc", DC),
    ("ac", AC),
    ("noise", NOISE),
    ("xf", XF),
    ("sp", SP),
    ("tran", TRAN),
    ("tdr", TDR),
    ("pz", PZ),
    ("envlp", ENVLP),
    ("pac", PAC),
    ("pdisto", PDISTO),
    ("pnoise", PNOISE),
    ("pss", PSS),
    ("pxf", PXF),
    ("sens", SENS),
    ("fourier", FOURIER),
    ("dcmatch", DCMATCH),
    ("stb", STB),
    ("sweep", SWEEP),
    ("montecarlo", MONTECARLO),
    ("section", SECTION),
    // --- builtin constants (KEEP uppercase; digit-free names only) ---
    ("M_DEGPERRAD", M_DEGPERRAD),
    ("M_E", M_E),
    ("M_PI", M_PI),
    ("M_TWO_PI", M_TWO_PI),
    ("P_C", P_C),
    ("P_H", P_H),
    ("P_K", P_K),
    ("P_Q", P_Q),
    // --- builtin functions (lower-cased; digit-free names only) ---
    ("abs", ABS),
    ("acos", ACOS),
    ("acosh", ACOSH),
    ("asin", ASIN),
    ("asinh", ASINH),
    ("atan", ATAN),
    ("atanh", ATANH),
    ("ceil", CEIL),
    ("cos", COS),
    ("cosh", COSH),
    ("exp", EXP),
    ("floor", FLOOR),
    ("fmod", FMOD),
    ("hypot", HYPOT),
    ("int", INT),
    ("log", LOG),
    ("nint", NINT),
    ("max", MAX),
    ("min", MIN),
    ("pow", POW),
    ("sin", SIN),
    ("sinh", SINH),
    ("sqrt", SQRT),
    ("tan", TAN),
    ("tanh", TANH),
    ("truncate", TRUNCATE),
    // --- save keywords (lower-cased) ---
    ("currents", CURRENTS),
    ("static", STATIC),
    ("displacement", DISPLACEMENT),
    ("dynamic", DYNAMIC),
    ("oppoint", OPPOINT),
    ("probe", PROBE),
    ("pwr", PWR),
    ("all", ALL),
];

struct TrieNode {
    value: Option<TokenKind>,
    is_key: bool,
    children: BTreeMap<char, usize>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode { value: None, is_key: false, children: BTreeMap::new() }
    }
}

pub struct KeywordTrie {
    nodes: Vec<TrieNode>,
}

impl KeywordTrie {
    pub fn new() -> Self {
        let mut trie = KeywordTrie { nodes: vec![TrieNode::new()] };
        for &(name, kind) in KEYWORDS {
            trie.insert(name, kind);
        }
        trie
    }

    fn insert(&mut self, key: &str, value: TokenKind) {
        let mut node = 0usize;
        for ch in key.chars() {
            node = match self.nodes[node].children.get(&ch) {
                Some(&next) => next,
                None => {
                    let next = self.nodes.len();
                    self.nodes.push(TrieNode::new());
                    self.nodes[node].children.insert(ch, next);
                    next
                }
            };
        }
        self.nodes[node].is_key = true;
        self.nodes[node].value = Some(value);
    }

    /// Look up an identifier's full text. Returns the keyword kind on an EXACT
    /// match, or `None` for a plain identifier. Mirrors the tail of the Spectre
    /// `lex_identifier`: no unique-prefix completion (unlike SPICE).
    pub fn lookup(&self, text: &str) -> Option<TokenKind> {
        let mut node = 0usize;
        for ch in text.chars() {
            node = *self.nodes[node].children.get(&ch)?;
        }
        if self.nodes[node].is_key {
            self.nodes[node].value
        } else {
            None
        }
    }
}

impl Default for KeywordTrie {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_keywords() {
        let t = KeywordTrie::new();
        assert_eq!(t.lookup("model"), Some(MODEL));
        assert_eq!(t.lookup("tran"), Some(TRAN));
        assert_eq!(t.lookup("parameters"), Some(PARAMETERS));
        assert_eq!(t.lookup("subckt"), Some(SUBCKT));
        assert_eq!(t.lookup("dc"), Some(DC));
    }

    #[test]
    fn builtin_constants_keep_uppercase() {
        let t = KeywordTrie::new();
        assert_eq!(t.lookup("M_PI"), Some(M_PI));
        assert_eq!(t.lookup("P_Q"), Some(P_Q));
        // Lower-case spelling is NOT a keyword.
        assert_eq!(t.lookup("m_pi"), None);
    }

    #[test]
    fn no_prefix_completion() {
        let t = KeywordTrie::new();
        // Unlike SPICE, a prefix does NOT complete: only the full word resolves.
        assert_eq!(t.lookup("param"), None);
        assert_eq!(t.lookup("tra"), None);
        assert_eq!(t.lookup("mod"), None);
    }

    #[test]
    fn digit_names_excluded() {
        let t = KeywordTrie::new();
        // Enum names containing a digit are excluded from reserved_words.
        assert_eq!(t.lookup("atan2"), None);
        assert_eq!(t.lookup("log10"), None);
        // ...but their digit-free siblings are present.
        assert_eq!(t.lookup("atan"), Some(ATAN));
        assert_eq!(t.lookup("log"), Some(LOG));
    }

    #[test]
    fn non_keywords() {
        let t = KeywordTrie::new();
        assert_eq!(t.lookup("foo"), None);
        assert_eq!(t.lookup("r1"), None);
        assert_eq!(t.lookup("vdd"), None);
    }

    #[test]
    fn end_is_key_with_children() {
        let t = KeywordTrie::new();
        // `end` is a key even though `ends` extends it.
        assert_eq!(t.lookup("end"), Some(END));
        assert_eq!(t.lookup("ends"), Some(ENDS));
    }
}
