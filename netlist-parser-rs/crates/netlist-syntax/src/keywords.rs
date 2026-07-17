//! Keyword recognition via a trie with unique-prefix completion.
//!
//! Ports `Tries.jl` + the `KW_TRIE`/`subtrie` logic at the end of
//! `lex_identifier` (tokenize/lexer.jl). The Julia trie is built from
//! `reserved_words` (every keyword `Kind`'s enum-name string). Matching descends
//! by character, then — crucially — completes a *unique prefix* to its key:
//!
//! ```text
//! while !node.is_key && node.children.len() == 1 { node = only_child }
//! if node.is_key { Some(kind) } else { None }
//! ```
//!
//! So `.param` → `PARAMETERS`, `.tra` → `TRAN`. The Julia enum's lowercase
//! `begin_*`/`end_*` marker names are also in `reserved_words`, but the lexer
//! upper-cases all input, so those lowercase-edged subtrees are unreachable and
//! cannot affect completion — we omit them.

use crate::syntax_kind::TokenKind;
use crate::syntax_kind::TokenKind::*;
use std::collections::BTreeMap;

/// Every keyword `TokenKind` and the (upper-case) source spelling of its Julia
/// enum name. Only real keywords — markers are inert (see module docs).
const KEYWORDS: &[(&str, TokenKind)] = &[
    ("DC", DC),
    ("AC", AC),
    ("TRAN", TRAN),
    ("IC", IC),
    ("TF", TF),
    ("NOISE", NOISE),
    ("PRINT", PRINT),
    ("PLOT", PLOT),
    ("MEASURE", MEASURE),
    ("IF", IF),
    ("ELSE", ELSE),
    ("ELSEIF", ELSEIF),
    ("ENDIF", ENDIF),
    ("MODEL", MODEL),
    ("LIB", LIB),
    ("INCLUDE", INCLUDE),
    ("SUBCKT", SUBCKT),
    ("END", END),
    ("ENDL", ENDL),
    ("ENDS", ENDS),
    ("PARAMETERS", PARAMETERS),
    ("CSPARAM", CSPARAM),
    ("OPTIONS", OPTIONS),
    ("TEMP", TEMP),
    ("WIDTH", WIDTH),
    ("GLOBAL", GLOBAL),
    ("DEV", DEV),
    ("LOT", LOT),
    ("SIMULATOR", SIMULATOR),
    ("LANG", LANG),
    ("SPECTRE", SPECTRE),
    ("SPICE", SPICE),
    ("TITLE", TITLE),
    ("DATA", DATA),
    ("ENDDATA", ENDDATA),
    ("ON", ON),
    ("OFF", OFF),
    ("HDL", HDL),
    ("FIND", FIND),
    ("DERIV", DERIV),
    ("WHEN", WHEN),
    ("AT", AT),
    ("TD", TD),
    ("RISE", RISE),
    ("FALL", FALL),
    ("CROSS", CROSS),
    ("LAST", LAST),
    ("AVG", AVG),
    ("MAX", MAX),
    ("MIN", MIN),
    ("PP", PP),
    ("RMS", RMS),
    ("INTEG", INTEG),
    ("TRIG", TRIG),
    ("VAL", VAL),
    ("TARG", TARG),
    ("PULSE", PULSE),
    ("SIN", SIN),
    ("EXP", EXP),
    ("PWL", PWL),
    ("SFFM", SFFM),
    ("AM", AM),
    ("TRNOISE", TRNOISE),
    ("TRRANDOM", TRRANDOM),
    ("POLY", POLY),
    ("TABLE", TABLE),
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

    /// Look up an identifier's text. Returns the completed keyword kind, or
    /// `None` if it is a plain identifier. Mirrors the tail of `lex_identifier`.
    pub fn lookup(&self, text: &str) -> Option<TokenKind> {
        let mut node = 0usize;
        for ch in text.chars() {
            node = *self.nodes[node].children.get(&ch)?;
        }
        // Unique-prefix completion.
        while !self.nodes[node].is_key && self.nodes[node].children.len() == 1 {
            node = *self.nodes[node].children.values().next().unwrap();
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
        assert_eq!(t.lookup("MODEL"), Some(MODEL));
        assert_eq!(t.lookup("TRAN"), Some(TRAN));
        assert_eq!(t.lookup("PARAMETERS"), Some(PARAMETERS));
    }

    #[test]
    fn unique_prefix_completion() {
        let t = KeywordTrie::new();
        // `.param` completes to PARAMETERS.
        assert_eq!(t.lookup("PARAM"), Some(PARAMETERS));
        assert_eq!(t.lookup("PARAME"), Some(PARAMETERS));
        // `.tra` completes to TRAN.
        assert_eq!(t.lookup("TRA"), Some(TRAN));
    }

    #[test]
    fn non_keywords() {
        let t = KeywordTrie::new();
        assert_eq!(t.lookup("FOO"), None);
        assert_eq!(t.lookup("R1"), None);
        assert_eq!(t.lookup("VIN"), None);
    }

    #[test]
    fn key_with_children_stays_put() {
        let t = KeywordTrie::new();
        // END is a key even though ENDL/ENDS/ENDIF extend it.
        assert_eq!(t.lookup("END"), Some(END));
        assert_eq!(t.lookup("ENDL"), Some(ENDL));
        assert_eq!(t.lookup("ENDS"), Some(ENDS));
    }
}
