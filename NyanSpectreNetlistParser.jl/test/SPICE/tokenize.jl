using Test

using NyanSpectreNetlistParser: SPICENetlistParser
using .SPICENetlistParser.SPICENetlistTokenize.Tokens
using .SPICENetlistParser.SPICENetlistTokenize: tokenize, Tokens, kind, next_token
using NyanLexers: is_triv

using DeepDiffs

token_test =
[
".TITLE THIS IS A TILE WITH ¤%&/)/& stuff" => [DOT, TITLE, TITLE_LINE]
"Vv-_-{} A B 0" => [IDENTIFIER_VOLTAGE, IDENTIFIER, IDENTIFIER, NUMBER]
"* MOSFET" => nothing
".GLOBAL VDD NET1" => [DOT, GLOBAL, IDENTIFIER, IDENTIFIER]
".MODEL BJT_modName NPN (BF=val)" => [DOT, MODEL, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ, VAL]
"Rname N1 N2 0.1 \$ comment" => [IDENTIFIER_RESISTOR, IDENTIFIER, IDENTIFIER, NUMBER]
"sky130.0" => [IDENTIFIER_SWITCH, DOT, NUMBER]
".param freq = 1Meg" => [DOT, PARAMETERS, IDENTIFIER, EQ, NUMBER]
".parameter freq = 1Meg" => [DOT, PARAMETERS, IDENTIFIER, EQ, NUMBER]
"*comment" => []
"    *comment" => []
";comment" => []
"    ;comment" => []
"1    ;comment" => [NUMBER]
"{1*1}" => [LBRACE, NUMBER, STAR, NUMBER, RBRACE]
".lib 'with spaces/sm141064.ngspice' nmos_6p0_t" => [DOT, LIB, STRING, IDENTIFIER]
".lib sm141064.ngspice nmos_6p0_t" => [DOT, LIB, IDENTIFIER, IDENTIFIER]
".include ./foo" => [DOT, INCLUDE, IDENTIFIER]
".include './foo.bar'" => [DOT, INCLUDE, STRING]
".param r_l='s*(r_length-2*r_dl)'" => [DOT, PARAMETERS, IDENTIFIER, EQ, PRIME, IDENTIFIER, STAR, LPAREN,
                                      IDENTIFIER, MINUS, NUMBER, STAR, IDENTIFIER, RPAREN, PRIME]
"Q1 Net-_Q1-C_ Net-_Q1-B_ 0 BC546B" => [IDENTIFIER_BIPOLAR_TRANSISTOR, IDENTIFIER, IDENTIFIER, NUMBER, IDENTIFIER]
"X1 v+ v- r={a-b} f-o-o=1" => [IDENTIFIER_SUBCIRCUIT_CALL, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ,
                               LBRACE, IDENTIFIER, MINUS, IDENTIFIER, RBRACE, IDENTIFIER, EQ, NUMBER]
".ic v( m_tn4:d )=  1.225e-08" => [DOT, IC, VAL, LPAREN, IDENTIFIER, COLON, IDENTIFIER, RPAREN, EQ, NUMBER]
"V1 vin 0 SIN (0, 1, 1k)" => [IDENTIFIER_VOLTAGE, IDENTIFIER, NUMBER, SIN, NUMBER, NUMBER, NUMBER]
"R1 (a b) r=\"foo+bar\"" => [IDENTIFIER_RESISTOR, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ, PRIME, IDENTIFIER, PLUS, IDENTIFIER, PRIME]
".MEAS TRAN res1 FIND V(out) AT=5m" => [DOT, MEASURE, TRAN, IDENTIFIER, FIND, VAL, LPAREN, IDENTIFIER, RPAREN, AT, EQ, NUMBER]
".tran 1ns 60ns" => [DOT, TRAN, NUMBER, NUMBER]
".model 1N3064 D" => [DOT, MODEL, NUMBER, IDENTIFIER]
# Additional NUMBER tokenization edge cases
"2N2222" => [NUMBER]  # Another model name starting with digits
"R1 n1 n2 r={1.5e-3}" => [IDENTIFIER_RESISTOR, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ, LBRACE, NUMBER, RBRACE]  # Scientific notation in expression
"R2 n1 n2 r={1foe-bar}" => [IDENTIFIER_RESISTOR, IDENTIFIER, IDENTIFIER, IDENTIFIER, EQ, LBRACE, NUMBER, MINUS, IDENTIFIER, RBRACE]  # Expression context limits consumption
"C1 n1 n2 1.5e-12F" => [IDENTIFIER_CAPACITOR, IDENTIFIER, IDENTIFIER, NUMBER]  # Scientific notation with unit
# Array tokenization cases
".OPTIONS montequantiles=[0.134 99.865]" => [DOT, OPTIONS, IDENTIFIER, EQ, LSQUARE, NUMBER, NUMBER, RSQUARE]  # Array in OPTIONS
".param vals=[1 2 3]" => [DOT, PARAMETERS, IDENTIFIER, EQ, LSQUARE, NUMBER, NUMBER, NUMBER, RSQUARE]  # Array in param
".OPTIONS someopt=[1,2]" => [DOT, OPTIONS, IDENTIFIER, EQ, LSQUARE, NUMBER, COMMA, NUMBER, RSQUARE]  # Array with comma
".param x=[1.5e-3]" => [DOT, PARAMETERS, IDENTIFIER, EQ, LSQUARE, NUMBER, RSQUARE]  # Single element array
# Base specifier number cases (Verilog-style)
"123'hAB" => [NUMBER]  # Hex base specifier
"8'hFF" => [NUMBER]  # 8-bit hex
"4'b1010" => [NUMBER]  # Binary base specifier
"8'o377" => [NUMBER]  # Octal base specifier
"16'h1234" => [NUMBER]  # 16-bit hex
".param myval=123'hAB" => [DOT, PARAMETERS, IDENTIFIER, EQ, NUMBER]  # Base specifier in param
# behavioural source cases
"EOS 7 1 POLY(1) 16 49 2E-3 1" => [IDENTIFIER_VOLTAGE_CONTROLLED_VOLTAGE, NUMBER, NUMBER, POLY, NUMBER, NUMBER, NUMBER, NUMBER, NUMBER]
"GD16 16 1 TABLE {V(16,1)} ((-100,-1p)(0,0)(1m,1u)(2m,1m))" => [IDENTIFIER_VOLTAGE_CONTROLLED_CURRENT, NUMBER, NUMBER, TABLE, LBRACE, VAL, LPAREN, NUMBER, COMMA, NUMBER, RPAREN, RBRACE, MINUS, NUMBER, MINUS, NUMBER, NUMBER, NUMBER, NUMBER, NUMBER, NUMBER, NUMBER]
"" => [ENDMARKER]
]

str = join((first(x) for x in token_test), '\n')
tokenized_kinds = filter(!is_triv, kind.(collect(tokenize(str, ERROR, next_token; case_sensitive=false))))
true_kinds = collect(Iterators.flatten([last(x) for x in token_test if last(x) !== nothing]))

if tokenized_kinds != true_kinds
    println(deepdiff(tokenized_kinds, true_kinds))
    error()
end
