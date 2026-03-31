using Test

using NyanSpectreNetlistParser
using NyanSpectreNetlistParser.SpectreNetlistTokenize.Tokens
using NyanSpectreNetlistParser.SpectreNetlistTokenize: tokenize, Tokens, kind, next_token
using NyanLexers: is_triv
using DeepDiffs

token_test =
[
"// foo" => nothing
"011" => [NUMBER]
"1.0" => [NUMBER]
"vdd!" => [IDENTIFIER]
"tan" => [TAN]
"tanh" => [TANH]
"march" => [MARCH]
"int" => [INT]
"2pf"=> [NUMBER]
"6.3ns" => [NUMBER]
"6_Ohms" => [NUMBER]
"0.3MHz" => [NUMBER]
"MHz" => [IDENTIFIER]
"a = 1 \\\nb=2" => [IDENTIFIER, EQ, NUMBER, ESCD_NEWLINE, IDENTIFIER, EQ, NUMBER]
"name info info=foo" => [IDENTIFIER, INFO, INFO, EQ, IDENTIFIER]
"tran tran tran=tran" => [TRAN, TRAN, TRAN, EQ, TRAN]
"save save=foo" => [SAVE, SAVE, EQ, IDENTIFIER]
"* comment" => nothing
"" => [ENDMARKER]
]

str = join((first(x) for x in token_test), '\n')
tokenized_kinds = filter(!is_triv, kind.(collect(tokenize(str, ERROR, next_token))))
true_kinds = collect(Iterators.flatten([last(x) for x in token_test if last(x) !== nothing]))

if tokenized_kinds != true_kinds
    println(deepdiff(tokenized_kinds, true_kinds))
    error()
end
