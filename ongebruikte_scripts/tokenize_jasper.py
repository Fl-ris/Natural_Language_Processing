"""
tokenize: Byte-Pair Encoding (BPE) tool voor Les 1: Tokenisatie.

Functionaliteit:

1) Een encoding leren uit een .txt tekstbestand en opslaan als .enc
   (optioneel direct dezelfde tekst naar tokens omzetten).

   Voorbeeld:
       tokenize learn \
           --text kanker_nl.txt \
           --enc kanker_nl.enc \
           --tokens kanker_nl.tok \
           --max-tokens 5000 \
           --min-freq 3 \
           --lowercase

2) Een .txt tekstbestand encoderen naar tokens (.tok) met een bestaande .enc encoding.

   Voorbeeld:
       tokenize encode \
           --text cancer_en.txt \
           --enc kanker_nl.enc \
           --tokens cancer_en_with_nl_enc.tok

3) Een .tok bestand met tokens decoderen naar leesbare tekst (.txt)
   met gebruik van dezelfde .enc encoding.

   Voorbeeld:
       tokenize decode \
           --tokens kanker_nl.tok \
           --enc kanker_nl.enc \
           --text kanker_nl_reconstructed.txt
"""


import argparse
from nlp_jasper import BPETokenizer, save_tokens, load_tokens


def cmd_learn(args):
    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer(
        max_tokens=args.max_tokens,
        min_freq=args.min_freq,
        lowercase=args.lowercase,
        merge_whitespace=args.merge_whitespace,
    )

    tokenizer.fit(text)
    tokenizer.save(args.enc)

    if args.tokens:
        ids = tokenizer.encode(text)
        save_tokens(args.tokens, ids)


def cmd_encode(args):
    tok = BPETokenizer.load(args.enc)

    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tok.encode(text)
    save_tokens(args.tokens, ids)


def cmd_decode(args):
    tok = BPETokenizer.load(args.enc)
    ids = load_tokens(args.tokens)
    text = tok.decode(ids)

    with open(args.text, "w", encoding="utf-8") as f:
        f.write(text)


def build_parser():
    p = argparse.ArgumentParser(add_help=True)

    sub = p.add_subparsers(dest="command", required=True)

    # learn
    pl = sub.add_parser("learn")
    pl.add_argument("--text", required=True)
    pl.add_argument("--enc", required=True)
    pl.add_argument("--tokens")
    pl.add_argument("--max-tokens", type=int, default=1000)
    pl.add_argument("--min-freq", type=int, default=2)
    pl.add_argument("--lowercase", action="store_true")
    pl.add_argument("--merge-whitespace", action="store_true")
    pl.set_defaults(func=cmd_learn)

    # encode
    pe = sub.add_parser("encode")
    pe.add_argument("--text", required=True)
    pe.add_argument("--enc", required=True)
    pe.add_argument("--tokens", required=True)
    pe.set_defaults(func=cmd_encode)

    # decode
    pd = sub.add_parser("decode")
    pd.add_argument("--tokens", required=True)
    pd.add_argument("--enc", required=True)
    pd.add_argument("--text", required=True)
    pd.set_defaults(func=cmd_decode)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
