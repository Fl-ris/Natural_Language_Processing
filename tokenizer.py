##################################
### Author: Mirte D, Floris M  ###
### Date: 04-12-2025           ###
### Version: 0.3               ###
##################################


import sys
import numpy
from collections import Counter
import pickle
from nlp import (file_reader, get_vocabulary, tokenizer,
                 sort_and_return_token, byte_pair_encoding,
                 save_encoding, load_encoding)


def input_parser():
    if len(sys.argv) <= 1 or sys.argv[1] in ("-h", "--help"):
        print("""Gebruik:
        python tokenizer.py learn  txt <min_freq> [enc_path]
        python tokenizer.py encode txt <enc_path>
        python tokenizer.py decode tok <enc_path>

        Voorbeeld:
        python tokenizer.py learn  data/story.txt 3
        python tokenizer.py encode data/story.txt encoding.enc
        python tokenizer.py decode data/story.tok encoding.enc
            """)

    action = sys.argv[1].lower()
    if action == "learn":
        txt_path = sys.argv[2]
        min_freq = int(sys.argv[3])
        enc_path = sys.argv[4] if len(sys.argv) > 4 else "encoding.enc"
        return action, txt_path, enc_path, min_freq
    elif action == "encode":
        txt_path = sys.argv[2]
        enc_path = sys.argv[3]
        return action, txt_path, enc_path, None
    elif action == "decode":
        tok_path = sys.argv[2]
        enc_path = sys.argv[3]
        return action, tok_path, enc_path, None
    else:
        raise ValueError("Onbekend commando... Type: --help ")
    
    
def learn_encoding(txt_path, min_freq, enc_path):
    word_list = file_reader(txt_path)

    for _ in range(9999):
        token_counts = tokenizer(word_list)
        pair = sort_and_return_token(token_counts, min_freq)
        if pair is None:
            break
        word_list = byte_pair_encoding(word_list, pair)

    vocab_counts, vocab_set = get_vocabulary(word_list)
    save_encoding(vocab_set, word_list, enc_path)

    print(f"Encoding opgeslagen in '{enc_path}'.")
    print(f"{len(vocab_set)} unieke tokens gevonden.")

def apply_encoding(txt_path, enc_path):
    enc_db = load_encoding(enc_path)
    vocab = enc_db["vocabulary"]
    word_list = file_reader(txt_path)


    while True:
        token_counts = tokenizer(word_list)
        pair = sort_and_return_token(token_counts, min_freq=1)
        if pair is None:
            break
        word_list = byte_pair_encoding(word_list, pair)

    tok_path = txt_path.rsplit(".", 1)[0] + ".tok"
    with open(tok_path, "w") as f:
        for w in word_list:
            f.write("".join(w))
    print(f"Tokens opgeslagen in '{tok_path}'.")


def decode_tokens(tok_path, enc_path):
    enc_db = load_encoding(enc_path)
    vocab = enc_db["vocabulary"]

    with open(tok_path, "r") as file:
        flat = list(file.read())

    def reverse_merge(tokens):
        changed = False
        out = []
        for t in tokens:
            if t in vocab and len(t) > 1:
                out.extend(list(t))
                changed = True
            else:
                out.append(t)
        return out, changed

    tokens = flat
    while True:
        tokens, changed = reverse_merge(tokens)
        if not changed:
            break

    txt = "".join(tokens).replace("  ", "\n")
    txt_path = tok_path.rsplit(".", 1)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(txt)
    print(f"Tekst teruggeconverteerd naar '{txt_path}'.")
    
    
if __name__ == "__main__":
    action, path, enc_path, min_freq = input_parser()
    if action == "learn":
        learn_encoding(path, min_freq, enc_path)
    elif action == "encode":
        apply_encoding(path, enc_path)
    elif action == "decode":
        decode_tokens(path, enc_path)