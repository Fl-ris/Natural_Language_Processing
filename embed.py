"""
embed.py — CBOW embeddings voor Floris/Mirte encoding.enc

Gebruik:
    python embed.py encoding.enc -H 50 -w 2 -o kanker_nl.emb
"""

import argparse
from nlp import (
    load_encoding,
    flatten_token_lists,
    make_cbow_examples_ids,
    train_cbow_mlp,
    extract_embeddings_from_mlp_ids,
    save_embeddings_with_vocab,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("enc_file", help="Het encoding.enc bestand van Floris/Mirte")
    parser.add_argument("-w", "--window", type=int, default=2)
    parser.add_argument("-H", "--hidden", type=int, default=20)
    parser.add_argument("-o", "--output")

    args = parser.parse_args()
    enc_path = args.enc_file

    # Output pad
    emb_path = args.output or enc_path.replace(".enc", ".emb")

    print(f"[INFO] Laad encoding: {enc_path}")
    enc = load_encoding(enc_path)

    vocab_list = sorted(enc["vocabulary"])
    token_to_id = {tok: i for i, tok in enumerate(vocab_list)}

    print("[INFO] Flatten tokens -> ID's")
    flat = flatten_token_lists(enc["text_tokens"])
    token_ids = [token_to_id[t] for t in flat]

    print("[INFO] CBOW dataset bouwen...")
    X, y, _ = make_cbow_examples_ids(token_ids, args.window)

    print(f"[INFO] Train MLP (hidden={args.hidden})...")
    clf = train_cbow_mlp(X, y, args.hidden)

    print("[INFO] Extract embeddings...")
    id_to_vec = extract_embeddings_from_mlp_ids(clf)

    print(f"[INFO] Opslaan naar {emb_path}")
    save_embeddings_with_vocab(emb_path, id_to_vec, vocab_list)

    print("[KLAAR] Embeding training voltooid.")


if __name__ == "__main__":
    main()

