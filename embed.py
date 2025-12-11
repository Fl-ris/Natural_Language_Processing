"""
embed.py — CBOW embedding trainer op Floris/Mirte BPE-encoding.

Gebruik:
    python embed.py encoding.enc -H 50 -w 2 -o kanker_nl.emb
"""

import argparse
import os

from nlp import (
    load_encoding,
    flatten_token_lists,
    make_cbow_examples_ids,
    train_cbow_mlp,
    extract_embeddings_from_mlp_ids,
    save_embeddings_with_vocab,
)


def main():
    parser = argparse.ArgumentParser(
    description="Train een CBOW-embeddingmodel op een .tok bestand (BPE-token-ID's)."
    )

    parser.add_argument(
        "tok_file",
        help="Pad naar het .tok databestand (spatiegescheiden ints)."
    )
    parser.add_argument(
        "enc_file",
        help="Pad naar het .enc BPE-encodingbestand."
    )

    parser.add_argument(
        "-w", "--window", type=int, default=2,
        help="Window-grootte n (links en rechts, default=2)."
    )
    parser.add_argument(
        "-H", "--hidden", type=int, default=10,
        help="Aantal neurons in de verborgen laag (default=10)."
    )
    parser.add_argument(
        "-o", "--output",
        help="Pad naar .emb outputbestand (default: zelfde naam als .tok maar .emb)."
    )


    args = parser.parse_args()

    tok_path = args.tok_file
    enc_path = args.enc_file

    if args.output is None:
        base, _ = os.path.splitext(enc_path)
        emb_path = base + ".emb"
    else:
        emb_path = args.output

    print(f"[INFO] Laad encoding uit {enc_path}")
    enc_db = load_encoding(enc_path)
    vocab_list = sorted(enc_db["vocabulary"])  # lijst tokens
    token_to_id = {tok: i for i, tok in enumerate(vocab_list)}

    print("[INFO] Flatten tokenized text naar token-ID's...")
    token_lists = enc_db["text_tokens"]
    flat_tokens = flatten_token_lists(token_lists)
    token_ids = [token_to_id[tok] for tok in flat_tokens if tok in token_to_id]

    print(f"[INFO] Aantal token-ID's: {len(token_ids)}")
    print(f"[INFO] Bouw CBOW trainingvoorbeelden (window={args.window})...")
    X, y, _ = make_cbow_examples_ids(token_ids, args.window)

    print(f"[INFO] Train MLP (hidden_dim={args.hidden})...")
    clf = train_cbow_mlp(X, y, args.hidden)

    print("[INFO] Haal embeddings per token-ID uit model...")
    id_to_vec = extract_embeddings_from_mlp_ids(clf)

    print(f"[INFO] Sla embeddings op naar {emb_path}")
    save_embeddings_with_vocab(emb_path, id_to_vec, vocab_list)

    print("[KLAAR] Embedding training voltooid.")


if __name__ == "__main__":
    main()
