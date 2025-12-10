"""
embed.py — CBOW embedding trainer

Dit script traint een eenvoudig CBOW-model op een .tok bestand met BPE-token-ID's.
Het model gebruikt sklearn's MLPClassifier om embeddings te leren. De embeddings
worden weggeschreven naar een .emb bestand, waarin elke regel bestaat uit:

    token <TAB> val1 <TAB> val2 <TAB> ... <TAB> valN

Gebruik (command line):
    python embed.py data.tok data.enc -w 2 -H 50 -o output.emb
"""

import argparse
import os

from nlp_jasper import (
    load_tokens,
    BPETokenizer,
    make_cbow_examples_ids,
    train_cbow_mlp,
    extract_embeddings_from_mlp_ids,
    save_embeddings_with_bpe,
)


def main():
    """
    Hoofdfunctie van embed.py.

    - Leest een .tok bestand (BPE token-ID's)
    - Leest het bijbehorende .enc BPE-encodingbestand
    - Maakt CBOW-trainingsvoorbeelden
    - Traineert een MLPClassifier met opgegeven hidden layer size
    - Haalt embeddings uit het model
    - Schrijft embeddings weg naar een .emb bestand

    Gebruik:
        python embed.py bestand.tok bestand.enc [-w WINDOW] [-H HIDDEN] [-o OUTPUT]
    """

    parser = argparse.ArgumentParser(
        description="Train een CBOW-embeddingmodel op een .tok bestand (BPE-token-ID's)."
    )
    parser.add_argument("tok_file", help="Pad naar het .tok databestand (spatiegescheiden ints).")
    parser.add_argument("enc_file", help="Pad naar het .enc BPE-encodingbestand (JSON).")

    parser.add_argument(
        "-w", "--window", type=int, default=2,
        help="Window-grootte n (aantal tokens links en rechts, default=2)."
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
    emb_path = args.output
    if emb_path is None:
        base, _ = os.path.splitext(tok_path)
        emb_path = base + ".emb"

    print(f"[INFO] Lees token-ID's uit {tok_path}")
    token_ids = load_tokens(tok_path)

    print(f"[INFO] Bouw CBOW trainingvoorbeelden (window={args.window})...")
    x, y, _ = make_cbow_examples_ids(token_ids, args.window)

    print(f"[INFO] Train MLP (hidden_dim={args.hidden})...")
    clf = train_cbow_mlp(x, y, args.hidden)

    print("[INFO] Haal embeddings per token-ID uit model...")
    id_to_vec = extract_embeddings_from_mlp_ids(clf)


    print(f"[INFO] Laad BPE-encoding uit {enc_path}")
    bpe = BPETokenizer.load(enc_path)

    print(f"[INFO] Sla embeddings op naar {emb_path}")
    save_embeddings_with_bpe(emb_path, id_to_vec, bpe)

    print("[KLAAR] Embedding training voltooid.")


if __name__ == "__main__":
    main()
