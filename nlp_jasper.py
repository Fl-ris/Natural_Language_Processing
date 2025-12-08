"""
nlp.py
Eenvoudige Byte-Pair Encoding (BPE) tokenizer voor Les 1: Tokenisatie.

- Werkt op teken-niveau.
- Start met losse karakters.
- Voegt steeds het meest frequente token-paar samen tot:
  * max_tokens is bereikt, of
  * de frequentie van het beste paar < min_freq.

Encodings (*.enc) worden opgeslagen als JSON met o.a.:
- config: max_tokens, min_freq, lowercase, merge_whitespace
- id_to_token: lijst van alle tokens (index = token-id)
- merges: lijst van [a, b, nieuw_token] in volgorde van samenvoegen

Tokens (*.tok) zijn spatiegescheiden integers.
"""
import json
from collections import Counter


class BPETokenizer:
    def __init__(self, max_tokens=1000, min_freq=2,
                 lowercase=False, merge_whitespace=False):
        self.max_tokens = max_tokens
        self.min_freq = min_freq
        self.lowercase = lowercase
        self.merge_whitespace = merge_whitespace

        self.id_to_token = []
        self.token_to_id = {}
        self.merges = []

    def _normalize(self, text):
        return text.lower() if self.lowercase else text

   
    def _count_pairs(self, tokens):
        pairs = []
        for i in range(len(tokens) - 1):
            a = tokens[i]
            b = tokens[i + 1]

            if not self.merge_whitespace and (a.isspace() or b.isspace()):
                continue

            pairs.append((a, b))

        return Counter(pairs)

    def _apply_merge(self, tokens, pair, new_token):
        a, b = pair
        output = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                output.append(new_token)
                i += 2
            else:
                output.append(tokens[i])
                i += 1
        return output

    def fit(self, text):
        text = self._normalize(text)
        tokens = list(text)
        vocab = set(tokens)
        self.merges = []

        while True:
            if len(vocab) >= self.max_tokens:
                break

            pair_counts = self._count_pairs(tokens)
            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]

            if best_freq < self.min_freq:
                break

            a, b = best_pair
            new_token = a + b
            vocab.add(new_token)
            self.merges.append([a, b, new_token])
            tokens = self._apply_merge(tokens, best_pair, new_token)

        # op het einde vocab sorteren
        self.id_to_token = sorted(vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}

    def encode(self, text):
        if not self.id_to_token:
            raise ValueError("Tokenizer niet getraind of geladen.")

        text = self._normalize(text)
        tokens = list(text)

        for a, b, new_token in self.merges:
            tokens = self._apply_merge(tokens, (a, b), new_token)

        ids = []
        for tok in tokens:
            if tok in self.token_to_id:
                ids.append(self.token_to_id[tok])
            else:
                for ch in tok:
                    if ch in self.token_to_id:
                        ids.append(self.token_to_id[ch])
        return ids

    def decode(self, token_ids):
        out = []
        for tid in token_ids:
            if 0 <= tid < len(self.id_to_token):
                out.append(self.id_to_token[tid])
            else:
                out.append("�")
        return "".join(out)

    def save(self, path):
        data = {
            "config": {
                "max_tokens": self.max_tokens,
                "min_freq": self.min_freq,
                "lowercase": self.lowercase,
                "merge_whitespace": self.merge_whitespace,
            },
            "id_to_token": self.id_to_token,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data["config"]
        obj = cls(
            max_tokens=cfg["max_tokens"],
            min_freq=cfg["min_freq"],
            lowercase=cfg["lowercase"],
            merge_whitespace=cfg["merge_whitespace"],
        )
        obj.id_to_token = data["id_to_token"]
        obj.token_to_id = {tok: i for i, tok in enumerate(obj.id_to_token)}
        obj.merges = data["merges"]
        return obj


def save_tokens(path, token_ids):
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(str(t) for t in token_ids))


def load_tokens(path):
    with open(path, "r", encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]