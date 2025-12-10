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

import numpy as np
from sklearn.neural_network import MLPClassifier

class BPETokenizer:
    """
    Eenvoudige Byte-Pair Encoding (BPE) tokenizer.

    - Werkt op teken-niveau (character-level)
    - Leert merges op basis van veelvoorkomende token-paren
    - Kan tekst encoderen naar token-ID's en decoderen naar tekst
    - Slaat modelconfiguratie, vocab en merges op in JSON
    """
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
        """
        Train de BPE-tokenizer op ruwe invoertekst.

        Deze functie:
        - normaliseert de tekst (lowercase indien ingesteld)
        - splitst de tekst in losse tekens (initiale tokens)
        - telt hoeveel keer elk token-paar voorkomt
        - voert iteratief de meest frequente merge uit
        - stopt wanneer:
                * max_tokens is bereikt, of
                * geen paar meer boven min_freq voorkomt
        - bouwt tenslotte de tokenlijst (id_to_token) en mapping (token_to_id)

        Parameters:
            text (str): De invoertekst waarop BPE-trainingsmerges worden geleerd.
        """
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

        self.id_to_token = sorted(vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}

    def encode(self, text):
        """
        Encodeer een stuk tekst naar BPE-token-ID's.

        Deze functie:
        - normaliseert de invoer (lowercase indien ingesteld),
        - splitst de tekst in karakters,
        - past alle geleerde BPE-merges toe in volgorde,
        - zet elk resulterend token om naar zijn numerieke ID.

        Als een token niet direct voorkomt in de vocab (bijv. na merges),
        wordt het opgesplitst in individuele karakters die wél bekend zijn.

        Parameters:
            text (str): De te encoderen tekst.

        Returns:
            List[int]: Een lijst van token-ID's die de tekst representeren.
        """
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
        """
        Decodeer een lijst van BPE-token-ID's terug naar tekst.

        Voor elk token-ID wordt het bijbehorende token opgezocht in
        `id_to_token`. Onbekende of ongeldige ID's worden vervangen
        door het vervangingssymbool '�'.

        Parameters:
            token_ids (List[int]): Lijst met numerieke token-ID's.

        Returns:
            str: De gereconstrueerde tekst, opgebouwd door alle tokens
            achter elkaar te plakken.
        """
        out = []
        for tid in token_ids:
            if 0 <= tid < len(self.id_to_token):
                out.append(self.id_to_token[tid])
            else:
                out.append("�")
        return "".join(out)

    def save(self, path):
        """
    Sla de getrainde BPE-tokenizer op naar een JSON-bestand.

    Het opgeslagen bestand bevat:
      - de configuratie (max_tokens, min_freq, lowercase, merge_whitespace)
      - de volledige vocabulaire (`id_to_token`)
      - de lijst van uitgevoerde merges in volgorde (`merges`)

    Dit JSON-bestand kan later worden geladen met `BPETokenizer.load()`.

    Parameters:
        path (str): Pad naar het JSON-bestand waarin de tokenizer
                    moet worden opgeslagen.
    """
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
        """
        Laad een opgeslagen BPE-tokenizer uit een JSON-bestand.

        Dit herstelt:
        - de configuratie (max_tokens, min_freq, lowercase, merge_whitespace)
        - de vocabulaire (`id_to_token` en `token_to_id`)
        - de lijst van merges in de juiste volgorde

        Hiermee wordt exact dezelfde tokenizer gereconstrueerd als waarmee
        het .tok-bestand oorspronkelijk gecodeerd werd.

        Parameters:
            path (str): Pad naar het JSON-bestand met de opgeslagen tokenizer.

        Returns:
            BPETokenizer: Een nieuwe instantie met geladen vocabulaire en merges.
        """
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
    """
    Sla een lijst van token-ID's op naar een tekstbestand.

    De token-ID's worden weggeschreven als één regel met spaties
    tussen de ID's, hetzelfde formaat als het `.tok` bestand dat
    in de opdrachten wordt gebruikt.

    Parameters:
        path (str): Pad naar het uitvoerbestand.
        token_ids (List[int]): Lijst van numerieke token-ID's die
                               moeten worden opgeslagen.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(str(t) for t in token_ids))

def load_tokens(path):
    """
    Laad een lijst van token-ID's uit een `.tok` tekstbestand.

    Het bestand bevat één regel met spatiegescheiden gehele getallen.
    Deze functie leest die regel in en zet elke waarde om naar een int.

    Parameters:
        path (str): Pad naar het .tok-bestand.

    Returns:
        List[int]: Lijst met token-ID's die in het bestand stonden.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


###Embeding
def make_cbow_examples_ids(token_ids, window):
    """
    Bouw CBOW-training op basis van een lijst token-ID's (ints).

    input: multi-hot vector van de context (links en rechts)
    output: index van het middelste token (int)
    """
    if not token_ids:
        raise ValueError("Lege tokenlijst.")

    vocab_size = max(token_ids) + 1
    x = []
    y = []

    for i in range(window, len(token_ids) - window):
        target_id = token_ids[i]
        context_ids = token_ids[i - window:i] + token_ids[i + 1:i + 1 + window]

        x_vec = np.zeros(vocab_size, dtype=np.float32)
        for cid in context_ids:
            if 0 <= cid < vocab_size:
                x_vec[cid] = 1.0
        x.append(x_vec)
        y.append(target_id)

    x = np.vstack(x)
    y = np.array(y, dtype=np.int64)
    return x, y, vocab_size

def train_cbow_mlp(x, y, hidden_dim):
    """
    Train een simpele MLP met één verborgen laag.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_dim,),
        activation="tanh",
        solver="adam",
        max_iter=20,
        random_state=0,
    )
    clf.fit(x, y)
    return clf

def extract_embeddings_from_mlp_ids(clf):
    """
    Maak embeddings alleen voor token-ID's die het model kent als klasse.

    clf.classes_ bevat de lijst token-ID's (ints) die als y voorkwamen.
    coefs_[1] heeft shape (hidden_dim, n_classes)
    kolom j hoort bij clf.classes_[j]
    """
    w_hidden_to_out = clf.coefs_[1]
    classes = clf.classes_

    id_to_vec = {}
    for j, tid in enumerate(classes):
        vec = w_hidden_to_out[:, j]
        id_to_vec[int(tid)] = vec
    return id_to_vec


def save_embeddings_with_bpe(path, id_to_vec, bpe_tokenizer):
    """
    Schrijf embeddings weg als TSV:
    token\tval1\tval2\t...\\n

    id_to_vec: dict[int, np.ndarray]
    bpe_tokenizer: BPETokenizer met .id_to_token
    """
    with open(path, "w", encoding="utf-8") as f:
        for tid, vec in id_to_vec.items():
            if 0 <= tid < len(bpe_tokenizer.id_to_token):
                tok = bpe_tokenizer.id_to_token[tid]
            else:
                tok = f"<UNK_{tid}>"
            vals = "\t".join(str(float(x)) for x in vec)
            f.write(f"{tok}\t{vals}\n")
