##################################
### Author: Mirte D            ###
### Date: 11-12-2025           ###
### Version: 0.1               ###
##################################

from tokenize import *
from nlp import *
import math
import argparse


def pre_process(docs):
    """
    Deze functie maakt de input klaar voor de verdere stappen.
    :param docs: een .txt bestand met daarin op elke regel een bestand.
    """
    line = docs.lower().strip()
    words = line.split()
    return [list(word) for word in words]


def count_encoding(doc, vocab):
    """
    Encode het opgegeven bestand met frequentie
    :param doc: een .txt bestand met daarin op elke regel een bestand.
    :param vocab: een lijst met alle woorden in het bestand.
    :return vector: een vector met daarin de encoding.
    """
    vector = [0] * len(vocab)
    for token in doc:
        if token in vocab:
            vector[vocab[token]] += 1
    return vector


def multi_hot(doc, vocab):
    """
    Encode het opgegeven bestand met multi-hot encoding
    :param doc: een .txt bestand met daarin op elke regel een bestand.
    :param vocab: een lijst met alle woorden in het bestand.
    :return vector: een vector met daarin de encoding.
    """
    vector = [0] * len(vocab)
    for token in doc:
        if token in vocab:
            vector[vocab[token]] = 1
    return vector


def tf_idf_vector(doc, vocab, idf_scores):
    """
    Encode het opgegeven bestand met TF-IDF
    :param doc: een .txt bestand met daarin op elke regel een bestand.
    :param vocab: een lijst met alle woorden in het bestand.
    :param idf_scores: de IDF-score
    :return tfidf_vector: een vector met daarin de encoding.
    """
    tf_vector = multi_hot(doc, vocab)
    tfidf_vector = [round(tf * idf_scores[i], 3) for i, tf in enumerate(tf_vector)]
    return tfidf_vector


def compute_idf(docs, vocab):
    """
    Berekent de IDF voor elk bestand in het bestand
    :param docs: een .txt bestand met daarin op elke regel een bestand.
    :param vocab: een lijst met alle woorden in het bestand.
    :return idf:
    """
    N = len(docs)
    df = [0] * len(vocab)

    for doc in docs:
        seen = set()
        for token in doc:
            if token in vocab and token not in seen:
                df[vocab[token]] += 1
                seen.add(token)
    idf = [math.log((N + 1) / (df_i + 1)) + 1 for df_i in df]
    return idf


def apply_bpe(doc, merges):
    """
    BPE uitvoeren op het gegeven document.
    :param doc: een document.
    :param merges: de merges in van tokens in het document.
    """
    tokens = doc[:]
    flat_tokens = []
    for merge in merges:
        tokens = byte_pair_encoding(tokens, merge)
    for word in tokens:
        for piece in word:
            flat_tokens.append(piece)
    return flat_tokens


def write_to_bow(vocabulary, vectors, path):
    """
    Schrijft de gegenereerde bag of words representatie weg naar .bow bestand.
    :param vocabulary: de gebruikte vocabulary.
    :param vectors: de vectoren met daarin de bag of words representatie.
    :param path: het pad wanneer het bestand wordt geschreven.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("Gebruikte vocabulary:\n")
        f.write(f"{vocabulary}\n\n")
        f.write("Bag of words representaties:\n")
        for vector in vectors:
            f.write(f"{vector}\n")


def main():
    parser = argparse.ArgumentParser(description="Bag of words encoder")
    parser.add_argument("filepath", nargs="+", help="Path(s) to corpus text file")
    parser.add_argument("--encoding", type=str, choices=["multi-hot", "count", "tfidf"],
                        default="count", help="Type of encoding")
    parser.add_argument("--minfreq", type=int,
                        default=2, help="Minimum frequency of tokens in the text")
    args = parser.parse_args()

    all_documents = []
    for path in args.filepath:
        with open(path, "r", encoding="utf-8") as f:
            all_documents.append([pre_process(line) for line in f])

    flat_tokens = [token for doc_list in all_documents for doc in doc_list for token in doc]

    merges = []
    min_frequency = args.minfreq

    for _ in range(9999):
        token_set = tokenizer(flat_tokens)
        new_token = sort_and_return_token(token_set, min_frequency)
        if new_token is None:
            break
        merges.append(new_token)
        flat_tokens = byte_pair_encoding(flat_tokens, new_token)
    vocab_count, vocabulary = get_vocabulary(flat_tokens)
    vocab_to_idx = {token: i for i, token in enumerate(vocabulary)}

    bpe_per_file = []
    for docs in all_documents:
        bpe_per_file.append([apply_bpe(doc, merges) for doc in docs])

    if args.encoding == "tfidf":
        idf_scores = compute_idf([doc for file_docs in bpe_per_file for doc in file_docs], vocab_to_idx)
    else:
        idf_scores = None

    for path, bpe_docs in zip(args.filepath, bpe_per_file):
        vectors = []
        for doc in bpe_docs:
            if args.encoding == "multi-hot":
                vectors.append(multi_hot(doc, vocab_to_idx))
            elif args.encoding == "count":
                vectors.append(count_encoding(doc, vocab_to_idx))
            elif args.encoding == "tfidf":
                vectors.append(tf_idf_vector(doc, vocab_to_idx, idf_scores))

        out_path = path.replace(".txt", ".bow")
        write_to_bow(vocabulary, vectors, out_path)
        print(f"Bag of words met het bestand: {path}")
        print(f"Vocabulary: {vocabulary}\n")
        print(f"Bag of words vectors ({args.encoding}): ")
        for vec in vectors:
            print(vec)
        print(f"\nBOW succesvol geschreven naar {out_path}")


if __name__ == "__main__":
    main()
