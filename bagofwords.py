# Stel dat een leesbaar *.txt tekstbestand gegeven is met daarin een corpus bestaande uit een reeks documenten. 
# Elk document staat op één regel (d.w.z. een document kan uit meerdere zinnen bestaan, maar bevat geen regeleinden). 
# Schrijf een python-script bagofwords dat kan worden aangeroepen vanaf de command line en dat elk document omzet in een 
# reeks getalwaarden. De gebruiker dient te kunnen aangeven of een multi-hot, telling/frequentie, of TF-IDF encoding gebruikt 
# dient te worden. Het resultaat dient te worden opgeslagen in een *.bow databestand met de bag-of-words representaties. Kies 
# zelf een geschikte bestandsindeling.

# Het dient ook mogelijk te zijn om meerdere leesbare *.txt tekstbestanden mee te geven. Die dienen dan allemaal gezamenlijk 
# te worden omgezet in hun eigen *.bow databestanden. Daarbij dient voor elk bestand dezelfde tokenisatie gebruikt te worden 
# en ook de TF-IDF berekening dient gebaseerd te worden op alle documenten tezamen (d.w.z. maak één byte-pair encoding op basis 
# van alle informatie in alle documenten gezamenlijk, en tel de frequenties van documenten die een zeker token bevatten over alle 
# documenten heen).

# Geef bondige documentatie van het gebruik weer als de gebruiker je script aanroept met bagofwords --help of bagofwords -h.

# Breng klassen en functies die van algemeen belang zijn onder in een aparte module nlp.py die gedeeld is over alle lessen, 
# en importeer deze in je script. Je script kan daarnaast eigen klassen en functies definiëren voor eigen gebruik.

# Eerste test met voorbeeld: https://www.datacamp.com/tutorial/python-bag-of-words-model?dc_referrer=https%3A%2F%2Fwww.google.com%2F 

from tokenize_mirte import *
import math
import argparse

def pre_process(docs):
    """
    Deze functie maakt de input klaar voor de verdere stappen.
    :param docs: een .txt bestand met daarin op elke regel een bestand.
    """
    docs = docs.lower().strip()
    return docs.split()


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


def apply_bpe(doc, merges):
    tokens = doc[:]
    flat_tokens = []
    for merge in merges:
        tokens = byte_pair_encoding(tokens, merge)
    for t in tokens:
        flat_tokens.append("".join(t))
    return flat_tokens


def main():
    corpus = ["Python is amazing and fun.", "Python is not just but also powerful", "Learning python is fun!"]

    documents = [pre_process(sentence) for sentence in corpus]

    flat_tokens = [token for doc in documents for token in doc]

    merges = []
    min_frequency = 2

    for _ in range(9999):
        token_set = tokenizer(flat_tokens)
        new_token = sort_token(token_set, min_frequency)
        if new_token is None:
            break
        merges.append(new_token)
        flat_tokens = byte_pair_encoding(flat_tokens, new_token)
    
    vocab_count, vocabulary = get_vocabulary(flat_tokens)
    vocab_to_idx = {token: i for i, token in enumerate(vocabulary)}

    bpe_documents = [apply_bpe(doc, merges) for doc in documents]

    bow_vectors = [count_encoding(doc, vocab_to_idx) for doc in bpe_documents]

    print("Vocabulary: ", vocabulary)
    print("Bag of words vectors:")
    for vec in bow_vectors:
        print(vec)

if __name__ == "__main__":
    main()
    