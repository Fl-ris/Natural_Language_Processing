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

corpus = ["Python is amazing and fun.", "Python is not just but also powerful", "Learning python is fun!"]
def pre_process(sentence):
    word_list = []
    sentence = sentence.lower().strip().split()
    word_list = [list(word) for word in sentence]
    return word_list


def create_bow_vector(sentence, vocab):
    vocab_list = list(vocab)
    vocab_to_idx = {word: i for i, word in enumerate(vocab_list)}
    vector = [0] * len(vocab_to_idx)

    for word in sentence:
        token = "".join(word)
        if token in vocab_to_idx:
            vector[vocab_to_idx[token]] += 1
    return vector 

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
print(vocabulary)
bow_vectors = [create_bow_vector(doc, vocabulary)for doc in documents]
print("Bag of words vectors:")
for vector in bow_vectors:
    print(vector)
