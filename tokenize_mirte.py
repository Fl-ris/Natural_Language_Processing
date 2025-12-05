# Schrijf een python-script tokenize dat kan worden aangeroepen vanaf de command line en 
# dat drie verschillende functionaliteiten implementeert:

# Het kan een gegeven leesbaar *.txt tekstbestand inlezen en hieruit door middel van byte-pair encoding een tokenisatie 
# afleiden. Deze gemaakte byte-pair encoding wordt weggeschreven naar een nieuw *.enc databestand (d.w.z. de correspondentie 
# tussen tekst en tokens, niet de omzetting van deze tekst zelf).

# Het kan een gegeven leesbaar *.txt tekstbestand inlezen en hierop een gemaakte byte-pair encoding uit een *.enc databestand 
# toepassen om dit om te zetten in tokens, en het resultaat vervolgens wegschrijven als een *.tok datafile met tokens.

# Het kan een gegeven *.tok bestand met tokens inlezen en hierop een gemaakte byte-pair encoding uit een *.enc bestand toepassen 
# om dit terug om te zetten in tekst, en het resultaat vervolgens wegschrijven als een leesbaar *.txt tekstbestand.

# Kies zelf geschikte bestandsindelingen voor de bestandstypen met encodings (*.enc) en met tokens (*.tok). De gebruiker 
# dient te kunnen aangeven hoeveel verschillende tokens er gemaakt mogen worden en/of hoe vaak een paar tokens ten minste 
# moet voorkomen om te worden samengevoegd tot een nieuw token. Bedenk ook zelf hoe je omgaat met interpunctie en andere 
# niet-alfanumerieke symbolen, of met verschillen in hoofdlettergebruik (of geef de gebruiker de mogelijkheid om hier keuzes 
# in te maken). Je mag ook de mogelijkheid geven functionaliteit te combineren (bv. een byte-pair encoding bepalen uit een 
# tekstdocument en dit tekstdocument gelijktijdig omzetten naar tokens in één aanroep).

# Geef bondige documentatie van het gebruik weer als de gebruiker je script aanroept met tokenize --help of tokenize -h.

# Breng klassen en functies die van algemeen belang zijn onder in een aparte module nlp.py die gedeeld is over alle lessen, 
# en importeer deze in je script. Je script kan daarnaast eigen klassen en functies definiëren voor eigen gebruik.

import sys
import numpy
from collections import Counter

def input_parser():
    text_path = None
    if(len(sys.argv) > 1):
        if(sys.argv[1] == "-h" or sys.argv[1] == "--help"):
            print("Gebruik:")
            print("input bestand blablabla...")
        else:
            text_path = sys.argv[1]
    return text_path


def file_reader(text_path):
    word_list = []
    with open(text_path) as text:
        for i in text:
            i = i.lower().strip() # Maak een lijst van woorden een maak ze allemaal lowercase.
            word_list.append(i)
    return word_list


def get_vocabulary(word_list):
    """
    Verkrijg alle unieke karakters in een text
    """
    vocabulary = set()
    for word in word_list:
        for char in word:
            for i in char:
                vocabulary.add(i)


    vocab_count = dict()
    for word in word_list:
        for char in word:
            if char in vocab_count:
                vocab_count[char] = vocab_count.get(char, 0) + 1
            else:
                vocab_count[char] = 1
    # print(vocab_count)
    return vocab_count


def tokenizer(word_list, vocabulary):
    token_set = set()
    
    for word in word_list:
        for i in range(0, len(word), 2):
            two_chars = word[i:i+2]
            token_set.add(two_chars)

    counted_tokens = {}
    for word in word_list:
        for i in range(0, len(word), 2):
            two_chars = word[i:i+2]
            if two_chars in token_set:
                counted_tokens[two_chars] = counted_tokens.get(two_chars, 0) + 1
            else:
                counted_tokens[two_chars] = 1
    return counted_tokens


def merge_dicts(counted_tokens, vocab_counted):
    """
    Docstring for merge_dicts
    :param counted_tokens: dictionary met getelde meerdere char tokens
    :param vocab_counted: dictionary met getelde chars
    """
    counted_tokens.update(vocab_counted)
    return counted_tokens

def sort_token_dict(counted_tokens):
    counter_sorted = dict(sorted(counted_tokens.items(), key = lambda item: item[1]))
    sorted_reversed = dict(reversed(list(counter_sorted.items())))
    # print(sorted_reversed)
    return sorted_reversed

                

def byte_pair_encoding(token_set, top_tokens):
    tokens_before = len(token_set) # Aantal tokens voor het optimaliseren

    for index, i in enumerate(token_set.keys()):
        if index > top_tokens:
            return 

        print(i)
    # for value in range(0, top_tokens):
    #     print(value)
        # print(token_set[value])
        #d = dict([(k, v) for k, v in d.items() if v != value])
    # print(d)



if __name__ == "__main__":
    text_path = input_parser()
    word_list = file_reader(text_path)
    vocabulary = get_vocabulary(word_list)
    token_set = tokenizer(word_list, vocabulary)
    counted_tokens = merge_dicts(token_set, vocabulary)
    sorted_reversed = sort_token_dict(counted_tokens)
    token_list = byte_pair_encoding(sorted_reversed, 10)
