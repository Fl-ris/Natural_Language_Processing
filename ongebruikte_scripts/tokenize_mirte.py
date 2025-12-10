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
from collections import Counter

def input_parser():
    text_path = None
    if(len(sys.argv) > 1):
        if(sys.argv[1] == "-h" or sys.argv[1] == "--help"):
            print("Gebruik:")
            print("Nog aanvullen")
        else:
            text_path = sys.argv[1]
            min_frequency = int(sys.argv[2])
    return text_path, min_frequency


def file_reader(text_path):
    word_list = []
    with open(text_path) as text:
        for line in text:
            words_split = line.strip().split()
            for i in words_split:
                i = list(i.lower())
                word_list.append(i)
    return word_list


def get_vocabulary(word_list):
    """
    Verkrijg alle unieke karakters in een text
    """
    vocabulary = set()
    for word in word_list:
        for char in word:
            vocabulary.add(char)


    vocab_count = dict()
    for word in word_list:
        for char in word:
            if char in vocab_count:
                vocab_count[char] = vocab_count.get(char, 0) + 1
            else:
                vocab_count[char] = 1
    return vocab_count, vocabulary


def tokenizer(word_list):
    token_count = {}
    
    for word in word_list:
        for i in range(len(word) -1):
            pair = (word[i], word[i+1])
            token_count[pair] = token_count.get(pair, 0) + 1
    return token_count


def sort_token(token_count, min_frequency):
    most_frequent_pair = None

    for pair, count in token_count.items():
        if count > min_frequency:
            most_frequent_pair = pair

    return most_frequent_pair
                

def byte_pair_encoding(word_list, pair_to_merge):
    merged_symbol = pair_to_merge[0] + pair_to_merge[1]
    new_word_list = []

    for word in word_list:
        i = 0
        new_word = []

        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair_to_merge:
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_word_list.append(new_word)
    return new_word_list


def write_to_enc(merges, path):
    with open(path, "w", encoding = "utf-8") as f:
        f.write("# BPE merges\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")


def encode_tok():
    pass


def decode_tok():
    pass


if __name__ == "__main__":
    textpath, min_frequency = input_parser()
    word_list = file_reader(textpath)
    

    merges = []

    for i in range(9999):
        token_set = tokenizer(word_list)
        new_token = sort_token(token_set, min_frequency)

        if new_token == None:
            break

        merges.append(new_token)
        new_word_list = byte_pair_encoding(word_list, new_token)
        word_list = new_word_list
    
    vocab_count, vocabulary = get_vocabulary(word_list)
    write_to_enc(merges, "write_test.enc")
    print(vocabulary)

    print(f"Er zitten {len(vocabulary)} unieke tokens in deze tekst.")


# TO DO:
# txt -> tok
# tok -> txt
