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
    return vocabulary

def tokenizer(word_list):
    token_set = dict()
    for word in word_list:
        for i in range(0,len(word),2):
            two_chars = word[i:i+2]
            token_set[two_chars] = i
    # print(token_set)
    return token_set
                

def byte_pair_encoding(token_set, top_tokens):
    tokens_before = len(token_set) # Aantal tokens voor het optimaliseren
    counter = {}
    res = dict(sorted(token_set.items(), key = lambda item: item[1]))
    res2 = dict(reversed(list(res.items())))
    for token in token_set:
        counter[token] = counter.get(token,0) + 1

if __name__ == "__main__":
    text_path = input_parser()
    word_list = file_reader(text_path)
    vocabulary = get_vocabulary(word_list)
    token_set = tokenizer(word_list)
    token_list = byte_pair_encoding(token_set, 50)
