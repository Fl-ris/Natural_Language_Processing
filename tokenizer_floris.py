#########################
### Author: Floris M  ###
### Date: 04-12-2025  ###
### Version: 0.2      ###
#########################


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
    token_set = set()
    for word in word_list:
        for i in range(0,len(word),2):
            two_chars = word[i:i+2]
            token_set.add(two_chars)

    return token_set
                
def byte_pair_encoding(token_set):
    tokens_before = len(token_set) # Aantal tokens voor het optimaliseren
    counter = {}
    for token in token_set:
        counter[token] = counter.get(token,0) + 1
    print(counter)





if __name__ == "__main__":
    text_path = input_parser()
    word_list = file_reader(text_path)
    vocabulary = get_vocabulary(word_list)
    token_set = tokenizer(word_list)
    token_list = byte_pair_encoding(token_set)



