#########################
### Author: Floris M  ###
### Date: 04-12-2025  ###
### Version: 0.2      ###
#########################


import sys
import numpy
from collections import Counter
import pickle


def input_parser():
    text_path = None
    if(len(sys.argv) > 1):
        if(sys.argv[1] == "-h" or sys.argv[1] == "--help"):
            print("Gebruik:")
            print("python tokenizer_floris.py /pad/naar/tekst minimale_frequentie_om_token_te_worden")
            print("voorbeeld: tokenizer_floris.py input/tokenizer_text_long.txt 3")
        else:
            text_path = sys.argv[1]
            min_freq = int(sys.argv[2])
    return text_path, min_freq


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
    # print(vocab_count)
    return vocab_count, vocabulary


def tokenizer(word_list):
    counted_tokens = {}
    
    for word in word_list:
        for i in range(len(word) - 1):
            # Tuple paar:
            pair = (word[i], word[i+1])
            
            if pair in counted_tokens:
                counted_tokens[pair] += 1
            else:
                counted_tokens[pair] = 1

    #print(counted_tokens)
    return counted_tokens

def sort_and_return_token(counted_tokens, min_freq):
    """
    Neemt een dict met token counts en selecteerd te hoogste twee om in een nieuw token te veranderen.

    :param counted_tokens:
    """

    if counted_tokens == None:
        return None
    

    counter_sorted = dict(reversed(list((sorted(counted_tokens.items(), key = lambda item: item[1])))))
    #print(counter_sorted)
    top = []
    for i in counter_sorted:
        top.append([i, counter_sorted[i]])


    # Neem de top twee tokens om deze te combineren:
    new_token = top.pop(0)

    # Een woord moet minimaal een n keer voorkomen voordat het een token mag worden:
    if new_token[1] < min_freq:
        return None

    return new_token[0]
                

def byte_pair_encoding(word_list, pair_to_merge):
    new_word_list = []
    merged = pair_to_merge[0] + pair_to_merge[1]

    for word in word_list:
        i = 0
        new_word = []
        while i < len(word):
            if i < len(word)-1 and word[i] == new_token[0] and word[i+1] == new_token[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_list.append(new_word)

    return new_word_list


def save_to_file(vocabulary, text_tokens):
    
    with open("encoding.enc", "w") as text:
        text.write(str(vocabulary))
    with open("story", "w") as text:
        text.write(str(text_tokens))
    
    db = {}
    db['vocabulary'] = vocabulary
    db['text_tokens'] = text_tokens
    
    dbfile = open('Vocab_Text_Tokens', 'ab')
    
    pickle.dump(db, dbfile)                    
    dbfile.close()

    pickle.dumps(vocabulary)
    


if __name__ == "__main__":


    text_path, min_freq = input_parser()
    word_list = file_reader(text_path)


    for i in range(9999):

        token_set = tokenizer(word_list)
        new_token = sort_and_return_token(token_set, min_freq)

        if new_token == None: # als er geen tokens meer te maken zijn, stop.
            break

        new_word_list = byte_pair_encoding(word_list,new_token)
        word_list = new_word_list

    
    aaa, vocabulary = get_vocabulary(new_word_list)
    save_to_file(vocabulary, new_word_list)
    print(get_vocabulary(new_word_list))
    #print(new_word_list)
    print(f"Er zitten {len(get_vocabulary(new_word_list))} unieke tokens in deze tekst.")

