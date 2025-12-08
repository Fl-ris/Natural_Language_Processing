#########################
### Author: Floris M  ###
### Date: 06-12-2025  ###
### Version: 0.1      ###
#########################


import numpy as np
import sys
import pickle
import random
from collections import Counter, defaultdict

def input_parser():
    pickle_path = None
    if(len(sys.argv) > 1):
        if(sys.argv[1] == "-h" or sys.argv[1] == "--help"):
            print("Gebruik: python ngram.py pad_naar_pickle")
            sys.exit()
        else:
            pickle_path = sys.argv[1]
    return pickle_path


def file_reader(pickle_path):

    dbfile = open("Vocab_Text_Tokens", "rb")    
    db = pickle.load(dbfile)

    dbfile.close()

    vocabulary_list = list(db["vocabulary"])
    story_text = list(db["text_tokens"])

    # Omdat de tokens in een lijst van lijsten staat: 
    story_token_list = [token for word in story_text for token in word]

    return vocabulary_list, story_token_list

def train_ngram_model(story_token_list, n=3):
    """
    Returns: Geneste dictionary {(ngram): {next_token: probability}}
    """

    transitions = defaultdict(Counter)
    context_size = n - 1

    # Tel hoevaak een token voorkomt:
    for i in range(len(story_token_list) - context_size):
        ngram = tuple(story_token_list[i : i + context_size]) 
        next_token = story_token_list[i + context_size]
        transitions[ngram][next_token] += 1

    # Counts naar waarschijnlijkheden:
    model = {}
    for ngram, counter in transitions.items():
        total_count = sum(counter.values())
        model[ngram] = {}
        for token, count in counter.items():
            model[ngram][token] = count / total_count

    return model

def generate_text(model, n, length):

    # Random start binnen de text:
    current_context = random.choice(list(model.keys()))
    output = list(current_context)
    

    for i in range(length):
        if current_context in model:
            possible_next_tokens = list(model[current_context].keys())
            probabilities = list(model[current_context].values())
            
            # Kies volgende token op basis van de lijst met probabilities:
            next_token = np.random.choice(possible_next_tokens, p=probabilities)
            output.append(next_token)
            
            current_context = tuple(output[-(n-1):])
        else:
            break

    return "".join(output)


if __name__ == "__main__":
    pickle_path = input_parser()
    vocabulary_list, story_token_list = file_reader(pickle_path)

    model = train_ngram_model(story_token_list, n=3)

    generated_text = generate_text(model, n=3, length=100)
    print(generated_text)
