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
from nlp import load_encoding, flatten_token_lists, train_ngram_model

def input_parser():
    """
    Verkrijg commandline argumenten:
    """
    
    pickle_path = None
    if(len(sys.argv) > 1):
        if(sys.argv[1] == "-h" or sys.argv[1] == "--help"):
            print("Gebruik: python ngram.py pad_naar_pickle")
            sys.exit()
        else:
            pickle_path = sys.argv[1]
    return pickle_path


def file_reader(pickle_path):
    """
    Lees pickle met vocab en text.
    """
    if not pickle_path:
        pickle_path = "encoding.enc"

    db = load_encoding(pickle_path)

    vocabulary_list = list(db["vocabulary"])
    story_text = db["text_tokens"]

    story_token_list = flatten_token_lists(story_text)

    return vocabulary_list, story_token_list



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
 