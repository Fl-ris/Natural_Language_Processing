#########################
### Author: Floris M  ###
### Date: 06-12-2025  ###
### Version: 0.1      ###
#########################


import numpy as np
import sys
import pickle

def input_parser():
    pickle_path = None
    if(len(sys.argv) > 1):
        if(sys.argv[1] == "-h" or sys.argv[1] == "--help"):
            print("Gebruik:")
        else:
            pickle_path = sys.argv[1]
    return pickle_path


def file_reader(pickle_path):
    """
    Docstring for file_reader
    
    :param vocab_path: pad naar het bestand met tokens
    :return vocubulary_list: een lijst met alle beschikbare tokens
    """

    dbfile = open('Vocab_Text_Tokens', 'rb')    
    db = pickle.load(dbfile)

    dbfile.close()

    vocabulary_list = list(db["vocabulary"])
    story_token_list = list(db["text_tokens"])

    return vocabulary_list, story_token_list


def prob_calc(vocabulary_list, story_token_list):
    print(story_token_list[0])



if __name__ == "__main__":
    pickle_path = input_parser()
    vocabulary_list, story_token_list = file_reader(pickle_path)

    prob_calc(vocabulary_list, story_token_list)
