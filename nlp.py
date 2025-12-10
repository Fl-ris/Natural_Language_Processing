import pickle
from collections import Counter, defaultdict


def file_reader(text_path):
    word_list = []
    with open(text_path) as text:
        for line in text:
            words_split = line.strip().split()
            for i in words_split:
                i = list(i.lower())
                i.append(" ") # Een spatie na elk woord, anders alle text aan elkaar vast...
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

    if not counted_tokens:
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
            if i < len(word)-1 and word[i] == pair_to_merge[0] and word[i+1] == pair_to_merge[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_list.append(new_word)

    return new_word_list


            
def save_encoding(vocab, tokenized_text, enc_path="encoding.enc"):
    db = {"vocabulary": vocab, "text_tokens": tokenized_text}
    with open(enc_path, "wb") as f:
        pickle.dump(db, f)


# def save_to_file(vocabulary, text_tokens):
    
#     with open("encoding.enc", "w") as text:
#         text.write(str(vocabulary))
#     with open("story", "w") as text:
#         text.write(str(text_tokens))
    
#     db = {}
#     db['vocabulary'] = vocabulary
#     db['text_tokens'] = text_tokens
    
#     dbfile = open('Vocab_Text_Tokens', 'ab')
    
#     pickle.dump(db, dbfile)                    
#     dbfile.close()

#     pickle.dumps(vocabulary)
    
def load_encoding(enc_path):
    """Laad pickle met vocab en"""
    with open(enc_path, "rb") as f:
        return pickle.load(f)



# ----------------------------------------------------------------------
# N‑gram helpers (originally in ngram.py)
# ----------------------------------------------------------------------
def flatten_token_lists(token_lists):
    """Turn a list‑of‑lists (words → list of chars) into a flat token list."""
    return [token for word in token_lists for token in word]

def train_ngram_model(token_list, n=3):
    """Return a nested dict {(n‑1)-gram → {next_token: prob}}."""
    transitions = defaultdict(Counter)
    context = n - 1
    for i in range(len(token_list) - context):
        ctx = tuple(token_list[i : i + context])
        nxt = token_list[i + context]
        transitions[ctx][nxt] += 1

    model = {}
    for ctx, ctr in transitions.items():
        total = sum(ctr.values())
        model[ctx] = {tok: cnt / total for tok, cnt in ctr.items()}
    return model